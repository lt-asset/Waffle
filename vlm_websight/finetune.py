import torch
from transformers import AutoProcessor
import os
import time
import json
import sys
from transformers import get_cosine_schedule_with_warmup
from transformers.deepspeed import HfDeepSpeedConfig
from peft import LoraConfig, get_peft_model
from VLM_WebSight import modeling_web
import deepspeed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import HierarchicalContrastiveHtmlDatasetInDict

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()

ds_config = json.load(open('ds_config.json', 'r'))
dschf = HfDeepSpeedConfig(ds_config)
torch.manual_seed(7)

PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
)
BAD_WORDS_IDS = PROCESSOR.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

model = modeling_web.WebForVisionText2Text.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
    torch_dtype=torch.bfloat16,
)
model.config.update({"web_attention_range": 2})

target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "output_proj", "o_proj"]
config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
)

model.model.gradient_checkpointing_enable({"use_reentrant": False})
model = get_peft_model(model, config)
model.model.model.vision_model.vision_model.encoder.gradient_checkpointing = True
model.print_trainable_parameters()

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Model Loaded, Start Loading Data, Parameters: {}'.format(sum(p.numel() for p in model.parameters())))

train_dataset = HierarchicalContrastiveHtmlDatasetInDict(
    PROCESSOR,
    'data_path', max_len=2048, dtype = torch.bfloat16
)

# Continue setting up your optimizer with the parameters that require gradients
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=500, num_training_steps=int(1.2 * len(train_dataset) / int(ds_config["train_batch_size"]))
)

engine, _, train_dataloader, _ = deepspeed.initialize(model=model, training_data=train_dataset, config_params=ds_config, optimizer=optimizer, lr_scheduler=scheduler)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print('Train Dataset: {}, Train Dataloader: {}'.format(len(train_dataset), len(train_dataloader)))

gpu_batch_size = int(ds_config["train_micro_batch_size_per_gpu"])
save_dir = f"write your save for the model here"
os.makedirs(save_dir,exist_ok=True)

engine.module.train()
group_size = 4
for epoch in range(1):
    start_time = time.time()
    train_loss = []
    train_contra_loss = []
    train_lm_loss = []
    for i, data in enumerate(train_dataloader):
        im_embeddings, text_embeddings = [], []
        lm_loss = 0
        for key in ["0", "1", "2", "3"]:
            # compute the image embeddings and average them
            pixel_values = data[key]["pixel_values"].squeeze(1).to(engine.device)
            input_ids = data[key]["input_ids"].to(engine.device)
            attention_mask = data[key]["attention_mask"].to(engine.device)
            web_attention_mask = data[key]["web_attention_mask"].to(engine.device)
            text_input_ids = data[key]["text_input_ids"].to(engine.device)
            text_attention_mask = data[key]["text_attention_mask"].to(engine.device)
            web_attention_mask_text = data[key]["web_attention_mask_text"].to(engine.device)
            labels = data[key]["labels"].to(engine.device)
            
            output_loss = engine.module(pixel_values = pixel_values,input_ids = input_ids, attention_mask = attention_mask, web_attention_mask = web_attention_mask, labels = labels).loss
            lm_loss = lm_loss + output_loss
            
            batch_size, num_images = pixel_values.size(0), pixel_values.size(1)
            pixel_values = pixel_values.contiguous().view(batch_size, num_images, *pixel_values.shape[2:])
            image_hidden_states = engine.module.model.model.vision_model(pixel_values=pixel_values).last_hidden_state
            image_hidden_states = engine.module.model.model.modality_projection(image_hidden_states)
            if engine.module.model.config.use_resampler:
                image_hidden_states = engine.module.model.model.perceiver_resampler(image_hidden_states)
            im_embeddings.append(torch.mean(image_hidden_states,dim=1))
            
            # compute the last hidden states of the last valid text token
            text_hidden_states = engine.module.model.model(input_ids = text_input_ids, attention_mask = text_attention_mask, web_attention_mask = web_attention_mask_text).last_hidden_state
            
            last_valid_token_indices = torch.arange(text_attention_mask.size(1)).to(engine.device) * text_attention_mask
            last_valid_token_indices = torch.argmax(last_valid_token_indices, dim=1)
            start_index = engine.module.model.model.image_seq_len + 2
            start_indices = torch.full_like(last_valid_token_indices, start_index)
            actual_start_indices = torch.min(torch.clamp(start_indices, 0, text_hidden_states.size(1) - 1), last_valid_token_indices)
            
            # Create a mask that includes positions from the start index to the last valid token
            mask = (torch.arange(text_hidden_states.size(1), device=engine.device)[None, :] >= actual_start_indices[:, None]) & \
                (torch.arange(text_hidden_states.size(1), device=engine.device)[None, :] <= last_valid_token_indices[:, None])
            masked_hidden_states = text_hidden_states * mask.unsqueeze(-1)
            mean_hidden_states = masked_hidden_states.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            text_embeddings.append(mean_hidden_states)
            
        im_matrix = torch.stack(im_embeddings)  # Shape [4, batch_size, hidden_size]
        text_matrix = torch.stack(text_embeddings)  # Shape [4, batch_size, hidden_size]
        im_matrix = im_matrix.permute(1, 0, 2)
        text_matrix = text_matrix.permute(1, 0, 2) # Shape [batch_size, 4, hidden_size]
        sim_matrix = torch.nn.LogSoftmax(dim=-1)(torch.nn.functional.cosine_similarity(im_matrix.unsqueeze(2), text_matrix.unsqueeze(1), dim=-1))
        
        # Shape [batch_size, 4, 4]
        gt = torch.eye(group_size).type_as(im_matrix) * 2 + torch.ones((group_size, group_size)).type_as(im_matrix) # Shape [batch_size, group_size, group_size]
        gt = torch.nn.functional.normalize(gt, dim=-1, p=1).view(-1, group_size).repeat(gpu_batch_size,1)  
        contra_loss = torch.nn.KLDivLoss(reduction="sum")(sim_matrix.view(-1, group_size), gt)
        lm_loss = lm_loss / group_size
        loss = 0.1 * contra_loss.mean() + lm_loss
        
        engine.backward(loss)
        engine.step()
        train_loss.append(loss.item())
        train_contra_loss.append(contra_loss.item())
        train_lm_loss.append(lm_loss.item())
        
        if i % 100 == 0:
            torch.cuda.empty_cache()
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print('epoch: {}, step: {}/{}, loss: {}, Con_loss: {}, Lm_loss: {}, lr: {}, time: {}s'.format(
                    epoch + 1, i, len(train_dataloader), round(sum(train_loss) / len(train_loss), 4), round(sum(train_contra_loss) / len(train_contra_loss), 4), 
                    round(sum(train_lm_loss) / len(train_lm_loss), 4), round(engine.optimizer.param_groups[0]['lr'], 8), int(time.time() - start_time)
                ))
            start_time = time.time()
            train_loss = []
        if i % 5000 == 0 and i > 0:
            engine.save_checkpoint(save_dir)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print('checkpoint saved')

    engine.save_checkpoint(save_dir)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('checkpoint saved')