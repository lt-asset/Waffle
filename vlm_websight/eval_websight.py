import torch
import os
from PIL import Image
from transformers import AutoProcessor
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
import time
from VLM_WebSight import modeling_web
from VLM_WebSight.generation_utils import TreeBuilder
import traceback

def convert_to_rgb(image):
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

# The processor is the same as the Idefics processor except for the BILINEAR interpolation,
# so this is a hack in order to redefine ONLY the transform method
def custom_transform(x):
    x = convert_to_rgb(x)
    x = to_numpy_array(x)
    x = resize(x, (960, 960), resample=PILImageResampling.BILINEAR)
    x = PROCESSOR.image_processor.rescale(x, scale=1 / 255)
    x = PROCESSOR.image_processor.normalize(
        x,
        mean=PROCESSOR.image_processor.image_mean,
        std=PROCESSOR.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x


torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda:0")
PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/VLM_WebSight_finetuned",
)
web_attention_range = 2
MODEL = modeling_web.VMistralForVisionText2Text.from_pretrained(
    f"Your finetuned model path here",
    torch_dtype=torch.bfloat16,
).to(DEVICE)

MODEL.config.update({"web_attention_range": web_attention_range})
image_seq_len = MODEL.config.perceiver_config.resampler_n_latents
BOS_TOKEN = PROCESSOR.tokenizer.bos_token
BAD_WORDS_IDS = PROCESSOR.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

MODEL.eval()
print(MODEL.config)

img_dir = "Inference dir containing the test images"
filenames = sorted(os.listdir(img_dir))
inputs = PROCESSOR.tokenizer(
    f"{BOS_TOKEN}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
    return_tensors="pt",
    add_special_tokens=False,
)

save_directory = f"your save path here"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    
ctr = 0
for filename in filenames:
    start_time = time.time()
    idx = filename.split(".")[0]
    print(ctr, idx)
    file_name = f"{idx}.html"
    file_path = os.path.join(save_directory, file_name)
    if os.path.exists(file_path): 
        ctr += 1
        continue
    
    image = Image.open(f'{img_dir}{filename}')
    inputs["pixel_values"] = PROCESSOR.image_processor([image], transform=custom_transform).to(dtype=torch.bfloat16)
    
    inputs_for_generation = {k: v.to(DEVICE) for k, v in inputs.items()}
    inputs_for_generation["web_attention_mask"] = None
    inputs_for_generation["html_tree"] = TreeBuilder(PROCESSOR.tokenizer)
    inputs_for_generation["html_tree"].web_attention_mask = inputs_for_generation["web_attention_mask"]
    
    try:
        generated_ids = MODEL.generate(**inputs_for_generation, bad_words_ids=BAD_WORDS_IDS, max_length=2048, num_beams=1, do_sample=True, top_p = 0.95, temperature=0.2)
        generated_text = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
        with open(file_path, "w") as file:
            file.write(generated_text)
        ctr += 1
        print(time.time()- start_time)
        
    except Exception as e:
        print(e)
        traceback.print_exc()
        ctr += 1
        continue
