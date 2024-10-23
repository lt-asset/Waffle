import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
from VLM_WebSight import modeling_web
from VLM_WebSight.generation_utils import TreeBuilder


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
    x = processor.image_processor.rescale(x, scale=1 / 255)
    x = processor.image_processor.normalize(
        x,
        mean=processor.image_processor.image_mean,
        std=processor.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x


processor = AutoProcessor.from_pretrained("HuggingFaceM4/VLM_WebSight_finetuned")

model_dir = "../models/Waffle-VLM-WebSight"
model = modeling_web.WebForVisionText2Text.from_pretrained(model_dir, torch_dtype=torch.bfloat16).cuda()
web_attention_range = 2     # use 2/8 = 1/4 attention heads for hierarchical attentio 
model.config.update({"web_attention_range": web_attention_range})

image_seq_len = model.config.perceiver_config.resampler_n_latents
BOS_TOKEN = processor.tokenizer.bos_token
BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

model.eval()

image_file = '../WebSight-Test/test-935.png'
image = Image.open(image_file)

inputs = processor.tokenizer(
    f"{BOS_TOKEN}<fake_token_around_image>{'<image>' * image_seq_len}<fake_token_around_image>",
    return_tensors="pt",
    add_special_tokens=False,
)
inputs["pixel_values"] = processor.image_processor([image], transform=custom_transform).to(dtype=torch.bfloat16)
inputs_for_generation = {k: v.cuda() for k, v in inputs.items()}
inputs_for_generation["web_attention_mask"] = None
inputs_for_generation["html_tree"] = TreeBuilder(processor.tokenizer)
inputs_for_generation["html_tree"].web_attention_mask = inputs_for_generation["web_attention_mask"]

generated_ids = model.generate(
    **inputs_for_generation, bad_words_ids=BAD_WORDS_IDS, max_length=2048, 
    num_return_sequences=1, do_sample=True, top_p=0.95, temperature=0.2
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
