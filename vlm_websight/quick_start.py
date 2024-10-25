import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
from utils import TreeBuilder


def convert_to_rgb(image):
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def inference_vlm_websight(image_path, html_path):
    
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

    model_dir = "lt-asset/Waffle_VLM_WebSight"
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    # model = modeling_web.WebForVisionText2Text.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
    
    assert model.config.web_attention_range == 2, "Waffle_VLM_WebSight is trained with hierarchical attention applied to 2 / 8 heads"
    # use 2/8 = 1/4 attention heads for hierarchical attention (as described in paper)
    model.eval()

    image_seq_len = model.config.perceiver_config.resampler_n_latents
    BOS_TOKEN = processor.tokenizer.bos_token
    BAD_WORDS_IDS = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    image = Image.open(image_path)
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
        num_return_sequences=1, do_sample=False
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    with open(html_path, 'w') as wp:
        wp.write(generated_text)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--html_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()

    inference_vlm_websight(args.image_path, args.html_path)
