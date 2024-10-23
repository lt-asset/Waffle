import torch
from PIL import Image
import random
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format, pad
from datasets import load_from_disk

random.seed(7)

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class HierarchicalContrastiveHtmlDatasetInDict(torch.utils.data.Dataset):
    def __init__(self, processor, data_path, max_len=2048, dtype = torch.bfloat16, seq_len=64):
        super().__init__()
        
        self.data = load_from_disk(data_path)
        self.data_path = data_path
        self.max_len = max_len
        self.processor = processor
        self.BOS_TOKEN = self.processor.tokenizer.bos_token
        self.PAD_TOKEN_ID = self.processor.tokenizer(self.processor.tokenizer.pad_token, add_special_tokens=False).input_ids[0]
        self.BAD_WORDS_IDS = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
        self.dtype = dtype
        self.seq_len = seq_len
        self.counter = 0
        self.cnt = 0
        print(f"Successfully loaded data: {len(self.data)}")
        
    def __getitem__(self, idx):
        return_dict = {}
        max_len = 2048
        item = self.data[idx]

        for sample_index in [str(i) for i in list(range(4))]:
            image = item[f"image_{sample_index}"]
            input_ids = item[sample_index]["input_ids"]
            input_ids = torch.LongTensor(input_ids)
            # 66 is hardcoded for the vlm-websight
            if sum(torch.isin(input_ids, torch.LongTensor(self.BAD_WORDS_IDS))).item() != 66:
                print(f"problematic, cnt is {self.cnt}", sum(torch.isin(input_ids, torch.LongTensor(self.BAD_WORDS_IDS))).item())
                self.cnt += 1
                return self.__getitem__((idx+1)%self.__len__())
            
            modified_attentions = torch.LongTensor(item[sample_index]["attention_mask"])
            labels = torch.LongTensor(item[sample_index]["labels"])
            attnetion_range = item[sample_index]["attnetion_range"]
            prompt_len = item[sample_index]["prompt_len"]
            pad_token_id = self.processor.tokenizer.pad_token_id
            pixel_values = self.processor.image_processor([image], transform=self.custom_transform).to(dtype=self.dtype)
            
            # Create the modified input_ids with the specified changes
            text_input_ids = torch.cat([input_ids[:1], input_ids[67:], torch.full((66,), pad_token_id)])
            text_attention_mask = torch.zeros(max_len,max_len)
            text_attention_mask[:1+attnetion_range, :1+attnetion_range] = torch.tril(torch.ones(1+attnetion_range,1+attnetion_range))
            text_attention_mask[1:1+attnetion_range, 1:1+attnetion_range] = modified_attentions[:attnetion_range, :attnetion_range]
            text_attention_mask_1d = torch.ones(text_input_ids.size())
            text_attention_mask_1d = text_attention_mask_1d.masked_fill(text_input_ids.eq(self.processor.tokenizer.pad_token_id), 0.0)
            
            attention_mask = torch.zeros(max_len,max_len)
            attention_mask[:prompt_len+attnetion_range, :prompt_len+attnetion_range] = torch.tril(torch.ones(prompt_len+attnetion_range,prompt_len+attnetion_range))
            attention_mask[prompt_len:prompt_len+attnetion_range, prompt_len:prompt_len+attnetion_range] = modified_attentions[:attnetion_range, :attnetion_range]
            attention_mask_1d = torch.ones(input_ids.size())
            attention_mask_1d = attention_mask_1d.masked_fill(input_ids.eq(self.processor.tokenizer.pad_token_id), 0.0)
            
            try:
                assert input_ids.shape[0] ==  max_len
                assert text_input_ids.shape[0] ==  max_len
                assert labels.shape[0] ==  max_len
                
                if torch.isclose(torch.zeros(1).to(self.dtype),torch.sum(pixel_values.sum(dim=(-1, -2, -3)))):
                    return self.__getitem__((idx+1)%len(self.data))
            except:
                print(f"???? invalid inputs, with cnt {self.cnt}")
                self.cnt+=1
                return self.__getitem__((idx+1)%len(self.data))
            
            return_dict[sample_index] = {
                "idx":idx,
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'text_input_ids': text_input_ids,
                'labels': labels,
                'web_attention_mask': attention_mask.type(torch.bool),
                'web_attention_mask_text': text_attention_mask.type(torch.bool),
                'attention_mask': attention_mask_1d.type(torch.bool),
                'text_attention_mask': text_attention_mask_1d.type(torch.bool),
            }
        return return_dict
    
    def __len__(self):
        return len(self.data)
    
    def custom_transform(self, x):
        x = convert_to_rgb(x)
        x = to_numpy_array(x)
        x = resize(x, (960, 960), resample=PILImageResampling.BILINEAR)
        x = self.processor.image_processor.rescale(x, scale=1 / 255)
        x = self.processor.image_processor.normalize(
            x,
            mean=self.processor.image_processor.image_mean,
            std=self.processor.image_processor.image_std
        )
        x = to_channel_dimension_format(x, ChannelDimension.FIRST)
        x = torch.tensor(x)
        return x
    