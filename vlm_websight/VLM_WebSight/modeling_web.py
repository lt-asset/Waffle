from dataclasses import dataclass
import inspect
import warnings
from typing import List, Optional, Tuple, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import (
    is_flash_attn_2_available
)
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from configuration_vmistral import VMistralConfig
from vision import SiglipVisionModel
from modeling_vmistral import *
from generation_utils import TreeBuilder, WebGenerationMixin
import time


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
    
@dataclass
class WebLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    html_tree: TreeBuilder = None


class WebAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: VMistralConfig, qk_layer_norms: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.qk_layer_norms = qk_layer_norms
        if self.qk_layer_norms:
            self.q_layer_norm = MistralRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_layer_norm = MistralRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.attention_dropout = config.attention_dropout

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        web_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use"
                " `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        if self.qk_layer_norms:
            query_states = self.q_layer_norm(query_states)
            key_states = self.k_layer_norm(key_states)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        web_attention_range = self.config.web_attention_range

        def split_tensor(tensor):
            if int(web_attention_range) == 8:
                return
            fraction = float(web_attention_range) / 8
            split_size_2 = int(self.num_heads * fraction)
            split_size_1 = self.num_heads - split_size_2
            return torch.split(tensor, [split_size_1, split_size_2], dim=1)
    
        if int(web_attention_range) != 8:
            query_states_1, query_states_2 = split_tensor(query_states)
            key_states_1, key_states_2 = split_tensor(key_states)
            value_states_1, value_states_2 = split_tensor(value_states)

            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                attn_output_1 = F.scaled_dot_product_attention(query_states_1, key_states_1, value_states_1, attn_mask=attention_mask)
                
                attn_output_2 = F.scaled_dot_product_attention(query_states_2, key_states_2, value_states_2, attn_mask=web_attention_mask)
            attn_output = torch.cat([attn_output_1, attn_output_2], dim=1)
        else:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask=web_attention_mask)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    

class WebFlashAttention2(WebAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """
        
class WebDecoderLayer(nn.Module):
    def __init__(self, config: VMistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            WebAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else WebFlashAttention2(config)
        )
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        web_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use"
                " `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            web_attention_mask=web_attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class WebPreTrainedModel(PreTrainedModel):
    config_class = VMistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WebDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = False
    

class WebModel(WebPreTrainedModel, VMistralModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: VMistralConfig
    """

    def __init__(self, config: VMistralConfig, vision_model=None):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.sliding_window = config.sliding_window

        self.embed_tokens = DecoupledEmbedding(
            num_embeddings=config.vocab_size,
            num_additional_embeddings=config.additional_vocab_size,
            embedding_dim=config.hidden_size,
            partially_freeze=config.freeze_text_layers,
            padding_idx=self.padding_idx,
        )

        # Load an uninitialized model and later in from_pretrained will load the pre-trained model -
        # this solves the losing of weights in `from_pretrained` on the main model
        self.vision_model = SiglipVisionModel(config.vision_config)

        # Dim projection - projecting from the vision dim to the text dim
        self.modality_projection = ModalityProjection(
            embed_dim_in=self.config.vision_config.hidden_size, embed_dim_out=self.config.hidden_size
        )

        # Perceiver Resampler
        if config.use_resampler:
            self.perceiver_resampler = PerceiverResampler(
                config.hidden_size,
                config.perceiver_config.resampler_depth,
                config.perceiver_config.resampler_n_heads,
                config.perceiver_config.resampler_head_dim,
                config.perceiver_config.resampler_n_latents,
                config.perceiver_config.qk_layer_norms_perceiver,
            )

        if config.use_resampler:
            self.image_seq_len = config.perceiver_config.resampler_n_latents
        else:
            self.image_seq_len = (
                config.vision_config.image_size // config.vision_config.patch_size
            ) ** 2  # TODO: pretty sure that does not work for CLIP models since there is the CLS token
        self.image_token_id = self.config.image_token_id

        self.layers = nn.ModuleList([WebDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False

        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        self.freeze_relevant_params(config)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        web_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VMistralBaseModelOutputWithPast]:
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype, device=input_ids.device)  # fp16 compatibility
            batch_size, num_images = pixel_values.size(0), pixel_values.size(1)
            
            # this change allows multi image in a single batch
            pixel_values = pixel_values.contiguous().view(batch_size, num_images, *pixel_values.shape[2:])
            # # Remove padding images - padding images are full 0.
            # real_images_inds = pixel_values.sum(dim=(-1, -2, -3)) != 0.0
            # print(real_images_inds)
            # pixel_values = pixel_values[real_images_inds]
            # # Get sequence from the vision encoder
            # print("shape_pixel", pixel_values.shape)
            image_hidden_states = self.vision_model(pixel_values=pixel_values).last_hidden_state

            # Modality projection
            image_hidden_states = self.modality_projection(image_hidden_states)

            if self.config.use_resampler:
                image_hidden_states = self.perceiver_resampler(image_hidden_states)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if past_key_values is None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist
            new_inp = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )
            inputs_embeds = new_inp["inputs_embeds"]

        # Can do add some token types embeddings here (image token vs text token)
        # something like inputs_embeds += self.token_types(token_types)

        # embed positions
        if (
            attention_mask is not None
            and hasattr(self.config, "_flash_attn_2_enabled")
            and self.config._flash_attn_2_enabled
            and past_key_values is not None
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        # We did not implement our model using Flash attn 2
        self.config._flash_attn_2_enabled = False
        if not getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            # attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
            web_attention_mask = web_attention_mask.unsqueeze(1)
            inverted_mask = 1.0 - web_attention_mask.to(inputs_embeds.dtype)
            web_attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), -1.e32
            )
            if input_ids is not None:
                bsz, L = input_ids.size()[:2]
                web_attention_mask = web_attention_mask[:, :, -L:, :]
        else:
            print("Exiting, wrong branch")
            exit()
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
            attention_mask[attention_mask == -float("inf")] = torch.finfo(self.dtype).min

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    web_attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    web_attention_mask=web_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, image_hidden_states]
                if v is not None
            )
        return VMistralBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            image_hidden_states=image_hidden_states,
        )
        
class WebForVisionText2Text(WebPreTrainedModel, WebGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, vision_model=None):
        super().__init__(config)
        self.model = WebModel(config, vision_model=vision_model)
        self.image_token_id = self.config.image_token_id
        self.lm_head = DecoupledLinear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            out_additional_features=config.additional_vocab_size,
            bias=False,
            partially_freeze=config.freeze_lm_head,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        web_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        html_tree = None,
    ) -> Union[Tuple, WebLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            web_attention_mask=web_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_hidden_states=image_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        # print(f"forward takes: {time.time()-start_time}")

        return WebLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            html_tree = html_tree
        )
        
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs
    ):
        image_hidden_states = kwargs.pop("image_hidden_states", None)
        if image_hidden_states is not None:
            kwargs["pixel_values"] = None
            
        inputs = prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        web_attention_mask, html_tree = None, kwargs.get("html_tree")
        
        if html_tree.web_attention_mask is None :
            attention_mask = inputs["attention_mask"]
            web_attention_mask = torch.tril(torch.ones((attention_mask.shape[-1], attention_mask.shape[-1]), dtype = attention_mask.dtype)).unsqueeze(0)
            html_tree.web_attention_mask = web_attention_mask
        else:
            html_tree = kwargs.get("html_tree")
            input_ids = inputs["input_ids"]
            tokenizer = html_tree.tokenizer
            cur_decoded_token = tokenizer.convert_tokens_to_string([" "]+tokenizer.convert_ids_to_tokens(input_ids[:,-1]))
            web_attn_range = html_tree.update_buffer([cur_decoded_token])
            bsz, L = html_tree.web_attention_mask.size()[:2]
            web_attention_mask = torch.zeros((bsz, L + 1, L + 1)).type_as(html_tree.web_attention_mask)
            web_attention_mask[:, :L, :L] = html_tree.web_attention_mask
            web_attn_range = torch.tensor(list(range(67))+[i + 67 for i in web_attn_range], dtype = web_attention_mask.dtype)
            web_attention_mask[:, -1, web_attn_range] = 1
            html_tree.web_attention_mask = web_attention_mask
            if html_tree.input_ids is None :
                html_tree.input_ids = input_ids
            else:
                html_tree.input_ids = torch.cat((html_tree.input_ids, input_ids), dim = 1)
        
        unwanted_kwargs = ["token_type_ids"]
        inputs.update({
            "web_attention_mask": web_attention_mask.to(inputs['attention_mask'].device),
            "html_tree": html_tree,
        })
        for kwarg in unwanted_kwargs:
            inputs.pop(kwarg, None)

        return inputs