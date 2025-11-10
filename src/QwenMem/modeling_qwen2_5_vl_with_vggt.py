import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2RMSNorm,
)
from configuration_qwen2_5_vl import Qwen2_5_VLConfig
from vggt.models.vggt import VGGT
from transformers.generation import GenerationMixin
from transformers.utils import (
    add_start_docstrings_to_model_forward,
)
from torch.nn import CrossEntropyLoss
from PIL import Image
from vggt.utils.load_fn import load_and_preprocess_images

@dataclass
class VGGTMergerConfig:
    input_dim: int = 2048
    output_dim: int = 1024
    patch_size: int = 14
    temporal_merge_size: int = 2
    spatial_merge_size: int = 2


class VGGTMerger(nn.Module):
    def __init__(self, config: VGGTMergerConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_merge_size = config.temporal_merge_size
        self.spatial_merge_size = config.spatial_merge_size
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        self.token_merge_in_dim = (
            self.input_dim * self.temporal_merge_size * (self.spatial_merge_size ** 2)
        )
        self.token_merge_ln_q = Qwen2RMSNorm(self.token_merge_in_dim)
        self.token_merge_mlp = nn.Sequential(
            nn.Linear(self.token_merge_in_dim, self.token_merge_in_dim),
            nn.GELU(),
            nn.Linear(self.token_merge_in_dim, self.output_dim),
        )

        self.vgg_to_language_ln_q = Qwen2RMSNorm(self.output_dim)
        self.vgg_to_language_mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def merge_tokens(self, tokens: torch.Tensor, images_shape: Tuple) -> torch.Tensor:
        H, W = images_shape[-2:]
        NUM_PATCH_H, NUM_PATCH_W = H // self.patch_size, W // self.patch_size
        B, F, S, D = tokens.shape

        pad_f = (-F) % self.temporal_merge_size
        if pad_f > 0:
            pad_tensor = tokens[:, -1:].expand(B, pad_f, S, D)
            tokens = torch.cat([tokens, pad_tensor], dim=1)
            F += pad_f

        tokens = tokens.view(B, F, NUM_PATCH_H, NUM_PATCH_W, D)
        usable_h = NUM_PATCH_H - (NUM_PATCH_H % self.spatial_merge_size)
        usable_w = NUM_PATCH_W - (NUM_PATCH_W % self.spatial_merge_size)
        tokens = tokens[:, :, :usable_h, :usable_w, :]
        
        tokens = tokens.view(
            B,
            F // self.temporal_merge_size,
            self.temporal_merge_size,
            usable_h // self.spatial_merge_size,
            self.spatial_merge_size,
            usable_w // self.spatial_merge_size,
            self.spatial_merge_size,
            D,
        )

        tokens = tokens.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        tokens = tokens.view(
            B,
            F // self.temporal_merge_size,
            usable_h // self.spatial_merge_size,
            usable_w // self.spatial_merge_size,
            self.temporal_merge_size * self.spatial_merge_size * self.spatial_merge_size * D,
        )

        tokens = self.token_merge_ln_q(tokens)
        tokens = self.token_merge_mlp(tokens)
        return tokens

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        images_shape: Tuple,
        media_type: str = "video",
    ) -> torch.Tensor:
        tokens = aggregated_tokens_list[-1][:, :, patch_start_idx:]
        
        if media_type == "images":
            tokens = tokens.repeat_interleave(2, dim=1)
        
        tokens = self.merge_tokens(tokens, images_shape)
        
        x = self.vgg_to_language_ln_q(tokens)
        x = self.vgg_to_language_mlp(x)
        
        return x


class VGGTEncoder(nn.Module):
    def __init__(
        self,
        config: VGGTMergerConfig,
        freeze: bool = True,
    ):
        super().__init__()
        
        self.config = config
        self.freeze = freeze
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Don't initialize VGGT yet - will be done later
        self.model = None
        self.merger = VGGTMerger(config)
        
    def _initialize_vggt(self):
        """Initialize VGGT model - called after parent model is loaded"""
        if self.model is None:
            self.model = VGGT.from_pretrained("facebook/VGGT-1B")
            self.model.camera_head = None
            self.model.track_head = None
            
            if self.freeze:
                for param in self.model.parameters():
                    param.requires_grad = False
    
    def _preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        return load_and_preprocess_images(images)

    def forward(
        self, 
        images: List[Image.Image], 
        media_type: str = "video"
    ) -> torch.Tensor:
        # Lazy initialization on first forward pass
        if self.model is None:
            self._initialize_vggt()
        
        images_tensor = self._preprocess_images(images).unsqueeze(0).to(self.device)
        images_shape = images_tensor.shape[-2:]
        
        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images_tensor)

        output = self.merger(
            aggregated_tokens_list=aggregated_tokens_list,
            patch_start_idx=patch_start_idx,
            images_shape=images_shape,
            media_type=media_type,
        )
        
        return output
    
    def to(self, device):
        """Override to method to update self.device"""
        self.device = device if isinstance(device, str) else str(device)
        return super().to(device)

class Qwen2_5_VLForConditionalGenerationWithVGGT(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        vggt_config = VGGTMergerConfig(output_dim=config.hidden_size)
        self.vggt = VGGTEncoder(vggt_config, freeze=True)

        self.vggt_fusion_weight = getattr(config, "vggt_fusion_weight", 0.3)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        self = cls(base.config)
        
        self.visual = base.visual
        self.model = base.model
        
        self.lm_head = base.lm_head
        
        self.vggt._initialize_vggt()
        self.vggt.to(base.device)

        self.rope_deltas = None
        
        return self

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_rope_index(self, *args, **kwargs):
        return self.model.get_rope_index(*args, **kwargs)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [Your existing implementation stays exactly the same]
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1],
                dtype=input_ids.dtype, device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                        
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                        
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        images_vggt: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                
                if images_vggt is not None:
                    try:
                        vggt_features = self.vggt.forward(images_vggt, media_type="images")
                        vggt_features = vggt_features.view(-1, vggt_features.shape[-1])
                        
                        if vggt_features.shape[0] == image_embeds.shape[0]:
                            image_embeds = image_embeds + self.vggt_fusion_weight * vggt_features
                    except Exception as e:
                        print(f"Warning: VGGT processing failed: {e}")
                
                mask = input_ids == self.config.image_token_id
                image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                mask = input_ids == self.config.video_token_id
                video_mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            images_vggt=images_vggt,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get number of images and videos per sample"""
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

__all__ = ["Qwen2_5_VLForConditionalGenerationWithVGGT"]