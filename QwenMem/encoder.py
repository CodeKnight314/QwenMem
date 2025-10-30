from dataclasses import dataclass
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from typing import List, Dict, Tuple
from PIL import Image
import torch
import torch.nn as nn


@dataclass
class VGGTMergerConfig:
    input_dim: int = 2048
    output_dim: int = 1024
    patch_size: int = 14
    temporal_merge_size: int = 2
    spatial_merge_size: int = 2


class VGGTNormalization(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.var(dim=-1, keepdim=True) + self.eps
        hidden_states = hidden_states * torch.rsqrt(variance)
        return hidden_states.to(input_dtype) * self.weight


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
        self.token_merge_ln_q = VGGTNormalization(self.token_merge_in_dim)
        self.token_merge_mlp = nn.Sequential(
            nn.Linear(self.token_merge_in_dim, self.token_merge_in_dim),
            nn.GELU(),
            nn.Linear(self.token_merge_in_dim, self.output_dim),
        )

        self.vgg_to_language_ln_q = VGGTNormalization(self.output_dim)
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
            NUM_PATCH_H // self.spatial_merge_size,
            NUM_PATCH_W // self.spatial_merge_size,
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
        model_name: str = "facebook/VGGT-1B",
        device: str = "auto",
        freeze: bool = True,
    ):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = VGGT.from_pretrained(model_name).to(device)
        self.merger = VGGTMerger(config).to(device)

        if freeze:
            self._freeze_model()

    def _preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        return load_and_preprocess_images(images)

    def _freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self, 
        images: List[Image.Image], 
        media_type: str = "video"
    ) -> torch.Tensor:
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