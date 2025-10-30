from QwenMem.vggt.models.vggt import VGGT
from QwenMem.vggt.utils.load_fn import load_and_preprocess_images
from typing import List, Dict
from PIL import Image
import torch
from typing import Tuple
import torch.nn as nn

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
    def __init__(self, config):
        super().__init__()

        self.patch_size = config.patch_size
        self.temporal_merge_size = config.temporal_merge_size
        self.spatial_merge_size = config.spatial_merge_size

        self.input_size = self.input_dim * self.temporal_merge_size * self.spatial_merge_size * self.spatial_merge_size
        self.token_merge_ln_q = VGGTNormalization(self.input_size)
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

        self.vgg_to_language_ln_q = VGGTNormalization(self.output_dim)
        self.vgg_to_language_mlp = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def merge_tokens(self, tokens: torch.Tensor, images_shape: Tuple) -> torch.Tensor:
        H, W = images_shape[-2:]
        NUM_H_PATCHES = H // self.patch_size
        NUM_W_PATCHES = W // self.patch_size

        B, F, S, D = tokens.shape
        tokens = tokens.view(B, F, S, NUM_H_PATCHES, NUM_W_PATCHES, D)
        tokens = tokens.view(
            B, 
            F//self.temporal_merge_size, 
            self.temporal_merge_size,
            NUM_H_PATCHES//self.spatial_merge_size,
            self.spatial_merge_size,
            NUM_W_PATCHES//self.spatial_merge_size,
            self.spatial_merge_size,
            D,
        )

        tokens = tokens.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()

        tokens = self.token_merge_ln_q(tokens)
        tokens = self.token_merge_mlp(tokens)
        return tokens

    def forward(self, tokens: torch.Tensor, images_shape: Tuple) -> torch.Tensor:
        tokens = self.merge_tokens(tokens, images_shape)
        tokens = self.vgg_to_language_ln_q(tokens)
        tokens = self.vgg_to_language_mlp(tokens)
        return tokens

class VGGTEncoder: 
    def __init__(self, config: Dict, model_name: str = "facebook/VGGT-1B", device: str = "cuda", freeze: bool = True):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = VGGT.from_pretrained(model_name).to(device)
        self.merger = VGGTMerger(config)
        if freeze:
            self._freeze_model()

    def _preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        return load_and_preprocess_images(images)

    def _freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _get_vgg_embeds(self, images: torch.Tensor, images_shape: Tuple) -> torch.Tensor:
        vgg_aggregator = self.model.aggregator
        output_list, patch_start_idx = vgg_aggregator(images)
        return self.merger(output_list, images_shape)

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        images = self._preprocess_images(images)
        images_shape = images.shape[-2:]
        input = self._get_vgg_embeds(images, images_shape)
        return input