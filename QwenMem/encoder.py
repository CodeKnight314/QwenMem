import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import numpy as np


class DinoV2Encoder(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)

    def forward(self, x: np.ndarray, train: bool = True):
        inputs = self.processor(images=x, return_tensors="pt")
        if not train:
            with torch.no_grad():
                outputs = self.model(**inputs).last_hidden_state
        else:
            outputs = self.model(**inputs).last_hidden_state
        return outputs[:, 1:, :]


class CLIPVisualEncoder(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(
            self.device
        )

    def forward(self, images: torch.Tensor, pool: bool = False):
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.device)
        else:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            pixel_values = inputs.pixel_values

        outputs = self.model(pixel_values, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state
        pooled = outputs.pooler_output

        if pool:
            return pooled
        return last_hidden
