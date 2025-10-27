import torch
import torch.nn as nn
from encoder import *
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
)
import numpy as np


# Cross Attention
# Gated Fusion


class QwenVLMem(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device

        self.spatial_encoder = DinoV2Encoder().to(self.device)

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct"
        ).to(self.device)

    def forward(self, x: np.ndarray, text: str):
        spatial_feats = self.spatial_encoder(x, train=False).to(self.device)

        inputs = self.processor(images=x, text=text, return_tensors="pt").to(
            self.device
        )

        outputs = self.model(**inputs)
        vision_text_feats = outputs.encoder_last_hidden_state

        return spatial_feats, vision_text_feats


class QwenMem(nn.Module):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qween/Qwen2.5-3B", output_hidden_states=True
        ).to(self.device)

    def forward(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)

        text_feats = outputs.hidden_states[-1]
        pooled_feats = text_feats.mean(dim=1)

        return text_feats, pooled_feats
