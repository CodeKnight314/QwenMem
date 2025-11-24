import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from functools import partial
from .blocks import DecoderBlock

class CUT3RStyleMemory(nn.Module):
    def __init__(
        self,
        state_size: int = 768,
        state_dim: int = 768,
        visual_dim: int = 1024,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        state_pe: str = "2d",
        rope=None,
    ):
        super().__init__()

        self.state_size = state_size
        self.state_dim = state_dim
        self.visual_dim = visual_dim
        self.num_heads = num_heads
        self.depth = depth
        self.state_pe = state_pe

        self.register_tokens = nn.Embedding(state_size, state_dim)
        self.decoder_embed_state = nn.Linear(state_dim, state_dim, bias=True)
        self.decoder_embed_visual = nn.Linear(visual_dim, state_dim, bias=True)

        self.dec_blocks_state = nn.ModuleList([
            DecoderBlock(
                dim=state_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                norm_mem=True,
                rope=rope
            )
            for _ in range(depth)
        ])

        self.dec_blocks = nn.ModuleList([
            DecoderBlock(
                dim=state_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                norm_mem=True,
                rope=rope
            )
            for _ in range(depth)
        ])

        # Freeze the decoder blocks
        for param in self.dec_blocks_state.parameters():
            param.requires_grad = False
        for param in self.dec_blocks.parameters():
            param.requires_grad = False

        self.dec_norm_state = norm_layer(state_dim)
        self.dec_norm_visual = norm_layer(state_dim)

    def _init_position_encoding(self):
        if self.state_pe == "2d":
            width = int(self.state_size ** 0.5)
            width = width + 1 if width % 2 == 1 else width
            pe = torch.tensor(
                [[i // width, i % width] for i in range(self.state_size)],
                dtype=torch.float32
            )
        elif self.state_pe == "1d":
            pe = torch.tensor([[i, i] for i in range(self.state_size)], dtype=torch.float32)
        else:
            self.register_buffer('state_pos', None)
            return
            
        self.register_buffer('state_pos', pe)

    def init_state(self, batch_size: int, device: torch.device):
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=device)
        )
        state_feat = state_feat.unsqueeze(0).expand(batch_size, -1, -1)
        state_feat = self.decoder_embed_state(state_feat)

        if hasattr(self, 'state_pos') and self.state_pos is not None:
            state_pos = self.state_pos.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        else: 
            state_pos = None

        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat: torch.Tensor,
        state_pos: Optional[torch.Tensor],
        visual_feat: torch.Tensor,
        visual_pos: Optional[torch.Tensor],
    ):
        visual_feat = self.decoder_embed_visual(visual_feat)
        final_output = [(state_feat, visual_feat)]

        for blk_state, blk_visual in zip(self.dec_blocks_state, self.dec_blocks):
            prev_state, prev_visual = final_output[-1]

            new_state, _ = blk_state(prev_state, prev_visual, state_pos, visual_pos)
            new_visual, _ = blk_visual(prev_visual, prev_state, visual_pos, state_pos)
            final_output.append((new_state, new_visual))

        updated_state = self.dec_norm_state(final_output[-1][0])
        updated_visual = self.dec_norm_visual(final_output[-1][1])
        return updated_state, updated_visual

    def forward(
        self,
        visual_feat: torch.Tensor,
        state_feat: Optional[torch.Tensor] = None,
        visual_pos: Optional[torch.Tensor] = None,
        state_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = visual_feat.shape[0]
        device = visual_feat.device

        if state_feat is None:
            state_feat, state_pos = self.init_state(batch_size, device)

        self._init_position_encoding()
        updated_state, updated_visual = self._recurrent_rollout(
            state_feat, state_pos, visual_feat, visual_pos
        )

        return updated_state, updated_visual

    def load_state_dict(self, checkpoint_path: str, strict: bool = False):
        """
        Load only transformer blocks (decoder blocks) from checkpoint.
        Embeddings and norms are not loaded.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        loaded_keys = []
        
        # Only load decoder blocks (transformer blocks)
        for i in range(self.depth):
            # Load dec_blocks_state
            prefix_src = f"dec_blocks_state.{i}."
            for k, v in state_dict.items():
                if k.startswith(prefix_src):
                    if k in self.state_dict():
                        if v.shape == self.state_dict()[k].shape:
                            self.state_dict()[k].copy_(v)
                            loaded_keys.append(k)
                        else:
                            print(f"Shape mismatch for {k}: "
                                  f"{v.shape} vs {self.state_dict()[k].shape}")
                            
            # Load dec_blocks
            prefix_src = f"dec_blocks.{i}."
            for k, v in state_dict.items():
                if k.startswith(prefix_src):
                    if k in self.state_dict():
                        if v.shape == self.state_dict()[k].shape:
                            self.state_dict()[k].copy_(v)
                            loaded_keys.append(k)
                        else:
                            print(f"Shape mismatch for {k}: "
                                  f"{v.shape} vs {self.state_dict()[k].shape}")
        
        print(f"Loaded {len(loaded_keys)} transformer block parameters from CUT3R checkpoint")
        
        if strict:
            # Only check decoder blocks for strict mode
            decoder_keys = {k for k in self.state_dict().keys() 
                           if k.startswith('dec_blocks_state.') or k.startswith('dec_blocks.')}
            loaded_set = set(loaded_keys)
            missing = decoder_keys - loaded_set
            if missing:
                print(f"Missing decoder block keys: {missing}")
                    
        return loaded_keys

if __name__ == "__main__":
    model = CUT3RStyleMemory()
    model.load_state_dict(checkpoint_path="checkpoint.pth")
    model.eval()

    visual_feat = torch.randn(1, 768, 1024)
    state_feat, state_pos = model.init_state(1, "cpu")
    updated_state, updated_visual = model(visual_feat, state_feat, state_pos)
    print(updated_state.shape, updated_visual.shape)