import torch
import torch.nn as nn


class BasicFusion(nn.Module):
    def __init__(
        self, text_dim: int, vis_dim: int, hidden_dim: int, num_heads: int = 8
    ):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, text_feats: torch.Tensor, vis_feats: torch.Tensor, pool: bool = False
    ):
        t = self.text_proj(text_feats)
        v = self.vis_proj(vis_feats)

        fused = self.norm(self.ff(torch.concat([t, v], dim=-1)))

        if pool:
            return fused.mean(dim=1)
        return fused


class XAttentionFusion(BasicFusion):
    def __init__(
        self, text_dim: int, vis_dim: int, hidden_dim: int, num_heads: int = 8
    ):
        super().__init__(text_dim, vis_dim, hidden_dim)

        self.cross = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, text_feats, vis_feats: torch.Tensor, pool: bool = False):
        t = self.text_proj(text_feats)
        v = self.vis_proj(vis_feats)

        fused, _ = self.cross(query=t, key=v, value=v)
        fused = self.norm(fused + t)
        fused = fused + self.ff(fused)

        if pool:
            return fused.mean(dim=1)
        return fused


class GatedFusion(BasicFusion):
    def __init__(
        self, text_dim: int, vis_dim: int, hidden_dim: int, num_heads: int = 8
    ):
        super().__init__(text_dim, vis_dim, hidden_dim)

        self.text_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.text_beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, text_feats: torch.Tensor, vis_feats: torch.Tensor, pool: bool = True
    ):
        t = self.text_proj(text_feats)
        v = self.vis_proj(vis_feats)

        t_context = t.mean(dim=1)

        gamma = torch.sigmoid(self.text_gamma(t_context)).unsqueeze(1)
        beta = self.text_beta(t_context).unsqueeze(1)

        fused = gamma * v + beta

        fused = self.norm(fused + v)

        if pool:
            return fused.mean(dim=1)
        return fused
