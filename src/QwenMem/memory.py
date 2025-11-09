import torch
import torch.nn as nn 

class QwenMemory(nn.Module):
    def __init__(self, dim=1024, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        self.state = nn.Parameter(torch.randn(1024, 1024))

        self.state_tokens_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.tokens_state_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        mlp_dim = int(dim * mlp_ratio)
        self.state_ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        self.tokens_ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        self.norm_state_1 = nn.LayerNorm(dim)
        self.norm_state_2 = nn.LayerNorm(dim)
        self.tokens_norm_1 = nn.LayerNorm(dim)
        self.tokens_norm_2 = nn.LayerNorm(dim)

    def forward(self, visual_tokens):
        batch_size = visual_tokens.shape[0]
        
        # First attention: tokens attend to state
        # Add batch dimension to state for attention
        s = self.norm_state_1(self.state).unsqueeze(0)  # (1, 1024, 1024)
        s = s.expand(batch_size, -1, -1)  # (batch_size, 1024, 1024)
        t = self.tokens_norm_1(visual_tokens)  # (batch_size, seq_len, 1024)

        state_attn_output, _ = self.tokens_state_attn(s, t, t)  # (batch_size, 1024, 1024)
        # Remove batch dimension and aggregate if needed
        self.state.data += state_attn_output.mean(dim=0)  # (1024, 1024)
        self.state.data += self.state_ffn(self.norm_state_2(self.state))

        # Second attention: state attends to tokens
        s = self.norm_state_2(self.state).unsqueeze(0)  # (1, 1024, 1024)
        s = s.expand(batch_size, -1, -1)  # (batch_size, 1024, 1024)
        t = self.tokens_norm_2(visual_tokens)  # (batch_size, seq_len, 1024)

        tokens_attn_output, _ = self.state_tokens_attn(t, s, s)  # (batch_size, seq_len, 1024)
        tokens = visual_tokens + tokens_attn_output
        tokens = tokens + self.tokens_ffn(self.tokens_norm_2(tokens))

        return tokens
