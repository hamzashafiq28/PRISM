import re
import torch
import torch.nn as nn

class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {'mm_projector_type': 'identity'}

class SimpleResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    mlp_gelu_match = re.match('^mlp(\\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    if projector_type == 'identity':
        return IdentityMap()
    raise ValueError(f'Unknown projector type: {projector_type}')

class SoftTokenMLP(nn.Module):

    def __init__(self, d_in: int, d_llm: int, n_tokens: int, hidden: int=1024):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_llm = d_llm
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, n_tokens * d_llm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return self.net(x).view(B, self.n_tokens, self.d_llm)

def build_soft_token_projector(d_in: int, d_llm: int, n_tokens: int, hidden: int=1024) -> SoftTokenMLP:
    return SoftTokenMLP(d_in=d_in, d_llm=d_llm, n_tokens=n_tokens, hidden=hidden)
