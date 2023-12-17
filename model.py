import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
# party id 57861845

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)
        
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.multi_head_attn(q, k ,v)
        return attn_output

class PreNorm(nn.Module):
    def __init__(self, fn, embed_dim):
        super().__init__()
        self.fn = fn 
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x, **kwargs):
        return self.f(self.ln(x), **kwargs)
        
class Project_Embedding(nn.Module):
    def __init__(self, img_size, patch_size, n_c, embed_dim):
        super().__init__()
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.fc = nn.Linear(n_c * patch_size ** 2, embed_dim)
        
    def forward(self, x):
        return self.fc(self.rearrange(x))
    
class MLP_Head(nn.Sequential):
    def __init__(self, embed_dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        

class VIT(nn.Module):
    def __init__(self, img_dim, patch_dim, embed_dim, n_channels , n_layers, n_heads, dropout=0.1):
        super().__init__()
        
        self.embed_