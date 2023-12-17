import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0)
        
        self.q = torch.nn.Linear(embed_dim, embed_dim)
        self.k = torch.nn.Linear(embed_dim, embed_dim)
        self.v = torch.nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.multi_head_attn(q, k ,v)
        return attn_output

class PreNorm(nn.Module):
    def __init__(self, embed_dim, fn):
        super().__init__()
        self.fn = fn 
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return res
        
class Project_Embedding(nn.Module):
    def __init__(self, patch_size, n_c, embed_dim):
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
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        

class VIT(nn.Module):
    def __init__(self, img_dim, patch_dim, embed_dim, n_channels , n_layers, n_heads, out_dim, dropout=0.1):
        super().__init__()
        
        self.height = img_dim[0]
        self.width = img_dim[1]
        self.patch_size = patch_dim
        self.embed_size = embed_dim
        self.channels = n_channels
        self.n_layers = n_layers
        self.heads = n_heads
   
        self.patch_embedings = Project_Embedding(patch_dim, n_channels, embed_dim)
        
        num_patches = img_dim[0] * img_dim[1]//(patch_dim**2)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embed_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        
        self.layers = nn.ModuleList([])
        
        for _ in range(n_layers):
            encoder_block = nn.Sequential(
                Residual(PreNorm(embed_dim, Attention(embed_dim = embed_dim, num_heads = n_heads, dropout=dropout))),
                Residual(PreNorm(embed_dim, MLP_Head(embed_dim = embed_dim, hidden_dim = embed_dim, dropout=dropout)))
            )
            self.layers.append(encoder_block)
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim)
        )
        
    def forward(self, img):
        x = self.patch_embedings(img)
        
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:,:(n+1)]
        
        for i in range(self.n_layers):
            x = self.layers[i](x)
            
        x = self.classification_head(x[:, 0, :])
        
        return x
        
        