import torch
from torch import nn
import torch.nn.functional as F

class SimpleConv(nn.Module):
    def __init__(self, pretrained_saliency_proj):
        super(SimpleConv, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11, 15), padding=(5, 7), groups=1)
        self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11, 15), padding=(5, 7), groups=1)
        self.conv2d_3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11, 15), padding=(5, 7), groups=1)
        self.conv2d_4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(11, 15), padding=(5, 7), groups=1)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(1)
        self.saliency_proj = pretrained_saliency_proj  # 256 -> 1 layer

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, feature_length, 256) -> (batch, 1, feature_length, 256)
        x = self.conv2d_1(x)  # Temporal conv (3x1)
        x = self.activation(x)
        x = self.conv2d_2(x)  # Another temporal conv
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv2d_3(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv2d_4(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        x = x.squeeze(1)  # (batch, 1, feature_length, 256) -> (batch, feature_length, 256)
        saliency = self.saliency_proj(x)  # Apply saliency projection
        return saliency

class SelfAttention(nn.Module):
    def __init__(self, feature_dim=256, num_heads=2):
        super(SelfAttention, self).__init__()
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(feature_dim)
        # self.batch_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        qkv = self.qkv(x)
        qkv = qkv.view(-1, x.size(1), self.num_heads, self.feature_dim // self.num_heads * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        attention = torch.matmul(q, k)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(-1, x.size(1), self.feature_dim)
        x = self.fc(x)
        x = self.activation(x)
        # x = self.batch_norm(x)
        
        return x

class SANet(nn.Module):
    def __init__(self, pretrained_saliency_proj, num_sa_blocks, feature_dim=256):
        super(SANet, self).__init__()
        for i in range(num_sa_blocks):
            setattr(self, f'self_attention_{i}', SelfAttention(feature_dim=feature_dim))
        self.saliency_proj = pretrained_saliency_proj
        self.num_sa_blocks = num_sa_blocks
        
    def forward(self, x):
        for i in range(self.num_sa_blocks):
            x = getattr(self, f'self_attention_{i}')(x)
        saliency = self.saliency_proj(x)
        return saliency
    
    
class TransformerBlock(nn.Module):
    def __init__(self, feature_dim=256, num_heads=2, ff_hidden_dim=256, dropout=0.2):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        # Linear projection for Q, K, V (similar to SelfAttention)
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.fc_attn = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_attn = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()
        
        # Feed-forward network (two linear layers with activation and dropout)
        self.ff = nn.Sequential(
            nn.Linear(feature_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(ff_hidden_dim, feature_dim)
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.dropout_ff = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        # Compute Q, K, V
        qkv = self.qkv(x)  # 
        qkv = qkv.view(x.size(0), x.size(1), self.num_heads, (self.feature_dim // self.num_heads) * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Rearrange for attention computation: (batch, num_heads, seq_len, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)
        attn = torch.matmul(q, k)  
        attn = self.softmax(attn)
        attn = self.dropout_attn(attn)
        
        # Apply attention weights
        attn_out = torch.matmul(attn, v)  
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(1), self.feature_dim)
        attn_out = self.fc_attn(attn_out)
        attn_out = self.activation(attn_out)
        
        # First residual connection + layer norm
        x = self.norm1(x + attn_out)
    
        # Feed-forward sub-layer
        ff_out = self.ff(x)
        ff_out = self.dropout_ff(ff_out)
        ff_out = self.activation(ff_out)
        
        # Second residual connection + layer norm
        out = self.norm2(x + ff_out)
        return out

class TransformerNet(nn.Module):
    def __init__(self, pretrained_saliency_proj, num_blocks, feature_dim=256, num_heads=2, ff_hidden_dim=128, dropout=0.2):
        super(TransformerNet, self).__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.saliency_proj = pretrained_saliency_proj
        self.num_blocks = num_blocks
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        saliency = self.saliency_proj(x)
        return saliency
