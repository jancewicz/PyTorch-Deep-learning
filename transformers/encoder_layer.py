import torch.nn as nn
import torch.nn.functional as F
from transformers.multihead_attention import MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # self-attention block
        attn_output, _ = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = self.norm1(src + self.dropout(attn_output))

        # feed-forward block
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout(ff))

        return src
