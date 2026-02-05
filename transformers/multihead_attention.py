import torch
import torch.nn as nn


"""
Multi-head attention is the architecture that allows the model to create various ways to interpret a single token.
Modern transformers vary from 8 to 16 MHA units.

What a single unit does is focus on a very specific meaning of the token.
Taking a simple example like translating "I like football" to Spanish -> "Me gusta el futbol", the token "like"
can be analyzed in various different contexts in each head.

Every MHA unit consists of three learned projections: Q, K, and V.
Query - What does this token need from the context?
Key - How do other tokens advertise their content?
Value - The actual information of the token.

For example, head 0 focuses on identifying the grammatical meaning of the token, while head 1 is trained to understand 
verb semantics and head 2 deciphers the emotional meaning of the word 'like'.

All these specific specializations of each head emerge from thousands of epochs during training.
Each head learns to specialize in something different during the training process, and the model is penalized 
if two heads start being responsible for the same thing, as this does not decrease the loss function and makes 
one head completely redundant.

Researchers tried to assign specific roles to each head before training, but it failed because it is very hard
to predict which patterns are going to be useful, therefore, they let the network learn the most important patterns by itself.
"""


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Stores number of heads (h) and computes the number of dimensions per head (d).
        Embedding size must be divisible by the number of heads.
        """
        super().__init__()
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, X):
        """
        Splits input along its last dimension (one split per head), converting
        3D tensor [B, L, h * d] to 4D tensor [B, L, h, d].
        B - batch size
        L - max len of input sequence
        h - num of heads
        d - num of dimensions per head
        The dimensions 1 and 2 are then swapped: [B, h, L, d].
        """
        return X.view(X.size(0), X.size(1), self.h, self.d).transpose(1, 2)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Apply linear transformation to query, key and value, pass the results through
        split_heads method.
        Compute the equation Attention(Q,K,V) = softmax(QK.T / √d_k) @ V with dropout on the weights.
        """
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(key))
        v = self.split_heads(self.v_proj(value))

        scores = q @ k.transpose(2, 3) / self.d**0.5

        if attn_mask:
            scores = scores.masked_fill(attn_mask, -torch.inf)
        if key_padding_mask:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -torch.inf)

        weights = scores.softmax(dim=-1)
        Z = self.dropout(weights) @ v
        # swap back dimensions and reshape tensor back to 3D (concat the outputs of al heads)
        Z = Z.transpose(1, 2)
        Z = Z.reshape(Z.size(0), Z.size(1), self.h * self.d)
        # apply output linear transformation
        return self.out_proj(Z), weights
