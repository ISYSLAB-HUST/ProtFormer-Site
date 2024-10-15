""" 
"""
import torch
from torch import nn, einsum
from einops import rearrange
import numpy as np


class OuterProductMean(nn.Module):
    def __init__(self, dim, pair_dim) -> None:
        super().__init__()

        self.reduction = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 16)
        )
        self.to_out = nn.Linear(16 * 16, pair_dim)

    def forward(self, x):
        x = self.reduction(x)
        x_outer = einsum('bmi,bnj->bmnij', x, x)
        out = self.to_out(rearrange(x_outer, '... i j -> ... (i j)'))

        return out


class RecyclingEmbedder(nn.Module):
    def __init__(self, pair_dim=20, dim=64):
        super(RecyclingEmbedder, self).__init__()
        self.norm_pair = nn.LayerNorm(pair_dim)
        self.norm_single = nn.LayerNorm(dim)

    def forward(self, reprs_prev):
        pair = self.norm_pair(reprs_prev['pair_repr'])
        single = self.norm_single(reprs_prev['single_repr'])
        return single, pair


class ProtFormerBlock(nn.Module):
    def __init__(
            self,
            dim,
            query_key_dim=64,
            value_dim=64,
            num_heads=8,
            pair_dim=20,
            dropout=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.pair_dim = pair_dim
        self.head_dim = query_key_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, query_key_dim, bias=False)
        self.to_k = nn.Linear(dim, query_key_dim, bias=False)
        self.to_v = nn.Linear(dim, query_key_dim)

        self.gating = nn.Sequential(
            nn.Linear(dim, value_dim),
            nn.Sigmoid()
        )
        self.to_out = nn.Sequential(
            nn.Linear(value_dim, value_dim),
            nn.Dropout(dropout)
        )
        self.pair_to_bias = nn.Sequential(
            nn.LayerNorm(self.pair_dim),
            nn.Linear(self.pair_dim, self.num_heads, bias=False)
        )

        self.OuterProductMean = OuterProductMean(value_dim, pair_dim)

    def forward(
            self,
            x,
            pair,
            mask=None
    ):
        normed_x = self.norm(x)
        q = self.to_q(normed_x)
        k = self.to_q(normed_x)
        v = self.to_v(normed_x)
        gate = self.gating(normed_x)

        # b: batch, l: length, h: head, d: dim

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # Calculate the dot product of q and k.
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scaling

        if pair is not None:
            sim = sim + rearrange(self.pair_to_bias(pair), "b i j h -> b h i j")

        if mask is not None:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -np.inf)

        attn = self.dropout(torch.softmax(sim, dim=-1))
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, "b h i d -> b i (h d)")
        out = out * gate

        out = self.to_out(out)

        pair = self.OuterProductMean(out)
        return out, pair


class ProtFormer(nn.Module):
    def __init__(self, dim: int,
                 num_layers: int,
                 n_hidden: int,
                 pair_dim: int,
                 dropout: float) -> None:
        super().__init__()

        self.sub_model = nn.ModuleList()
        self.sub_model.append(
            ProtFormerBlock(dim, n_hidden, n_hidden, pair_dim=pair_dim, dropout=dropout))
        for i in range(num_layers - 1):
            self.sub_model.append(
                ProtFormerBlock(n_hidden, n_hidden, n_hidden, pair_dim=pair_dim, dropout=dropout)
            )

    def forward(self, x, pair, mask=None):
        for layer in self.sub_model:
            x, pair = layer(x, pair, mask)
        return x, pair


class SSpredictor(nn.Module):
    def __init__(self,
                 dim: int,
                 num_layers: int,
                 n_hidden: int,
                 pair_dim: int,
                 dropout: float):
        super().__init__()

        self.reduction = nn.Linear(dim, n_hidden)
        self.proformer = ProtFormer(n_hidden, num_layers, n_hidden, pair_dim, dropout)
        self.recyle_embedding = RecyclingEmbedder(pair_dim=pair_dim, dim=n_hidden)

        # Task block

        self.ss2 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden, out_features=2),
        ])

    def forward(self, reprs, mask=None, num_recycle=3) -> dict:
        reprs_prev = None
        reprs["single_repr"] = self.reduction(reprs["single_repr"])

        for c in range(1 + num_recycle):
            if reprs_prev is None:
                reprs_prev = {
                    'pair_repr': torch.zeros_like(reprs["pair_repr"]),
                    'single_repr': torch.zeros_like(reprs["single_repr"]),
                }

            rec_x, rec_pair = self.recyle_embedding(reprs_prev)
            reprs["single_repr"], reprs["pair_repr"] = rec_x + reprs["single_repr"], reprs["pair_repr"] + rec_pair

            if c < num_recycle:
                x, pair = self.proformer(reprs["single_repr"], reprs["pair_repr"])
                reprs_prev = {
                    'single_repr': x.detach(),
                    'pair_repr': pair.detach(),
                }
            else:
                x, _ = self.proformer(reprs["single_repr"], reprs["pair_repr"])

        # hidden neurons to classes

        ss2 = self.ss2(x)
        return {"ss2": ss2}
    
