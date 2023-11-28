from __future__ import annotations

import math

import einops
from fancy_einsum import einsum
import torch as t
from torch import nn
import torch.nn.functional as F


from config import Config
from utils import to_one_hot
    
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty(cfg.n_ctx, cfg.d_model))
        nn.init.normal_(self.W_pos)
    
    def forward(self, tokens: t.Tensor) -> t.Tensor:
        pe = self.W_pos[:tokens.size(-1), :]
        batch_pe = einops.repeat(pe, "pos d_model -> batch pos d_model", batch=tokens.size(0))
        return batch_pe.clone()

    @classmethod
    def test(cls) -> None:
        cfg = Config.dummy()
        pe = cls(cfg)
        x = cfg.random_tokens()
        y = pe(x)
        assert x.ndim == 2
        assert y.ndim == 3
        assert x.shape[:2] == y.shape[:2]
        assert y.size(-1) == cfg.d_model
        print("PASSED!")

class Embed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E)
        
    def forward(self, tokens: t.Tensor) -> None:
        assert tokens.ndim == 2
        assert tokens.max() <= self.cfg.d_vocab - 1
        assert tokens.size(1) <= self.cfg.n_ctx
        
        tokens_one_hot = to_one_hot(tokens, self.cfg.d_vocab)
        # print(tokens_one_hot.shape, self.cfg.d_vocab)
        return einsum(
            "batch pos d_vocab, d_vocab d_model -> batch pos d_model",
            tokens_one_hot,
            self.W_E
        ) * math.sqrt(self.cfg.d_model)
    
    @classmethod
    def test(cls) -> None:
        cfg = Config.dummy()
        embed = cls(cfg)
        tokens = cfg.random_tokens()
        embs = embed(tokens)
        assert embs.isnan().sum().item() == 0
        # print(f"{embs.shape = }")
        print("PASSED!")
        
class Unembed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty(cfg.d_model, cfg.d_vocab))
        nn.init.normal_(self.W_U)
        self.b_U = nn.Parameter(t.zeros(cfg. d_vocab))
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return einsum(
            "batch pos d_model, d_model d_vocab -> batch pos d_vocab",
            x,
            self.W_U
        ) + self.b_U
    
    @classmethod
    def test(cls) -> None:
        cfg = Config.dummy()
        ue = cls(cfg)
        x = cfg.random_resid()
        y = ue(x)
        print(f"PASSED! {tuple(y.shape)}")
        # print(y.isnan().sum())
        
class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model)
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.W_out(self.dropout(F.relu(self.W_in(x))))
    
    @classmethod
    def test(cls) -> None:
        cfg = Config.dummy()
        mlp = cls(cfg)
        x = cfg.random_resid()
        y = mlp(x)
        assert x.shape == y.shape
        assert y.isnan().sum().item() == 0
        print(f"PASSED! shape: {tuple(x.shape)}")

class SelfAttention(nn.Module):
    IGNORE: t.Tensor
    
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        # Q
        self.W_Q = nn.Parameter(t.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        nn.init.normal_(self.W_Q)
        self.b_Q = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        # K
        self.W_K = nn.Parameter(t.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        nn.init.normal_(self.W_K)
        self.b_K = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        # V
        self.W_V = nn.Parameter(t.empty(cfg.d_model, cfg.n_heads, cfg.d_head))
        nn.init.normal_(self.W_V)
        self.b_V = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        # O
        self.W_O = nn.Parameter(t.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        nn.init.normal_(self.W_O)
        self.b_O = nn.Parameter(t.zeros(cfg.d_model))
        # buffer
        self.register_buffer("IGNORE", t.tensor(-1e6))
    
    def apply_causal_mask(self, attn_scores: t.Tensor) -> t.Tensor:
        assert attn_scores.ndim == 4
        assert attn_scores.size(2) == attn_scores.size(3)
        return attn_scores.where(
            t.ones_like(attn_scores).triu().flip(-1) == 0,
            self.IGNORE
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        assert x.ndim == 3 # batch pos d_model
        assert x.size(2) == self.cfg.d_model
        assert x.size(1) <= self.cfg.n_ctx
        
        q = einsum(
            "batch pos d_model, d_model n_heads d_head -> batch pos n_heads d_head",
            x, 
            self.W_Q
        ) + self.b_Q
        k = einsum(
            "batch pos d_model, d_model n_heads d_head -> batch pos n_heads d_head",
            x, 
            self.W_K
        ) + self.b_K
        v = einsum(
            "batch pos d_model, d_model n_heads d_head -> batch pos n_heads d_head",
            x, 
            self.W_V
        ) + self.b_V
        
        attn_scores = einsum(
            "batch q_pos n_heads d_head, batch k_pos n_heads d_head -> batch n_heads q_pos k_pos", 
            q,
            k
        ) / math.sqrt(self.cfg.d_head)
        attn_scores_masked = self.apply_causal_mask(attn_scores)
        attn_pattern = attn_scores_masked.softmax(-1)
        
        z = einsum(
            "batch n_heads q_pos k_pos, batch k_pos n_heads d_head -> batch q_pos n_heads d_head",
            attn_pattern,
            v
        )
        
        # print(f"W_O: {tuple(self.W_O.shape)}\nz: {tuple(z.shape)}")
        
        o = einsum(
            "n_heads d_head d_model, batch pos n_heads d_head -> batch pos d_model", 
            self.W_O,
            z
        ) + self.b_O
        return o
        
    
    @classmethod
    def test(cls) -> None:
        cfg = Config.dummy()
        x = cfg.random_resid()
        attn = cls(cfg)
        y = attn(x)
        assert x.shape == y.shape
        assert y.isnan().sum().item() == 0
        print(f"Passed! shape: {tuple(x.shape)}")
        
SelfAttention.test()
## LayerNorm
def normalize(x: t.Tensor, epsilon: float = 1e-6) -> t.Tensor:
    return (x - x.mean(-1, keepdim=True)) / (x.std() + epsilon)

class LayerNorm(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.scale = nn.Parameter(t.empty(cfg.d_model))
        nn.init.normal_(self.scale)
        self.translation = nn.Parameter(t.zeros(cfg.d_model))
        
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return normalize(x, self.cfg.epsilon) * self.scale + self.translation
        
    @classmethod
    def test(cls) -> None:
        cfg = Config.dummy()
        ln = cls(cfg)
        x = cfg.random_resid()
        y = ln(x)
        assert x.shape == y.shape
        assert y.isnan().sum().item() == 0
        print("PASSED!")
        
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x + self.mlp(self.ln2(self.attn(self.ln1(x))))
    
    @classmethod
    def test(cls) -> None:
        cfg = Config.dummy()
        tb = cls(cfg)
        x = cfg.random_resid()
        y = tb(x)
        assert x.shape == y.shape
        assert y.isnan().sum().item() == 0
        print("PASSED!")

class Transformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.pos_embed = PosEmbed(cfg)
        self.embed = Embed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)
        
    def forward(self, tokens: t.Tensor) -> t.Tensor:
        assert tokens.ndim == 2
        assert tokens.size(1) <= self.cfg.n_ctx
        assert tokens.dtype == t.int64
        assert tokens.max() <= self.cfg.d_vocab
        
        pos_embeddings = self.pos_embed(tokens)
        embeddings = self.embed(tokens)
        resid = pos_embeddings + embeddings
        for block_i, block in enumerate(self.blocks):
            resid = block(resid)
        # print("check", resid.isnan().sum().item())
        logits = self.unembed(resid)
        return logits
    
    @classmethod
    def test(cls) -> None:
        cfg = Config.dummy(n_layers=2)
        transformer = cls(cfg)
        tokens = cfg.random_tokens()
        logits = transformer(tokens)
        preds = logits.argmax(-1)
        nans = logits.isnan().sum().item()
        assert nans == 0, f"{nans = }"
        print("PASSED!")
        