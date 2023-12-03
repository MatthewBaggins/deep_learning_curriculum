from __future__ import annotations

import math

from colorama import Fore, Style
import einops
from fancy_einsum import einsum
import torch as t
from torch import nn
import torch.nn.functional as F

from config import Config #pylint:disable=import-error
    
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.zeros(cfg.n_ctx, cfg.d_model))
    
    def forward(
        self,
        tokens: t.Tensor # [batch pos]
    ) -> t.Tensor:
        # Take as many positional encodings as there are tokens
        pe = self.W_pos[:tokens.size(-1), :] # [pos d_model]
        # Repeat across the batch dimension
        batch_pe = einops.repeat(pe, "pos d_model -> batch pos d_model", batch=tokens.size(0)) # [batch pos d_model]
        return batch_pe
        

    @classmethod
    def test(cls) -> bool:
        cfg = Config.dummy()
        pe = cls(cfg)
        x = cfg.random_tokens()
        y = pe(x)
        try:
            assert x.ndim == 2
            assert y.ndim == 3
            assert x.shape[:2] == y.shape[:2]
            assert y.size(-1) == cfg.d_model
            return True
        except AssertionError as e:
            print(e)
            return False
        
    

class Embed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=cfg.init_range)
        
    def forward(
        self,
        tokens: t.Tensor # [batch pos]
    ) -> t.Tensor:
        # Use token indices to get correct embeddings
        embedded = self.W_E[tokens, :] # [batch pos d_model]
        return embedded
    
    @classmethod
    def test(cls) -> bool:
        cfg = Config.dummy()
        embed = cls(cfg)
        tokens = cfg.random_tokens()
        embs = embed(tokens)
        try:
            assert embs.isnan().sum().item() == 0
            return True
        except AssertionError as e:
            print(e)
            return False
        
class Unembed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty(cfg.d_model, cfg.d_vocab))
        nn.init.normal_(self.W_U, std=cfg.init_range)
        self.b_U = nn.Parameter(t.zeros(cfg.d_vocab))
        
    def forward(
        self,
        resid_final: t.Tensor # [batch pos d_model]
    ) -> t.Tensor:
        logits = einsum(
            "batch pos d_model, d_model d_vocab -> batch pos d_vocab",
            resid_final,
            self.W_U
        ) + self.b_U
        return logits
    
    @classmethod
    def test(cls) -> bool:
        cfg = Config.dummy()
        ue = cls(cfg)
        x = cfg.random_resid()
        y = ue(x)
        try:
            assert y.isnan().sum().item() == 0
            return True
        except AssertionError as e:
            print(e)
            return False
        
class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        # In
        self.W_in = nn.Parameter(t.empty(cfg.d_model, cfg.d_mlp))
        nn.init.normal_(self.W_in, std=cfg.init_range)
        self.b_in = nn.Parameter(t.zeros(cfg.d_mlp))
        # Out
        self.W_out = nn.Parameter(t.empty(cfg.d_mlp, cfg.d_model))
        nn.init.normal_(self.W_out, std=cfg.init_range)
        self.b_out = nn.Parameter(t.zeros(cfg.d_model))
    
    def forward(
        self,
        resid_mid: t.Tensor # [batch pos d_model]
    ) -> t.Tensor:
        # Cast to MLP space
        pre_act = einsum(
            "batch pos d_model, d_model d_mlp -> batch pos d_mlp",
            resid_mid,
            self.W_in
        ) + self.b_in
        # Apply the activation function (ReLU)
        post_act = F.relu(pre_act)
        # Cast back to model space
        out = einsum(
            "batch pos d_mlp, d_mlp d_model -> batch pos d_model",
            post_act,
            self.W_out
        ) + self.b_out
        return out
    
    @classmethod
    def test(cls) -> bool:
        cfg = Config.dummy()
        mlp = cls(cfg)
        x = cfg.random_resid()
        y = mlp(x)
        try:
            assert x.shape == y.shape
            assert y.isnan().sum().item() == 0
            return True
        except AssertionError as e:
            print(e)
            return False

class SelfAttention(nn.Module):
    IGNORE: t.Tensor
    
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        
        # Query
        self.W_Q = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_Q, std=cfg.init_range)
        self.b_Q = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        
        # Key
        self.W_K = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_K, std=cfg.init_range)
        self.b_K = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        
        # Value
        self.W_V = nn.Parameter(t.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_V, std=cfg.init_range)
        self.b_V = nn.Parameter(t.zeros(cfg.n_heads, cfg.d_head))
        
        # Output
        self.W_O = nn.Parameter(t.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        nn.init.normal_(self.W_O, std=cfg.init_range)
        self.b_O = nn.Parameter(t.zeros(cfg.d_model))
        
        # Buffer
        self.register_buffer("IGNORE", t.tensor(-1e5))
    
    def forward(
        self,
        resid_pre: t.Tensor # [batch pos d_model]
    ) -> t.Tensor:
        q = einsum(
            "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head",
            resid_pre,
            self.W_Q
        ) + self.b_Q
        k = einsum(
            "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head",
            resid_pre,
            self.W_K
        ) + self.b_K
        v = einsum(
            "batch pos d_model, n_heads d_model d_head -> batch pos n_heads d_head",
            resid_pre,
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
        o = einsum(
            "batch pos n_heads d_head, n_heads d_head d_model -> batch pos d_model",
            z,
            self.W_O
        ) + self.b_O
        return o
        
    def apply_causal_mask(
        self,
        attn_scores: t.Tensor # [batch n_heads q_pos k_pos]
    ) -> t.Tensor:
        mask = t.triu(t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores_masked = attn_scores.masked_fill(mask, self.IGNORE)
        return attn_scores_masked
        
    
    @classmethod
    def test(cls) -> bool:
        cfg = Config.dummy()
        x = cfg.random_resid()
        attn = cls(cfg)
        y = attn(x)
        try:
            assert x.shape == y.shape
            assert y.isnan().sum().item() == 0
            return True
        except AssertionError as e:
            print(e)
            return False
        
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.empty(cfg.d_model))
        nn.init.normal_(self.w, std=cfg.init_range)
        self.b = nn.Parameter(t.zeros(cfg.d_model))
    
    def forward(
        self,
        resid: t.Tensor # [batch pos d_model]
    ) -> t.Tensor:
        resid_centered = resid - einops.reduce(resid, "batch pos d_model -> batch pos 1", "mean")
        scale = (einops.reduce(resid_centered.pow(2), "batch pos d_model -> batch pos 1", "mean") + self.cfg.ln_eps) ** 0.5
        resid_normalized = (resid_centered / scale) * self.w + self.b
        return resid_normalized
        
    @classmethod
    def test(cls) -> bool:
        cfg = Config.dummy()
        ln = cls(cfg)
        x = cfg.random_resid()
        y = ln(x)
        try:
            assert x.shape == y.shape
            assert y.isnan().sum().item() == 0
            return True
        except AssertionError as e:
            print(e)
            return False
        
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = SelfAttention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)
        
    def forward(
        self,
        resid: t.Tensor # [batch pos d_model]
    ) -> t.Tensor:
        return self.mlp(self.ln2(self.attn(self.ln1(resid))))
        
    
    @classmethod
    def test(cls) -> bool:
        cfg = Config.dummy()
        tb = cls(cfg)
        x = cfg.random_resid()
        y = tb(x)
        try:
            assert x.shape == y.shape
            assert y.isnan().sum().item() == 0
            return True
        except AssertionError as e:
            return False

class Transformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.pos_embed = PosEmbed(cfg)
        self.embed = Embed(cfg)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(cfg)
                for _ in range(cfg.n_layers)
            ]
        )
        self.unembed = Unembed(cfg)
    
    def forward(
        self,
        tokens: t.Tensor # [batch pos]
    ) -> t.Tensor:
        pos_embed = self.pos_embed(tokens)
        embed = self.embed(tokens)
        resid = pos_embed + embed
        for block in self.blocks:
            resid = resid + block(resid)
        logits = self.unembed(resid)
        return logits
    
    @classmethod
    def test(cls) -> bool:
        cfg = Config.dummy(n_layers=2)
        transformer = cls(cfg)
        tokens = cfg.random_tokens()
        logits = transformer(tokens)
        _preds = logits.argmax(-1)
        nans = logits.isnan().sum().item()
        try:
            assert nans == 0, f"{nans = }"
            return True
        except AssertionError as e:
            print(e)
            return False
        
def _validate_model() -> None:
    modules = [g for g in globals().values() if isinstance(g, type) and issubclass(g, nn.Module)]
    for module in modules:
        msg = module.__name__ + ": " + (Fore.GREEN + "PASSED!" if module.test() else Fore.RED + "FAILED!") + Style.RESET_ALL # type: ignore
        print(msg)


def main() -> None:
    _validate_model()

if __name__ == "__main__":
    main()
