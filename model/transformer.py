"""
A small GPT-style decoder-only Transformer, written by hand.

Only nn.Module, nn.Parameter and nn.Linear are used from torch.nn. Everything else
(embeddings, layer norm, causal self-attention, softmax, GELU, dropout, cross-entropy)
is implemented below so it can be explained line by line.

Architecture (pre-norm):
    tokens -> token embedding + position embedding -> dropout
           -> N x [ x = x + Attn(LN(x));  x = x + FFN(LN(x)) ]
           -> final LN -> logits = x @ token_embedding.T   (weight tying)
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


def get_device():
    """Auto-detect the best device: Apple GPU (mps), then NVIDIA (cuda), then CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    context_len: int = 256
    n_layers: int = 6
    d_model: int = 384
    n_heads: int = 6
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Hand-written building blocks
# ---------------------------------------------------------------------------
def softmax(x, dim=-1):
    """Numerically stable softmax: subtract the max before exponentiating."""
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def gelu(x):
    """GELU activation (tanh approximation used by GPT-2)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def dropout(x, p, training):
    """Inverted dropout: zero each value with probability p and scale the rest by 1/(1-p)."""
    if not training or p == 0.0:
        return x
    keep = (torch.rand_like(x) > p).to(x.dtype)
    return x * keep / (1.0 - p)


def cross_entropy(logits, targets, ignore_index=-100):
    """
    Softmax cross-entropy, by hand.
      logits:  (N, V) raw scores
      targets: (N,)   correct token ids; positions equal to ignore_index are skipped
    loss_i = log(sum_j exp(logit_ij)) - logit_i[target_i]     (averaged over kept positions)
    """
    mask = targets != ignore_index
    logits, targets = logits[mask], targets[mask]
    m = logits.max(dim=-1, keepdim=True).values                       # for stability
    log_sum_exp = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
    correct = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (log_sum_exp - correct).mean()


class LayerNorm(nn.Module):
    """Normalise each vector to mean 0 / variance 1, then apply a learned scale and shift."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token can only look at itself and earlier tokens."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)   # queries, keys, values in one matmul
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)      # output projection
        self.p = cfg.dropout
        # Causal mask: lower-triangular matrix of ones. mask[i, j] = 1 if j <= i.
        mask = torch.tril(torch.ones(cfg.context_len, cfg.context_len))
        self.register_buffer("mask", mask.view(1, 1, cfg.context_len, cfg.context_len), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # (B, h, T, T)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # block the future
        weights = softmax(scores, dim=-1)
        weights = dropout(weights, self.p, self.training)
        out = weights @ v                                                       # (B, h, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)                    # merge heads
        return dropout(self.proj(out), self.p, self.training)


class FeedForward(nn.Module):
    """Position-wise MLP: expand to 4x width, GELU, project back."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.p = cfg.dropout

    def forward(self, x):
        return dropout(self.proj(gelu(self.fc(x))), self.p, self.training)


class Block(nn.Module):
    """One pre-norm Transformer block with residual connections."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Embeddings are plain learnable matrices; looking up a token = indexing a row.
        self.tok_emb = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(cfg.context_len, cfg.d_model) * 0.01)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = LayerNorm(cfg.d_model)
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if name.endswith("weight") and p.dim() == 2:          # Linear layers
                std = 0.02
                if name.endswith("proj.weight"):                   # residual projections (GPT-2 trick)
                    std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
                nn.init.normal_(p, mean=0.0, std=std)
            elif name.endswith("bias") and "ln" not in name:
                nn.init.zeros_(p)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token ids
        targets: (B, T) next-token ids, or -100 where no loss should be computed
        returns  logits (B, T, V) and loss (or None)
        """
        B, T = idx.shape
        assert T <= self.cfg.context_len, "sequence longer than context"
        x = self.tok_emb[idx] + self.pos_emb[:T]                  # (B, T, C)
        x = dropout(x, self.cfg.dropout, self.training)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.t()                             # weight tying: reuse the embedding

        loss = None
        if targets is not None:
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def save_checkpoint(path, model, optimizer=None, step=0, extra=None):
    torch.save({
        "config": asdict(model.cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "extra": extra or {},
    }, path)


def load_model(path, device="cpu"):
    """Load a checkpoint and return (model, checkpoint_dict)."""
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


if __name__ == "__main__":
    model = GPT(GPTConfig())
    print("GPT config:", GPTConfig())
    print("Parameters: {:,} ({:.2f}M)".format(model.num_params(), model.num_params() / 1e6))
    print("Device:", get_device())
