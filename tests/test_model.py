"""Tests for the hand-written Transformer."""
import torch
import torch.nn.functional as F

from model.transformer import GPT, GPTConfig, cross_entropy, softmax, LayerNorm

SMALL = GPTConfig(vocab_size=100, context_len=16, n_layers=2, d_model=32, n_heads=4, dropout=0.1)


def test_output_shapes():
    model = GPT(SMALL)
    idx = torch.randint(0, 100, (3, 10))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (3, 10, 100)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_causal_mask_blocks_future():
    """Changing a later token must not change the outputs at earlier positions."""
    torch.manual_seed(0)
    model = GPT(SMALL).eval()
    a = torch.randint(0, 100, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 100, (4,))      # change only positions 8..11
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[0, :8], lb[0, :8], atol=1e-5)
    assert not torch.allclose(la[0, 8:], lb[0, 8:])


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(50, 100) * 3
    targets = torch.randint(0, 100, (50,))
    assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-5)
    # with ignored positions (used for answer-only loss)
    targets[::3] = -100
    assert torch.allclose(cross_entropy(logits, targets),
                          F.cross_entropy(logits, targets, ignore_index=-100), atol=1e-5)


def test_softmax_and_layernorm_match_torch():
    x = torch.randn(4, 7)
    assert torch.allclose(softmax(x), torch.softmax(x, dim=-1), atol=1e-6)
    ln = LayerNorm(7)
    assert torch.allclose(ln(x), F.layer_norm(x, (7,)), atol=1e-5)


def test_weight_tying_and_param_count():
    model = GPT(GPTConfig())
    n = model.num_params()
    assert 10e6 <= n <= 15e6, n
    # output layer uses the token embedding matrix, so there is no separate output weight
    assert not any("head" in name for name, _ in model.named_parameters())


def test_can_overfit_tiny_batch():
    """Sanity check that gradients flow: loss should drop fast on one repeated batch."""
    torch.manual_seed(0)
    model = GPT(SMALL)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, 100, (4, 16))
    x, y = idx[:, :-1], idx[:, 1:]
    first = None
    for _ in range(60):
        _, loss = model(x, y)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
