"""Tests for sampling and answer parsing."""
import torch

from model.generate import sample_next, split_source, generate
from model.transformer import GPT, GPTConfig
from tokenizer.bpe import BPETokenizer


def test_greedy_picks_argmax():
    logits = torch.tensor([0.1, 5.0, 0.3, 2.0])
    assert sample_next(logits, temperature=0) == 1


def test_top_k_only_samples_top_tokens():
    torch.manual_seed(0)
    logits = torch.tensor([3.0, 2.9, -1.0, -2.0, 0.0])
    picks = {sample_next(logits, temperature=1.0, top_k=2) for _ in range(200)}
    assert picks <= {0, 1} and len(picks) == 2


def test_split_source():
    out = split_source(" Employees get 12 days of casual leave per year. [HR Leave Policy §3]")
    assert out == {"answer": "Employees get 12 days of casual leave per year.", "source": "HR Leave Policy §3"}
    refusal = split_source("I don't know. This is not covered in the company policies.")
    assert refusal["source"] == ""


def test_generate_respects_max_tokens():
    tok = BPETokenizer()
    tok.train("hello world " * 20, vocab_size=270, verbose=False)
    model = GPT(GPTConfig(vocab_size=270, context_len=16, n_layers=1, d_model=16, n_heads=2))
    text = generate(model, tok, "hello", max_new_tokens=30, temperature=1.0, top_k=10)
    assert isinstance(text, str)
    # each generated token decodes to at most len(longest token) characters
    assert len(text) <= 30 * max(len(v) for v in tok.vocab.values())
