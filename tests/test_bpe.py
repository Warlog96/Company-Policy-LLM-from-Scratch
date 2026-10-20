"""Tests for the from-scratch BPE tokenizer."""
import os

import pytest

from tokenizer.bpe import BPETokenizer, TOKENIZER_PATH, merge_pair

SAMPLES = [
    "Employees get 12 days of casual leave per year. [HR Leave Policy §3]",
    "Q: How many sick leaves do I get?\nA: 10 days. <eos>",
    "Unicode works too: café, naïve, 日本語, emoji 🎉, tabs\tand   spaces.\n\n",
    "",
    "Once upon a time, there was a little dog named Max.<eos><pad>",
]


def small_tokenizer():
    tok = BPETokenizer()
    tok.train("the cat sat on the mat. the cat ate the rat. " * 50, vocab_size=300, verbose=False)
    return tok


def test_merge_pair():
    assert merge_pair([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip_small(text):
    tok = small_tokenizer()
    assert tok.decode(tok.encode(text)) == text


def test_merges_compress():
    tok = small_tokenizer()
    text = "the cat sat on the mat."
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_special_tokens(tmp_path):
    tok = small_tokenizer()
    ids = tok.encode("hi <eos>")
    assert ids[-1] == tok.eos_id
    assert tok.vocab_size == 300
    # save / load gives identical results
    path = str(tmp_path / "tok.json")
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    for s in SAMPLES:
        assert tok2.encode(s) == tok.encode(s)


@pytest.mark.skipif(not os.path.exists(TOKENIZER_PATH), reason="trained tokenizer not built yet")
@pytest.mark.parametrize("text", SAMPLES)
def test_roundtrip_trained(text):
    tok = BPETokenizer.load(TOKENIZER_PATH)
    assert tok.vocab_size == 4096
    assert tok.decode(tok.encode(text)) == text
