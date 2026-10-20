"""
Text generation + the answer() function used by the API, UI and evaluation.

Sampling is written by hand:
  1. divide logits by the temperature (lower = more confident / less random)
  2. top-k: keep only the k highest-scoring tokens
  3. softmax -> probabilities, then sample one token with the inverse-CDF method
temperature = 0 means greedy decoding (always pick the most likely token).

CLI:
  python -m model.generate "How many casual leaves do I get?"
  python -m model.generate "..." --ckpt checkpoints/quick/qa.pt --temperature 0
"""
import argparse
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.transformer import load_model, get_device, softmax  # noqa: E402
from tokenizer.bpe import BPETokenizer, TOKENIZER_PATH  # noqa: E402

DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "qa.pt")
QUICK_CKPT = os.path.join(ROOT, "checkpoints", "quick", "qa.pt")
SOURCE_RE = re.compile(r"\s*\[([^\[\]]+)\]\s*$")     # "... [HR Leave Policy §3]" at the end


def sample_next(logits, temperature=0.8, top_k=20):
    """Pick the next token id from a (V,) vector of logits."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k is not None and top_k < logits.numel():
        kth_best = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    probs = softmax(logits, dim=-1)
    # inverse-CDF sampling: draw u ~ U(0,1), return the first token whose cumulative prob > u
    cdf = torch.cumsum(probs, dim=-1)
    u = torch.rand(1, device=logits.device) * cdf[-1]
    return int(min((cdf < u).sum().item(), logits.numel() - 1))


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens=100, temperature=0.8, top_k=20, device="cpu"):
    """Continue `prompt` until <eos> or max_new_tokens. Returns only the new text."""
    model.eval()
    ids = tok.encode(prompt)
    new_ids = []
    ctx = model.cfg.context_len
    for _ in range(max_new_tokens):
        x = torch.tensor([(ids + new_ids)[-ctx:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = sample_next(logits[0, -1].float().cpu(), temperature, top_k)
        if nxt == tok.eos_id:
            break
        new_ids.append(nxt)
    return tok.decode(new_ids)


class KnowledgeBot:
    """Loads the model + tokenizer once and answers questions."""

    def __init__(self, ckpt_path=None, device=None):
        ckpt_path = ckpt_path or os.environ.get("COMPANY_AI_CKPT") or DEFAULT_CKPT
        if not os.path.exists(ckpt_path) and os.path.exists(QUICK_CKPT):
            print("[warning] {} not found, using the --quick test model".format(ckpt_path))
            ckpt_path = QUICK_CKPT
        self.device = device or get_device()
        self.tok = BPETokenizer.load(TOKENIZER_PATH)
        self.model, ckpt = load_model(ckpt_path, self.device)
        self.model.eval()
        self.ckpt_path = ckpt_path
        self.info = {
            "checkpoint": os.path.relpath(ckpt_path, ROOT),
            "parameters": self.model.num_params(),
            "layers": self.model.cfg.n_layers,
            "d_model": self.model.cfg.d_model,
            "heads": self.model.cfg.n_heads,
            "context": self.model.cfg.context_len,
            "vocab": self.model.cfg.vocab_size,
            "final_val_loss": ckpt.get("extra", {}).get("final_val"),
            "device": self.device,
        }

    def answer(self, question, temperature=0.2, top_k=5, max_new_tokens=100):
        prompt = "Q: {}\nA:".format(" ".join(question.strip().split()))
        text = generate(self.model, self.tok, prompt, max_new_tokens, temperature, top_k, self.device)
        return split_source(text)


def split_source(text):
    """'Employees get 12 days ... [HR Leave Policy §3]' -> answer + source."""
    text = text.strip()
    m = SOURCE_RE.search(text)
    if m:
        return {"answer": text[:m.start()].strip(), "source": m.group(1).strip()}
    return {"answer": text, "source": ""}


_BOT = None


def answer(question, **kwargs):
    """answer(question) -> {"answer": str, "source": str}. The model is loaded on first use."""
    global _BOT
    if _BOT is None:
        _BOT = KnowledgeBot()
    return _BOT.answer(question, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    bot = KnowledgeBot(args.ckpt)
    out = bot.answer(args.question, temperature=args.temperature, top_k=args.top_k)
    print("Answer:", out["answer"])
    print("Source:", out["source"] or "(none)")


if __name__ == "__main__":
    main()
