"""
Training script for all three stages.

  Stage 1  pretrain   next-token prediction on English stories (learn the language)
  Stage 2  knowledge  company facts as plain statements + ~15% stories (learn the facts)
  Stage 3  qa         Q&A examples, loss ONLY on answer tokens (learn to answer)

Each stage starts from the previous stage's final checkpoint.

Examples:
  python -m train.train --stage pretrain
  python -m train.train --stage pretrain --resume      # continue after an interruption
  python -m train.train --stage knowledge
  python -m train.train --stage qa
  python -m train.train --stage all --quick            # ~50 steps per stage, smoke test

Outputs:
  checkpoints/<stage>.pt          final weights of a stage (input to the next stage)
  checkpoints/<stage>_latest.pt   periodic checkpoint for --resume
  logs/<stage>_log.csv            step, lr, train_loss, val_loss, tokens/sec
  logs/<stage>_loss.png           loss plot
(--quick writes to checkpoints/quick/ and logs/quick/ so it never overwrites real runs)
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.transformer import GPT, GPTConfig, get_device, save_checkpoint, load_model  # noqa: E402
from tokenizer.bpe import BPETokenizer, TOKENIZER_PATH  # noqa: E402

DATA = os.path.join(ROOT, "data")
TOKEN_CACHE = os.path.join(DATA, "tokens")

# Default hyper-parameters per stage. --quick overrides max_steps.
STAGES = {
    "pretrain":  dict(lr=6e-4, max_steps=8000, warmup=300, batch_size=32, eval_every=250, init_from=None),
    "knowledge": dict(lr=1e-4, max_steps=2000, warmup=100, batch_size=32, eval_every=100, init_from="pretrain"),
    "qa":        dict(lr=5e-5, max_steps=3000, warmup=100, batch_size=32, eval_every=100, init_from="knowledge"),
}
CORPUS_MIX = 0.15   # fraction of stage-2 sequences that are story text


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_corpus_tokens(tok, split, max_mb=None):
    """Tokenize data/corpus/<split>.txt once and cache it as a uint16 .npy file."""
    os.makedirs(TOKEN_CACHE, exist_ok=True)
    tag = "" if max_mb is None else "_{}mb".format(max_mb)
    cache = os.path.join(TOKEN_CACHE, "corpus_{}{}.npy".format(split, tag))
    if os.path.exists(cache):
        return np.load(cache)
    path = os.path.join(DATA, "corpus", split + ".txt")
    print("Tokenizing {} (one-time, cached to {}) ...".format(path, cache))
    ids, read, t0 = [], 0, time.time()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            ids.extend(tok.encode(line.rstrip("\n")))
            read += len(line)
            if max_mb is not None and read > max_mb * 1e6:
                break
            if i % 50000 == 0 and i:
                print("  {:.0f} MB -> {:.1f}M tokens ({:.0f}s)".format(read / 1e6, len(ids) / 1e6, time.time() - t0))
    arr = np.array(ids, dtype=np.uint16)
    np.save(cache, arr)
    print("  {:,} tokens".format(len(arr)))
    return arr


def random_windows(tokens, batch_size, T):
    """Random (x, y) windows from a long token stream; y is x shifted by one."""
    starts = np.random.randint(0, len(tokens) - T - 1, size=batch_size)
    x = np.stack([tokens[s:s + T] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1:s + T + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def load_knowledge_tokens(tok):
    with open(os.path.join(DATA, "processed", "knowledge.txt"), encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.shuffle(lines)
    return np.array(tok.encode("\n".join(lines)), dtype=np.uint16)


def load_qa(tok, split):
    """Tokenize Q&A examples into (ids, n_prompt_tokens). The prompt is 'Q: ...\\nA:'."""
    out = []
    with open(os.path.join(DATA, "processed", split + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            cut = text.index("\nA:") + len("\nA:")
            prompt_ids = tok.encode(text[:cut])
            answer_ids = tok.encode(text[cut:])
            out.append((prompt_ids + answer_ids, len(prompt_ids)))
    return out


def qa_batch(examples, pad_id):
    """
    Pad a list of examples into a batch. Targets are -100 (ignored) for:
      - question tokens (we only train on the answer)
      - padding
    """
    T = max(len(ids) for ids, _ in examples) - 1
    x = torch.full((len(examples), T), pad_id, dtype=torch.long)
    y = torch.full((len(examples), T), -100, dtype=torch.long)
    for i, (ids, n_prompt) in enumerate(examples):
        ids = torch.tensor(ids)
        L = len(ids) - 1
        x[i, :L] = ids[:-1]
        y[i, :L] = ids[1:]
        y[i, :n_prompt - 1] = -100          # positions that predict question tokens
    return x, y


class Batcher:
    """Returns train/val batches for the chosen stage."""

    def __init__(self, stage, tok, batch_size, T, quick):
        self.stage, self.bs, self.T, self.pad = stage, batch_size, T, tok.pad_id
        corpus_mb = 3 if quick else None
        if stage in ("pretrain", "knowledge"):
            self.corpus_train = load_corpus_tokens(tok, "train", corpus_mb)
            self.corpus_val = load_corpus_tokens(tok, "val", 1 if quick else None)
        if stage == "knowledge":
            self.knowledge = load_knowledge_tokens(tok)
        if stage in ("knowledge", "qa"):
            self.qa_train = load_qa(tok, "train")
            self.qa_val = load_qa(tok, "val")

    def train_batch(self):
        if self.stage == "pretrain":
            return random_windows(self.corpus_train, self.bs, self.T)
        if self.stage == "knowledge":
            # ~85% fact statements, ~15% stories so the model keeps its English
            n_corpus = int(round(self.bs * CORPUS_MIX))
            xk, yk = random_windows(self.knowledge, self.bs - n_corpus, self.T)
            xc, yc = random_windows(self.corpus_train, n_corpus, self.T)
            return torch.cat([xk, xc]), torch.cat([yk, yc])
        return qa_batch(random.sample(self.qa_train, self.bs), self.pad)

    def val_batches(self, n=20):
        """Validation: story text for stage 1, held-out Q&A answers for stages 2-3."""
        if self.stage == "pretrain":
            rng = np.random.RandomState(0)          # same windows every time -> comparable numbers
            for _ in range(n):
                starts = rng.randint(0, len(self.corpus_val) - self.T - 1, size=self.bs)
                x = np.stack([self.corpus_val[s:s + self.T] for s in starts]).astype(np.int64)
                y = np.stack([self.corpus_val[s + 1:s + self.T + 1] for s in starts]).astype(np.int64)
                yield torch.from_numpy(x), torch.from_numpy(y)
        else:
            for i in range(0, len(self.qa_val), self.bs):
                yield qa_batch(self.qa_val[i:i + self.bs], self.pad)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def lr_at(step, max_lr, warmup, max_steps):
    """Linear warmup, then cosine decay down to 10% of max_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, batcher, device):
    model.eval()
    losses = [model(x.to(device), y.to(device))[1].item() for x, y in batcher.val_batches()]
    model.train()
    return float(np.mean(losses))


def plot_losses(log_path, png_path, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps, tr, vs, va = [], [], [], []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            tr.append(float(row["train_loss"]))
            if row["val_loss"]:
                vs.append(int(row["step"]))
                va.append(float(row["val_loss"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, tr, color="#2a78d6", linewidth=2, label="train loss")
    ax.plot(vs, va, color="#eb6834", linewidth=2, marker="o", markersize=4, label="val loss")
    ax.set_title("Stage: {}".format(stage), loc="left")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def train_stage(stage, args):
    hp = dict(STAGES[stage])
    if args.max_steps:
        hp["max_steps"] = args.max_steps
    if args.quick:
        hp.update(max_steps=50, warmup=10, eval_every=25)
    ckpt_dir = os.path.join(ROOT, "checkpoints", "quick" if args.quick else "")
    log_dir = os.path.join(ROOT, "logs", "quick" if args.quick else "")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, stage + ".pt")
    latest_path = os.path.join(ckpt_dir, stage + "_latest.pt")
    log_path = os.path.join(log_dir, stage + "_log.csv")

    device = get_device()
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)
    tok = BPETokenizer.load(TOKENIZER_PATH)

    # --- model: resume, or start from the previous stage, or from scratch
    start_step = 0
    ckpt = None
    if args.resume and os.path.exists(latest_path):
        model, ckpt = load_model(latest_path, device)
        start_step = ckpt["step"]
        print("Resuming {} from step {}".format(stage, start_step))
    elif hp["init_from"]:
        prev = os.path.join(ckpt_dir, hp["init_from"] + ".pt")
        if not os.path.exists(prev):
            sys.exit("Missing {} - run stage '{}' first.".format(prev, hp["init_from"]))
        model, _ = load_model(prev, device)
        print("Initialised from", prev)
    else:
        model = GPT(GPTConfig(vocab_size=tok.vocab_size)).to(device)
    model.train()
    print("Stage {} | device {} | {:,} parameters".format(stage, device, model.num_params()))

    # --- optimizer: weight decay on matrices only (not on biases / layer-norm)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=hp["lr"], betas=(0.9, 0.95))
    if ckpt is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])

    batcher = Batcher(stage, tok, hp["batch_size"], model.cfg.context_len, args.quick)

    # --- log file (append when resuming)
    new_log = not (args.resume and start_step > 0 and os.path.exists(log_path))
    if not new_log:
        # drop rows logged after the checkpoint we resume from (they will be redone)
        with open(log_path) as f:
            rows = list(csv.reader(f))
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerows([rows[0]] + [r for r in rows[1:] if int(r[0]) <= start_step])
    log_f = open(log_path, "w" if new_log else "a", newline="")
    logger = csv.writer(log_f)
    if new_log:
        logger.writerow(["step", "lr", "train_loss", "val_loss", "tokens_per_sec", "elapsed_sec"])

    best_val = ckpt["extra"].get("best_val", float("inf")) if ckpt else float("inf")
    t0 = time.time()
    tokens_seen = 0
    val_loss = None
    for step in range(start_step, hp["max_steps"]):
        lr = lr_at(step, hp["lr"], hp["warmup"], hp["max_steps"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = batcher.train_batch()
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += x.numel()

        last = step == hp["max_steps"] - 1
        val_str = ""
        if (step + 1) % hp["eval_every"] == 0 or last:
            val_loss = evaluate(model, batcher, device)
            val_str = "{:.4f}".format(val_loss)
            best_val = min(best_val, val_loss)
        if step % 10 == 0 or val_str:
            elapsed = time.time() - t0
            tps = tokens_seen / max(elapsed, 1e-6)
            logger.writerow([step + 1, "{:.2e}".format(lr), "{:.4f}".format(loss.item()), val_str,
                             int(tps), int(elapsed)])
            log_f.flush()
            if step % 50 == 0 or val_str:
                print("step {:5d}/{} | lr {:.2e} | loss {:.4f}{} | {:,.0f} tok/s".format(
                    step + 1, hp["max_steps"], lr, loss.item(),
                    " | val " + val_str if val_str else "", tps))
        if (step + 1) % args.ckpt_every == 0 or last:
            save_checkpoint(latest_path, model, optimizer, step + 1, {"best_val": best_val})

    log_f.close()
    final_val = val_loss if val_loss is not None else (ckpt["extra"].get("final_val") if ckpt else None)
    save_checkpoint(final_path, model, None, hp["max_steps"],
                    {"stage": stage, "final_val": final_val, "best_val": best_val})
    plot_losses(log_path, os.path.join(log_dir, stage + "_loss.png"), stage)
    print("Saved {} | final val loss {}".format(final_path, final_val))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["pretrain", "knowledge", "qa", "all"])
    parser.add_argument("--quick", action="store_true", help="~50 steps per stage to test the pipeline")
    parser.add_argument("--resume", action="store_true", help="continue from <stage>_latest.pt")
    parser.add_argument("--max_steps", type=int, default=None, help="override the stage's step count")
    parser.add_argument("--ckpt_every", type=int, default=500)
    args = parser.parse_args()
    stages = ["pretrain", "knowledge", "qa"] if args.stage == "all" else [args.stage]
    for s in stages:
        train_stage(s, args)


if __name__ == "__main__":
    main()
