"""
Byte-level BPE tokenizer, written from scratch (no tokenizer libraries).

How it works:
  1. Every string is turned into UTF-8 bytes, so the base vocabulary is the 256 byte values
     and ANY text can be encoded (no "unknown" token needed).
  2. Training repeatedly finds the most frequent adjacent pair of tokens and merges it
     into a new token, until the vocabulary is full.
  3. Encoding applies the learned merges in the order they were learned.
  4. Decoding just concatenates the bytes of each token.

Vocabulary layout (total = vocab_size = 4096):
  0..255                 raw bytes
  256..4093              learned merges
  4094, 4095             special tokens <eos>, <pad>

Before BPE we split text into "chunks" (words with their leading space, numbers, punctuation)
with a regex, so merges never cross word boundaries. Training counts each unique chunk once
together with its frequency, which makes training fast.

Usage:
  python -m tokenizer.bpe            # trains on a ~5 MB mix of corpus + company Q&A text
  python -m tokenizer.bpe --force    # retrain even if bpe.json exists (invalidates trained models)
"""
import json
import os
import re
from collections import Counter, defaultdict

SPECIAL_TOKENS = ["<eos>", "<pad>"]

# Pre-tokenization pattern: contractions, words with optional leading space,
# numbers (up to 3 digits), punctuation runs, and whitespace.
SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self):
        self.merges = {}            # (id_a, id_b) -> new_id, in learning order
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes
        self.special = {}           # "<eos>" -> id
        self.pattern = re.compile(SPLIT_PATTERN)
        self.cache = {}             # chunk string -> list of ids (speeds up encoding)
        self.size = 256             # total vocabulary size (set by train / load)

    # ------------------------------------------------------------------ train
    def train(self, text, vocab_size=4096, verbose=True):
        num_merges = vocab_size - 256 - len(SPECIAL_TOKENS)

        # Remove special-token strings from training text: they get their own ids.
        for s in SPECIAL_TOKENS:
            text = text.replace(s, " ")

        # Count unique chunks: "the" appearing 50,000 times is processed once with weight 50,000.
        chunk_counts = Counter(self.pattern.findall(text))
        words = [list(c.encode("utf-8")) for c in chunk_counts]   # each word = list of token ids
        freqs = list(chunk_counts.values())

        # pair -> total count, and pair -> which words contain it
        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += freqs[wi]
                where[pair].add(wi)

        for m in range(num_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)     # most frequent pair
            new_id = 256 + m
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            # Only words that contain `best` need to be updated.
            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                # remove this word's old pair counts
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    where[pair].discard(wi)
                # merge every occurrence of `best` in this word
                w = merge_pair(w, best, new_id)
                words[wi] = w
                # add the new pair counts back
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += f
                    where[pair].add(wi)
            where.pop(best, None)

            if verbose and (m + 1) % 500 == 0:
                print("  merge {}/{}: {!r}".format(m + 1, num_merges, self.vocab[new_id].decode("utf-8", "replace")))

        # Special tokens go at the very end of the vocabulary.
        for i, s in enumerate(SPECIAL_TOKENS):
            self.special[s] = vocab_size - len(SPECIAL_TOKENS) + i
        self.size = vocab_size
        self.cache = {}

    # ----------------------------------------------------------------- encode
    def _encode_chunk(self, chunk):
        """Apply merges to one chunk, always merging the earliest-learned pair first."""
        if chunk in self.cache:
            return self.cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # find the pair with the lowest merge id (= learned earliest)
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = merge_pair(ids, best, self.merges[best])
        self.cache[chunk] = ids
        return ids

    def encode(self, text):
        """String -> list of token ids. Special tokens like <eos> are recognised."""
        ids = []
        special_re = "(" + "|".join(re.escape(s) for s in self.special) + ")"
        for part in re.split(special_re, text):
            if part in self.special:
                ids.append(self.special[part])
            elif part:
                for chunk in self.pattern.findall(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    # ----------------------------------------------------------------- decode
    def decode(self, ids):
        """List of token ids -> string."""
        inv_special = {v: k for k, v in self.special.items()}
        out = bytearray()
        for i in ids:
            if i in inv_special:
                out.extend(inv_special[i].encode("utf-8"))
            else:
                out.extend(self.vocab[i])
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------- properties
    @property
    def eos_id(self):
        return self.special["<eos>"]

    @property
    def pad_id(self):
        return self.special["<pad>"]

    @property
    def vocab_size(self):
        return self.size

    # -------------------------------------------------------------- save/load
    def save(self, path):
        data = {
            "vocab_size": self.size,
            "pattern": SPLIT_PATTERN,
            "merges": [[a, b, new] for (a, b), new in self.merges.items()],
            "special_tokens": self.special,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.pattern = re.compile(data["pattern"])
        for a, b, new in data["merges"]:            # rebuild vocab in learning order
            tok.merges[(a, b)] = new
            tok.vocab[new] = tok.vocab[a] + tok.vocab[b]
        tok.special = data["special_tokens"]
        tok.size = data["vocab_size"]
        return tok


def merge_pair(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_PATH = os.path.join(ROOT, "tokenizer", "bpe.json")


def build_training_sample(corpus_mb=4.0, company_mb=1.0):
    """~5 MB: 4 MB of stories + company text repeated up to ~1 MB (so company terms get tokens)."""
    with open(os.path.join(ROOT, "data", "corpus", "train.txt"), encoding="utf-8") as f:
        corpus = f.read(int(corpus_mb * 1e6))
    company = ""
    for name in ["train.txt", "knowledge.txt"]:
        with open(os.path.join(ROOT, "data", "processed", name), encoding="utf-8") as f:
            company += f.read()
    reps = max(1, int(company_mb * 1e6 / len(company)))
    return corpus + "\n" + company * reps


def main():
    import sys
    import time
    # The model's embedding rows are tied to these token ids. Retraining the tokenizer after
    # pretraining would silently scramble them, so we refuse unless --force is given.
    if os.path.exists(TOKENIZER_PATH) and "--force" not in sys.argv:
        tok = BPETokenizer.load(TOKENIZER_PATH)
        print("Tokenizer already exists ({}, vocab {}). Keeping it.".format(TOKENIZER_PATH, tok.vocab_size))
        print("Use --force to retrain (only before stage 1 pretraining!).")
        return
    text = build_training_sample()
    print("Training BPE on {:.1f} MB of text ...".format(len(text.encode("utf-8")) / 1e6))
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(text, vocab_size=4096)
    tok.save(TOKENIZER_PATH)
    print("Saved {} (vocab {}) in {:.0f}s".format(TOKENIZER_PATH, tok.vocab_size, time.time() - t0))

    sample = "Q: How many casual leaves do I get per year?\nA: Employees get 12 days of casual leave per year. [HR Leave Policy §3] <eos>"
    ids = tok.encode(sample)
    print("Example: {} chars -> {} tokens".format(len(sample), len(ids)))
    print([tok.decode([i]) for i in ids])


if __name__ == "__main__":
    main()
