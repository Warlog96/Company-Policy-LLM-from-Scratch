"""
Download a general English corpus (TinyStories) for stage-1 language pretraining.

This is ONLY raw text data (short children's stories), not a model.
We stream the public file and stop after --mb megabytes (~140 MB ~= 30M BPE tokens),
so we never download the whole 1.9 GB file.

Stories are separated by "<|endoftext|>" in the source; we replace that with our own <eos>.

Outputs:
  data/corpus/train.txt   pretraining text
  data/corpus/val.txt     held-out text for validation loss
"""
import argparse
import os

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "corpus")
BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_URL = BASE + "TinyStories-train.txt"
VAL_URL = BASE + "TinyStories-valid.txt"


def stream_text(url, out_path, max_bytes):
    """Stream `url` to `out_path`, stopping after max_bytes, ending on a story boundary."""
    written = 0
    buffer = ""
    with requests.get(url, stream=True, timeout=60) as r, open(out_path, "w", encoding="utf-8") as out:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1 << 20, decode_unicode=False):
            buffer += chunk.decode("utf-8", errors="ignore")
            # write only complete stories; keep the unfinished tail in the buffer
            parts = buffer.split("<|endoftext|>")
            buffer = parts.pop()
            for story in parts:
                story = " ".join(story.split())          # normalise whitespace to single spaces
                if story:
                    out.write(story + " <eos>\n")
                    written += len(story) + 7
            print("\r  {}: {:.1f} MB".format(os.path.basename(out_path), written / 1e6), end="", flush=True)
            if written >= max_bytes:
                break
    print()
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=float, default=140, help="megabytes of training text to keep")
    parser.add_argument("--val_mb", type=float, default=2, help="megabytes of validation text")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Downloading TinyStories (text only) ...")
    stream_text(TRAIN_URL, os.path.join(OUT_DIR, "train.txt"), int(args.mb * 1e6))
    stream_text(VAL_URL, os.path.join(OUT_DIR, "val.txt"), int(args.val_mb * 1e6))
    print("Done. Files in", OUT_DIR)


if __name__ == "__main__":
    main()
