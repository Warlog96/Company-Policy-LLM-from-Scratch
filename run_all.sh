#!/usr/bin/env bash
# Run the full pipeline: data -> tokenizer -> tests -> 3 training stages -> evaluation.
#
#   ./run_all.sh            full run (stage 1 takes a few hours)
#   ./run_all.sh --quick    ~50 steps per stage, checks everything connects (a few minutes)
set -euo pipefail
cd "$(dirname "$0")"

QUICK=""
CKPT="checkpoints/qa.pt"
if [[ "${1:-}" == "--quick" ]]; then
  QUICK="--quick"
  CKPT="checkpoints/quick/qa.pt"
fi

# 0. environment
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

echo "== 1. Generate Q&A data from the CSV"
python data_prep/generate_qa.py

echo "== 2. Download TinyStories text (skipped if already present)"
if [[ ! -f data/corpus/train.txt ]]; then
  python data_prep/download_corpus.py
fi

echo "== 3. Train the BPE tokenizer"
python -m tokenizer.bpe

echo "== 4. Unit tests"
python -m pytest -q

echo "== 5. Train: pretrain -> knowledge -> qa"
python -m train.train --stage pretrain  $QUICK
python -m train.train --stage knowledge $QUICK
python -m train.train --stage qa        $QUICK

echo "== 6. Evaluate"
python -m eval.evaluate --ckpt "$CKPT"

echo "== 7. Try it"
python -m model.generate "How many casual leaves do I get?" --ckpt "$CKPT"

echo
echo "Done. Start the app with:"
echo "  COMPANY_AI_CKPT=$CKPT uvicorn app.api:app --port 8000"
echo "  streamlit run app/ui.py"
