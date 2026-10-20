# Arcvale Knowledge Base AI — a small language model built from scratch

Employees ask questions about company policy and get an answer **with a source reference**,
e.g. `Employees get 12 days of casual leave per year. [HR Leave Policy §3]`.

All knowledge lives **inside the model's weights**. The model learned it by training on a CSV of
company facts. There is **no RAG, no vector database and no retrieval** at answer time.

Everything is written by hand: a byte-level BPE tokenizer, a GPT-style Transformer (embeddings,
attention, causal mask, layer norm, feed-forward, softmax cross-entropy) and a sampler.
PyTorch is only used for tensors, autograd, `nn.Module` / `nn.Parameter` / `nn.Linear` and AdamW.
No pretrained weights are used anywhere.

## How it works

| Step | File | What happens |
|---|---|---|
| Facts | `data/company_facts.csv` | 155 fictional policy facts (topic, question, answer, source) |
| Q&A data | `data_prep/generate_qa.py` | 6–7 paraphrases per question × 2–3 answer phrasings, 50 out-of-scope refusals; 1 paraphrase per fact held out for testing |
| Corpus | `data_prep/download_corpus.py` | ~141 MB of TinyStories text (≈35M tokens) to learn English |
| Tokenizer | `tokenizer/bpe.py` | byte-level BPE, 4096 tokens incl. `<eos>`, `<pad>` |
| Model | `model/transformer.py` | 6 layers, d_model 384, 6 heads, context 256, **12.3M parameters**, tied output layer |
| Training | `train/train.py` | stage 1 `pretrain` → stage 2 `knowledge` → stage 3 `qa` (loss on answer tokens only) |
| Answering | `model/generate.py` | `answer(question) -> {"answer", "source"}` with temperature / top-k sampling |
| Evaluation | `eval/evaluate.py` | answer / source / refusal accuracy + hallucination examples |
| App | `app/api.py`, `app/ui.py` | FastAPI `POST /ask` + Streamlit chat page |

Training example format: `Q: <question>\nA: <answer> [<source>] <eos>`

## Commands (in order)

```bash
# 1. Install (Python 3.9+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate data
python data_prep/generate_qa.py          # CSV -> data/processed/{train,val,test}.jsonl, knowledge.txt
python data_prep/download_corpus.py      # TinyStories text -> data/corpus/

# 3. Train the tokenizer (+ run all unit tests)
python -m tokenizer.bpe
python -m pytest -q

# 4. Quick end-to-end test (~2 minutes, writes to checkpoints/quick/ and logs/quick/)
python -m train.train --stage all --quick
python -m eval.evaluate --ckpt checkpoints/quick/qa.pt

# 5. Real training (each stage loads the previous stage's checkpoint)
python -m train.train --stage pretrain    # ~2 h on an Apple M5 (8000 steps, 65M tokens)
python -m train.train --stage knowledge   # ~30 min
python -m train.train --stage qa          # ~10 min
#    interrupted? add --resume to continue from checkpoints/<stage>_latest.pt

# 6. Evaluate -> eval/results.json and eval/report.md
python -m eval.evaluate

# 7. Ask from the command line
python -m model.generate "How many casual leaves do I get?"

# 8. Run the API (terminal 1)
uvicorn app.api:app --port 8000
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "How long is maternity leave?"}'

# 9. Run the UI (terminal 2) -> http://localhost:8501
streamlit run app/ui.py
```

Or run everything with `./run_all.sh` (or `./run_all.sh --quick`).

## Outputs

- `checkpoints/<stage>.pt` final weights per stage; `checkpoints/<stage>_latest.pt` for `--resume`
- `logs/<stage>_log.csv` train/val loss, learning rate and tokens/sec; `logs/<stage>_loss.png` loss plot
- `eval/results.json`, `eval/report.md`

## Training details

| Stage | Data | LR (warmup + cosine) | Steps | Val loss measured on |
|---|---|---|---|---|
| 1 `pretrain` | TinyStories | 6e-4 | 8000 × 32 × 256 tokens | held-out stories |
| 2 `knowledge` | policy statements + 15% stories | 1e-4 | 2000 | held-out Q&A answers |
| 3 `qa` | Q&A examples, question tokens masked | 5e-5 | 3000 | held-out Q&A answers |

All stages use AdamW (β = 0.9/0.95, weight decay 0.1), gradient clipping 1.0, dropout 0.1, and auto-detect
the device (`mps` → `cuda` → `cpu`). Override the step count with `--max_steps N`.

## Notes and limitations

- The company (Arcvale Technologies) and all of its policies are fictional.
- A 12M-parameter model can still produce fluent but wrong answers. The UI shows a disclaimer, and
  `eval/report.md` lists hallucination examples.
- The test set checks that the model has learned facts in a way that **generalises across wordings**.
  It does not test facts the model never saw. Questions outside the policies should get the refusal
  answer `I don't know. This is not covered in the company policies.`
- To add or change a policy, edit the CSV and retrain stages 2–3. Stage 1 does not need to be repeated.
