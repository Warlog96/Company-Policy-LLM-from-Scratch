"""
Evaluate the trained model. Three kinds of held-out questions, none used in training:

  template    a held-out template wording of every fact ("Quick question: Am I allowed to ...")
  natural     a held-out hand-written wording of every fact ("how many WFH days per week")
  oos         out-of-scope questions the model has never seen (should be refused)
  real-world  data/real_world_test.csv: a separate hand-written set (typos, slang, situations)

Metrics
  answer accuracy   in-scope: every key value of the reference (numbers, times, amounts,
                    emails) appears in the model's answer. For facts without numbers,
                    >= 60% of the reference's content words must appear.
                    (real-world set: the "must_contain" text must appear)
  source accuracy   in-scope: predicted [source] tag exactly equals the reference.
  refusal accuracy  out-of-scope: the model says "I don't know".
  false refusals    in-scope questions the model wrongly refused.
  hallucinations    a confident answer that is wrong.

Decoding is greedy (temperature 0) so results are reproducible.

  python -m eval.evaluate                      # uses checkpoints/qa.pt
  python -m eval.evaluate --ckpt checkpoints/quick/qa.pt
"""
import argparse
import csv
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.generate import KnowledgeBot  # noqa: E402

TEST_PATH = os.path.join(ROOT, "data", "processed", "test.jsonl")
REAL_WORLD_PATH = os.path.join(ROOT, "data", "real_world_test.csv")
OUT_JSON = os.path.join(ROOT, "eval", "results.json")
OUT_MD = os.path.join(ROOT, "eval", "report.md")
KINDS = ["template", "natural", "oos", "real-world"]

STOPWORDS = set("""a an the and or of to in on at for by with from is are be been was were
can may must will your you their there this that these those after before into than then
each every per any all not no yes our its it as up out only""".split())


def normalise(s):
    return " ".join(s.lower().replace(",", "").split())


def key_values(answer):
    """Numbers / times / amounts / emails that must appear in a correct answer."""
    vals = re.findall(r"[\w.]+@[\w.]+\w", answer)                    # emails
    text = re.sub(r"[\w.]+@[\w.]+\w", " ", answer)
    # 12, 1.5, 9:30, 80%, 1,500, and ordinals like 21st / 20th (kept whole so "21st" matches "21st")
    vals += [v.rstrip(".") for v in re.findall(r"\d[\d,.:]*(?:%|st|nd|rd|th)?", text)]
    return [normalise(v) for v in vals]


def content_words(answer):
    words = re.findall(r"[a-z]+", answer.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def is_refusal(text):
    return "i don't know" in text.lower() or "not covered in the company policies" in text.lower()


def answer_correct(pred, ref):
    p = normalise(pred)
    keys = key_values(ref)
    if keys:
        # whole-token match, so "2" does not match inside "12"
        return all(re.search(r"(?<![\w.]){}(?![\w])".format(re.escape(k)), p) for k in keys)
    words = content_words(ref)
    hits = sum(1 for w in words if w in p)
    return bool(words) and hits / len(words) >= 0.6


def load_tests():
    """All test questions as dicts with: question, kind, in_scope, answer, source, topic, must_contain."""
    tests = []
    with open(TEST_PATH, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            tests.append({"question": ex["question"], "kind": ex["kind"], "in_scope": ex["in_scope"],
                          "answer": ex["answer"], "source": ex["source"], "topic": ex["topic"],
                          "must_contain": None})
    if os.path.exists(REAL_WORLD_PATH):
        with open(REAL_WORLD_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tests.append({"question": r["question"], "kind": "real-world", "in_scope": bool(r["source"]),
                              "answer": r["must_contain"], "source": r["source"], "topic": "real-world",
                              "must_contain": r["must_contain"] or None})
    return tests


def pct(xs):
    return round(100.0 * sum(xs) / max(1, len(xs)), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    bot = KnowledgeBot(args.ckpt)
    tests = load_tests()
    rows, t0 = [], time.time()
    for i, ex in enumerate(tests):
        out = bot.answer(ex["question"], temperature=0)
        refused = is_refusal(out["answer"])
        row = dict(ex, pred_answer=out["answer"], pred_source=out["source"], refused=refused)
        if ex["in_scope"]:
            if ex["must_contain"]:
                ok = normalise(ex["must_contain"]) in normalise(out["answer"])
            else:
                ok = answer_correct(out["answer"], ex["answer"])
            row["answer_correct"] = (not refused) and ok
            row["source_correct"] = out["source"] == ex["source"]
            row["hallucination"] = (not refused) and not row["answer_correct"]
        else:
            row["refusal_correct"] = refused
            row["hallucination"] = not refused
        rows.append(row)
        if (i + 1) % 50 == 0:
            print("  {}/{} evaluated".format(i + 1, len(tests)))

    # ---- metrics per kind of question
    by_kind = {}
    for kind in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        if not rs:
            continue
        ins = [r for r in rs if r["in_scope"]]
        oos = [r for r in rs if not r["in_scope"]]
        m = {"n_in_scope": len(ins), "n_out_of_scope": len(oos)}
        if ins:
            m.update(answer_accuracy=pct([r["answer_correct"] for r in ins]),
                     source_accuracy=pct([r["source_correct"] for r in ins]),
                     answer_and_source_accuracy=pct([r["answer_correct"] and r["source_correct"] for r in ins]),
                     false_refusal_rate=pct([r["refused"] for r in ins]),
                     hallucination_rate=pct([r["hallucination"] for r in ins]))
        if oos:
            m["refusal_accuracy"] = pct([r["refusal_correct"] for r in oos])
        by_kind[kind] = m

    per_topic = {}
    for r in rows:
        if r["in_scope"] and r["kind"] in ("template", "natural"):
            t = per_topic.setdefault(r["topic"], {"n": 0, "answer": 0, "source": 0})
            t["n"] += 1
            t["answer"] += r["answer_correct"]
            t["source"] += r["source_correct"]

    metrics = {"checkpoint": bot.info["checkpoint"], "by_kind": by_kind,
               "eval_seconds": round(time.time() - t0, 1)}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "per_topic": per_topic,
                   "hallucination_examples": [r for r in rows if r["hallucination"]][:30],
                   "all": rows}, f, indent=2, ensure_ascii=False)
    write_report(metrics, per_topic, rows)
    print(json.dumps(metrics, indent=2))
    print("Saved", OUT_JSON, "and", OUT_MD)


def md_cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def write_report(m, per_topic, rows):
    def cell(kind, key):
        v = m["by_kind"].get(kind, {}).get(key)
        return "-" if v is None else "{}%".format(v)

    bk = m["by_kind"]
    L = ["# Evaluation report", "",
         "Checkpoint: `{}`. Greedy decoding. None of these questions were used in training.".format(m["checkpoint"]),
         "", "## Summary", "",
         "| Metric | Template wording | Natural wording | Real-world set |", "|---|---|---|---|",
         "| In-scope questions | {} | {} | {} |".format(*[bk.get(k, {}).get("n_in_scope", 0) for k in ("template", "natural", "real-world")])]
    for label, key in [("Answer accuracy", "answer_accuracy"), ("Source accuracy", "source_accuracy"),
                       ("Answer + source correct", "answer_and_source_accuracy"),
                       ("False refusals", "false_refusal_rate"), ("Hallucination rate", "hallucination_rate")]:
        L.append("| {} | {} | {} | {} |".format(label, cell("template", key), cell("natural", key), cell("real-world", key)))
    L += ["", "| Out-of-scope | Questions | Refusal accuracy |", "|---|---|---|",
          "| Held-out out-of-scope set | {} | {} |".format(bk.get("oos", {}).get("n_out_of_scope", 0), cell("oos", "refusal_accuracy")),
          "| Real-world out-of-scope | {} | {} |".format(bk.get("real-world", {}).get("n_out_of_scope", 0), cell("real-world", "refusal_accuracy")),
          "", "**Template** = a held-out reworded template question for every fact. "
          "**Natural** = a held-out hand-written rewording for every fact. "
          "**Real-world** = a separate hand-written set with slang, typos and situations (`data/real_world_test.csv`).",
          "", "## Per topic (template + natural)", "", "| Topic | Questions | Answer acc. | Source acc. |", "|---|---|---|---|"]
    for t, v in sorted(per_topic.items()):
        L.append("| {} | {} | {:.0f}% | {:.0f}% |".format(t, v["n"], 100 * v["answer"] / v["n"], 100 * v["source"] / v["n"]))

    hard = [r for r in rows if r["kind"] in ("natural", "real-world", "oos")]
    good = [r for r in hard if r.get("answer_correct") and r.get("source_correct") or r.get("refusal_correct")]
    bad = [r for r in hard if r["hallucination"] or (r["in_scope"] and not (r["answer_correct"] and r["source_correct"]))]
    for title, items in [("10 good answers (natural / real-world)", good[:10]), ("Bad answers / hallucinations", bad[:15])]:
        L += ["", "## " + title, "", "| Set | Question | Model answer | Model source | Expected |", "|---|---|---|---|---|"]
        for r in items:
            if r["in_scope"]:
                expected = (r["answer"] if not r["must_contain"] else "contains: " + r["must_contain"]) + " [{}]".format(r["source"])
            else:
                expected = "refusal"
            L.append("| {} | {} | {} | {} | {} |".format(r["kind"], md_cell(r["question"]), md_cell(r["pred_answer"]),
                                                       md_cell(r["pred_source"] or "-"), md_cell(expected)))
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
