"""
Turn the company policy CSVs into training text for the model.

Inputs (in data/):
  company_facts.csv         topic, question, answer, source   (200 facts)
  company_paraphrases.csv   5 natural, hand-written rewordings per fact ("|"-separated)
  out_of_scope.csv          questions the policies do NOT cover (answered with a refusal)
  real_world_test.csv       hand-written evaluation questions - NEVER used for training

Outputs (in data/processed/):
  train.jsonl / val.jsonl / test.jsonl   Q&A examples (+ metadata for evaluation)
  train.txt                              the same training examples as plain text (easy to inspect)
  knowledge.txt                          facts written as plain statements (used in stage 2)

Every Q&A example is formatted exactly as:
  Q: <question>\nA: <answer> [<source>] <eos>

Where the training questions come from (per fact):
  - template paraphrases: word rewrites ("Can I" -> "Am I allowed to") x wrappers ("Quick question: ...")
  - natural paraphrases: the hand-written rewordings ("how many WFH days per week")
  - noisy copies: lower-case / no punctuation / one typo, so real typing still works

Split strategy ("split by fact"): every fact appears in training, but for EVERY fact we hold back
one template wording and one natural wording for the test set (and some for validation).
So the test asks about known facts with wording the model has never seen.
Each test row has a "kind": "template", "natural" or "oos" (out of scope).
"""
import csv
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CSV_PATH = os.path.join(DATA, "company_facts.csv")
PARA_PATH = os.path.join(DATA, "company_paraphrases.csv")
OOS_PATH = os.path.join(DATA, "out_of_scope.csv")
REAL_WORLD_PATH = os.path.join(DATA, "real_world_test.csv")
OUT_DIR = os.path.join(DATA, "processed")

COMPANY = "Arcvale"
EOS = "<eos>"
REFUSAL = "I don't know. This is not covered in the company policies."

# ---------------------------------------------------------------------------
# Paraphrasing: (1) word-level rewrites that keep the meaning, (2) wrappers
# ---------------------------------------------------------------------------
REWRITES = [
    (r"\bdo I get\b", "does an employee get"),
    (r"\bCan I\b", "Am I allowed to"),
    (r"\bcan I\b", "am I allowed to"),
    (r"\bWhat is\b", "What's"),
    (r"\bper year\b", "each year"),
    (r"\bper year\b", "annually"),
    (r"\bper month\b", "each month"),
    (r"\bIs there\b", "Do we have"),
    (r"\bHow do I\b", "What is the process to"),
    (r"\bWhen do I get\b", "When will I receive"),
    (r"\bWhat happens if I\b", "What is the consequence if I"),
    (r"\bthe company\b", COMPANY),
    (r"\boffice\b", "workplace"),
    (r"\ballowed\b", "permitted"),
    (r"\bleaves\b", "leave days"),
    (r"\bmy work laptop\b", "my company laptop"),
    (r"\bHow long is\b", "What is the duration of"),
    (r"\bHow often\b", "How frequently"),
]

WRAPPERS = [
    "{q}",
    "Can you tell me this: {q}",
    "Quick question: {q}",
    "Hi, {ql}",
    "I'm a new employee. {q}",
    "As an Arcvale employee, {ql}",
    "Please help. {q}",
    "HR question - {q}",
    "{ql_noqm}",
    "I wanted to know, {ql}",
    "Hello! {q} Thanks.",
    "Regarding {topic}: {q}",
    "Could you check the policy? {q}",
]


def lower_first(s):
    """Lowercase the first letter ('How many' -> 'how many'), but keep 'I ...' as is."""
    if s.startswith("I ") or len(s) < 2 or s[1].isupper():
        return s
    return s[0].lower() + s[1:]


def rewrite_variants(q):
    """The question itself plus every single-rule rewrite that changes it."""
    variants = [q]
    for pattern, repl in REWRITES:
        new = re.sub(pattern, repl, q, count=1)
        if new != q and new not in variants:
            variants.append(new)
    return variants


def paraphrases(q, topic, n, rng):
    """Return n distinct paraphrases of q. The original question is always first."""
    pool = []
    for v in rewrite_variants(q):
        for w in WRAPPERS:
            p = w.format(q=v, ql=lower_first(v), ql_noqm=lower_first(v).rstrip("?"), topic=topic)
            if p != q and p not in pool:
                pool.append(p)
    rng.shuffle(pool)
    return [q] + pool[: n - 1]


def answer_phrasings(answer, source, rng):
    """2-3 phrasings of the same answer. The fact itself never changes."""
    policy_name = source.split("§")[0].strip()   # "HR Leave Policy §3" -> "HR Leave Policy"
    options = [
        answer,
        "As per company policy, " + lower_first(answer),
        "According to the " + policy_name + ", " + lower_first(answer),
    ]
    return options[: rng.choice([2, 3])]


def format_example(question, answer, source=None):
    """The one and only training format."""
    if source:
        return "Q: {}\nA: {} [{}] {}".format(question, answer, source, EOS)
    return "Q: {}\nA: {} {}".format(question, answer, EOS)


# ---------------------------------------------------------------------------
# Noise: real employees type in lower case, skip punctuation and make typos
# ---------------------------------------------------------------------------
def casual(q):
    """'What is the dress code?' -> 'what is the dress code'"""
    return re.sub(r"[?!.]+$", "", q).lower().strip()


def typo(q, rng):
    """Swap two neighbouring letters inside one longer word: 'casual' -> 'csaual'."""
    words = q.split()
    long_words = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not long_words:
        return casual(q)
    i = rng.choice(long_words)
    w = words[i]
    j = rng.randint(1, len(w) - 3)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def noisy(q, rng):
    return casual(q) if rng.random() < 0.5 else typo(q, rng)


def norm(q):
    """Normalise a question for leak checks."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", q.lower()).split())


# Plain-statement templates for stage 2 ("knowledge" stage)
STATEMENT_TEMPLATES = [
    "{answer} [{source}]",
    "Arcvale {topic} policy: {answer} [{source}]",
    "{source} says: {answer}",
    "At Arcvale Technologies, {answer_l} This is stated in {source}.",
]


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rng = random.Random(1234)
    facts = read_csv(CSV_PATH)
    natural = {r["question"]: [p.strip() for p in r["paraphrases"].split("|") if p.strip()]
               for r in read_csv(PARA_PATH)}
    missing = [f["question"] for f in facts if f["question"] not in natural]
    assert not missing, "facts without natural paraphrases: {}".format(missing)
    oos = [r["question"] for r in read_csv(OOS_PATH)]
    real_world = set(norm(r["question"]) for r in read_csv(REAL_WORLD_PATH))

    train, val, test = [], [], []

    def add(rows, q, meta, kind, answers):
        for a in answers:
            src = meta["source"] or None
            rows.append(dict(meta, question=q, kind=kind, text=format_example(q, a, src)))

    for fact_id, row in enumerate(facts):
        q, a, src, topic = row["question"], row["answer"], row["source"], row["topic"]
        meta = {"fact_id": fact_id, "topic": topic, "answer": a, "source": src, "in_scope": True}

        # 1) template paraphrases: original + 5 variants, one variant held out for test
        tq = paraphrases(q, topic, 6, rng)
        # held-out wording must differ from the original even after lower-casing / removing "?"
        test_template = next(x for x in reversed(tq) if norm(x) != norm(q))
        train_qs = [x for x in tq if norm(x) != norm(test_template)]

        # 2) natural paraphrases: one held out for test, one for validation on ~40% of facts
        nat = natural[q][:]
        rng.shuffle(nat)
        test_natural = nat[0]
        val_natural = [nat[1]] if rng.random() < 0.4 else []
        nat_train = [x for x in nat[1:] if x not in val_natural]
        train_qs += nat_train

        # 3) noisy copies of the original + natural questions, and some natural ones with a wrapper
        for x in [q] + nat_train:
            if rng.random() < 0.7:
                train_qs.append(noisy(x, rng))
        for x in nat_train:
            if rng.random() < 0.3:
                w = rng.choice(WRAPPERS[1:8])
                train_qs.append(w.format(q=x, ql=lower_first(x), ql_noqm=x.rstrip("?"), topic=topic))

        for x in dict.fromkeys(train_qs):                       # de-duplicate, keep order
            add(train, x, meta, "train", answer_phrasings(a, src, rng))
        for x in val_natural:
            add(val, x, meta, "natural", [a])
        add(test, test_template, meta, "template", [a])
        add(test, test_natural, meta, "natural", [a])

    # Out-of-scope questions: disjoint train / val / test questions
    rng.shuffle(oos)
    oos_test, oos_val, oos_train = oos[:15], oos[15:22], oos[22:]
    oos_meta = {"fact_id": -1, "topic": "out of scope", "answer": REFUSAL, "source": "", "in_scope": False}
    for q in oos_train:
        variants = [q, rng.choice(WRAPPERS[1:8]).format(q=q, ql=lower_first(q), ql_noqm=q.rstrip("?"), topic="general"),
                    casual(q), typo(q, rng)]
        for x in dict.fromkeys(variants):
            add(train, x, oos_meta, "train", [REFUSAL])
    for q in oos_val:
        add(val, q, oos_meta, "oos", [REFUSAL])
    for q in oos_test:
        add(test, q, oos_meta, "oos", [REFUSAL])

    # Leak checks: no test / validation / real-world question may appear in training
    train_norm = set(norm(ex["question"]) for ex in train)
    leaks = [ex["question"] for ex in test + val if norm(ex["question"]) in train_norm]
    assert not leaks, "test/val leak: {}".format(leaks[:5])
    rw_leaks = [q for q in real_world if q in train_norm]
    assert not rw_leaks, "real-world test leak: {}".format(rw_leaks)

    rng.shuffle(train)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(OUT_DIR, name + ".jsonl"), "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(ex["text"] for ex in train) + "\n")

    # Knowledge statements (stage 2): each fact written as plain sentences
    statements = []
    for row in facts:
        for t in STATEMENT_TEMPLATES:
            s = t.format(answer=row["answer"], answer_l=lower_first(row["answer"]),
                         source=row["source"], topic=row["topic"])
            statements.append(s + " " + EOS)
    with open(os.path.join(OUT_DIR, "knowledge.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(statements) + "\n")

    def count(rows, kind):
        return sum(1 for e in rows if e["kind"] == kind)

    n_oos_train = sum(1 for e in train if not e["in_scope"])
    print("facts: {} | train examples: {} ({} refusals) | unique train questions: {}".format(
        len(facts), len(train), n_oos_train, len(train_norm)))
    print("val: {} | test: {} (template {}, natural {}, out-of-scope {})".format(
        len(val), len(test), count(test, "template"), count(test, "natural"), count(test, "oos")))
    print("knowledge statements: {} | real-world test questions kept out of training: {}".format(
        len(statements), len(real_world)))
    print("example:\n" + train[0]["text"])


if __name__ == "__main__":
    main()
