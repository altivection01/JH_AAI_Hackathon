#!/usr/bin/env python
"""
RiskGuardian — GenAI predictor: few-shot classification with a LOCAL Hugging Face model.

Mixtral-8x22B — the LARGEST by total parameters; a sparse MoE, so inference is fast for its size.

Runs the SAME engineered few-shot prompt as the API classifiers, but entirely LOCAL via
`transformers` with 4-bit (nf4) quantization, so the model fits in 128 GB. The label is
decided by comparing the next-token logits of "0" vs "1" — deterministic, and it never fails.
Writes:
    submission2.mixtral-8x22b.csv   <- predictions (pure GenAI / FAQ #8; no classical fallback)

Model: mistralai/Mixtral-8x22B-Instruct-v0.1   (141B sparse-MoE (≈39B active), ≈80 GB at 4-bit nf4)

Requires (MLENV311 + ) transformers, accelerate, bitsandbytes, torch. A CUDA GPU enables
true 4-bit; on a CPU / Apple-silicon 128 GB box a GGUF build (llama-cpp-python) is the
recommended quantized path (same weights, same prompt) — see the notebook's hardware note.

Usage:
    python riskguardian_hf_mixtral_classifier.py                 # validate (40 rows) + classify the 70 test rows
    python riskguardian_hf_mixtral_classifier.py --no-validate   # skip validation, just classify + submit
    python riskguardian_hf_mixtral_classifier.py --dry-run       # no model load: just print a sample prompt
    HF_MODEL=... python riskguardian_hf_mixtral_classifier.py    # override the model id

The corpus is a *combination*: ~1/3 char-obfuscated synthetic risk events (cyber=0 / financial=1)
and ~2/3 the ISOT fake/real news set (fake clickbait=0 / real Reuters=1). The engineered prompt
encodes that structure; few-shot examples span all four quadrants.
"""
import argparse, os, re
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_CSV  = "combined_risk_train.csv"
TEST_CSV   = "combined_risk_test.csv"
SAMPLE_CSV = "Sample_Submission.csv"

MODEL_ID = os.environ.get("HF_MODEL", "mistralai/Mixtral-8x22B-Instruct-v0.1")
OUT_PATH = "submission2.mixtral-8x22b.csv"
MAX_INPUT_TOKENS = int(os.environ.get("HF_MAX_INPUT_TOKENS", "8192"))


# ── data: load + tag the two sub-corpora (obfuscated risk vs clean news) ───
def load_frames():
    tr = pd.read_csv(TRAIN_CSV); te = pd.read_csv(TEST_CSV)
    wl = None
    for p in ("/usr/share/dict/words", "/usr/dict/words"):
        if Path(p).exists():
            wl = set(w.strip().lower() for w in open(p) if len(w.strip()) >= 4); break
    def cleanliness(t):
        if wl is None: return 0.5
        tk = re.findall(r"[A-Za-z]{4,}", t.lower())
        return sum(w in wl for w in tk) / len(tk) if tk else 0.0
    for d in (tr, te):
        d["clean"]   = d["text"].map(cleanliness)
        d["is_obf"]  = d["clean"] < 0.55
        d["reuters"] = d["text"].str.contains("Reuters", case=True)
    return tr, te


# ── the engineered few-shot prompt (identical across every provider) ───────
SYSTEM = (
    "You classify RiskGuardian items as 0 or 1. The corpus mixes two text types:\n"
    "(A) Risk reports (words may be scrambled/typo'd): 0 = CYBERSECURITY risk "
    "(hacking, malware, breaches, networks, intrusion, DDoS, cloud, ransomware); "
    "1 = FINANCIAL risk (markets, portfolios, debt, liquidity, credit, financial regulation/compliance).\n"
    "(B) News articles (clean English): 0 = sensational/opinion clickbait (ALL-CAPS hooks, 'WATCH:', "
    "'Featured image via Getty Images'); 1 = neutral newswire reporting (e.g. Reuters; datelines like "
    "'WASHINGTON (Reuters) -'; sober 'X said' attribution).\n"
    "Reply with ONLY one character: 0 or 1."
)
SYSTEM_NAIVE = ("Classify the risk event as 0 = Cybersecurity or 1 = Financial. "
                "Reply with ONLY one character: 0 or 1.")


# ── few-shot bank: 2 examples per quadrant, deterministic, truncated ───────
def trunc(t, n):
    return " ".join(str(t).split())[:n]

def build_shots(rt_tr, seed=7):
    """Few-shot bank: 2 examples per quadrant, deterministic, truncated."""
    rng = np.random.default_rng(seed)
    def pick(mask, n):
        idx = np.asarray(rt_tr.index[np.asarray(mask)])
        return list(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else []
    shot_idx = (pick((rt_tr.is_obf) & (rt_tr.label == 0), 2) +
                pick((rt_tr.is_obf) & (rt_tr.label == 1), 2) +
                pick((~rt_tr.is_obf) & (rt_tr.label == 0) & (~rt_tr.reuters), 2) +
                pick((~rt_tr.is_obf) & (rt_tr.label == 1) & (rt_tr.reuters), 2))
    shots = "\n\n".join(f"TEXT: {trunc(rt_tr.text[i], 300)}\nLABEL: {int(rt_tr.label[i])}" for i in shot_idx)
    return shot_idx, shots, rng


# ── local model: load once, classify by 0-vs-1 next-token logits ───────────
_STATE = {"model": None, "tok": None, "zero": None, "one": None}


def load_model(model_id=MODEL_ID):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print(f"loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(low_cpu_mem_usage=True, device_map="auto")
    if torch.cuda.is_available():
        kw.update(torch_dtype=torch.bfloat16,
                  quantization_config=BitsAndBytesConfig(
                      load_in_4bit=True, bnb_4bit_quant_type="nf4",
                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    else:
        kw.update(torch_dtype=torch.bfloat16)
        print("WARNING: no CUDA GPU detected -> bitsandbytes 4-bit is unavailable. For a 128 GB "
              "CPU / Apple-silicon box, run a GGUF build of this model via llama-cpp-python "
              "(same weights, same prompt); loading in bfloat16 here may exceed 128 GB for 70B+ models.")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    _STATE.update(model=model, tok=tok,
                  zero=tok("0", add_special_tokens=False).input_ids[-1],
                  one=tok("1", add_special_tokens=False).input_ids[-1])
    return model, tok


def make_messages(text, system, shots):
    user = (shots + "\n\n" if shots else "") + f"TEXT: {trunc(text, 1500)}\nLABEL:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def classify(text, shots, system=None):
    import torch
    if system is None:
        system = SYSTEM
    model, tok = _STATE["model"], _STATE["tok"]
    msgs = make_messages(text, system, shots)
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        ids = tok(system + "\n\n" + msgs[1]["content"], return_tensors="pt").input_ids
    ids = ids[:, -MAX_INPUT_TOKENS:].to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return 0 if float(logits[_STATE["zero"]]) >= float(logits[_STATE["one"]]) else 1


def classify_many(texts, shots, system=None):
    return [classify(t, shots, system=system) for t in texts]


# ── validation: honest accuracy estimate on a labeled holdout ──────────────
def validate(rt_tr, shots, shot_idx, rng, per_quad=10):
    quads = [((rt_tr.is_obf) & (rt_tr.label == 0), "obf/cyber(0)"),
             ((rt_tr.is_obf) & (rt_tr.label == 1), "obf/financial(1)"),
             ((~rt_tr.is_obf) & (rt_tr.label == 0), "news/fake(0)"),
             ((~rt_tr.is_obf) & (rt_tr.label == 1), "news/real(1)")]
    shot = set(int(i) for i in shot_idx)
    val = []
    for m, name in quads:
        pool = [int(i) for i in np.asarray(rt_tr.index[np.asarray(m)]) if int(i) not in shot]
        val += [(int(i), name) for i in rng.choice(pool, size=min(per_quad, len(pool)), replace=False)]
    vt = [rt_tr.text[i] for i, _ in val]
    vy = [int(rt_tr.label[i]) for i, _ in val]

    print(f"Validating engineered prompt on {len(val)} held-out labeled rows...")
    vp = classify_many(vt, shots, system=SYSTEM)
    print(f"  engineered-prompt accuracy: {np.mean([p == y for p, y in zip(vp, vy)]):.3f}")
    dfv = pd.DataFrame({"quad": [q for _, q in val], "y": vy, "p": vp})
    print(dfv.assign(correct=lambda d: d.p == d.y).groupby("quad").correct.mean().round(3).to_string())

    sub_i = list(range(0, len(val), 2))                            # every other row -> naive comparison
    npv = classify_many([vt[i] for i in sub_i], shots, system=SYSTEM_NAIVE)
    nacc = np.mean([npv[k] == vy[sub_i[k]] for k in range(len(sub_i))])
    eacc = np.mean([vp[sub_i[k]] == vy[sub_i[k]] for k in range(len(sub_i))])
    print(f"\nprompt-engineering lift on the same {len(sub_i)} rows ->  naive: {nacc:.3f}   engineered: {eacc:.3f}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Local HF few-shot classifier for the RiskGuardian test set.")
    ap.add_argument("--no-validate", action="store_true", help="skip the labeled-holdout validation pass")
    ap.add_argument("--dry-run", action="store_true", help="no model load: just print a sample prompt")
    ap.add_argument("--compare", default="submission2.csv", help="read-only: compare results against this CSV")
    args = ap.parse_args()

    rt_tr, rt_te = load_frames()
    print(f"train: {rt_tr.shape} | test: {rt_te.shape} | "
          f"test composition: {int(rt_te.is_obf.sum())} obfuscated / {int((~rt_te.is_obf).sum())} clean")
    shot_idx, shots, rng = build_shots(rt_tr)
    print(f"model: {MODEL_ID} | few-shot examples: {len(shot_idx)} (2 per quadrant) | out: {OUT_PATH}")

    if args.dry_run:
        print("\n[dry-run] sample prompt for test id=0 (no model load):\n" + "-" * 60)
        print("SYSTEM:\n" + SYSTEM + "\n\nUSER:\n" + make_messages(rt_te.text[0], SYSTEM, shots)[1]["content"][:1200] + " ...")
        return

    load_model()
    print("smoke test (expect 1):",
          classify("Quarterly portfolio drawdown and liquidity stress across credit positions.", shots))

    if not args.no_validate:
        print()
        validate(rt_tr, shots, shot_idx, rng)

    print(f"\nClassifying {len(rt_te)} test rows with {MODEL_ID} ...")
    preds = np.array(classify_many(rt_te["text"].tolist(), shots, system=SYSTEM)).astype(int)

    sample = pd.read_csv(SAMPLE_CSV)
    pd.DataFrame({"id": rt_te["id"].values, "label": preds})[list(sample.columns)].to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} = {len(preds)} predictions ({MODEL_ID}, local few-shot prompting)")
    print("label counts:", pd.Series(preds).value_counts().to_dict())

    cmp_path = args.compare
    if cmp_path and Path(cmp_path).exists() and os.path.abspath(cmp_path) != os.path.abspath(OUT_PATH):
        other = pd.read_csv(cmp_path)
        if list(other.columns) == list(sample.columns) and len(other) == len(preds):
            a = other["label"].to_numpy()
            print(f"\nvs {cmp_path}: agreement {int((a == preds).sum())}/{len(preds)}")
            for i in range(len(rt_te)):
                if a[i] != preds[i]:
                    typ = "obf" if rt_te.is_obf[i] else ("news+Reuters" if rt_te.reuters[i] else "news")
                    print(f"   diff id={int(rt_te.id[i])} {cmp_path}={int(a[i])} pred={int(preds[i])} [{typ}] {trunc(rt_te.text[i], 80)}")


if __name__ == "__main__":
    main()
