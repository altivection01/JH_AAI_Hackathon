#!/usr/bin/env python
"""
RiskGuardian — GenAI predictor: few-shot LLM classification with Anthropic Claude.

NOTE: PLEASE DISREGARD PRIOR SUBMISSIONS- we were using vanilla ML techniques (TF-IDF + LinearSVC) 
NOT GENAI as instructed in the FAQ. 
THIS VERSION implements a few-shot prompting approach with an Anthropic Claude model: 

It few-shot-prompts Claude to label each test row 0 (cybersecurity / fake-news) or
1 (financial / real-Reuters-news), and writes:
    submission.csv            <- LLM predictions (PRIMARY, GenAI / FAQ #8)
    submission_classical.csv  <- TF-IDF + LinearSVC backstop (~99.99% CV); also fills any failed call

The corpus is a *combination*: ~1/3 char-obfuscated synthetic risk events (cyber=0 / financial=1)
and ~2/3 the ISOT fake/real news set (fake clickbait=0 / real Reuters=1). The engineered prompt
encodes that structure; few-shot examples span all four quadrants.

Usage (run with the MLENV311 interpreter):
    python riskguardian_llm_classifier.py                 # validate (40 rows) + classify the 70 test rows
    python riskguardian_llm_classifier.py --no-validate   # skip validation, just classify + submit
    python riskguardian_llm_classifier.py --dry-run       # no API calls: classical only + show a sample prompt
    ANTHROPIC_MIN_INTERVAL=1.0 python riskguardian_llm_classifier.py --workers 6   # tune pacing/concurrency

Requires ANTHROPIC_API_KEY in the environment or in ./config.json.
"""
import argparse, json, os, re, time, threading, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

TRAIN_CSV  = "combined_risk_train.csv"
TEST_CSV   = "combined_risk_test.csv"
SAMPLE_CSV = "Sample_Submission.csv"
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")


# ── config / API key ──────────────────────────────────────────────────────
def load_api_key():
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k and Path("config.json").exists():
        k = json.load(open("config.json")).get("ANTHROPIC_API_KEY")
    return k


# ── data: load + tag the two sub-corpora (obfuscated risk vs clean news) ───
def load_frames():
    tr = pd.read_csv(TRAIN_CSV)
    te = pd.read_csv(TEST_CSV)
    wl = None
    for p in ("/usr/share/dict/words", "/usr/dict/words"):
        if Path(p).exists():
            wl = set(w.strip().lower() for w in open(p) if len(w.strip()) >= 4)
            break

    def cleanliness(t):
        if wl is None:
            return 0.5
        tk = re.findall(r"[A-Za-z]{4,}", t.lower())
        return sum(w in wl for w in tk) / len(tk) if tk else 0.0

    for d in (tr, te):
        d["clean"]   = d["text"].map(cleanliness)
        d["is_obf"]  = d["clean"] < 0.55
        d["reuters"] = d["text"].str.contains("Reuters", case=True)
    return tr, te


# ── classical backstop: word + char_wb TF-IDF -> LinearSVC (Part XVI) ──────
def text_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word",    ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300_000)),
    ])


def classical_predict(rt_tr, rt_te):
    pipe = Pipeline([("feats", text_features()), ("clf", LinearSVC(C=1.0))]).fit(rt_tr.text, rt_tr.label)
    return np.asarray(pipe.predict(rt_te.text)).astype(int)


# ── LLM few-shot classifier (Anthropic Claude) — Part XVII ─────────────────
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


# Global pacer so all worker threads stay under the API rate limit.
_rl_lock = threading.Lock()
_rl_last = [0.0]
MIN_INTERVAL = float(os.environ.get("ANTHROPIC_MIN_INTERVAL", "1.5"))  # ~40 req/min


def _throttle():
    with _rl_lock:
        wait = _rl_last[0] + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _rl_last[0] = time.time()


def make_user_prompt(text, shots):
    return (shots + "\n\n" if shots else "") + f"TEXT: {trunc(text, 1500)}\nLABEL:"


def classify(text, api_key, shots, system=SYSTEM, model=ANTHROPIC_MODEL, retries=8):
    payload = {"model": model, "max_tokens": 5, "temperature": 0, "system": system,
               "messages": [{"role": "user", "content": make_user_prompt(text, shots)}]}
    delay = 2.0
    for a in range(retries):
        try:
            _throttle()
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json=payload, timeout=60)
            if r.status_code == 200:
                m = re.search(r"[01]", r.json()["content"][0]["text"])
                return int(m.group()) if m else -1
            if r.status_code in (429, 500, 502, 503, 529):          # rate-limited / transient -> back off
                ra = r.headers.get("retry-after")
                wait = float(ra) if (ra and ra.replace(".", "", 1).isdigit()) else delay
                time.sleep(min(wait, 30) + random.uniform(0, 0.5))
                delay = min(delay * 2, 30)
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 30)
    print("  gave up on one row after", retries, "tries")
    return -1                                                       # caller falls back to classical


def classify_many(texts, api_key, shots, system=SYSTEM, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda t: classify(t, api_key, shots, system=system), texts))


# ── validation: honest accuracy estimate on a labeled holdout ──────────────
def validate(rt_tr, api_key, shots, shot_idx, rng, workers=4, per_quad=10):
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
    vp = classify_many(vt, api_key, shots, system=SYSTEM, workers=workers)
    ok = [p == y for p, y in zip(vp, vy) if p != -1]
    print(f"  engineered-prompt accuracy: {np.mean(ok):.3f}   (call failures: {sum(p == -1 for p in vp)})")
    dfv = pd.DataFrame({"quad": [q for _, q in val], "y": vy, "p": vp})
    print(dfv.assign(correct=lambda d: d.p == d.y).groupby("quad").correct.mean().round(3).to_string())

    sub_i = list(range(0, len(val), 2))                            # every other row -> naive comparison
    npv = classify_many([vt[i] for i in sub_i], api_key, shots, system=SYSTEM_NAIVE, workers=workers)
    nacc = np.mean([npv[k] == vy[sub_i[k]] for k in range(len(sub_i)) if npv[k] != -1])
    eacc = np.mean([vp[sub_i[k]] == vy[sub_i[k]] for k in range(len(sub_i)) if vp[sub_i[k]] != -1])
    print(f"\nprompt-engineering lift on the same {len(sub_i)} rows ->  naive: {nacc:.3f}   engineered: {eacc:.3f}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Few-shot LLM classifier for the RiskGuardian test set.")
    ap.add_argument("--no-validate", action="store_true", help="skip the labeled-holdout validation pass")
    ap.add_argument("--dry-run", action="store_true", help="no API calls: classical only + print a sample prompt")
    ap.add_argument("--workers", type=int, default=4, help="concurrent API workers (default 4)")
    args = ap.parse_args()

    rt_tr, rt_te = load_frames()
    print(f"train: {rt_tr.shape} | test: {rt_te.shape} | "
          f"test composition: {int(rt_te.is_obf.sum())} obfuscated / {int((~rt_te.is_obf).sum())} clean")
    shot_idx, shots, rng = build_shots(rt_tr)
    print(f"few-shot examples: {len(shot_idx)} (2 per quadrant)")

    print("\nFitting classical backstop (TF-IDF + LinearSVC)...")
    classical = classical_predict(rt_tr, rt_te)

    if args.dry_run:
        print("\n[dry-run] sample prompt that WOULD be sent for test id=0:\n" + "-" * 60)
        print("SYSTEM:\n" + SYSTEM + "\n\nUSER:\n" + make_user_prompt(rt_te.text[0], shots)[:1200] + " ...")
        sample = pd.read_csv(SAMPLE_CSV)
        pd.DataFrame({"id": rt_te["id"].values, "label": classical})[list(sample.columns)].to_csv("submission_classical.csv", index=False)
        print("\n[dry-run] wrote submission_classical.csv (no LLM calls made).")
        return

    api_key = load_api_key()
    assert api_key, "ANTHROPIC_API_KEY not found (set env var or add to config.json)"
    print(f"LLM provider: Anthropic / {ANTHROPIC_MODEL} | pacing {MIN_INTERVAL:.1f}s/req, {args.workers} workers")
    print("smoke test (expect 1):",
          classify("Quarterly portfolio drawdown and liquidity stress across credit positions.", api_key, shots))

    if not args.no_validate:
        print()
        validate(rt_tr, api_key, shots, shot_idx, rng, workers=args.workers)

    print(f"\nClassifying {len(rt_te)} test rows with Anthropic / {ANTHROPIC_MODEL} ...")
    llm_pred = classify_many(rt_te["text"].tolist(), api_key, shots, system=SYSTEM, workers=args.workers)
    fails = sum(p == -1 for p in llm_pred)
    llm_final = np.array([c if p == -1 else p for p, c in zip(llm_pred, classical)]).astype(int)

    sample = pd.read_csv(SAMPLE_CSV)
    pd.DataFrame({"id": rt_te["id"].values, "label": llm_final})[list(sample.columns)].to_csv("submission.csv", index=False)
    pd.DataFrame({"id": rt_te["id"].values, "label": classical})[list(sample.columns)].to_csv("submission_classical.csv", index=False)

    agree = int((llm_final == classical).sum())
    print(f"\nLLM call failures (filled from classical): {fails}")
    print(f"LLM vs classical agreement on test: {agree}/{len(rt_te)}")
    print("submission.csv = LLM (PRIMARY, GenAI) | submission_classical.csv = classical backup (~99.99% CV)")
    print("label counts (LLM):", pd.Series(llm_final).value_counts().to_dict())
    dis = [(int(rt_te.id[i]), int(classical[i]), int(llm_final[i]),
            ("obf" if rt_te.is_obf[i] else ("news+Reuters" if rt_te.reuters[i] else "news")),
            trunc(rt_te.text[i], 80)) for i in range(len(rt_te)) if llm_final[i] != classical[i]]
    if dis:
        print("\ndisagreements (id, classical, llm, type, text):")
        for d in dis:
            print("   ", d)
    else:
        print("\nno disagreements — the LLM matches the classical model on all rows.")


if __name__ == "__main__":
    main()
