#!/usr/bin/env python
"""
RiskGuardian — GenAI predictor: few-shot LLM classification on Cerebras (open-source >300B model).

Sibling of riskguardian_llm_classifier.py (which uses Anthropic Claude). This version runs the SAME
engineered few-shot prompt against a large open-source model served on Cerebras — default
`zai-glm-4.7` (GLM-4.7, ~355B-parameter MoE) — so the two providers' results can be compared.

It few-shot-prompts the model to label each test row 0 (cybersecurity / fake-news) or
1 (financial / real-Reuters-news), and writes:
    submission.cerebras.<model>.csv   <- LLM predictions (pure GenAI / FAQ #8; no classical fallback)

NOTE: GLM-4.7 is a *reasoning* model on Cerebras — it emits chain-of-thought in a separate `reasoning`
field and the final answer in `content`, so we request a generous max_tokens and parse `content`.

The corpus is a *combination*: ~1/3 char-obfuscated synthetic risk events (cyber=0 / financial=1)
and ~2/3 the ISOT fake/real news set (fake clickbait=0 / real Reuters=1). The engineered prompt
encodes that structure; few-shot examples span all four quadrants.

Usage (run with the MLENV311 interpreter):
    python riskguardian_cerebras_classifier.py                 # validate (40 rows) + classify the 70 test rows
    python riskguardian_cerebras_classifier.py --no-validate   # skip validation, just classify + submit
    python riskguardian_cerebras_classifier.py --dry-run       # no API calls: just show a sample prompt
    CEREBRAS_MODEL=zai-glm-4.7 CEREBRAS_MIN_INTERVAL=2.0 python riskguardian_cerebras_classifier.py --workers 4

Requires CEREBRAS_API_KEY in the environment or in ./config.json.
"""
import argparse, json, os, re, time, threading, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

TRAIN_CSV  = "combined_risk_train.csv"
TEST_CSV   = "combined_risk_test.csv"
SAMPLE_CSV = "Sample_Submission.csv"

CEREBRAS_URL   = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "zai-glm-4.7")   # >300B open-source MoE
MAX_TOKENS     = int(os.environ.get("CEREBRAS_MAX_TOKENS", "4000"))  # room for reasoning + the final answer


# ── config / API key ──────────────────────────────────────────────────────
def load_api_key():
    k = os.environ.get("CEREBRAS_API_KEY")
    if not k and Path("config.json").exists():
        k = json.load(open("config.json")).get("CEREBRAS_API_KEY")
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


# ── LLM few-shot classifier (Cerebras / open-source model) ─────────────────
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
MIN_INTERVAL = float(os.environ.get("CEREBRAS_MIN_INTERVAL", "2.0"))  # ~30 req/min (reasoning -> more tokens/call)


def _throttle():
    with _rl_lock:
        wait = _rl_last[0] + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _rl_last[0] = time.time()


def make_user_prompt(text, shots):
    return (shots + "\n\n" if shots else "") + f"TEXT: {trunc(text, 1500)}\nLABEL:"


def _parse_label(content):
    """Final answer lives in `content` (reasoning is in a separate field). Strip any stray <think> and read 0/1."""
    if not content:
        return -1
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    m = re.search(r"[01]", content)
    return int(m.group()) if m else -1


def classify(text, api_key, shots, system=SYSTEM, model=CEREBRAS_MODEL, retries=8):
    # OpenAI-style chat-completions: system+user as messages, large max_tokens for the reasoning model.
    payload = {"model": model, "max_tokens": MAX_TOKENS, "temperature": 0,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": make_user_prompt(text, shots)}]}
    delay = 2.0
    for a in range(retries):
        try:
            _throttle()
            r = requests.post(
                CEREBRAS_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload, timeout=120)
            if r.status_code == 200:
                return _parse_label(r.json()["choices"][0]["message"].get("content"))
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
    return -1                                                       # unresolved; main() retries then defaults to 0


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
    ap = argparse.ArgumentParser(description="Few-shot LLM classifier (Cerebras open-source model) for the RiskGuardian test set.")
    ap.add_argument("--no-validate", action="store_true", help="skip the labeled-holdout validation pass")
    ap.add_argument("--dry-run", action="store_true", help="no API calls: just print a sample prompt")
    ap.add_argument("--workers", type=int, default=4, help="concurrent API workers (default 4)")
    ap.add_argument("--compare", default="submission.csv", help="read-only: compare results against this CSV (e.g. the Claude submission)")
    args = ap.parse_args()

    model_slug = CEREBRAS_MODEL.replace("/", "-")
    out_path = f"submission.cerebras.{model_slug}.csv"

    rt_tr, rt_te = load_frames()
    print(f"train: {rt_tr.shape} | test: {rt_te.shape} | "
          f"test composition: {int(rt_te.is_obf.sum())} obfuscated / {int((~rt_te.is_obf).sum())} clean")
    shot_idx, shots, rng = build_shots(rt_tr)
    print(f"model: Cerebras / {CEREBRAS_MODEL} | few-shot examples: {len(shot_idx)} (2 per quadrant) | out: {out_path}")

    if args.dry_run:
        print("\n[dry-run] sample prompt for test id=0 (no API calls):\n" + "-" * 60)
        print("SYSTEM:\n" + SYSTEM + "\n\nUSER:\n" + make_user_prompt(rt_te.text[0], shots)[:1200] + " ...")
        return

    api_key = load_api_key()
    assert api_key, "CEREBRAS_API_KEY not found (set env var or add to config.json)"
    print(f"pacing {MIN_INTERVAL:.1f}s/req, {args.workers} workers, max_tokens={MAX_TOKENS}")
    print("smoke test (expect 1):",
          classify("Quarterly portfolio drawdown and liquidity stress across credit positions.", api_key, shots))

    if not args.no_validate:
        print()
        validate(rt_tr, api_key, shots, shot_idx, rng, workers=args.workers)

    print(f"\nClassifying {len(rt_te)} test rows with Cerebras / {CEREBRAS_MODEL} ...")
    llm_pred = classify_many(rt_te["text"].tolist(), api_key, shots, system=SYSTEM, workers=args.workers)

    # Pure GenAI: retry any failed calls sequentially (no classical backstop).
    fail_idx = [i for i, p in enumerate(llm_pred) if p == -1]
    if fail_idx:
        print(f"retrying {len(fail_idx)} failed row(s) sequentially...")
        for i in fail_idx:
            llm_pred[i] = classify(rt_te.text[i], api_key, shots)
    unresolved = [int(rt_te.id[i]) for i, p in enumerate(llm_pred) if p == -1]
    if unresolved:
        print(f"WARNING: {len(unresolved)} row(s) unclassified after retries; defaulting to 0: ids={unresolved}")
    llm_final = np.array([0 if p == -1 else p for p in llm_pred]).astype(int)

    sample = pd.read_csv(SAMPLE_CSV)
    pd.DataFrame({"id": rt_te["id"].values, "label": llm_final})[list(sample.columns)].to_csv(out_path, index=False)
    print(f"\nWrote {out_path} = {len(llm_final)} LLM predictions (Cerebras / {CEREBRAS_MODEL}, few-shot prompting)")
    print("label counts:", pd.Series(llm_final).value_counts().to_dict())

    # Read-only cross-model comparison (does NOT alter predictions).
    cmp_path = args.compare
    if cmp_path and Path(cmp_path).exists() and os.path.abspath(cmp_path) != os.path.abspath(out_path):
        other = pd.read_csv(cmp_path)
        if list(other.columns) == list(sample.columns) and len(other) == len(llm_final):
            a = other["label"].to_numpy()
            agree = int((a == llm_final).sum())
            print(f"\nvs {cmp_path}: agreement {agree}/{len(llm_final)}")
            for i in range(len(rt_te)):
                if a[i] != llm_final[i]:
                    typ = "obf" if rt_te.is_obf[i] else ("news+Reuters" if rt_te.reuters[i] else "news")
                    print(f"   diff id={int(rt_te.id[i])} {cmp_path}={int(a[i])} {model_slug}={int(llm_final[i])} [{typ}] {trunc(rt_te.text[i], 80)}")


if __name__ == "__main__":
    main()
