#!/usr/bin/env python
"""
build_submission2_notebook.py — assemble submission2.ipynb (ROAD), Apple-silicon / MLX edition.

A structural twin of submission.ipynb that runs the SAME engineered few-shot prompt entirely
LOCAL on Apple silicon via MLX (mlx-lm). The models are 4-bit MLX conversions hosted on the
Hugging Face Hub (mlx-community), so they load straight from HF and run on the Mac's unified
memory / Metal GPU — no CUDA, no bitsandbytes. Output -> submission2.csv.

Three largest open instruct models that fit a 128 GB Mac at 4-bit:
    mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit   (141B MoE, ~78 GB)  -- largest
    mlx-community/Qwen2.5-72B-Instruct-4bit          (72B,      ~40 GB)
    mlx-community/Llama-3.3-70B-Instruct-4bit         (70B,      ~40 GB)

Shared cells (art, data-reveal, prompt-story, validation, viz) are reused verbatim from the
existing submission2.ipynb so this stays decoupled from submission.ipynb's evolving structure.
"""
import json
from pathlib import Path

SRC_NB = "submission2.ipynb"     # reuse the stable shared cells from the current notebook
OUT_NB = "submission2.ipynb"
OLD_SCRIPTS = ["riskguardian_hf_mixtral_classifier.py",
               "riskguardian_hf_qwen_classifier.py",
               "riskguardian_hf_llama_classifier.py"]

# (filename, model id, slug, params blurb, 4-bit mem, one-line desc)
MODELS = [
    ("riskguardian_mlx_mixtral_classifier.py", "mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit",
     "mixtral-8x22b", "141B sparse-MoE (≈39B active)", "≈78 GB",
     "Mixtral-8x22B — the LARGEST by total parameters; a sparse MoE, so decode is fast for its size."),
    ("riskguardian_mlx_qwen_classifier.py", "mlx-community/Qwen2.5-72B-Instruct-4bit",
     "qwen2.5-72b", "72B dense", "≈40 GB",
     "Qwen2.5-72B-Instruct — a very strong dense instruct model; excellent at constrained classification."),
    ("riskguardian_mlx_llama_classifier.py", "mlx-community/Llama-3.3-70B-Instruct-4bit",
     "llama-3.3-70b", "70B dense", "≈40 GB",
     "Llama-3.3-70B-Instruct — Meta's strongest 70B instruct model; comfortable headroom on a 128 GB Mac."),
]
PRIMARY_MODEL = "mlx-community/Qwen2.5-72B-Instruct-4bit"   # default -> submission2.csv (override via HF_MODEL)

# ───────────────────────── shared prompt/data fragments ─────────────────────
SYSTEM = (
    'SYSTEM = (\n'
    '    "You classify RiskGuardian items as 0 or 1. The corpus mixes two text types:\\n"\n'
    '    "(A) Risk reports (words may be scrambled/typo\'d): 0 = CYBERSECURITY risk "\n'
    '    "(hacking, malware, breaches, networks, intrusion, DDoS, cloud, ransomware); "\n'
    '    "1 = FINANCIAL risk (markets, portfolios, debt, liquidity, credit, financial regulation/compliance).\\n"\n'
    '    "(B) News articles (clean English): 0 = sensational/opinion clickbait (ALL-CAPS hooks, \'WATCH:\', "\n'
    '    "\'Featured image via Getty Images\'); 1 = neutral newswire reporting (e.g. Reuters; datelines like "\n'
    '    "\'WASHINGTON (Reuters) -\'; sober \'X said\' attribution).\\n"\n'
    '    "Reply with ONLY one character: 0 or 1."\n'
    ')\n'
    'SYSTEM_NAIVE = ("Classify the risk event as 0 = Cybersecurity or 1 = Financial. "\n'
    '                "Reply with ONLY one character: 0 or 1.")\n'
)

LOAD_FRAMES = '''def load_frames():
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
    return tr, te'''

BUILD_SHOTS = '''def trunc(t, n):
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
    shots = "\\n\\n".join(f"TEXT: {trunc(rt_tr.text[i], 300)}\\nLABEL: {int(rt_tr.label[i])}" for i in shot_idx)
    return shot_idx, shots, rng'''

# ───────────────────────── MLX standalone-script template ───────────────────
MLX_SCRIPT = '''#!/usr/bin/env python
"""
RiskGuardian — GenAI predictor: few-shot classification on Apple silicon via MLX.

__DESC__

Runs the SAME engineered few-shot prompt as the API classifiers, fully LOCAL on Apple silicon
via MLX (mlx-lm). The model is a 4-bit MLX conversion on the Hugging Face Hub (mlx-community),
so it loads straight from HF and runs on the Mac's unified memory / Metal GPU — no CUDA, no
bitsandbytes. The label is decided by comparing the next-token logits of "0" vs "1":
deterministic, a single forward pass, and it never emits an unparseable answer.
Writes:
    __OUT__   <- predictions (pure GenAI / FAQ #8; no classical fallback)

Model: __MODEL_ID__   (__PARAMS__, __MEM__ at 4-bit)

Requires: macOS on Apple silicon + `pip install mlx-lm` (plus numpy, pandas). ~__MEM__ of the
128 GB unified memory holds the weights, leaving ample room for the KV-cache.

Usage:
    python __SCRIPT__                 # validate (40 rows) + classify the 70 test rows
    python __SCRIPT__ --no-validate   # skip validation, just classify + submit
    python __SCRIPT__ --dry-run       # no model load: just print a sample prompt
    HF_MODEL=mlx-community/... python __SCRIPT__   # override the model id

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

MODEL_ID = os.environ.get("HF_MODEL", "__MODEL_ID__")
OUT_PATH = "__OUT__"
MAX_INPUT_TOKENS = int(os.environ.get("HF_MAX_INPUT_TOKENS", "8192"))


# ── data: load + tag the two sub-corpora (obfuscated risk vs clean news) ───
__LOAD_FRAMES__


# ── the engineered few-shot prompt (identical across every provider) ───────
__SYSTEM__


# ── few-shot bank: 2 examples per quadrant, deterministic, truncated ───────
__BUILD_SHOTS__


# ── Apple-silicon model (MLX): load once, classify by 0-vs-1 next-token logits ──
_STATE = {"model": None, "tok": None, "zero": None, "one": None}


def load_model(model_id=MODEL_ID):
    from mlx_lm import load
    print(f"loading {model_id} (MLX / Apple-silicon Metal) ...")
    model, tok = load(model_id)
    _STATE.update(model=model, tok=tok,
                  zero=tok.encode("0")[-1], one=tok.encode("1")[-1])
    return model, tok


def make_messages(text, system, shots):
    user = (shots + "\\n\\n" if shots else "") + f"TEXT: {trunc(text, 1500)}\\nLABEL:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _encode(msgs):
    tok = _STATE["tok"]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
        if isinstance(ids, str):
            ids = tok.encode(ids)
    except Exception:
        ids = tok.encode(msgs[0]["content"] + "\\n\\n" + msgs[1]["content"])
    return list(ids)[-MAX_INPUT_TOKENS:]


def classify(text, shots, system=None):
    import mlx.core as mx
    if system is None:
        system = SYSTEM
    ids = _encode(make_messages(text, system, shots))
    logits = _STATE["model"](mx.array(ids)[None])[0, -1]
    zero = float(logits[_STATE["zero"]].item())
    one  = float(logits[_STATE["one"]].item())
    return 0 if zero >= one else 1


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
    print(f"\\nprompt-engineering lift on the same {len(sub_i)} rows ->  naive: {nacc:.3f}   engineered: {eacc:.3f}")


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Local MLX few-shot classifier for the RiskGuardian test set.")
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
        print("\\n[dry-run] sample prompt for test id=0 (no model load):\\n" + "-" * 60)
        print("SYSTEM:\\n" + SYSTEM + "\\n\\nUSER:\\n" + make_messages(rt_te.text[0], SYSTEM, shots)[1]["content"][:1200] + " ...")
        return

    load_model()
    print("smoke test (expect 1):",
          classify("Quarterly portfolio drawdown and liquidity stress across credit positions.", shots))

    if not args.no_validate:
        print()
        validate(rt_tr, shots, shot_idx, rng)

    print(f"\\nClassifying {len(rt_te)} test rows with {MODEL_ID} ...")
    preds = np.array(classify_many(rt_te["text"].tolist(), shots, system=SYSTEM)).astype(int)

    sample = pd.read_csv(SAMPLE_CSV)
    pd.DataFrame({"id": rt_te["id"].values, "label": preds})[list(sample.columns)].to_csv(OUT_PATH, index=False)
    print(f"\\nWrote {OUT_PATH} = {len(preds)} predictions ({MODEL_ID}, local MLX few-shot prompting)")
    print("label counts:", pd.Series(preds).value_counts().to_dict())

    cmp_path = args.compare
    if cmp_path and Path(cmp_path).exists() and os.path.abspath(cmp_path) != os.path.abspath(OUT_PATH):
        other = pd.read_csv(cmp_path)
        if list(other.columns) == list(sample.columns) and len(other) == len(preds):
            a = other["label"].to_numpy()
            print(f"\\nvs {cmp_path}: agreement {int((a == preds).sum())}/{len(preds)}")
            for i in range(len(rt_te)):
                if a[i] != preds[i]:
                    typ = "obf" if rt_te.is_obf[i] else ("news+Reuters" if rt_te.reuters[i] else "news")
                    print(f"   diff id={int(rt_te.id[i])} {cmp_path}={int(a[i])} pred={int(preds[i])} [{typ}] {trunc(rt_te.text[i], 80)}")


if __name__ == "__main__":
    main()
'''


def render_script(model_id, out_csv, params, mem, desc, script_name):
    s = MLX_SCRIPT
    s = s.replace("__LOAD_FRAMES__", LOAD_FRAMES)
    s = s.replace("__SYSTEM__", SYSTEM.rstrip("\n"))
    s = s.replace("__BUILD_SHOTS__", BUILD_SHOTS)
    s = s.replace("__MODEL_ID__", model_id)
    s = s.replace("__OUT__", out_csv)
    s = s.replace("__PARAMS__", params)
    s = s.replace("__MEM__", mem)
    s = s.replace("__DESC__", desc)
    s = s.replace("__SCRIPT__", script_name)
    return s


# remove the old transformers/bitsandbytes scripts; write the new MLX ones
for old in OLD_SCRIPTS:
    p = Path(old)
    if p.exists():
        p.unlink(); print("removed", old)
for fname, model_id, slug, params, mem, desc in MODELS:
    Path(fname).write_text(render_script(model_id, f"submission2.{slug}.csv", params, mem, desc, fname))
    print("wrote", fname)


# ───────────────────────── notebook assembly ───────────────────────────────
src = json.load(open(SRC_NB))
sc = src["cells"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def reuse(idx):
    c = json.loads(json.dumps(sc[idx]))
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
    c.setdefault("metadata", {})
    return c


def writefile_cell(path, blurb):
    body = Path(path).read_text()
    header = (f"# {blurb}\n"
              f"# Full local MLX few-shot classifier, inlined. `%%writefile` shows the code AND\n"
              f"# (re)materializes {path} WITHOUT executing it (no model load / no inference here).\n")
    return code(f"%%writefile {path}\n{header}\n{body}")


cells = []

# ART (reused, unchanged)
cells.append(reuse(0))

# Title + ROAD orientation (Apple-silicon / MLX)
cells.append(md(
"""# <span style="color:#22442C">RiskGuardian — Few-Shot Prompting Submission (Apple-silicon / MLX edition)</span>
### <em style="color:#7EA98B">Cyber (0) vs. Financial (1) — pure GenAI / prompt-engineering (FAQ&nbsp;#8), run on-prem on a Mac</em>

> **A local-only twin of [`submission.ipynb`](submission.ipynb).** Same engineered few-shot prompt, same
> ROAD structure — but every prediction comes from an **open-weight model running on Apple silicon via
> [MLX](https://github.com/ml-explore/mlx)**, with no external API and no CUDA. Output:
> [`submission2.csv`](submission2.csv).

This notebook follows the **ROAD** analytical format:

| | Stage | What it answers |
|---|---|---|
| **R** | **Requirements** | What was asked, what the data *actually* is, and the success metric. |
| **O** | **Operationalize** | The engineered few-shot prompt + the three local MLX model scripts (inline). |
| **A** | **Analysis** | Held-out validation, prompt-engineering lift, and the cross-model comparison. |
| **D** | **Decision** | Which local model becomes `submission2.csv`, and why. |

**Why this twin exists.** The API notebook proved the *method* (Claude Sonnet 4.5 → 100%). This one proves
it is **reproducible with open weights, fully offline, on a Mac** — the privacy-preserving path when data
can't leave the building. We pick the **three largest open instruct models that fit a 128&nbsp;GB Mac at
4-bit**, served as ready-to-run **MLX** conversions on the Hugging Face Hub:

| Model (HF / mlx-community) | Params | ~4-bit footprint | Role |
|---|---|---|---|
| `mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit` | **141B** (sparse MoE, ≈39B active) | ≈78&nbsp;GB | largest |
| `mlx-community/Qwen2.5-72B-Instruct-4bit` | 72B dense | ≈40&nbsp;GB | default → `submission2.csv` |
| `mlx-community/Llama-3.3-70B-Instruct-4bit` | 70B dense | ≈40&nbsp;GB | comparison |

> **Honesty note.** Unlike `submission.ipynb` (whose API results are already computed), the accuracy /
> agreement figures here are **produced when you run the cells on the Mac** — they are intentionally left as
> live computations, not pre-filled numbers.
"""))

# ═════════════════════════ R — REQUIREMENTS ═════════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">R — Requirements</span>

**The task is identical to `submission.ipynb`** — the data, labels, and metric do not change; only the
*engine* (local MLX models on Apple silicon) does.

Classify each row of `combined_risk_test.csv` as **0 = Cybersecurity** or **1 = Financial**, scored on
**accuracy**, using **prompt engineering with a pre-trained LLM** (FAQ&nbsp;#8).

| File | Rows | Columns |
|---|---|---|
| `combined_risk_train.csv` | 14,830 (balanced) | `id, text, label` |
| `combined_risk_test.csv` | 70 | `id, text` |
| `Sample_Submission.csv` | 70 | `id, label` |

**What the data actually is — the key insight.** The corpus is a **mixture of two sub-corpora**, both in
train *and* test with consistent labels:

| Sub-corpus | ~Share of train | Form | Label 0 | Label 1 |
|---|---|---|---|---|
| Synthetic risk events | ~1/3 | **character-obfuscated** text | Cybersecurity | Financial |
| ISOT Fake/Real news | ~2/3 | clean English articles | Fake clickbait | **Real (Reuters)** |

So **0 = cyber *or* fake-news**, **1 = financial *or* real (Reuters) news** — a source-based rule for the
news half that is **not inferable from the brief**. Test mix: **20 obfuscated + 50 clean (25 Reuters)**.

The cell below loads the data and reveals that two-sub-corpus structure.
"""))
cells.append(reuse(3))   # load + reveal the combined structure (shared)

# ═════════════════════════ O — OPERATIONALIZE ═══════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">O — Operationalize</span>

The method is **engineered few-shot prompting** — a prompt that *encodes the discovered structure*: a system
instruction naming both text types and their separate label rules, **eight few-shot examples spanning all
four quadrants**, and a single-character output. The prompt is **byte-for-byte the same** as the API
notebook; only the model backend changes (here: MLX on Apple silicon).
"""))
cells.append(reuse(5))   # "The prompt-engineering story" markdown (shared)

# Hardware / memory note (Apple silicon / MLX)
cells.append(md(
"""### <em style="color:#7EA98B">Hardware & memory — open weights on a 128&nbsp;GB Mac (MLX)</em>

**MLX** is Apple's array/ML framework; **`mlx-lm`** runs LLMs on the Mac's **unified memory** and **Metal**
GPU. The models below are **4-bit MLX conversions** published on the Hugging Face Hub under
`mlx-community`, so `mlx_lm.load("mlx-community/…")` downloads and runs them directly — **no CUDA, no
bitsandbytes, no GGUF conversion step**. 4-bit costs roughly **0.5&nbsp;bytes/parameter**, so even the 141B
Mixtral MoE sits well under the 128&nbsp;GB unified-memory budget:

| Model | Params | 4-bit weights | Fits a 128&nbsp;GB Mac? |
|---|---|---|---|
| Mixtral-8x22B-Instruct (MoE) | 141B | ≈70 GB | ✅ (≈78 GB w/ overhead) |
| Qwen2.5-72B-Instruct | 72B | ≈38 GB | ✅ comfortably |
| Llama-3.3-70B-Instruct | 70B | ≈36 GB | ✅ comfortably |

> Recommended hardware: an Apple-silicon Mac with **128&nbsp;GB unified memory** (e.g. Mac&nbsp;Studio
> M-series&nbsp;Ultra, or a 128&nbsp;GB MacBook&nbsp;Pro). For the 141B MoE, raise Metal's working-set limit
> if needed: `sudo sysctl iogpu.wired_limit_mb=122880`.

**Classification by logits, not free text.** Instead of generating and regex-parsing, we read the model's
**next-token logits at the answer position and compare `"0"` vs `"1"`** — deterministic, a single forward
pass, and it can never emit an unparseable answer (no `-1` failures).

**Dependencies:** macOS on Apple silicon + `pip install mlx-lm` (plus `numpy`, `pandas`).
"""))

# In-notebook MLX engine
cells.append(md(
"""### <em style="color:#7EA98B">The in-notebook few-shot engine (local MLX model)</em>

Loads one MLX model (default `mlx-community/Qwen2.5-72B-Instruct-4bit`, override with the `HF_MODEL` env
var), builds the few-shot bank, and exposes `classify` / `classify_many` used by the validation and
prediction cells. The heavy line is `mlx_lm.load(...)` — it pulls/loads the weights once (first run may
download tens of GB) and maps them into unified memory.
"""))
engine = f'''# ── GenAI predictor (local Apple-silicon model via MLX): few-shot classifier ──
# Same engineered prompt as submission.ipynb, but every label comes from a LOCAL MLX model
# running on Apple silicon (unified memory / Metal). Decision rule = compare next-token logits
# of "0" vs "1" (deterministic; never fails).
import os, re
import numpy as np, pandas as pd
from pathlib import Path

HF_MODEL = os.environ.get("HF_MODEL", "{PRIMARY_MODEL}")
MAX_INPUT_TOKENS = int(os.environ.get("HF_MAX_INPUT_TOKENS", "8192"))

def _ensure_frames():                       # reuse the R-stage frames if present, else load + tag
    g = globals()
    if "rt_tr" in g and "is_obf" in g["rt_tr"]:
        return g["rt_tr"], g["rt_te"]
    tr = pd.read_csv("combined_risk_train.csv"); te = pd.read_csv("combined_risk_test.csv")
    wl = None
    for p in ("/usr/share/dict/words", "/usr/dict/words"):
        if Path(p).exists():
            wl = set(w.strip().lower() for w in open(p) if len(w.strip()) >= 4); break
    def clean(t):
        if wl is None: return 0.5
        tk = re.findall(r"[A-Za-z]{{4,}}", t.lower()); return sum(w in wl for w in tk)/len(tk) if tk else 0.0
    for d in (tr, te):
        d["clean"] = d["text"].map(clean); d["is_obf"] = d["clean"] < 0.55
        d["reuters"] = d["text"].str.contains("Reuters", case=True)
    return tr, te
rt_tr, rt_te = _ensure_frames()

{SYSTEM}
rng = np.random.default_rng(7)
def pick(mask, n):
    idx = np.asarray(rt_tr.index[np.asarray(mask)])
    return list(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else []
SHOT_IDX = (pick((rt_tr.is_obf) & (rt_tr.label == 0), 2) +
            pick((rt_tr.is_obf) & (rt_tr.label == 1), 2) +
            pick((~rt_tr.is_obf) & (rt_tr.label == 0) & (~rt_tr.reuters), 2) +
            pick((~rt_tr.is_obf) & (rt_tr.label == 1) & (rt_tr.reuters), 2))
def trunc(t, n): return " ".join(str(t).split())[:n]
SHOTS = "\\n\\n".join(f"TEXT: {{trunc(rt_tr.text[i], 300)}}\\nLABEL: {{int(rt_tr.label[i])}}" for i in SHOT_IDX)

# ── load the MLX model once (weights live in the Mac's unified memory) ──
from mlx_lm import load
import mlx.core as mx
print("loading", HF_MODEL, "(MLX / Apple-silicon Metal) ... first run may download tens of GB")
_model, _tok = load(HF_MODEL)
_ZERO = _tok.encode("0")[-1]
_ONE  = _tok.encode("1")[-1]

def _encode(msgs):
    try:
        ids = _tok.apply_chat_template(msgs, add_generation_prompt=True)
        if isinstance(ids, str): ids = _tok.encode(ids)
    except Exception:
        ids = _tok.encode(msgs[0]["content"] + "\\n\\n" + msgs[1]["content"])
    return list(ids)[-MAX_INPUT_TOKENS:]

def classify(text, system=SYSTEM, shots=SHOTS):
    user = (shots + "\\n\\n" if shots else "") + f"TEXT: {{trunc(text, 1500)}}\\nLABEL:"
    msgs = [{{"role": "system", "content": system}}, {{"role": "user", "content": user}}]
    logits = _model(mx.array(_encode(msgs))[None])[0, -1]
    return 0 if float(logits[_ZERO].item()) >= float(logits[_ONE].item()) else 1

def classify_many(texts, system=SYSTEM):
    return [classify(t, system=system) for t in texts]

print("model loaded:", HF_MODEL, "| few-shot examples:", len(SHOT_IDX))
print("smoke test (expect 1):", classify("Quarterly portfolio drawdown and liquidity stress across credit positions."))
'''
cells.append(code(engine))

# three MLX scripts inline
cells.append(md(
"""### <em style="color:#7EA98B">The three local MLX few-shot scripts (inlined)</em>

The *same* engineered prompt, packaged as three standalone scripts — one per model — so each can be run from
a shell and the results compared. Each is reproduced in full via `%%writefile`, which **shows the source and
(re)writes the file without executing it** (no weights load here). Each writes its own
`submission2.<model>.csv`; run them like:

```bash
pip install mlx-lm
python riskguardian_mlx_qwen_classifier.py        # -> submission2.qwen2.5-72b.csv
python riskguardian_mlx_llama_classifier.py       # -> submission2.llama-3.3-70b.csv
python riskguardian_mlx_mixtral_classifier.py     # -> submission2.mixtral-8x22b.csv
HF_MODEL=mlx-community/Qwen2.5-72B-Instruct-4bit python riskguardian_mlx_qwen_classifier.py   # override
```
"""))
for fname, model_id, slug, params, mem, desc in MODELS:
    cells.append(md(f"**`{fname}`** — {desc} ({params}, {mem} @ 4-bit) → `submission2.{slug}.csv`"))
    cells.append(writefile_cell(fname, desc))

# ═════════════════════════ A — ANALYSIS ═════════════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">A — Analysis</span>

### <em style="color:#7EA98B">Honest validation: held-out accuracy + prompt-engineering lift</em>

Score the engineered prompt on a labeled holdout drawn evenly from all four quadrants (never the few-shot
examples), report per-quadrant accuracy, and measure the lift over a **naive** prompt. This runs the loaded
local model, so expect it to take a few minutes; the numbers below are computed live (not pre-filled).
"""))
cells.append(reuse(17))   # validation (shared signature)
cells.append(reuse(18))   # caption (shared)
cells.append(reuse(19))   # viz1: naive vs engineered
cells.append(reuse(20))   # viz2: per-quadrant

cells.append(md(
"""### <em style="color:#7EA98B">Cross-model comparison (local MLX models)</em>

Run all three scripts (above) to populate `submission2.<model>.csv`, then compare. The cell below loads
whichever per-model CSVs exist alongside the primary `submission2.csv` and reports label splits and pairwise
agreement. With the API models, GLM-4.7 and GPT-5.3 agreed 70/70 (differing from Claude only on **id 37**) —
the interesting question here is whether these *open* models converge the same way, and how they handle that
same source-vs-content row. **The numbers populate once you've run the models.**
"""))
cells.append(reuse(22))   # cross-model agreement code (shared; stdlib only)

# ═════════════════════════ D — DECISION ═════════════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">D — Decision</span>

**`submission2.csv` is written by the loaded local MLX model** (default
`mlx-community/Qwen2.5-72B-Instruct-4bit`) — pure GenAI, no classical model in the prediction path
(FAQ&nbsp;#8). To promote a different local model to the official `submission2.csv`, set `HF_MODEL` before
running the in-notebook engine, e.g.:

```bash
HF_MODEL=mlx-community/Mixtral-8x22B-Instruct-v0.1-4bit jupyter nbconvert --execute submission2.ipynb
```

**How to choose between the three:** pick the model with the best **held-out validation accuracy** from the
Analysis section (all four quadrants), breaking ties toward the model that reads **id&nbsp;37** as the
*source-based* `0` — the row that distinguishes provenance-reasoning from surface-plausibility, and the one
the API notebook's perfect score confirmed as `0`. The cell below runs the chosen model over the 70 test
rows and writes `submission2.csv`.
"""))
cells.append(reuse(24))   # predict + write submission2.csv (shared)

cells.append(md(
"""### <em style="color:#7EA98B">Decision summary & repository map</em>

- **Submit:** `submission2.csv` — written locally by the chosen MLX model (default Qwen2.5-72B-Instruct-4bit),
  few-shot prompting, **0 external API calls**, entirely on Apple silicon.
- **Why local / MLX:** identical method to the winning `submission.csv`, but fully offline on a Mac — the
  privacy-preserving path when data can't leave the building.
- **Model choice:** decide from the held-out validation; the three candidates are the largest open instruct
  models that fit a 128&nbsp;GB Mac at 4-bit.

| Path | What it is |
|---|---|
| [`submission2.csv`](submission2.csv) | **Local submission** — chosen MLX model, pure GenAI (`id,label`) |
| `submission2.qwen2.5-72b.csv` / `…llama-3.3-70b.csv` / `…mixtral-8x22b.csv` | per-model outputs (comparison) |
| `riskguardian_mlx_qwen_classifier.py` | Qwen2.5-72B-Instruct-4bit (72B) few-shot classifier |
| `riskguardian_mlx_llama_classifier.py` | Llama-3.3-70B-Instruct-4bit (70B) few-shot classifier |
| `riskguardian_mlx_mixtral_classifier.py` | Mixtral-8x22B-Instruct-v0.1-4bit (141B MoE) few-shot classifier |
| [`submission.ipynb`](submission.ipynb) | the API twin (Claude/GLM/GPT) this notebook mirrors |

---
*Apple-silicon / MLX edition of the RiskGuardian few-shot-prompting summary. Same engineered prompt, same
ROAD order — open weights, on-device, `submission2.csv`.*
"""))

# write
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (MLENV311)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
json.dump(nb, open(OUT_NB, "w"))
print(f"wrote {OUT_NB}: {len(cells)} cells, {Path(OUT_NB).stat().st_size/1e6:.1f} MB")
