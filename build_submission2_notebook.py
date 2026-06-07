#!/usr/bin/env python
"""
build_submission2_notebook.py — assemble submission2.ipynb (ROAD), HF-local edition.

A structural duplicate of submission.ipynb that swaps the three *API* few-shot
classifiers (Anthropic / Cerebras / OpenAI) for three *local Hugging Face*
few-shot classifiers, runnable on a 128 GB machine via 4-bit quantization:

    mistralai/Mixtral-8x22B-Instruct-v0.1   (141B MoE, ~80 GB @ 4-bit)  -- largest
    Qwen/Qwen2.5-72B-Instruct               (72B,      ~42 GB @ 4-bit)
    meta-llama/Llama-3.3-70B-Instruct       (70B,      ~40 GB @ 4-bit)

The same engineered few-shot prompt is used. Output -> submission2.csv.
Reuses the banner art + the Requirements / prompt-engineering / viz blocks from
submission.ipynb; writes the three HF scripts to disk and inlines them.
"""
import json
from pathlib import Path

SRC_NB = "submission.ipynb"          # reuse art + shared analysis cells from here
OUT_NB = "submission2.ipynb"

# (filename, model id, slug, total-params blurb, 4-bit mem, one-line desc)
MODELS = [
    ("riskguardian_hf_mixtral_classifier.py", "mistralai/Mixtral-8x22B-Instruct-v0.1",
     "mixtral-8x22b", "141B sparse-MoE (≈39B active)", "≈80 GB",
     "Mixtral-8x22B — the LARGEST by total parameters; a sparse MoE, so inference is fast for its size."),
    ("riskguardian_hf_qwen_classifier.py", "Qwen/Qwen2.5-72B-Instruct",
     "qwen2.5-72b", "72B dense", "≈42 GB",
     "Qwen2.5-72B-Instruct — a very strong dense instruct model; excellent at constrained classification."),
    ("riskguardian_hf_llama_classifier.py", "meta-llama/Llama-3.3-70B-Instruct",
     "llama-3.3-70b", "70B dense", "≈40 GB",
     "Llama-3.3-70B-Instruct — Meta's strongest 70B instruct model; comfortable headroom on 128 GB."),
]
PRIMARY_MODEL = "Qwen/Qwen2.5-72B-Instruct"   # default for submission2.csv (override via HF_MODEL)

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

# ───────────────────────── HF standalone-script template ────────────────────
HF_SCRIPT = '''#!/usr/bin/env python
"""
RiskGuardian — GenAI predictor: few-shot classification with a LOCAL Hugging Face model.

__DESC__

Runs the SAME engineered few-shot prompt as the API classifiers, but entirely LOCAL via
`transformers` with 4-bit (nf4) quantization, so the model fits in 128 GB. The label is
decided by comparing the next-token logits of "0" vs "1" — deterministic, and it never fails.
Writes:
    __OUT__   <- predictions (pure GenAI / FAQ #8; no classical fallback)

Model: __MODEL_ID__   (__PARAMS__, __MEM__ at 4-bit nf4)

Requires (MLENV311 + ) transformers, accelerate, bitsandbytes, torch. A CUDA GPU enables
true 4-bit; on a CPU / Apple-silicon 128 GB box a GGUF build (llama-cpp-python) is the
recommended quantized path (same weights, same prompt) — see the notebook's hardware note.

Usage:
    python __SCRIPT__                 # validate (40 rows) + classify the 70 test rows
    python __SCRIPT__ --no-validate   # skip validation, just classify + submit
    python __SCRIPT__ --dry-run       # no model load: just print a sample prompt
    HF_MODEL=... python __SCRIPT__    # override the model id

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
    user = (shots + "\\n\\n" if shots else "") + f"TEXT: {trunc(text, 1500)}\\nLABEL:"
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
        ids = tok(system + "\\n\\n" + msgs[1]["content"], return_tensors="pt").input_ids
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
    print(f"\\nprompt-engineering lift on the same {len(sub_i)} rows ->  naive: {nacc:.3f}   engineered: {eacc:.3f}")


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
    print(f"\\nWrote {OUT_PATH} = {len(preds)} predictions ({MODEL_ID}, local few-shot prompting)")
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


def render_script(model_id, out_csv, slug, params, mem, desc, script_name):
    s = HF_SCRIPT
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


# write the three standalone HF scripts to disk
for fname, model_id, slug, params, mem, desc in MODELS:
    out_csv = f"submission2.{slug}.csv"
    Path(fname).write_text(render_script(model_id, out_csv, slug, params, mem, desc, fname))
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
              f"# Full local HF few-shot classifier, inlined. `%%writefile` shows the code AND\n"
              f"# (re)materializes {path} WITHOUT executing it (no model load / no inference here).\n")
    return code(f"%%writefile {path}\n{header}\n{body}")


cells = []

# ART (reused from the top of submission.ipynb)
cells.append(reuse(0))

# Title + ROAD orientation
cells.append(md(
"""# <span style="color:#22442C">RiskGuardian — Few-Shot Prompting Submission (local Hugging Face edition)</span>
### <em style="color:#7EA98B">Cyber (0) vs. Financial (1) — pure GenAI / prompt-engineering (FAQ&nbsp;#8), run entirely on-prem</em>

> **A local-only twin of [`submission.ipynb`](submission.ipynb).** Same engineered few-shot prompt, same
> ROAD structure — but every prediction comes from an **open-weight Hugging Face model running on your own
> 128&nbsp;GB box**, with no external API. Output: [`submission2.csv`](submission2.csv).

This notebook follows the **ROAD** analytical format:

| | Stage | What it answers |
|---|---|---|
| **R** | **Requirements** | What was asked, what the data *actually* is, and the success metric. |
| **O** | **Operationalize** | The engineered few-shot prompt + the three local HF model scripts (inline). |
| **A** | **Analysis** | Held-out validation, prompt-engineering lift, and the cross-model comparison. |
| **D** | **Decision** | Which local model becomes `submission2.csv`, and why. |

**Why this twin exists.** The API notebook proved the *method* (Claude Sonnet 4.5 → 100%). This one proves
the method is **reproducible with open weights, fully offline** — useful when data can't leave the building.
We pick the **three largest open instruct models that fit 128&nbsp;GB at 4-bit** and likely give solid results:

| Model | Params | ~4-bit footprint | Role |
|---|---|---|---|
| `mistralai/Mixtral-8x22B-Instruct-v0.1` | **141B** (sparse MoE, ≈39B active) | ≈80&nbsp;GB | largest |
| `Qwen/Qwen2.5-72B-Instruct` | 72B dense | ≈42&nbsp;GB | default → `submission2.csv` |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B dense | ≈40&nbsp;GB | comparison |

> **Honesty note.** Unlike `submission.ipynb` (whose API results are already computed), the accuracy /
> agreement figures here are **produced when you run the cells locally** — they are intentionally left as
> live computations, not pre-filled numbers.
"""))

# ═════════════════════════ R — REQUIREMENTS ═════════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">R — Requirements</span>

**The task is identical to `submission.ipynb`** — the data, labels, and metric do not change; only the
*engine* (local HF models) does.

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
notebook; only the model backend changes.
"""))
cells.append(reuse(5))   # "The prompt-engineering story" markdown (shared)

# Hardware / memory note
cells.append(md(
"""### <em style="color:#7EA98B">Hardware & memory — fitting open weights in 128&nbsp;GB</em>

Every model below is loaded with **4-bit (nf4) quantization** (`bitsandbytes`), which costs roughly
**0.5&nbsp;bytes/parameter** plus a little overhead. That keeps even the 141B Mixtral MoE under ~80&nbsp;GB,
leaving headroom for the KV-cache on a 128&nbsp;GB machine:

| Model | Params | 4-bit weights | Fits 128&nbsp;GB? |
|---|---|---|---|
| Mixtral-8x22B-Instruct | 141B (MoE) | ≈70 GB | ✅ (≈80 GB w/ overhead) |
| Qwen2.5-72B-Instruct | 72B | ≈36 GB | ✅ comfortably |
| Llama-3.3-70B-Instruct | 70B | ≈35 GB | ✅ comfortably |

**Classification by logits, not free text.** Instead of generating and regex-parsing, we read the model's
**next-token logits at the answer position and compare `"0"` vs `"1"`** — deterministic, single forward
pass, and it can never emit an unparseable answer (no `-1` failures).

**Backends.** The scripts use `transformers` + `bitsandbytes`, which need a **CUDA GPU** for true 4-bit
(weights live in VRAM and/or spill to the 128&nbsp;GB system RAM via `device_map="auto"`). On a **CPU /
Apple-silicon 128&nbsp;GB** box, bitsandbytes 4-bit isn't available — there, run a **GGUF build of the same
model via `llama-cpp-python`** (identical weights, identical prompt); 4-bit GGUF (Q4_K_M) has the same
footprint and is the standard CPU/unified-memory path.

**Dependencies:** `transformers`, `accelerate`, `bitsandbytes`, `torch` (plus `numpy`, `pandas`).
"""))

# In-notebook HF engine
cells.append(md(
"""### <em style="color:#7EA98B">The in-notebook few-shot engine (local model)</em>

Loads one model (default `Qwen/Qwen2.5-72B-Instruct`, override with the `HF_MODEL` env var), builds the
few-shot bank, and exposes `classify` / `classify_many` used by the validation and prediction cells. The
heavy line is `from_pretrained(...)` — it pulls/loads the weights once and can take several minutes.
"""))
engine = f'''# ── GenAI predictor (local Hugging Face model): few-shot prompt classifier ──
# Same engineered prompt as submission.ipynb, but every label comes from a LOCAL model.
# Decision rule = compare next-token logits of "0" vs "1" (deterministic; never fails).
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

# ── load the model once (4-bit on GPU; spills to the 128 GB system RAM as needed) ──
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
print("loading", HF_MODEL, "... (first load pulls/initializes the weights; can take minutes)")
_tok = AutoTokenizer.from_pretrained(HF_MODEL)
_kw = dict(low_cpu_mem_usage=True, device_map="auto")
if torch.cuda.is_available():
    _kw.update(torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
else:
    _kw.update(torch_dtype=torch.bfloat16)
    print("WARNING: no CUDA GPU -> 4-bit bitsandbytes unavailable. On a 128 GB CPU/Apple-silicon box, "
          "use a GGUF build via llama-cpp-python (same weights & prompt). bf16 here may exceed 128 GB for 70B+.")
_model = AutoModelForCausalLM.from_pretrained(HF_MODEL, **_kw).eval()
_ZERO = _tok("0", add_special_tokens=False).input_ids[-1]
_ONE  = _tok("1", add_special_tokens=False).input_ids[-1]

def classify(text, system=SYSTEM, shots=SHOTS):
    user = (shots + "\\n\\n" if shots else "") + f"TEXT: {{trunc(text, 1500)}}\\nLABEL:"
    msgs = [{{"role": "system", "content": system}}, {{"role": "user", "content": user}}]
    try:
        ids = _tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    except Exception:
        ids = _tok(system + "\\n\\n" + user, return_tensors="pt").input_ids
    ids = ids[:, -MAX_INPUT_TOKENS:].to(_model.device)
    with torch.no_grad():
        logits = _model(ids).logits[0, -1]
    return 0 if float(logits[_ZERO]) >= float(logits[_ONE]) else 1

def classify_many(texts, system=SYSTEM):
    return [classify(t, system=system) for t in texts]

print("model loaded:", HF_MODEL, "| few-shot examples:", len(SHOT_IDX))
print("smoke test (expect 1):", classify("Quarterly portfolio drawdown and liquidity stress across credit positions."))
'''
cells.append(code(engine))

# three HF scripts inline
cells.append(md(
"""### <em style="color:#7EA98B">The three local-model few-shot scripts (inlined)</em>

The *same* engineered prompt, packaged as three standalone scripts — one per model — so each can be run from
a shell and the results compared. Each is reproduced in full via `%%writefile`, which **shows the source and
(re)writes the file without executing it** (no weights load here). Each writes its own
`submission2.<model>.csv`; run them like:

```bash
python riskguardian_hf_qwen_classifier.py        # -> submission2.qwen2.5-72b.csv
python riskguardian_hf_llama_classifier.py       # -> submission2.llama-3.3-70b.csv
python riskguardian_hf_mixtral_classifier.py     # -> submission2.mixtral-8x22b.csv
HF_MODEL=Qwen/Qwen2.5-72B-Instruct python riskguardian_hf_qwen_classifier.py   # override freely
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
cells.append(reuse(16))   # validation (shared signature: classify_many(vt, system=...))
cells.append(reuse(17))   # caption (shared)
cells.append(reuse(18))   # viz1: naive vs engineered (reuses nacc/eacc)
cells.append(reuse(19))   # viz2: per-quadrant (reuses dfv)

cells.append(md(
"""### <em style="color:#7EA98B">Cross-model comparison (local models)</em>

Run all three scripts (above) to populate `submission2.<model>.csv`, then compare. The cell below loads
whichever per-model CSVs exist alongside the primary `submission2.csv` and reports label splits and pairwise
agreement. With the API models, GLM-4.7 and GPT-5.3 agreed 70/70 (differing from Claude only on **id 37**) —
the interesting question here is whether these *open* models converge the same way, and how they handle that
same source-vs-content row. **The numbers populate once you've run the models.**
"""))
cells.append(code(
'''# ── Cross-model agreement, recomputed from the local submission CSVs (read-only) ──
import csv
from pathlib import Path

FILES = {
    "submission2.csv (primary)": "submission2.csv",
    "Qwen2.5-72B":   "submission2.qwen2.5-72b.csv",
    "Llama-3.3-70B": "submission2.llama-3.3-70b.csv",
    "Mixtral-8x22B": "submission2.mixtral-8x22b.csv",
}

def load(path):
    return {int(r["id"]): int(r["label"]) for r in csv.DictReader(open(path))}

preds = {name: load(p) for name, p in FILES.items() if Path(p).exists()}
if not preds:
    print("No submission2*.csv yet — run the Decision cell and/or the three scripts first.")
else:
    ids = sorted(next(iter(preds.values())))
    print("label split per model (0=cyber/fake, 1=financial/real):")
    for name, d in preds.items():
        v = list(d.values()); print(f"  {name:26s} {{0: {v.count(0)}, 1: {v.count(1)}}}")
    names = list(preds)
    if len(names) > 1:
        print("\\npairwise agreement / 70:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = preds[names[i]], preds[names[j]]
                agree = sum(a[k] == b[k] for k in ids)
                diffs = [k for k in ids if a[k] != b[k]]
                print(f"  {names[i]:26s} vs {names[j]:26s}: {agree}/70" +
                      (f"   (disagree on ids {diffs})" if diffs else "   (identical)"))
'''))

# ═════════════════════════ D — DECISION ═════════════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">D — Decision</span>

**`submission2.csv` is written by the loaded local model** (default `Qwen/Qwen2.5-72B-Instruct`) — pure
GenAI, no classical model in the prediction path (FAQ&nbsp;#8). To promote a different local model to the
official `submission2.csv`, set `HF_MODEL` before running the in-notebook engine, e.g.:

```bash
HF_MODEL=mistralai/Mixtral-8x22B-Instruct-v0.1 jupyter nbconvert --execute submission2.ipynb
```

**How to choose between the three:** pick the model with the best **held-out validation accuracy** from the
Analysis section (all four quadrants), breaking ties toward the model that reads **id&nbsp;37** as the
*source-based* `0` — that is the row that distinguishes provenance-reasoning from surface-plausibility, and
the one the API notebook's perfect score confirmed as `0`. The cell below runs the chosen model over the 70
test rows and writes `submission2.csv`.
"""))
cells.append(code(
'''# ── Predict the 70 test rows with the local model = submission2.csv ─────
# Pure GenAI / prompt engineering (FAQ #8): the local LLM is the SOLE source of every label.
print("Classifying 70 test rows with", HF_MODEL, "...")
preds = np.array(classify_many(rt_te["text"].tolist(), system=SYSTEM)).astype(int)

sample = pd.read_csv("Sample_Submission.csv")
out = pd.DataFrame({"id": rt_te["id"].values, "label": preds})[list(sample.columns)]
out.to_csv("submission2.csv", index=False)

# Validate the submission FORMAT (FAQ #1, #2).
checks = {
    "exactly 2 columns":              out.shape[1] == 2,
    f"columns == {list(sample.columns)}": list(out.columns) == list(sample.columns),
    "row count == 70":                len(out) == 70,
    "ids match test set (and order)": list(out["id"]) == list(rt_te["id"]),
    "labels in {0,1}":                set(int(v) for v in out["label"].unique()) <= {0, 1},
}
for k, v in checks.items(): print("PASS" if v else "FAIL", "-", k)
print(f"\\nWrote submission2.csv = {len(preds)} predictions ({HF_MODEL}, local few-shot prompting)")
print("label counts:", pd.Series(preds).value_counts().to_dict())
display(out.head(8))
'''))

cells.append(md(
"""### <em style="color:#7EA98B">Decision summary & repository map</em>

- **Submit:** `submission2.csv` — written locally by the chosen open model (default Qwen2.5-72B-Instruct),
  few-shot prompting, **0 external API calls**.
- **Why local:** identical method to the winning `submission.csv`, but fully offline — the privacy-preserving
  path when data can't leave the building.
- **Model choice:** decide from the held-out validation; the three candidates are the largest open instruct
  models that fit 128&nbsp;GB at 4-bit.

| Path | What it is |
|---|---|
| [`submission2.csv`](submission2.csv) | **Local submission** — chosen HF model, pure GenAI (`id,label`) |
| `submission2.qwen2.5-72b.csv` / `…llama-3.3-70b.csv` / `…mixtral-8x22b.csv` | per-model outputs (comparison) |
| `riskguardian_hf_qwen_classifier.py` | Qwen2.5-72B-Instruct (72B) few-shot classifier |
| `riskguardian_hf_llama_classifier.py` | Llama-3.3-70B-Instruct (70B) few-shot classifier |
| `riskguardian_hf_mixtral_classifier.py` | Mixtral-8x22B-Instruct (141B MoE) few-shot classifier |
| [`submission.ipynb`](submission.ipynb) | the API twin (Claude/GLM/GPT) this notebook mirrors |

---
*Local Hugging Face edition of the RiskGuardian few-shot-prompting summary. Same engineered prompt, same ROAD
order — open weights, on-prem, `submission2.csv`.*
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
