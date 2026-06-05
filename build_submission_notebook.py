#!/usr/bin/env python
"""
build_submission_notebook.py — assemble submission.ipynb in ROAD analytical format.

ROAD = Requirements -> Operationalize -> Analysis -> Decision.

The notebook summarizes the few-shot-prompting results for the RiskGuardian
hackathon (Claude Sonnet 4.5 won at 100%). It:
  * reuses the banner ART from the top of RiskGuardian_Cyber_Risk_Assessment.ipynb,
  * reuses the few-shot analysis blocks (Part XVI/XVII) from that same notebook,
  * inlines the three provider few-shot scripts (Anthropic / Cerebras / OpenAI)
    as %%writefile cells so the full source is visible *and* materializable
    without firing API calls on run.
"""
import json
from pathlib import Path

SRC_NB = "RiskGuardian_Cyber_Risk_Assessment.ipynb"
OUT_NB = "submission.ipynb"
SCRIPTS = [
    ("riskguardian_llm_classifier.py",      "Anthropic Claude — `claude-sonnet-4-5` (the winning submission)"),
    ("riskguardian_cerebras_classifier.py", "Cerebras — `zai-glm-4.7` (~355B open-source MoE, comparison)"),
    ("riskguardian_openai_classifier.py",   "OpenAI — `gpt-5.3-chat-latest` (comparison)"),
]

src = json.load(open(SRC_NB))
src_cells = src["cells"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


def reuse(idx):
    """Copy a source cell verbatim, stripping any saved outputs/counts."""
    c = json.loads(json.dumps(src_cells[idx]))
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    c.setdefault("metadata", {})
    return c


def writefile_cell(path, blurb):
    body = Path(path).read_text()
    header = (f"# {blurb}\n"
              f"# Full few-shot classifier source, inlined. `%%writefile` shows the code AND\n"
              f"# (re)materializes {path} on the disk WITHOUT executing it (no API calls fire here).\n")
    return code(f"%%writefile {path}\n{header}\n{body}")


cells = []

# ───────────────────────── ART (reused from the top of the source notebook) ──
cells.append(reuse(0))

# ───────────────────────── Title + ROAD orientation ─────────────────────────
cells.append(md(
"""# <span style="color:#22442C">RiskGuardian — Few-Shot Prompting Submission</span>
### <em style="color:#7EA98B">Cyber (0) vs. Financial (1) risk classification — pure GenAI / prompt-engineering (FAQ&nbsp;#8)</em>

> **Result: a perfect score (100%) on the 70-row hold-out test set.**
> Final submission: [`submission.csv`](submission.csv) — pure few-shot prompting with **Anthropic Claude** (`claude-sonnet-4-5`).

This notebook is laid out in **ROAD** analytical format:

| | Stage | What it answers |
|---|---|---|
| **R** | **Requirements** | What was asked, what the data *actually* is, and the success metric. |
| **O** | **Operationalize** | The engineered few-shot prompt, and the three provider scripts that run it (inline). |
| **A** | **Analysis** | Held-out validation, prompt-engineering lift, and the cross-model comparison. |
| **D** | **Decision** | Which model we submitted, why, and the one row that separated the field. |

**TL;DR.** The task looked like "cyber vs. financial," but the corpus is secretly a **combination of two
sub-corpora** — and for ~2/3 of it the real signal is **fake-vs-real news**, not topic. We encoded that
hidden structure into a few-shot prompt and ran the *identical* prompt through three frontier models.
**Claude Sonnet 4.5 scored 100%.** The headline finding: **GLM-4.7 and GPT-5.3 made byte-identical
predictions (70/70 with each other), including the same single mistake on row 37** — while Claude (and a
classical reference) got that row right.
"""))

# ═══════════════════════════════ R — REQUIREMENTS ════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">R — Requirements</span>

**The task.** Classify each risk event in `combined_risk_test.csv` as **0 = Cybersecurity** or
**1 = Financial**, scored on **accuracy**. The FAQ specifies the method: **prompt engineering with a
pre-trained large language model** (FAQ&nbsp;#8) — a GenAI solution, *not* a classical discriminative model.

| File | Rows | Columns |
|---|---|---|
| `combined_risk_train.csv` | 14,830 (balanced: 7,466 × 0, 7,364 × 1) | `id, text, label` |
| `combined_risk_test.csv` | 70 | `id, text` |
| `Sample_Submission.csv` | 70 | `id, label` |

**What the data actually is — the key insight.** The released corpus is a **mixture of two very different
sub-corpora**, both present in train *and* test with consistent labels:

| Sub-corpus | ~Share of train | Form | Label 0 | Label 1 |
|---|---|---|---|---|
| Synthetic risk events | ~1/3 | **character-obfuscated** text (e.g. *"CybernseRuirty"*, *"financal rtanactions"*) | Cybersecurity | Financial |
| ISOT Fake/Real news | ~2/3 | clean English articles | Fake clickbait | **Real (Reuters)** |

So the real label rule is:

- **0 = cyber risk *or* fake-news clickbait**
- **1 = financial risk *or* real (Reuters) news**

For the news half the ground truth is **source-based**: a `(Reuters)` dateline ⇒ 1; sensational/clickbait
styling (ALL-CAPS hooks, "Featured image via Getty Images") ⇒ 0. This rule is **not inferable from the
brief** — it only emerges from inspecting the training labels. The test set is the same mix:
**20 obfuscated + 50 clean (25 of which are Reuters)**.

The cell below loads the data and *reveals* that two-sub-corpus structure (a dictionary-hit ratio cleanly
separates obfuscated risk reports from clean news; within the clean news, `Reuters` presence tracks the
0/1 label almost perfectly).
"""))
cells.append(reuse(48))   # load + reveal the combined structure

# ═══════════════════════════════ O — OPERATIONALIZE ══════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">O — Operationalize</span>

The winning method is **engineered few-shot prompting**: a prompt that *encodes the discovered structure*
rather than the brief's surface framing —

1. A **system prompt** describing *both* text types and their separate label rules — including the news
   source rule (neutral newswire/Reuters ⇒ 1, sensational clickbait ⇒ 0).
2. **Eight few-shot examples spanning all four quadrants** (obfuscated-cyber, obfuscated-financial,
   fake-news, real-news), so the model sees the pattern, not just the description.
3. A **constrained output**: reply with a single character, `0` or `1`. Temperature 0.
"""))
cells.append(reuse(54))   # "The prompt-engineering story" markdown

cells.append(md(
"""### <em style="color:#7EA98B">The in-notebook few-shot engine (Anthropic Claude)</em>

This is the live, runnable predictor: it builds the few-shot bank (2 examples per quadrant, deterministic),
paces requests under the API rate limit, and few-shot-prompts the model for a single-character label.
Requires `ANTHROPIC_API_KEY` in the environment or `config.json`.
"""))
cells.append(reuse(55))   # in-notebook GenAI predictor

# ── the three provider scripts, inline ──
cells.append(md(
"""### <em style="color:#7EA98B">The three provider few-shot scripts (inlined)</em>

The *same* engineered prompt was run through three providers so the results can be compared head-to-head.
Each script is reproduced in full below via `%%writefile`, which **displays the source inline and
(re)writes the file to disk without executing it** — so running these cells will not fire any API calls.
Run a script from a shell to actually classify (e.g. `python riskguardian_llm_classifier.py`).

- **`riskguardian_llm_classifier.py`** — Anthropic Claude, the *pure-GenAI* primary; writes `submission.csv`.
- **`riskguardian_cerebras_classifier.py`** — same prompt on Cerebras `zai-glm-4.7` (~355B MoE, a *reasoning* model).
- **`riskguardian_openai_classifier.py`** — same prompt on OpenAI, auto-adapting per-model API quirks.
"""))
for path, blurb in SCRIPTS:
    cells.append(md(f"**`{path}`** — {blurb}"))
    cells.append(writefile_cell(path, blurb))

# ═══════════════════════════════ A — ANALYSIS ════════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">A — Analysis</span>

### <em style="color:#7EA98B">Honest validation: held-out accuracy + prompt-engineering lift</em>

We score the engineered prompt on a labeled holdout drawn evenly from all four quadrants (never the
few-shot examples), report per-quadrant accuracy, and count call failures. The same rows are also scored
with a **naive** prompt ("classify as cyber=0 / financial=1") to measure the lift from encoding structure.
"""))
cells.append(reuse(56))   # validation
cells.append(reuse(57))   # caption
cells.append(reuse(58))   # viz1: naive vs engineered
cells.append(reuse(59))   # viz2: per-quadrant

cells.append(md(
"""### <em style="color:#7EA98B">Held-out result</em>

| Prompt | Accuracy (held-out) |
|---|---|
| Naive ("classify as cyber=0 / financial=1") | ~0.85–0.95 |
| **Engineered (structure + few-shot)** | **1.000** |

A naive prompt mislabels much of the news half (it has no reason to map "Obama clickbait → 0"). The
engineered prompt nails all four quadrants. **This structure-encoding is what won it.**

### <em style="color:#7EA98B">Cross-model comparison — the remarkable part</em>

We ran the **identical prompt** through three frontier models (plus a classical TF-IDF + LinearSVC
reference). All four agree on **69 / 70** test rows. Every disagreement collapses to **a single row: id 37.**

| Model | Label split | Hold-out acc | Naive → engineered | id 37 |
|---|---|---|---|---|
| Classical TF-IDF + LinearSVC | {0:34, 1:36} | 99.99% CV | — | **0** |
| **Claude Sonnet 4.5** (submitted) | {0:34, 1:36} | 1.000 | 0.85 → 1.00 | **0** |
| GLM-4.7 (Cerebras, ~355B MoE) | {0:33, 1:37} | 1.000 | 0.95 → 1.00 | **1** |
| GPT-5.3-chat-latest (OpenAI) | {0:33, 1:37} | 1.000 | 0.95 → 1.00 | **1** |

Pairwise agreement reveals exactly **two camps**: `classical == Claude` → **70/70**, and
`GLM-4.7 == GPT-5.3` → **70/70**. GLM-4.7 (Z.ai, served on Cerebras) and GPT-5.3 (OpenAI) are different
vendors, different architectures, different training pipelines — yet they produced **byte-identical
predictions across all 70 rows**, right down to **the same lone error on row 37**.

The cell below recomputes that agreement directly from the three submission CSVs.
"""))
cells.append(code(
'''# ── Cross-model agreement, recomputed from the submission CSVs (read-only) ──
import csv
from pathlib import Path

FILES = {
    "Claude (submitted)": "submission.csv",
    "GLM-4.7 (Cerebras)": "submission.cerebras.zai-glm-4.7.csv",
    "GPT-5.3 (OpenAI)":   "submission.openai.gpt-5.3-chat-latest.csv",
}

def load(path):
    return {int(r["id"]): int(r["label"]) for r in csv.DictReader(open(path))}

preds = {name: load(p) for name, p in FILES.items() if Path(p).exists()}
ids = sorted(next(iter(preds.values())))

print("label split per model (0=cyber/fake, 1=financial/real):")
for name, d in preds.items():
    v = list(d.values())
    print(f"  {name:22s} {{0: {v.count(0)}, 1: {v.count(1)}}}")

names = list(preds)
print("\\npairwise agreement / 70:")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = preds[names[i]], preds[names[j]]
        agree = sum(a[k] == b[k] for k in ids)
        diffs = [k for k in ids if a[k] != b[k]]
        print(f"  {names[i]:22s} vs {names[j]:22s}: {agree}/70" +
              (f"   (disagree on id {diffs})" if diffs else "   (identical)"))
'''))

# ═══════════════════════════════ D — DECISION ════════════════════════════════
cells.append(md(
"""---
## <span style="color:#40634A">D — Decision</span>

**We submitted Claude Sonnet 4.5** — pure GenAI, no classical model anywhere in the prediction path
(FAQ&nbsp;#8). It scored **100%** on the 70-row test set, confirming **row 37 = 0**.

### <em style="color:#7EA98B">Row 37 — the one that separated them</em>

> *"Publisher Announces Books By The Obamas Are Officially On The Way — Penguin Random House announced…"*

A clean, **non-Reuters** article (with tell-tale obfuscation artifacts like *"Oenguin"*, *"MArkus Dohle"*).

- **GLM-4.7 and GPT-5.3 said 1** — they reasoned from *content*: this reads like a sober, factual report, therefore "real news."
- **Claude and the classical SVM said 0** — they followed the *source fingerprint*: no Reuters dateline ⇒ it belongs to the fake-news half.

The dataset's ground truth is source-based, so **0 is correct** — and the perfect score confirmed it. The
newer models' shared instinct to judge *plausibility over provenance* is exactly the failure mode the
engineered prompt was designed to prevent, and it's notable that two independent frontier models failed it
*identically*.

### <em style="color:#7EA98B">Decision summary</em>

- **Submit:** `submission.csv` — Claude `claude-sonnet-4-5`, few-shot prompting, **100%**.
- **Why Claude:** it reasoned from *provenance* (source fingerprint), not surface plausibility, and got the
  single hardest row right where two newer models slipped.
- **Confidence:** the held-out validation (100% across all four quadrants, three independent models, plus a
  99.99%-CV classical reference that agrees 70/70 with Claude) is the stronger evidence that the method
  generalizes within this dataset's structure — the 70-item perfect score is a small-sample confirmation.

The read-only diagnostic below contrasts the classical reference against the LLM submission — the quickest
way to eyeball the id=37 disagreement. It writes nothing and is not part of the graded prediction path.
"""))
cells.append(reuse(62))   # diagnostic: classical vs LLM submission

cells.append(md(
"""### <em style="color:#7EA98B">Repository map</em>

| Path | What it is |
|---|---|
| [`submission.csv`](submission.csv) | **Winning submission** — Claude `claude-sonnet-4-5`, pure GenAI (`id,label`) |
| `submission.cerebras.zai-glm-4.7.csv` | GLM-4.7 predictions (comparison) |
| `submission.openai.gpt-5.3-chat-latest.csv` | GPT-5.3 predictions (comparison) |
| `riskguardian_llm_classifier.py` | **Primary** few-shot classifier — Anthropic Claude (pure GenAI) |
| `riskguardian_cerebras_classifier.py` | Same prompt on Cerebras `zai-glm-4.7` |
| `riskguardian_openai_classifier.py` | Same prompt on OpenAI |
| `RiskGuardian_Cyber_Risk_Assessment.ipynb` | Full analysis notebook (this is the few-shot summary extract) |

---
*Few-shot-prompting summary for the RiskGuardian Solutions hackathon. The engineered prompt, the three
provider scripts, the held-out validation, and the cross-model comparison are all reproduced above in ROAD
order.*
"""))

# ───────────────────────── assemble + write ─────────────────────────────────
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
