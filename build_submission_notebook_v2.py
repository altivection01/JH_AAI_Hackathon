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

v2: reproduces the *structure* of the committed `submission.ipynb` (the
restructured ROAD ordering, the Problem-Statement / R / O / A / D cells from
an inlined JSON block, the shared-imports cell, the enlarged figure sizes, and
the cross-model agreement chart). It emits UN-EXECUTED cells: outputs and
embedded chart images are absent and only appear after the notebook is run.
The committed submission.ipynb remains the source of truth for outputs.
"""
import json
from pathlib import Path
import re as _re, base64 as _b64, hashlib as _hl, os as _os

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


# The committed submission strips the shared import block from each inlined
# script body (the imports are hoisted into the standalone imports cell). The
# on-disk .py files keep their imports; we strip them only in the inlined copy.
_WRITEFILE_IMPORTS = (
    "import argparse, json, os, re, time, threading, random\n"
    "from pathlib import Path\n"
    "from concurrent.futures import ThreadPoolExecutor\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import requests\n")

def writefile_cell(path, blurb):
    body = Path(path).read_text()
    body = body.replace(_WRITEFILE_IMPORTS, "", 1)   # hoisted to the imports cell
    header = (f"# {blurb}\n"
              f"# Full few-shot classifier source, inlined. `%%writefile` shows the code AND\n"
              f"# (re)materializes {path} on the disk WITHOUT executing it (no API calls fire here).\n")
    return code(f"%%writefile {path}\n{header}\n{body}")


# ── Pre-extracted hand-edited ROAD text blocks (from the committed notebook) ──
# These are ALREADY reflowed (single-line paragraphs) in an inlined JSON block.
import json as _vjson
_V2_BLOCKS_JSON = r'''{"problem_statement": "## Problem Statement\n\nIn an era of rapid digital transformation, organizations across industries such as finance, healthcare, and energy face unprecedented risks that can impact their operations, reputation, and compliance. These risks range from cybersecurity breaches that compromise sensitive data to financial irregularities that disrupt business continuity. Managing these risks effectively requires not only real-time insights but also predictive capabilities to anticipate potential threats and mitigate their impact.\n\nRiskGuardian Solutions, a global leader in consulting and risk management services, helps clients navigate these challenges by providing tailored solutions. RiskGuardian\u2019s clients depend on advanced risk management frameworks to safeguard their operations, maintain regulatory compliance, and protect their assets. With the growing complexity of the risk landscape, accurately identifying and classifying potential risks has become a critical need for organizations aiming to remain resilient and competitive.\n", "R_section": "## <span style=\"color:#40634A\">R \u2014 Requirements</span>\n**Can GenAI-only methods be used to discriminate between Cyber Risk or Fake News (clickbait) on this dataset at a high degree of accuracy?**\n\nYes. Only one metric truly matters in this exercise: accuracy. We test several LLMs, identify one that achieved 100% accuracy with few-shot prompting, and compare the results to other LLMs, identifying one failure mode for some models.\n\n", "O_header": "---\n## <span style=\"color:#40634A\">O \u2014 Operationalize</span>\n", "O_task": "**The task.** Classify each risk event in `combined_risk_test.csv` as **0 = Cybersecurity** or **1 = Financial**, scored on **accuracy**. The FAQ specifies the method: **prompt engineering with a pre-trained large language model** (FAQ&nbsp;#8) \u2014 a GenAI solution, *not* a classical discriminative model.\n\n### Data Overview\n\n| File | Rows | Columns |\n|---|---|---|\n| `combined_risk_train.csv` | 14,830 (balanced: 7,466 \u00d7 0, 7,364 \u00d7 1) | `id, text, label` |\n| `combined_risk_test.csv` | 70 | `id, text` |\n| `Sample_Submission.csv` | 70 | `id, label` |\n\n**What the data actually is \u2014 the key insight.** The released corpus is a **mixture of two very different sub-corpora**, both present in train *and* test with consistent labels:\n\n| Sub-corpus | ~Share of train | Form | Label 0 | Label 1 |\n|---|---|---|---|---|\n| Synthetic risk events | ~1/3 | **character-obfuscated** text (e.g. *\"CybernseRuirty\"*, *\"financal rtanactions\"*) | Cybersecurity | Financial |\n| ISOT Fake/Real news | ~2/3 | clean English articles | Fake clickbait | **Real (Reuters)** |\n\nSo the real label rule is:\n\n- **0 = cyber risk *or* fake-news clickbait**\n- **1 = financial risk *or* real (Reuters) news**\n\nFor the news half the ground truth is **source-based**: a `(Reuters)` dateline \u21d2 1; sensational/clickbait styling (ALL-CAPS hooks, \"Featured image via Getty Images\") \u21d2 0. This rule is **not inferable from the brief** \u2014 it only emerges from inspecting the training labels. The test set is the same mix: **20 obfuscated + 50 clean (25 of which are Reuters)**.\n\nThe cell below loads the data and *reveals* that two-sub-corpus structure (a dictionary-hit ratio cleanly separates obfuscated risk reports from clean news; within the clean news, `Reuters` presence tracks the 0/1 label almost perfectly).", "A_intro": "### <em style=\"color:#7EA98B\">The prompt-engineering story</em>\n\n**The hard part of this task wasn't classifying \u2014 it was figuring out what the labels mean.**\n\n**<span style=\"color:#7EA98B\">1.</span> The corpus is two datasets in a trench coat.** About one-third of the rows are obfuscated synthetic risk reports (characters deliberately scrambled and typo'd); the other two-thirds are clean news articles from the ISOT fake/real news set. The `0/1` label means *different things* in each: for risk reports, `0` = cybersecurity and `1` = financial; for news, `0` = sensational/fake and `1` = neutral newswire.\n\n**<span style=\"color:#7EA98B\">2.</span> Why a naive prompt fails.** \"Classify as Cybersecurity (0) or Financial (1)\" \u2014 the obvious reading of the brief \u2014 is meaningless for two-thirds of the data, because the news rows aren't about cyber or finance at all. A naive prompt scores only ~0.85; the brief alone never tells you the news rule is fake-vs-real.\n\n**<span style=\"color:#7EA98B\">3.</span> The engineered prompt.** We encode the structure we discovered: a system instruction that names both text types and their separate label rules, plus a few-shot bank with two worked examples per quadrant (obfuscated-cyber, obfuscated-financial, news-fake, news-real). Temperature 0, single-character output. This lifts held-out accuracy from ~0.85 (naive) to 1.00 (engineered).\n\n**<span style=\"color:#7EA98B\">4.</span> Reasoning from content, not leakage.** The literal substring \"Reuters\" separates real from fake almost perfectly in training \u2014 but that's label leakage that need not hold on unseen data. Our prompt is built to reason from *content* \u2014 tone, sourcing, dateline style \u2014 not to grep for a magic word. Evidence: on id=37, a calm, clean, **non-Reuters** story about an Obama book deal, the model correctly reads it as the fake-news half, where two newer models over-trusted the \"sounds real\" surface and got it wrong.\n\n**<span style=\"color:#7EA98B\">5.</span> Honest validation.** We score on a labeled holdout drawn evenly from all four quadrants (never the few-shot examples), report per-quadrant accuracy, and count call failures. The submission is written from the LLM only \u2014 no classical model in the prediction path (FAQ #8).", "D_section": "---\n## <span style=\"color:#40634A\">D \u2014 Decision</span>\n**It is clear that the business goal of successful discrimination can be met** The lowest accuracy of the few-shot LLM models against the test dataset was 98.57%, and Anthropic Sonnet 4.5 achieved 100%. The code to implement this solution in a scalable way was simple, and can be easily maintained. The business recommendation is to deploy using **Claude Sonnet 4.5** \u2014 pure GenAI, no classical model anywhere in the prediction path (FAQ&nbsp;#8). It scored **100%** on the 70-row test set, confirming **row 37 = 0**.\n\n### <em style=\"color:#7EA98B\">Row 37 \u2014 the one that separated them</em>\n\n> *\"Publisher Announces Books By The Obamas Are Officially On The Way \u2014 Penguin Random House announced\u2026\"*\n\nA clean, **non-Reuters** article (with tell-tale obfuscation artifacts like *\"Oenguin\"*, *\"MArkus Dohle\"*).\n\n- **GLM-4.7 and GPT-5.3 said 1** \u2014 they reasoned from *content*: this reads like a sober, factual report, therefore \"real news.\"\n- **Claude and the classical SVM said 0** \u2014 they followed the *source fingerprint*: no Reuters dateline \u21d2 it belongs to the fake-news half.\n\nThe dataset's ground truth is source-based, so **0 is correct** \u2014 and the perfect score confirmed it. The newer models' shared instinct to judge *plausibility over provenance* is exactly the failure mode the engineered prompt was designed to prevent, and it's notable that two independent frontier models failed it *identically*.\n\n### <em style=\"color:#7EA98B\">Decision summary</em>\n\n- **Submitted:** `submission.csv` \u2014 Claude `claude-sonnet-4-5`, few-shot prompting, **100%**.\n- **Why Claude:** it reasoned from *provenance* (source fingerprint), not surface plausibility, and got the single hardest row right where two newer models slipped.\n- **Confidence:** the held-out validation (100% across all four quadrants, three independent models, plus a 99.99%-CV classical reference that agrees 70/70 with Claude) is the stronger evidence that the method generalizes within this dataset's structure \u2014 the 70-item perfect score is a small-sample confirmation.\n\n### <em style=\"color:#7EA98B\">Business value</em>\n\n**The user.** A RiskGuardian analyst triaging an inbound queue of risk events \u2014 each item must be routed to the **cybersecurity** or **financial** workstream before it can be actioned; misrouting delays response and muddies compliance reporting.\n\n**The value.** Prediction is a single few-shot LLM call per event \u2014 sub-second, **no model to train, retrain, or host**, and the entire decision rule is one auditable prompt. Extending to a new risk domain is a prompt edit (a few worked examples), not a labeling-and-retraining cycle, so the approach scales at near-zero marginal engineering cost \u2014 the explain-and-recommend profile a consulting client expects.\n\nThe read-only diagnostic below contrasts the classical reference against the LLM submission \u2014 the quickest way to eyeball the id=37 disagreement. It writes nothing and is not part of the graded prediction path.\n", "chart_md_sub": "### <em style=\"color:#7EA98B\">Cross-model agreement \u2014 the two camps, visualized</em>\n\nThe same agreement as a matrix. Two blocks of perfect agreement emerge \u2014 **{Claude, classical}** and **{GLM-4.7, GPT-5.3}** \u2014 and every cross-camp pair differs on **exactly one row, id 37**: the single test item that separated the field.\n", "chart_md_main": "### <em style=\"color:#7EA98B\">Cross-model agreement \u2014 the two camps</em>\n\nRunning the *identical* engineered prompt through three frontier models (plus the classical TF-IDF + LinearSVC reference) and comparing the resulting submissions row-by-row reveals two blocks of perfect agreement \u2014 **{Claude, classical}** and **{GLM-4.7, GPT-5.3}**. Every cross-camp pair differs on **exactly one row, id 37** \u2014 the lone test item that separated the field. The chart below recomputes this directly from the saved submission CSVs (read-only; no API calls).\n", "chart_code": "# \u2500\u2500 Viz 3: cross-model agreement matrix \u2014 the \"two camps\" (read-only CSVs) \u2500\u2500\nimport os, csv\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom matplotlib.colors import ListedColormap, BoundaryNorm\n\nFILES = {\n    \"Claude\\n(submitted)\":    \"submission.csv\",\n    \"Classical\\nTF-IDF+SVM\":  \"submission_classical.csv\",\n    \"GLM-4.7\\n(Cerebras)\":    \"submission.cerebras.zai-glm-4.7.csv\",\n    \"GPT-5.3\\n(OpenAI)\":      \"submission.openai.gpt-5.3-chat-latest.csv\",\n}\ndef _load(p):\n    return {int(r[\"id\"]): int(r[\"label\"]) for r in csv.DictReader(open(p))}\npreds = {name: _load(path) for name, path in FILES.items() if os.path.exists(path)}\nnames = list(preds)\nids   = sorted(next(iter(preds.values())))\nN     = len(ids)\nn     = len(names)\n\n# Pairwise agreement count (out of N); disagreement is 0 within a camp, 1 across camps.\nagree    = np.array([[sum(preds[a][k] == preds[b][k] for k in ids) for b in names] for a in names])\ndisagree = N - agree\n\nfig, ax = plt.subplots(figsize=(7.2, 6.2))\ncmap = ListedColormap([\"#2E5B40\", \"#D55E00\"])          # green = identical, orange = differs\nnorm = BoundaryNorm([-0.5, 0.5, max(1, disagree.max()) + 0.5], cmap.N)\nax.imshow(disagree, cmap=cmap, norm=norm)\n\nax.set_xticks(range(n)); ax.set_yticks(range(n))\nax.set_xticklabels(names, fontsize=10); ax.set_yticklabels(names, fontsize=10)\nax.set_xticks(np.arange(-.5, n, 1), minor=True); ax.set_yticks(np.arange(-.5, n, 1), minor=True)\nax.grid(which=\"minor\", color=\"white\", linewidth=2); ax.tick_params(which=\"minor\", length=0)\nfor i in range(n):\n    for j in range(n):\n        ax.text(j, i, f\"{agree[i, j]}/{N}\", ha=\"center\", va=\"center\",\n                color=\"white\", fontsize=12, fontweight=\"bold\")\nax.set_title(\"Cross-model agreement on the 70-row test set\\n\"\n             \"Two camps \u2014 they differ on exactly one row (id 37)\",\n             fontsize=12, fontweight=\"bold\")\nfig.tight_layout()\nos.makedirs(\"figures\", exist_ok=True)\nfig.savefig(\"figures/cross_model_agreement.png\", dpi=150, bbox_inches=\"tight\")\nplt.show()\n"}'''
BLOCKS = _vjson.loads(_V2_BLOCKS_JSON)

def reuse_strip_imports(idx, old, new=""):
    """reuse(idx) but with an import block (hand-)removed from the source.
    The committed submission hoists shared imports into one cell and strips them
    from the reused code cells. No-op (with a guard) if `old` isn't present."""
    c = reuse(idx)
    s = "".join(c["source"])
    if old in s:
        s = s.replace(old, new, 1)
    c["source"] = s.splitlines(keepends=True)
    return c

def reuse_replace(idx, old, new):
    """reuse(idx) with a single literal substring swap (e.g. enlarged figsize)."""
    c = reuse(idx)
    s = "".join(c["source"])
    if old in s:
        s = s.replace(old, new, 1)
    c["source"] = s.splitlines(keepends=True)
    return c

cells = []

# 0 ── ART (reused from the top of the source notebook) ──
cells.append(reuse(0))

# 1 ── Title + ROAD orientation ──
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

# 2 ── Problem Statement (hand-added block) ──
cells.append(md(BLOCKS["problem_statement"]))

# ═══════════════════════════════ R — REQUIREMENTS ════════════════════════════
# 3 ── R — Requirements (question) ──
cells.append(md(BLOCKS["R_section"]))

# ═══════════════════════════════ O — OPERATIONALIZE ══════════════════════════
# 4 ── O — Operationalize (header) ──
cells.append(md(BLOCKS["O_header"]))
# 5 ── O task + data overview ──
cells.append(md(BLOCKS["O_task"]))

# 6 ── "The prompt-engineering story" (A_intro block == source cell) ──
cells.append(md(BLOCKS["A_intro"]))

# 7 ── shared imports (hoisted out of the reused code cells) ──
cells.append(code(
"""import os, re, json, time, requests, threading, random, argparse
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import anthropic"""))

# 8 ── load + reveal the combined structure (imports stripped -> cell 7) ──
cells.append(reuse_strip_imports(48, "import re\nfrom pathlib import Path\n"))

# 9 ── the in-notebook few-shot engine (markdown) ──
cells.append(md(
"""### <em style="color:#7EA98B">The in-notebook few-shot engine (Anthropic Claude)</em>

This is the live, runnable predictor: it builds the few-shot bank (2 examples per quadrant, deterministic),
paces requests under the API rate limit, and few-shot-prompts the model for a single-character label.
Requires `ANTHROPIC_API_KEY` in the environment or `config.json`.
"""))
# 10 ── in-notebook GenAI predictor (imports stripped -> cell 7) ──
cells.append(reuse_strip_imports(
    55,
    "\nimport os, re, json, time, requests\nimport numpy as np, pandas as pd\n"
    "from pathlib import Path\nfrom concurrent.futures import ThreadPoolExecutor\n\n",
    "\n\n\n"))

# 11 ── the three provider few-shot scripts (markdown) ──
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
# 12-17 ── the three provider scripts, inline (md + %%writefile each) ──
for path, blurb in SCRIPTS:
    cells.append(md(f"**`{path}`** — {blurb}"))
    cells.append(writefile_cell(path, blurb))

# ═══════════════════════════════ A — ANALYSIS ════════════════════════════════
# 18 ── A — Analysis header (with embedded assumption-test reframe) ──
cells.append(md(
"""---
## <span style="color:#40634A">A — Analysis</span>

### <em style="color:#7EA98B">Honest validation: held-out accuracy + prompt-engineering lift</em>

We score the engineered prompt on a labeled holdout drawn evenly from all four quadrants (never the few-shot examples), report per-quadrant accuracy, and count call failures. The same rows are also scored with a **naive** prompt ("classify as cyber=0 / financial=1") to measure the lift from encoding structure. This holdout is really an **assumption test**: the engineered prompt hard-codes our discovered claim that the corpus is *two sub-corpora with different label rules* — if that assumption were wrong, per-quadrant accuracy would crack on at least one quadrant. It does not (all four hit 1.00), which is what licenses the leaderboard score as *structure we validated* rather than luck on 70 rows.
"""))
# 19 ── validation (reuse) ──
cells.append(reuse(56))
# 20 ── caption (plain — NO reframe here; the source cell carries it) ──
cells.append(md("Prompt-engineering lift and per-quadrant accuracy, from the held-out validation above."))
# 21 ── viz1: naive vs engineered (enlarged figure) ──
cells.append(reuse_replace(58, "figsize=(5, 4)", "figsize=(12, 8)"))
# 22 ── viz2: per-quadrant ──
cells.append(reuse(59))

# 23 ── held-out result + cross-model comparison (markdown) ──
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
# 24 ── cross-model agreement code (recomputed from the CSVs) ──
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
# 25 ── cross-model agreement chart (markdown block) ──
cells.append(md(BLOCKS["chart_md_sub"]))
# 26 ── cross-model agreement chart (code block) ──
cells.append(code(BLOCKS["chart_code"]))

# ═══════════════════════════════ D — DECISION ════════════════════════════════
# 27 ── D — Decision (hand-added block) ──
cells.append(md(BLOCKS["D_section"]))
# 28 ── diagnostic: classical vs LLM submission (reuse) ──
cells.append(reuse(62))

# 29 ── Repository map ──
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

# ── post-process: reflow + green-numbers + logo-swap on every markdown cell ──
# Mirrors the committed notebook's hand-edits. The block strings are already
# reflowed (so _reflow is a no-op on them) and the reused source cells are too;
# the pipeline is applied uniformly for consistency. The assumption-test reframe
# already lives inside the A_intro / D_section blocks, so it is NOT re-added.
_GREEN = "#7EA98B"
_ASSETS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "assets")
def _is_boundary(l):
    s = l.lstrip()
    if s == '': return True
    if s[:1] in ('|', '#', '>', '<'): return True
    if s.startswith('```'): return True
    if _re.match(r'(?:[-*_]\s*){3,}$', s): return True
    return False
def _is_list_item(l):
    s = l.lstrip(); return bool(_re.match(r'[-*+]\s', s) or _re.match(r'\d+\.\s', s))
def _reflow(src):
    out = []; buf = []
    def flush():
        if buf: out.append(' '.join(x.strip() for x in buf)); buf.clear()
    for l in src.split('\n'):
        if l.strip() == '': flush(); out.append('')
        elif _is_boundary(l): flush(); out.append(l)
        elif _is_list_item(l): flush(); buf.append(l)
        else: buf.append(l)
    flush(); return '\n'.join(out)
def _green_numbers(src):
    return _re.sub(r'\*\*(\d+)\.(\s)', r'**<span style="color:' + _GREEN + r'">\1.</span>\2', src)
_white_b64 = _b64.b64encode(open(_os.path.join(_ASSETS, "jhu-hackathon-logo-white-bg.png"), "rb").read()).decode()
_alpha_md5 = _hl.md5(open(_os.path.join(_ASSETS, "the_plot_thickens_logo_alpha.png"), "rb").read()).hexdigest()
def _swap_alpha_logo(src):
    m = _re.search(r'base64,([A-Za-z0-9+/=]+)', src)
    if m and _hl.md5(_b64.b64decode(m.group(1))).hexdigest() == _alpha_md5:
        return src[:m.start(1)] + _white_b64 + src[m.end(1):]
    return src

for c in cells:
    if c["cell_type"] != "markdown":
        continue
    s = "".join(c["source"])
    s = _swap_alpha_logo(s)
    s = s.replace("## Problem Statement",
                  '## <span style="color:#40634A">Problem Statement</span>')
    s = _reflow(s)
    s = _green_numbers(s)
    c["source"] = s.splitlines(keepends=True)

# ───────────────────────── footer: left-aligned white-bg logo + rule ─────────
cells.append(md('<div align="left"><img src="data:image/png;base64,' + _white_b64 +
                '" alt="the_plotly_thickens — RiskGuardian (JHU AAI Hackathon)" '
                'width="300" /></div>'))
cells.append(md("---"))

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
