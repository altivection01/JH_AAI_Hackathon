# RiskGuardian — Cyber vs. Financial Risk Classification

> **Result: a perfect score (100%) on the 70-row hold-out test set.**
> Final submission: [`submission.csv`](submission.csv) — pure few-shot prompting with **Anthropic Claude** (`claude-sonnet-4-5`).

---

## TL;DR

- The task looked like "classify text as cybersecurity (0) vs. financial (1)." The data was secretly a **combination of two corpora**, and the real signal for ~2/3 of it was **fake-vs-real news**, not topic.
- We first solved it with **classical ML** (TF-IDF + Linear SVM) at **99.99% cross-val accuracy** — a very sharp knife, but the FAQ required a **GenAI / prompt-engineering** approach. We brought a knife to a gunfight.
- We pivoted to **pure few-shot LLM prompting**, encoding the dataset's true structure into the prompt. That submission scored **100%**.
- We then ran the *same* prompt through three frontier models. The most striking finding: **GLM-4.7 and GPT-5.3 made byte-identical predictions (70/70 with each other), including the same single mistake on row 37** — while Claude and the classical model got that row right.

---

## The task

Classify each risk event in `combined_risk_test.csv` as **0 = Cybersecurity** or **1 = Financial**, scored on **accuracy**. The FAQ specified the method: **prompt engineering with a pre-trained large language model**.

| File | Rows | Columns |
|---|---|---|
| `combined_risk_train.csv` | 14,830 (balanced: 7,466 × 0, 7,364 × 1) | `id, text, label` |
| `combined_risk_test.csv` | 70 | `id, text` |
| `Sample_Submission.csv` | 70 | `id, label` |

## What the data actually was (the key insight)

The released corpus is a **mixture of two very different sub-corpora**, both present in train *and* test with consistent labels:

| Sub-corpus | ~Share of train | Form | Label 0 | Label 1 |
|---|---|---|---|---|
| Synthetic risk events | ~1/3 | **character-obfuscated** text (e.g. *"CybernseRuirty"*, *"financal rtanactions"*) | Cybersecurity | Financial |
| ISOT Fake/Real News | ~2/3 | clean English articles | Fake clickbait | **Real (Reuters)** |

So the label rule is really:

- **0 = cyber risk *or* fake-news clickbait**
- **1 = financial risk *or* real (Reuters) news**

For the news half, the ground truth is **source-based**: a `(Reuters)` dateline ⇒ 1; sensational/clickbait styling (ALL-CAPS hooks, "Featured image via Getty Images") ⇒ 0. This rule is *not inferable from the brief* — it only emerges from inspecting the training labels. The test set is the same mix: **20 obfuscated + 50 clean (25 of which are Reuters)**.

## Our journey: a (sharp) knife to a gunfight

**Phase 1 — Classical ML.** A `FeatureUnion` of word (1–2gram) and **character `char_wb` (3–5gram)** TF-IDF features into a `LinearSVC`. The character n-grams are the trick: they stay informative even when per-character scrambling destroys whole-word tokens.

- **5-fold CV accuracy: 0.9999** (1.0000 on the obfuscated subset, 0.9998 on the clean subset)
- LinearSVC, Logistic Regression, and a structural Reuters/cleanliness rule all agreed 70/70 on the test set.

It was an excellent classifier — and the **wrong weapon**. The FAQ asked for a GenAI prompt-engineering solution; a discriminative linear model, however accurate, doesn't satisfy that. Sharp knife, gunfight.

**Phase 2 — Pure GenAI.** We rebuilt the predictor as **few-shot LLM prompting**, with the classical model retained only as a reference (and later removed entirely, so the submission is 100% GenAI).

## The winning method: engineered few-shot prompting

The prompt **encodes the discovered structure** rather than the brief's surface framing:

1. A system prompt describing *both* text types and their label rules — including the news source rule (neutral newswire/Reuters ⇒ 1, sensational clickbait ⇒ 0).
2. **Eight few-shot examples spanning all four quadrants** (obfuscated-cyber, obfuscated-financial, fake-news, real-news), so the model sees the pattern, not just the description.
3. A constrained output: reply with a single character, `0` or `1`.

**This structure-encoding is what won it.** We measured the lift against a naive "cyber vs. financial" prompt on a held-out labeled set:

| Prompt | Accuracy (held-out) |
|---|---|
| Naive ("classify as cyber=0 / financial=1") | ~0.85–0.95 |
| **Engineered (structure + few-shot)** | **1.000** |

A naive prompt mislabels much of the news half (it has no reason to map "Obama clickbait → 0"). The engineered prompt nails it.

## Cross-model comparison (the remarkable part)

We ran the **identical prompt** through three frontier models (plus the classical reference). All four agree on **69 / 70** test rows. Every disagreement collapses to **a single row: id 37.**

| Model | Label split | Hold-out acc | Naive → engineered | id 37 |
|---|---|---|---|---|
| Classical TF-IDF + LinearSVC | {0:34, 1:36} | 99.99% CV | — | **0** |
| **Claude Sonnet 4.5** (submitted) | {0:34, 1:36} | 1.000 | 0.85 → 1.00 | **0** |
| GLM-4.7 (Cerebras, ~355B MoE) | {0:33, 1:37} | 1.000 | 0.95 → 1.00 | **1** |
| GPT-5.3-chat-latest (OpenAI) | {0:33, 1:37} | 1.000 | 0.95 → 1.00 | **1** |

Pairwise agreement reveals exactly **two camps**:

- `classical == Claude` → **70/70**
- `GLM-4.7 == GPT-5.3` → **70/70**

**The homogeneity of the newer models is the headline.** GLM-4.7 (Z.ai, served on Cerebras) and GPT-5.3 (OpenAI) are different vendors, different architectures, different training pipelines — yet they produced **byte-identical predictions across all 70 rows**, right down to **the same lone error on row 37**. Two independent frontier models converged on the same answer *and* the same mistake.

### Row 37 — the one that separated them

> *"Publisher Announces Books By The Obamas Are Officially On The Way — Penguin Random House announced…"*

A clean, **non-Reuters** article (with tell-tale obfuscation artifacts like *"Oenguin"*, *"MArkus Dohle"*).

- **GLM-4.7 and GPT-5.3 said 1** — they reasoned from *content*: this reads like a sober, factual report, therefore "real news."
- **Claude and the classical SVM said 0** — they followed the *source fingerprint*: no Reuters dateline ⇒ it belongs to the fake-news half.

The dataset's ground truth is source-based, so **0 is correct** — and the perfect score confirmed it. The newer models' shared instinct to judge *plausibility over provenance* is exactly the failure mode the engineered prompt was designed to prevent, and it's notable that two of them failed it *identically*.

## Results

- **`submission.csv` (Claude) scored 100% on the 70-row test set.**
- This confirms **row 37 = 0**: the source-fingerprint reasoning (Claude + classical SVM + Reuters rule) was right; the two newer LLMs were the only ones that slipped, on the single hardest row.

## Repository contents

| Path | What it is |
|---|---|
| [`submission.csv`](submission.csv) | **Winning submission** — Claude `claude-sonnet-4-5`, pure GenAI (`id,label`) |
| `submission.cerebras.zai-glm-4.7.csv` | GLM-4.7 predictions (comparison) |
| `submission.openai.gpt-5.3-chat-latest.csv` | GPT-5.3 predictions (comparison) |
| `riskguardian_llm_classifier.py` | **Primary** few-shot classifier — Anthropic Claude (pure GenAI) |
| `riskguardian_cerebras_classifier.py` | Same prompt on Cerebras `zai-glm-4.7` (~355B, only >300B model on the key) |
| `riskguardian_openai_classifier.py` | Same prompt on OpenAI (default `gpt-5.3-chat-latest`; override via `OPENAI_MODEL`) |
| `RiskGuardian_Cyber_Risk_Assessment.ipynb` | Analysis notebook — **Part XVI** (classical reference) + **Part XVII** (in-notebook LLM) |
| `build_notebook.py` | Regenerates the notebook |
| `combined_risk_train.csv` / `combined_risk_test.csv` / `Sample_Submission.csv` | Provided data |
| `config.json` | API keys — **git-ignored**; you must supply your own (see below) |

> Note: `submission_classical.csv` may exist as a leftover from the classical reference; the official submission is `submission.csv`.

## How to run

**Environment:** Python 3.11 with `numpy`, `pandas`, `requests` (the pure-GenAI scripts), plus `scikit-learn` for the classical reference / notebook. In this project that's the `MLENV311` conda env.

**API keys:** create `config.json` (already git-ignored) with the providers you want to use:

```json
{
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "CEREBRAS_API_KEY":  "csk-...",
  "OPENAI_API_KEY":    "sk-proj-..."
}
```

**Generate the (winning) submission:**

```bash
python riskguardian_llm_classifier.py                # validate on a hold-out, then classify the 70 test rows
python riskguardian_llm_classifier.py --no-validate  # skip validation, just write submission.csv
python riskguardian_llm_classifier.py --dry-run      # no API calls: print a sample prompt
```

**Compare other providers** (each writes its own `submission.<provider>.<model>.csv` and diffs against `submission.csv`):

```bash
python riskguardian_cerebras_classifier.py
python riskguardian_openai_classifier.py
OPENAI_MODEL=gpt-5.5 python riskguardian_openai_classifier.py    # swap models freely
```

Each script paces requests under the provider's rate limit (`*_MIN_INTERVAL`), backs off on 429s, and — for OpenAI — auto-adapts the per-model API quirks (`max_completion_tokens` vs `max_tokens`, and whether `temperature=0` is allowed).

## Honest notes & limitations

- The label rule for the news half is **source-based** (Reuters ⇒ real), not topical. Our prompt makes this explicit; without it, accuracy drops ~10–15 points.
- The synthetic portion is **deliberately obfuscated**; character n-grams (classical) and large LLMs (GenAI) both see through it, but brittle whole-word methods would not.
- A perfect score on 70 items is a small-sample result. The hold-out validation (100% across all four quadrants, three independent models) is the stronger evidence that the method generalizes within this dataset's structure.
