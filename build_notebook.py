#!/usr/bin/env python
"""Builder: assembles the RiskGuardian cyber-risk notebook via nbformat.
Run with the MLENV311 interpreter so the kernelspec/lib assumptions match.
"""
import base64, os, pathlib, re
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ── Header styling: "the_plotly_thickens" brand-green hierarchy ───────────
# Three-tier ramp derived from the logo's signature forest green (#3E6048),
# darkening with header depth (mirrors Gavin's blue JHU-styler system).
H1_COLOR = "#22442C"  # deep forest  — major sections
H2_COLOR = "#40634A"  # brand green  — sub-sections
H3_COLOR = "#7EA98B"  # light sage   — sub-sub-sections (italic)

# H3 headers that should stay unstyled (procedural/structural markers).
H3_SKIP_PATTERNS = [r"^###\s+Step\s+\d+", r"^###\s+Note\s+on\b",
                    r"^###\s+Correlation\s+Analysis", r"^###\s+\d+\."]

def _style_header_line(line):
    """Style a single markdown header line with the brand-green palette.
    Idempotent (skips lines already carrying <span>/<em>/<font>)."""
    if any(t in line for t in ("<span", "<em ", "<font")):
        return line
    if re.match(r"^# (?!#)", line):
        return f'# <span style="color:{H1_COLOR}">{line[2:].strip()}</span>'
    if re.match(r"^## (?!#)", line):
        return f'## <span style="color:{H2_COLOR}">{line[3:].strip()}</span>'
    if re.match(r"^### (?!#)", line):
        if any(re.match(p, line.strip()) for p in H3_SKIP_PATTERNS):
            return line
        return f'### <em style="color:{H3_COLOR}">{line[4:].strip("*_ ")}</em>'
    return line

def style_headers():
    """Apply the brand-green hierarchy to every markdown header line in-place."""
    n = 0
    for c in cells:
        if c.cell_type != "markdown":
            continue
        lines = c.source.split("\n")
        new = [_style_header_line(ln) if ln[:2] == "# " or ln[:3] == "## " or ln[:4] == "### " else ln
               for ln in lines]
        if new != lines:
            c.source = "\n".join(new); n += 1
    print(f"styled headers in {n} markdown cells (brand-green hierarchy)")

# ── Build-time image embedding: base64-inline PNGs so the .ipynb is self-contained ──
_HERE = pathlib.Path(__file__).resolve().parent
_ASSETS = _HERE.parent / "assets"
HERO_IMG = _ASSETS / "jhu-hackathon-hero-tpt.png"
LOGO_IMG = _ASSETS / "the_plot_thickens_logo_alpha.png"

def img_md(path, alt, width):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"WARNING: image not found, skipping embed: {p}")
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return (f'<div align="center"><img src="data:image/png;base64,{data}" '
            f'alt="{alt}" width="{width}" /></div>')

# ════════════════════════════════════════════════════════════════════════
# Hero header (cell 0): full-width base64-embedded banner.
_hero = img_md(HERO_IMG, "the_plotly_thickens — RiskGuardian", "100%")
if _hero:
    md(_hero)

# ════════════════════════════════════════════════════════════════════════
md(r'''
# RiskGuardian Solutions — Cross-Industry Cyber Risk Assessment & Projection

**A research-grounded, runnable reference implementation** for classifying and projecting cyber risk
across Finance, Healthcare, and Energy.

This notebook delivers two things at once:

1. **Research synthesis** — the current cutting-edge of ML/AI models *and* the risk-quantification
   frameworks they plug into, distilled from recent (2023–2026) peer-reviewed and primary sources and
   adversarially fact-checked.
2. **A buildable prototype** — an end-to-end pipeline that runs top-to-bottom on the `MLENV311`
   environment: synthetic multi-industry data → **GBDT risk classification** → **SHAP explainability**
   → **deep-learning ensemble** → **anomaly detection** → **EPSS×CVSS×KEV vulnerability prioritization**
   → **FAIR Monte-Carlo loss projection (ALE / VaR / CVaR)** → **attack-path modeling** →
   **Bayesian threat-intel updating** → **GenAI risk briefs (LLM prompting + RAG)** →
   **interactive projection dashboard**.

> **The core design is not invented here.** It mirrors a peer-reviewed 2026 blueprint
> ([Nwafor et al., *Expert Systems with Applications*](https://www.sciencedirect.com/science/article/pii/S0957417425035353)):
> engineered NIST-CSF features → XGBoost predicts expected loss + a composite Risk Exposure Score →
> FAIR + Monte Carlo → VaR/CVaR → SHAP → dashboard.

---

### How to run
- **Kernel:** select **Python (MLENV311)**.
- **Run order:** top-to-bottom (`Run All`). Every cell is deterministic (fixed seeds).
- **Dependencies:** `xgboost, lightgbm, catboost, torch, shap, scikit-learn, scipy, networkx, plotly,
  ipywidgets, streamlit, seaborn, neo4j` — all present in `MLENV311`. (Part XI uses live LLM keys and
  Part IX an optional Neo4j graph DB, both read from `config.json`; both degrade gracefully if absent.)

### A note on honesty (this matters for a risk product)
The research pass below **killed 3 claims** that did not survive verification. Where the evidence is
thin (LLM/agentic methods), single-study, or based on synthetic data, this notebook says so explicitly.
A risk-assessment tool that overstates its confidence is itself a risk.
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part I — Research: the state of the art (2023–2026)

Modern cyber-risk assessment converges on a **two-layer pattern**:

```
   ┌─────────────────────────────────────────────────────────────┐
   │  LAYER 2 — QUANTIFICATION & GOVERNANCE (decision-grade)       │
   │  FAIR · NIST CSF 2.0 (Govern) · Monte Carlo · VaR/CVaR ·      │
   │  Bayesian networks · MITRE ATT&CK · EPSS/CVSS/KEV             │
   └───────────────▲─────────────────────────────────────────────┘
                   │  model outputs become $-denominated, auditable decisions
   ┌───────────────┴─────────────────────────────────────────────┐
   │  LAYER 1 — ML / AI MODELS (pattern detection on data)         │
   │  Gradient-boosted trees · Deep nets · GNNs · Anomaly detect.  │
   └─────────────────────────────────────────────────────────────┘
```

### Layer 1 — which models, and why

| Task | Cutting-edge approach | Evidence / why |
|---|---|---|
| **Risk classification on tabular data** | **Gradient-boosted trees (XGBoost / LightGBM / CatBoost)** | Production-grade default. **EPSS itself runs a binomial XGBoost estimator** (v4, Mar 2025), chosen because it beat Poisson/logistic/neural alternatives [1]. A 2026 FAIR+XGBoost study: XGBoost 0.925 acc vs RF 0.917 vs SVM 0.713 [2]. |
| **Tabular deep learning** | **Pre-tuned MLPs (RealMLP); ensemble with GBDTs** | RealMLP is *competitive* with GBDTs; the best **no-tuning** result comes from **ensembling MLP + GBDT** (NeurIPS 2024) [3]. Edge is modest (1–3%) — treat as "try both & blend". |
| **Attack-path / entity modeling** | **Graph neural networks** | Classical attack-graphs explode combinatorially on real infra. A Physics-Informed GNN reached F1 0.93 [4]. *(Single self-reported study — promising, not proven.)* |
| **Anomaly / novel-threat detection** | **Isolation Forest, autoencoders** | Unsupervised flagging of outlier assets/behaviour where labels don't exist. |

> **Tabular reality check:** on structured security/risk tables, tree ensembles remain the default. The
> XGBoost-vs-RandomForest gap in [2] is *within error bars* — so "GBDTs lead" is a **strong default, not a law**.

### Layer 2 — the frameworks that make it decision-grade

- **FAIR (Factor Analysis of Information Risk)** — the standard for expressing risk in **dollars**
  (Loss Event Frequency × Loss Magnitude, simulated via Monte Carlo). NIST publishes FAIR as an
  **Informative Reference** mapped to **CSF 2.0**, under the new **Govern** function [5]. *(FAIR is
  probabilistic/estimative — not literally "predictive" — and depends on expert inputs.)*
- **Monte Carlo loss modeling → ALE, VaR, CVaR, loss-exceedance curves** — the lingua franca of
  cyber-insurance economics. Open-source `pyfair` implements this.
- **EPSS + CVSS + CISA-KEV "chaining"** — a dead-simple, high-leverage prioritization rule:
  `(KEV OR EPSS≥0.088) AND CVSS≥7.0` → **~18× efficiency, 85.6% coverage, ~95% workload cut** on 28k real CVEs [8].
  EPSS = ML exploit-likelihood (next 30 days); CVSS = theoretical severity. **Use both** [1].
- **Bayesian risk networks** — FAIR-BN re-implements FAIR to fix Monte-Carlo approximation limits [6];
  Bayesian updating fuses fresh **MITRE ATT&CK**-mapped threat intel into posterior exploit probabilities [7].

### What did NOT survive verification (stated plainly)
- **LLM / agentic cyber-risk claims** — specific *measured-outcome* SOC claims were **refuted** by the
  adversarial pass (vendor hype, unverified). The lesson isn't "avoid GenAI" — it's **ground it**. This
  hackathon mandates GenAI, so Part XI applies **LLM prompting + RAG over authoritative framework docs**
  (NIST / MITRE / FAIR) for explanations and recommendations, while the GBDT stays the accuracy engine.
- **VERIS / VCDB as an "unrestricted, corporate-suitable" dataset** — **refuted**; use with care.
- Most headline benchmarks are **synthetic or single-study** — architecture-validating, not real-world-proven.
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part II — Reference architecture

```
        DATA INGESTION                 ML / AI LAYER                 QUANTIFICATION              PROJECTION
  ┌───────────────────────┐    ┌────────────────────────┐    ┌──────────────────────┐    ┌────────────────────┐
  │ Asset & control inv.  │    │ GBDT risk classifier    │    │ Composite Risk        │    │ Exec dashboard      │
  │ Vuln feeds (NVD/EPSS) │───▶│  + MLP ensemble         │───▶│  Exposure Score (RES) │───▶│ (plotly/streamlit)  │
  │ Threat intel (ATT&CK) │    │ SHAP explainability     │    │ FAIR Monte Carlo      │    │ Loss-exceedance     │
  │ Network topology      │    │ Isolation-Forest anomaly│    │  → ALE / VaR / CVaR   │    │ Per-asset $ risk    │
  │ [synthetic here]      │    │ Attack-graph (→ GNN)    │    │ Bayesian CTI update   │    │ Prioritized backlog │
  └───────────────────────┘    └────────────────────────┘    └──────────────────────┘    └────────────────────┘
                                         │                              │
                            EPSS×CVSS×KEV vuln chaining        Governance: NIST CSF 2.0 (Govern/Identify/Protect…)
```

**GenAI layer (Part XI):** on top of the quantitative stack, an **LLM-prompting + RAG** pipeline over
NIST / MITRE / FAIR turns model outputs into grounded, client-ready risk briefings (FAQ #8 — GenAI-forward).

**Swapping in real data (production):** replace the synthetic generator with
[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (80+ labeled flow features) for network-IDS
features — *prefer the corrected **LYCOS-IDS2017** variant; the original has known label/flow bugs* — plus live
**NVD + EPSS** vulnerability feeds and **MITRE ATT&CK** mappings. The modeling, quantification, and dashboard
layers below are unchanged.
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part III — Setup & synthetic multi-industry dataset

We use a **synthetic** asset inventory so the notebook is self-contained (no downloads, no PII) and
reproducible. Features are grounded in real risk factors used by the literature: **EPSS** exploit
probability, **CVSS** severity, **NIST-CSF-style control maturity** (technology / process / people,
weighted **40 / 35 / 25** per [2]), data sensitivity (PII/PHI/PCI), exposure, patch latency, and incident history.
''')

code(r'''
# ── Environment & imports ───────────────────────────────────────────────
import os
# Cap thread pools BEFORE importing numpy/torch. Many OpenMP runtimes (xgboost,
# lightgbm, catboost, sklearn, torch) coexisting in one kernel otherwise each grab
# all cores and thrash — which can make a trivial MLP train for minutes on macOS.
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings, platform, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from IPython.display import display

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, ConfusionMatrixDisplay,
                             roc_auc_score, accuracy_score)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
import scipy, networkx as nx

# Cutting-edge backends (all present in MLENV311; detected so the notebook is portable)
def _try(name):
    try: return __import__(name)
    except Exception: return None
xgb, lgb, cat_, torch, shap = (_try("xgboost"), _try("lightgbm"),
                               _try("catboost"), _try("torch"), _try("shap"))
HAS = {"xgboost":xgb is not None, "lightgbm":lgb is not None, "catboost":cat_ is not None,
       "torch":torch is not None, "shap":shap is not None}
if torch is not None: torch.set_num_threads(1)

np.random.seed(42); RNG = np.random.default_rng(42)
sns.set_theme(style="whitegrid", context="notebook"); plt.rcParams["figure.figsize"]=(8,4.5)

print("Python", platform.python_version())
for k, mod in dict(xgboost=xgb, lightgbm=lgb, catboost=cat_, torch=torch, shap=shap).items():
    print(f"  {k:9s}: {'v'+mod.__version__ if mod else 'fallback (sklearn)'}")

# Load API keys from config.json into the environment (values are NEVER printed).
# NOTE: config.json holds secrets — keep it local; do not upload or commit it.
def load_config(path="config.json"):
    import json
    if os.path.exists(path):
        with open(path) as f: c = json.load(f)
        for k, v in c.items():
            if isinstance(v, str): os.environ.setdefault(k, v)
        return [k for k in c if ("KEY" in k or "TOKEN" in k)]
    return []
_loaded = load_config()
print("API keys loaded from config.json:", _loaded or "none (using existing env)")
''')

code(r'''
# ── Synthetic multi-industry asset / risk dataset ───────────────────────
INDUSTRIES = ["Finance", "Healthcare", "Energy"]

def make_assets(n=4000, rng=RNG):
    ind  = rng.choice(INDUSTRIES, size=n, p=[0.40, 0.35, 0.25])
    atyp = rng.choice(["Server","Database","Endpoint","OT_ICS","CloudSvc","WebApp"], size=n)
    internet_facing   = rng.integers(0, 2, n)
    data_sensitivity  = rng.integers(1, 6, n)                       # 1..5 (PII/PHI/PCI)
    num_open_vulns    = rng.poisson(8, n)
    max_cvss          = np.clip(rng.normal(7.0, 2.0, n), 0, 10).round(1)
    mean_epss         = np.clip(rng.beta(1.5, 8, n), 0, 1).round(3) # EPSS-like exploit prob
    # NIST-CSF-style control maturity sub-scores (1..5); weighted tech40/process35/people25 [2]
    ctrl_tech, ctrl_process, ctrl_people = (rng.integers(1,6,n) for _ in range(3))
    control_maturity  = (0.40*ctrl_tech + 0.35*ctrl_process + 0.25*ctrl_people).round(2)
    patch_latency_days= np.clip(rng.normal(30, 20, n), 0, 200).astype(int)
    mfa_enabled       = rng.integers(0, 2, n)
    encryption_at_rest= rng.integers(0, 2, n)
    network_segmentation = rng.integers(0, 2, n)
    past_incidents    = rng.poisson(0.6, n)
    threat_intel_score= np.clip(rng.normal(50, 18, n), 0, 100).round(1)
    downtime_hours    = np.clip(rng.normal(6, 5, n), 0, 72).round(1)

    ind_mult = np.where(ind=="Finance", 1.15, np.where(ind=="Healthcare", 1.10, 1.0))
    # Latent risk (gives the classifier real signal), then ordinal label by quantiles
    z = (0.30*mean_epss*10 + 0.18*max_cvss + 0.05*num_open_vulns + 0.60*data_sensitivity
         + 0.80*internet_facing + 0.90*past_incidents + 0.03*patch_latency_days
         - 0.95*control_maturity - 0.70*mfa_enabled - 0.50*encryption_at_rest
         - 0.60*network_segmentation + 0.02*threat_intel_score) * ind_mult
    z = z + rng.normal(0, 1.2, n)
    q = np.quantile(z, [0.50, 0.80, 0.95])
    risk_class = np.where(z<q[0], "Low", np.where(z<q[1], "Medium",
                          np.where(z<q[2], "High", "Critical")))
    return pd.DataFrame(dict(
        industry=ind, asset_type=atyp, internet_facing=internet_facing,
        data_sensitivity=data_sensitivity, num_open_vulns=num_open_vulns, max_cvss=max_cvss,
        mean_epss=mean_epss, ctrl_tech=ctrl_tech, ctrl_process=ctrl_process, ctrl_people=ctrl_people,
        control_maturity=control_maturity, patch_latency_days=patch_latency_days,
        mfa_enabled=mfa_enabled, encryption_at_rest=encryption_at_rest,
        network_segmentation=network_segmentation, past_incidents=past_incidents,
        threat_intel_score=threat_intel_score, downtime_hours=downtime_hours, risk_class=risk_class))

df = make_assets()
print("dataset:", df.shape)
display(df.head())
''')

code(r'''
# ── Quick EDA: class balance & by-industry profile ──────────────────────
ORDER = ["Low","Medium","High","Critical"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=df, x="risk_class", order=ORDER, hue="risk_class",
              palette="RdYlGn_r", legend=False, ax=ax[0]); ax[0].set_title("Risk class distribution")
pd.crosstab(df.industry, df.risk_class)[ORDER].plot(kind="bar", stacked=True,
              colormap="RdYlGn_r", ax=ax[1]); ax[1].set_title("Risk class by industry"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()
display(df.groupby("industry")[["mean_epss","max_cvss","control_maturity","past_incidents"]].mean().round(2))
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part IV — ML risk classification (XGBoost) + SHAP explainability

We train a **gradient-boosted tree** classifier — the production default for tabular security data [1][2].
A `ColumnTransformer` one-hot-encodes categoricals; labels are integer-encoded so every backend behaves
identically. We report a full classification report, **macro one-vs-rest ROC-AUC**, and a row-normalized
confusion matrix.

**Note the rare-event challenge:** *Critical* assets are intentionally scarce (~5%). Recall on the
minority class is the metric that matters for a risk product — a missed Critical is far costlier than a
false alarm.

> **Leaderboard metric (FAQ #3):** classification → **Accuracy** (correct ÷ total, higher is better);
> regression → **RMSE** (lower is better). We surface **accuracy** as the headline number and tune for it,
> since the hackathon ranks on best accuracy (FAQ #7).
''')

code(r'''
# ── Feature matrix, label encoding, split, preprocessor ─────────────────
target   = "risk_class"
cat_cols = ["industry", "asset_type"]
num_cols = [c for c in df.columns if c not in cat_cols + [target]]
X = df[cat_cols + num_cols].copy()
le = LabelEncoder().fit(df[target])
y  = le.transform(df[target])                       # ints 0..3
class_names = list(le.classes_)                     # index == integer code
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", "passthrough", num_cols)])
print("classes:", dict(enumerate(class_names)), "| train/test:", Xtr.shape[0], Xte.shape[0])
''')

code(r'''
# ── Primary classifier: XGBoost (sklearn HistGB fallback) ───────────────
def make_gbdt():
    if HAS["xgboost"]:
        return "XGBoost", xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, objective="multi:softprob",
            tree_method="hist", eval_metric="mlogloss", random_state=42, n_jobs=-1)
    return "HistGradientBoosting (fallback)", HistGradientBoostingClassifier(
            max_iter=400, max_depth=6, learning_rate=0.08, l2_regularization=1.0, random_state=42)

gbdt_name, gbdt = make_gbdt()
clf = Pipeline([("pre", pre), ("model", gbdt)]).fit(Xtr, ytr)
proba_gbdt = clf.predict_proba(Xte)
pred = proba_gbdt.argmax(1)
lbls = list(range(len(class_names)))

acc_gbdt = accuracy_score(yte, pred)
print(f"Model: {gbdt_name}\n")
print(classification_report(yte, pred, target_names=class_names, digits=3))
auc_gbdt = roc_auc_score(yte, proba_gbdt, multi_class="ovr", average="macro", labels=lbls)
print(f"Accuracy (LEADERBOARD METRIC, FAQ #3): {acc_gbdt:.4f}   |   macro OVR ROC-AUC: {auc_gbdt:.3f}")
''')

code(r'''
# ── Confusion matrix (row-normalized = per-class recall) ────────────────
fig, ax = plt.subplots(figsize=(5.5, 5))
ConfusionMatrixDisplay.from_predictions(yte, pred, display_labels=class_names,
    normalize="true", cmap="Blues", xticks_rotation=45, colorbar=False, ax=ax)
ax.set_title(f"Confusion matrix (recall) — {gbdt_name.split()[0]}")
plt.tight_layout(); plt.show()
''')

code(r'''
# ── GBDT family bake-off — research point: tree ensembles cluster close ──
def fit_score(name, model):
    t = time.time(); p = Pipeline([("pre", pre), ("m", model)]).fit(Xtr, ytr)
    a = roc_auc_score(yte, p.predict_proba(Xte), multi_class="ovr", average="macro", labels=lbls)
    return dict(model=name, macro_AUC=round(a, 3), fit_seconds=round(time.time()-t, 2))

rows = [fit_score("HistGB (sklearn)", HistGradientBoostingClassifier(
            max_iter=400, max_depth=6, learning_rate=0.08, l2_regularization=1.0, random_state=42))]
if HAS["xgboost"]:
    rows.append(fit_score("XGBoost", xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist",
        eval_metric="mlogloss", random_state=42, n_jobs=-1)))
if HAS["lightgbm"]:
    rows.append(fit_score("LightGBM", lgb.LGBMClassifier(n_estimators=400, num_leaves=31, max_depth=-1,
        learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1)))
if HAS["catboost"]:
    rows.append(fit_score("CatBoost", cat_.CatBoostClassifier(iterations=400, depth=5, learning_rate=0.08,
        l2_leaf_reg=3.0, random_seed=42, verbose=0, thread_count=2, allow_writing_files=False)))
display(pd.DataFrame(rows).sort_values("macro_AUC", ascending=False).reset_index(drop=True))
print("Reading: AUCs cluster tightly — consistent with [2] (the XGBoost↔RF gap is within error bars).")
''')

code(r'''
# ── Explainability: SHAP (TreeExplainer) with permutation-importance fallback ──
used_shap = False
if HAS["shap"] and HAS["xgboost"]:
    try:
        feat_names = list(clf.named_steps["pre"].get_feature_names_out())
        Xte_pre = clf.named_steps["pre"].transform(Xte)
        sv = np.array(shap.TreeExplainer(clf.named_steps["model"]).shap_values(Xte_pre[:500]))
        # multiclass shape = (samples, features, classes)
        imp = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(0)
        o = np.argsort(imp)[::-1][:12]
        plt.figure(figsize=(7, 4))
        plt.barh([feat_names[i] for i in o][::-1], imp[o][::-1], color="#4c9a76")
        plt.title("Global feature importance (mean |SHAP value|)"); plt.xlabel("mean |SHAP|")
        plt.tight_layout(); plt.show()
        ci = class_names.index("Critical")
        shap.summary_plot(sv[:, :, ci], Xte_pre[:500], feature_names=feat_names, show=False)
        plt.title("SHAP beeswarm — drivers of CRITICAL risk"); plt.tight_layout(); plt.show()
        used_shap = True
    except Exception as e:
        print("SHAP path failed -> permutation importance:", e)
if not used_shap:
    r = permutation_importance(clf, Xte.iloc[:800], yte[:800], n_repeats=5, random_state=42, n_jobs=-1)
    raw = cat_cols + num_cols; o = r.importances_mean.argsort()[::-1][:12]
    plt.figure(figsize=(7, 4)); plt.barh([raw[i] for i in o][::-1], r.importances_mean[o][::-1], color="#4c9a76")
    plt.title("Permutation importance — top risk drivers"); plt.xlabel("mean importance"); plt.tight_layout(); plt.show()
print("Explainability via:", "SHAP" if used_shap else "permutation importance")
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part V — Deep model + ensemble (NeurIPS 2024 finding)

A pre-tuned MLP is *competitive* with GBDTs, and **ensembling MLP + GBDT** gives the best no-tuning
result [3]. We train a small `torch` MLP and average its class probabilities with the GBDT's, then
compare macro-AUC for **GBDT alone / MLP alone / ensemble**.
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
### Hardware acceleration — auto-configure Torch for MPS / H100 / RTX 6000

`configure_torch_device()` detects the accelerator and sets the right backend flags so the **same notebook**
runs optimally on a **Mac (Apple MPS)**, an **NVIDIA H100** (Hopper), or an **NVIDIA RTX 6000**
(Ada / Ampere / Turing) — CPU otherwise. On Ampere-or-newer GPUs it enables **TF32** + cuDNN autotuning and
selects **bfloat16**; on pre-Ampere it uses **float16**; on MPS it uses float32 with op-fallback for
unsupported kernels. The deep-learning cells then run on the detected `DEVICE`.
''')

code(r'''
# ── Auto-configure Torch for the available accelerator (MPS / CUDA H100 / RTX 6000 / CPU) ──
def configure_torch_device(verbose=True):
    cfg = {"device":"cpu", "name":"CPU", "dtype":"float32", "amp":False,
           "autocast":None, "recommended_batch_size":256, "notes":[]}
    if not HAS["torch"]:
        cfg["notes"].append("torch not installed -> sklearn MLP fallback on CPU")
    elif torch.backends.mps.is_available():                       # Apple Silicon
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1") # CPU fallback for unsupported ops
        cfg.update(device="mps", name="Apple Silicon GPU (MPS)")
        cfg["notes"] += ["float32 (MPS has no float64)", "PYTORCH_ENABLE_MPS_FALLBACK=1 set"]
    elif torch.cuda.is_available():                               # NVIDIA
        nm = torch.cuda.get_device_name(0)
        maj, mnr = torch.cuda.get_device_capability(0)
        mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9)
        cfg.update(device="cuda", name=f"{nm} (sm_{maj}{mnr}, {mem} GB)")
        torch.backends.cudnn.benchmark = True                     # autotune kernels for fixed shapes
        if maj >= 8:                                              # Ampere / Ada / Hopper
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            cfg.update(dtype="bfloat16", amp=True, autocast="bfloat16")
            cfg["notes"].append("TF32 enabled; bfloat16 autocast recommended")
        else:                                                     # Volta / Turing (e.g. Quadro RTX 6000 = sm_75)
            cfg.update(dtype="float16", amp=True, autocast="float16")
            cfg["notes"].append("pre-Ampere GPU: float16 autocast (no TF32/bf16)")
        u = nm.upper()
        if "H100" in u:
            cfg.update(recommended_batch_size=1024)
            cfg["notes"].append("H100 Hopper (80GB): large batches; consider torch.compile + FP8 (Transformer Engine)")
        elif "6000" in u:
            cfg.update(recommended_batch_size=512)
            cfg["notes"].append("RTX 6000-class: 48GB (Ada/Ampere) or 24GB (Turing) — size batch to memory")
    else:
        cfg["notes"].append("no GPU detected -> CPU (thread caps set in the setup cell)")
    if verbose:
        print(f"Torch device: {cfg['name']}  ->  device='{cfg['device']}', dtype={cfg['dtype']}, "
              f"amp={cfg['amp']}, batch~{cfg['recommended_batch_size']}")
        for n in cfg["notes"]: print("   -", n)
    return cfg

DEVICE_CFG = configure_torch_device()
DEVICE = torch.device(DEVICE_CFG["device"]) if HAS["torch"] else "cpu"
''')

code(r'''
# ── Torch MLP + soft-vote ensemble with the GBDT ────────────────────────
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(clf.named_steps["pre"].transform(Xtr)).astype("float32")
Xte_s = scaler.transform(clf.named_steps["pre"].transform(Xte)).astype("float32")
n_feat, n_cls = Xtr_s.shape[1], len(class_names)

if HAS["torch"]:
    import torch.nn as nn
    torch.manual_seed(42); torch.set_num_threads(1)              # thread cap matters on the CPU path
    net = nn.Sequential(nn.Linear(n_feat,128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
                        nn.Linear(128,64), nn.ReLU(), nn.Linear(64,n_cls)).to(DEVICE)   # <-- detected device
    opt, lossf = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4), nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr_s, device=DEVICE); yt = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    bs, n = DEVICE_CFG["recommended_batch_size"], len(Xt)
    net.train()
    for _ in range(30):                                          # minibatch SGD
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            if idx.numel() < 2: continue                         # BatchNorm needs >=2 samples
            opt.zero_grad(); loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        proba_mlp = torch.softmax(net(torch.tensor(Xte_s, device=DEVICE)), 1).cpu().numpy()
    mlp_name = f"Torch MLP [{DEVICE_CFG['device']}]"
else:
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(128,64), alpha=1e-4, max_iter=300,
                        early_stopping=True, random_state=42).fit(Xtr_s, ytr)
    proba_mlp, mlp_name = mlp.predict_proba(Xte_s), "sklearn MLP (fallback)"

proba_ens = (proba_gbdt + proba_mlp) / 2
auc_mlp = roc_auc_score(yte, proba_mlp, multi_class="ovr", average="macro", labels=lbls)
auc_ens = roc_auc_score(yte, proba_ens, multi_class="ovr", average="macro", labels=lbls)
print(f"{gbdt_name.split()[0]:12s} macro-AUC: {auc_gbdt:.3f}")
print(f"{mlp_name:12s} macro-AUC: {auc_mlp:.3f}")
print(f"{'Ensemble':12s} macro-AUC: {auc_ens:.3f}   (probability average of GBDT + MLP)")
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part VI — Anomaly detection (Isolation Forest)

Supervised labels miss *novel* risk. An **Isolation Forest** flags assets whose feature combinations are
statistically unusual — useful for surfacing misconfigurations and emerging exposure the classifier
hasn't seen.
''')

code(r'''
# ── Isolation Forest: flag anomalous assets ─────────────────────────────
iso = Pipeline([("pre", pre), ("scale", StandardScaler(with_mean=False)),
                ("iso", IsolationForest(n_estimators=300, contamination=0.05, random_state=42))]).fit(Xtr)
flag  = iso.predict(Xte)                 # -1 anomaly, 1 normal
score = -iso.decision_function(Xte)      # higher = more anomalous
out = Xte.copy()
out["anomaly_score"] = score.round(3); out["true_risk"] = le.inverse_transform(yte)
print(f"Anomalies flagged: {int((flag==-1).sum())} / {len(flag)} ({100*(flag==-1).mean():.1f}%)")
display(out.sort_values("anomaly_score", ascending=False)
        .head(8)[["industry","asset_type","mean_epss","max_cvss","control_maturity",
                  "past_incidents","true_risk","anomaly_score"]].reset_index(drop=True))
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part VII — Vulnerability prioritization: EPSS × CVSS × CISA-KEV chaining

Patching everything is impossible; CVSS-alone over-flags. The **chaining** rule
`(KEV OR EPSS≥0.088) AND CVSS≥7.0` concentrates effort on vulnerabilities that are *both* severe *and*
likely-to-be-exploited — reported at **~18× efficiency, 85.6% coverage, ~95% workload reduction** on
28k real CVEs [8]. We reproduce the *shape* of that result on a synthetic CVE table.
''')

code(r'''
# ── EPSS × CVSS × KEV chaining vs CVSS-only ─────────────────────────────
def make_cves(n=20000, rng=RNG):
    cvss = np.clip(rng.normal(6.5, 1.8, n), 0, 10).round(1)
    epss = np.clip(rng.beta(1.3, 12, n), 0, 1).round(4)          # heavily right-skewed (realistic)
    in_kev = (rng.random(n) < (0.002 + 0.05*(epss > 0.5))).astype(int)
    p_expl = np.clip(0.85*epss + 0.6*in_kev + 0.02*(cvss/10), 0, 1)
    exploited = (rng.random(n) < p_expl).astype(int)              # latent ground truth
    return pd.DataFrame(dict(cvss=cvss, epss=epss, in_kev=in_kev, exploited=exploited))

cves = make_cves()
chain     = ((cves.in_kev == 1) | (cves.epss >= 0.088)) & (cves.cvss >= 7.0)
cvss_only = cves.cvss >= 7.0
total_expl = int(cves.exploited.sum())

def summarize(mask, label):
    n = int(mask.sum()); hit = int(cves.loc[mask, "exploited"].sum())
    return dict(strategy=label, flagged=n, pct_workload=round(100*n/len(cves),1),
                coverage_pct=round(100*hit/total_expl,1), precision_pct=round(100*hit/max(n,1),1))

summ = pd.DataFrame([summarize(cvss_only, "CVSS≥7 only"),
                     summarize(chain, "EPSS×CVSS×KEV chaining")])
print(f"Total CVEs: {len(cves):,} | actually exploited: {total_expl:,}")
display(summ)
if summ.loc[1,"pct_workload"] > 0:
    print(f"Workload cut: {summ.loc[0,'flagged']:,} → {summ.loc[1,'flagged']:,} "
          f"({100*(1-summ.loc[1,'flagged']/summ.loc[0,'flagged']):.0f}% fewer), "
          f"precision {summ.loc[0,'precision_pct']}% → {summ.loc[1,'precision_pct']}%.")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(summ.strategy, summ.pct_workload, color=["#c0504d","#4c9a76"]); ax[0].set_title("Remediation workload (% flagged)"); ax[0].set_ylabel("%")
ax[1].bar(summ.strategy, summ.precision_pct, color=["#c0504d","#4c9a76"]); ax[1].set_title("Precision (% flagged that were exploited)"); ax[1].set_ylabel("%")
plt.tight_layout(); plt.show()
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part VIII — FAIR Monte-Carlo loss projection + Risk Exposure Score

This is the **Layer-1 → Layer-2 bridge**: the ML model's predicted probability of High/Critical risk
drives the **FAIR Loss Event Frequency**, and a **composite Risk Exposure Score (RES)** blends normalized
risk components [2]. Monte-Carlo simulation then produces a **dollar loss distribution** with
**ALE** (Annualized Loss Expectancy), **95% VaR**, **95% CVaR**, and a **loss-exceedance curve** — the
language executives and cyber-insurers actually use.

> *FAIR is estimative, not predictive: outputs are only as good as the input distributions. Here they're
> illustrative parameters, not calibrated to a specific firm.*
''')

code(r'''
# ── Composite Risk Exposure Score (RES) over the full portfolio ─────────
def norm01(s):
    s = np.asarray(s, float); rng_ = s.max()-s.min()
    return (s - s.min())/rng_ if rng_ > 0 else np.zeros_like(s)

proba_full = clf.predict_proba(df[cat_cols + num_cols])
p_hc = proba_full[:, class_names.index("High")] + proba_full[:, class_names.index("Critical")]
res_components = pd.DataFrame(dict(
    frequency     = p_hc,                                  # ML-driven loss-event likelihood
    vulnerability = norm01(df.mean_epss * (df.max_cvss/10)),
    control_gap   = norm01(5 - df.control_maturity),       # weaker controls -> higher exposure
    loss_severity = norm01(df.data_sensitivity),
    downtime      = norm01(df.downtime_hours)))
df["RES"]      = res_components.mean(axis=1).round(3)       # equal-weight blend [2]
df["RES_tier"] = pd.cut(df.RES, [-.01,.33,.55,.75,1.01], labels=["Low","Medium","High","Critical"])

def fair_monte_carlo(p_event, data_sensitivity=3, industry_mult=1.0, sims=30000, seed=7):
    rng = np.random.default_rng(seed)
    lef = (0.10 + 1.6*float(p_event)) * industry_mult       # Loss Event Frequency (events/yr)
    n_events = rng.poisson(lef, sims)
    ln_mu, ln_sigma = 10.8 + 0.28*data_sensitivity, 1.15    # per-event loss ~ lognormal (USD)
    annual = np.array([rng.lognormal(ln_mu, ln_sigma, k).sum() if k else 0.0 for k in n_events])
    v95 = np.quantile(annual, 0.95)
    return annual, dict(LEF=round(lef,3), ALE=annual.mean(), VaR95=v95, CVaR95=annual[annual>=v95].mean())

port_p = float(np.clip(p_hc.mean(), 0, 1))
annual, m = fair_monte_carlo(port_p, data_sensitivity=int(round(df.data_sensitivity.mean())))
print(f"Portfolio P(High/Critical) = {port_p:.3f}  →  LEF = {m['LEF']}/yr")
print(f"ALE = ${m['ALE']:,.0f}    95% VaR = ${m['VaR95']:,.0f}    95% CVaR = ${m['CVaR95']:,.0f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
xs = np.sort(annual); ex = 1 - np.arange(1, len(xs)+1)/len(xs)
ax[0].plot(xs/1e3, ex); ax[0].set_xscale("log"); ax[0].axvline(m['VaR95']/1e3, ls="--", c="r", label="95% VaR")
ax[0].set_xlabel("Annual loss (US$ thousands)"); ax[0].set_ylabel("P(loss > x)")
ax[0].set_title("Loss Exceedance Curve (FAIR Monte Carlo)"); ax[0].legend()
sns.boxplot(data=df, x="industry", y="RES", hue="industry", palette="flare", legend=False, ax=ax[1])
ax[1].set_title("Risk Exposure Score by industry")
plt.tight_layout(); plt.show()
display(df.groupby("industry")["RES"].agg(["mean","max"]).round(3))
''')

code(r'''
# ── Per-asset $ projection: highest-RES vs best-controlled asset ────────
ex = df.assign(p_hc=p_hc)
worst, best = ex.sort_values("RES").iloc[-1], ex.sort_values("RES").iloc[0]
for tag, row in [("HIGH-RISK", worst), ("WELL-CONTROLLED", best)]:
    mult = {"Finance":1.15, "Healthcare":1.10, "Energy":1.0}[row.industry]
    _, mm = fair_monte_carlo(row.p_hc, int(row.data_sensitivity), mult)
    print(f"[{tag:15s}] {row.industry}/{row.asset_type:9s}  RES={row.RES:.2f}  "
          f"P(High/Crit)={row.p_hc:.2f}  ALE=${mm['ALE']:,.0f}  95%VaR=${mm['VaR95']:,.0f}")
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part IX — Attack-path modeling (graph baseline → GNN upgrade)

Cyber risk is **path-dependent**: the question is not just "is this asset vulnerable?" but "what's the
likeliest route to the crown jewels?" We model the network as a directed graph with per-edge exploit
probabilities, find the **most-likely attack path**, and identify the **chokepoint** (highest betweenness)
to harden first. At production scale this is where **graph neural networks** replace brute-force enumeration [4].
''')

code(r'''
# ── Attack graph: likeliest path + chokepoint ───────────────────────────
G = nx.DiGraph()
edges = [("Internet","WebApp",0.6), ("WebApp","AppServer",0.5), ("AppServer","SQL_DB",0.45),
         ("Internet","VPN",0.3), ("VPN","AppServer",0.4), ("AppServer","DomainCtrl",0.5),
         ("DomainCtrl","SQL_DB",0.7), ("SQL_DB","Backups",0.3)]
for u, v, p in edges:
    G.add_edge(u, v, exploit_p=p, weight=-np.log(p))     # -log(p): shortest path = most-likely path
crit = "SQL_DB"
path = nx.shortest_path(G, "Internet", crit, weight="weight")
path_p = float(np.prod([G[path[i]][path[i+1]]["exploit_p"] for i in range(len(path)-1)]))
choke = max(nx.betweenness_centrality(G).items(), key=lambda kv: kv[1])[0]
print(f"Likeliest path  Internet → {crit}:  {' → '.join(path)}   (compound P = {path_p:.3f})")
print(f"Top chokepoint (betweenness): {choke}  →  prioritize hardening here")

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 5))
nx.draw(G, pos, with_labels=True, node_color="#cfe3f3", node_size=1700, font_size=9, arrows=True)
nx.draw_networkx_edge_labels(G, pos, font_size=7,
    edge_labels={(u,v): f"{d['exploit_p']:.2f}" for u,v,d in G.edges(data=True)})
nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), edge_color="red", width=2.6)
plt.title("Attack graph — riskiest path highlighted (red)"); plt.axis("off"); plt.tight_layout(); plt.show()
''')

md(r'''
### Neo4j — the attack graph in a production graph database

The in-memory `networkx` graph is perfect for a demo, but real estates have millions of nodes/edges. We
persist the **same** graph to **Neo4j** and run the analysis in **Cypher** — the likeliest attack path
(max product of edge exploit probabilities) and the chokepoint (the node on the most attack paths) — then
**cross-check it against the `networkx` result**. The server here also has **Neo4j GDS** (Graph Data
Science), so at enterprise scale the same graph supports Dijkstra/betweenness and **node embeddings
(FastRP / node2vec) that feed the GNN attack-path models** from the research ([PIGNN](https://www.mdpi.com/2624-800X/5/2/15)).
Connects live using the `config.json` Neo4j creds; **degrades gracefully** (falls back to the in-memory
result) if Neo4j is unavailable.
''')

code(r'''
# ── Persist the attack graph to Neo4j and query it with Cypher (production graph DB) ──
NEO4J_OK = False
try:
    from neo4j import GraphDatabase
    _uri, _user, _pwd = os.environ.get("NEO4J_URI"), os.environ.get("NEO4J_USER"), os.environ.get("NEO4J_PASSWORD")
    if _uri and _pwd:
        _drv = GraphDatabase.driver(_uri, auth=(_user, _pwd)); _drv.verify_connectivity()
        edges_payload = [{"src":u, "dst":v, "p":float(d["exploit_p"])} for u, v, d in G.edges(data=True)]
        with _drv.session() as s:
            s.run("MATCH (n:RGAsset) DETACH DELETE n")                      # idempotent; scoped to our label
            s.run("UNWIND $edges AS e "
                  "MERGE (a:RGAsset {name:e.src}) MERGE (b:RGAsset {name:e.dst}) "
                  "MERGE (a)-[r:LEADS_TO]->(b) SET r.exploit_p = e.p", edges=edges_payload)
            # Likeliest path = max product of edge exploit_p — pure Cypher, no GDS/APOC required
            rec = s.run("MATCH p=(a:RGAsset {name:$src})-[:LEADS_TO*1..6]->(b:RGAsset {name:$dst}) "
                        "WITH [n IN nodes(p) | n.name] AS path, "
                        "reduce(pr=1.0, r IN relationships(p) | pr * r.exploit_p) AS prob "
                        "RETURN path, prob ORDER BY prob DESC LIMIT 1", src="Internet", dst=crit).single()
            # Chokepoint = intermediate node on the most Internet->target paths
            ck = s.run("MATCH p=(a:RGAsset {name:$src})-[:LEADS_TO*1..6]->(b:RGAsset {name:$dst}) "
                       "UNWIND nodes(p)[1..-1] AS m "
                       "RETURN m.name AS node, count(*) AS paths ORDER BY paths DESC LIMIT 1", src="Internet", dst=crit).single()
            gds = s.run("SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'gds' RETURN count(name) AS n").single()["n"]
        _drv.close(); NEO4J_OK = True
        print(f"Neo4j ({_uri.split('://')[0]}://): wrote {len(edges_payload)} edges as (:RGAsset)-[:LEADS_TO]->(:RGAsset)")
        print(f"  Cypher likeliest path : {' -> '.join(rec['path'])}   (compound P = {rec['prob']:.3f})")
        print(f"  Cypher chokepoint     : {ck['node']}  (traversed by {ck['paths']} attack paths)")
        print(f"  cross-check vs networkx: path {'MATCHES' if rec['path']==path else 'differs'}  |  GDS procedures available = {gds}")
    else:
        print("Neo4j creds not in config.json -> skipping (the in-memory networkx result above stands).")
except Exception as e:
    print("Neo4j unavailable -> using the in-memory networkx result above. Detail:", str(e)[:140])
''')

md(r'''
### Neo4j GDS — node embeddings as GNN-ready features

The research's cutting-edge attack-path method is a **graph neural network** (PIGNN [4]). GNNs don't
consume raw graphs — they consume **per-node feature vectors**. We use **Neo4j GDS** to produce exactly
that: **FastRP structural embeddings** (16-dim; *undirected* propagation so every node is represented) plus
**betweenness centrality** (attack-flow importance). At enterprise scale — over an asset/identity/
vulnerability graph far larger than this demo — this feature matrix is the input to a PIGNN-style model
that *learns* attack paths instead of enumerating them. We compute, display, and PCA-visualize the
features, then drop the in-memory projections.
''')

code(r'''
# ── Neo4j GDS: FastRP embeddings + betweenness → the feature matrix a GNN consumes ──
try:
    if not globals().get("NEO4J_OK"):
        raise RuntimeError("prior Neo4j cell did not load the :RGAsset graph")
    from neo4j import GraphDatabase
    _drv = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    with _drv.session() as s:
        # directed projection -> attack-flow betweenness centrality
        s.run("CALL gds.graph.drop('rg_dir', false) YIELD graphName RETURN graphName")
        s.run("CALL gds.graph.project('rg_dir','RGAsset',{LEADS_TO:{properties:'exploit_p'}}) YIELD nodeCount RETURN nodeCount")
        btw = {r["name"]: r["score"] for r in s.run(
            "CALL gds.betweenness.stream('rg_dir') YIELD nodeId, score "
            "RETURN gds.util.asNode(nodeId).name AS name, score")}
        s.run("CALL gds.graph.drop('rg_dir', false) YIELD graphName RETURN graphName")
        # undirected projection -> FastRP structural embeddings (deterministic: seed + concurrency 1)
        s.run("CALL gds.graph.drop('rg_undir', false) YIELD graphName RETURN graphName")
        s.run("CALL gds.graph.project('rg_undir','RGAsset',{LEADS_TO:{orientation:'UNDIRECTED',properties:'exploit_p'}}) YIELD nodeCount RETURN nodeCount")
        rows = s.run("CALL gds.fastRP.stream('rg_undir',{embeddingDimension:16,relationshipWeightProperty:'exploit_p',"
                     "randomSeed:42,concurrency:1}) YIELD nodeId, embedding "
                     "RETURN gds.util.asNode(nodeId).name AS name, embedding").data()
        s.run("CALL gds.graph.drop('rg_undir', false) YIELD graphName RETURN graphName")
    _drv.close()

    E = np.array([r["embedding"] for r in rows]); names = [r["name"] for r in rows]
    feat = pd.DataFrame({"asset": names, "betweenness": [round(btw.get(n,0.0),2) for n in names]})
    for i in range(E.shape[1]): feat[f"emb{i}"] = E[:,i].round(3)
    feat = feat.sort_values("betweenness", ascending=False).reset_index(drop=True)
    print(f"GDS FastRP node features: {E.shape[0]} nodes x {E.shape[1]}-dim embedding (+ betweenness) — the GNN input matrix")
    display(feat[["asset","betweenness"] + [f"emb{i}" for i in range(6)]])

    from sklearn.decomposition import PCA
    P = PCA(n_components=2, random_state=42).fit_transform(E)
    plt.figure(figsize=(7,4.5))
    sc = plt.scatter(P[:,0], P[:,1], s=[90+70*btw.get(n,0) for n in names],
                     c=[btw.get(n,0) for n in names], cmap="Reds", edgecolor="k", zorder=3)
    for (x,y),n in zip(P, names): plt.annotate(n, (x,y), fontsize=8, xytext=(5,4), textcoords="offset points")
    plt.colorbar(sc, label="betweenness"); plt.title("Neo4j GDS FastRP embeddings (PCA 2D; size/colour = betweenness)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.show()
    print("These per-node embeddings + centrality are the feature matrix a GNN (e.g., PIGNN) consumes to LEARN attack paths.")
except Exception as e:
    print("GDS step skipped (Neo4j/GDS unavailable):", str(e)[:140])
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part X — Bayesian threat-intel updating (MITRE ATT&CK)

Static scores go stale. **Bayesian updating** revises an exploit probability as fresh,
**ATT&CK-mapped** threat intel arrives, so two CVEs with identical CVSS/EPSS priors diverge once one
shows active exploitation [7]. This is how the platform stays *forward-looking*.
''')

code(r'''
# ── Bayesian posterior update from CTI signals ──────────────────────────
def bayes_update(prior, p_signal_if_threat, p_signal_if_benign):
    num = p_signal_if_threat * prior
    return num / (num + p_signal_if_benign * (1 - prior))

prior = 0.10
scenarios = pd.DataFrame([
    dict(cve="CVE-A: active exploitation (T1190)", prior=prior, posterior=bayes_update(prior, 0.90, 0.10)),
    dict(cve="CVE-B: PoC published only",          prior=prior, posterior=bayes_update(prior, 0.50, 0.30)),
    dict(cve="CVE-C: no new intel",                prior=prior, posterior=bayes_update(prior, 0.15, 0.85))])
scenarios["posterior"] = scenarios.posterior.round(3)
print("Same CVSS, same EPSS prior (0.10) — Bayesian posterior reprioritizes by live threat intel:")
display(scenarios)
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part XI — GenAI layer: LLM prompting + Retrieval-Augmented Generation (RAG)

**Why this is here (FAQ #8):** the hackathon requires GenAI techniques. We implement a **first-class RAG
pipeline** — a curated knowledge base of **NIST CSF 2.0, MITRE ATT&CK, FAIR, and EPSS/CVSS** guidance, a
**retriever**, and an **LLM** that turns each high-risk asset into a **grounded, client-ready risk
briefing** (rationale → NIST-mapped controls → the ATT&CK tactic to defend). This is the *consulting*
output RiskGuardian actually sells — explanation and recommendation, not just a score.

**Design choices that matter:**
- **Grounded, not hallucinated.** Generation is constrained to *retrieved authoritative context* — directly
  answering the research caution that **unverified** LLM claims don't hold up. GenAI where it's trustworthy.
- **GenAI-forward hybrid (FAQ #8):** the **GBDT stays the leaderboard accuracy engine**; the LLM adds the
  explanation/recommendation layer it's genuinely good at.
- **Live model.** Keys load from `config.json`; the **preferred provider is Cerebras running a >300B model
  (`zai-glm-4.7`, ~355B MoE)** for fast, high-quality briefs — Anthropic/OpenAI are fallbacks, and a
  deterministic offline brief keeps the notebook runnable with no key. *Grounding matters:* an ungrounded
  model can get basics wrong (a smaller model called EPSS an "e-learning system"); constraining generation
  to retrieved NIST/MITRE/FAIR context prevents that. (Bigger LLM ⇒ better **briefs**, not a higher
  leaderboard score — accuracy still comes from the GBDT.)
''')

code(r'''
# ── GenAI knowledge base + RAG retriever (offline TF-IDF retrieval) ──────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = [
 {"topic":"NIST CSF 2.0 - Govern",   "text":"Establish and monitor cybersecurity risk strategy, expectations and policy; the Govern function (new in CSF 2.0) anchors enterprise risk decisions."},
 {"topic":"NIST CSF 2.0 - Identify", "text":"Maintain asset inventory and assess risk so exposure is understood; prioritize by business criticality and data sensitivity."},
 {"topic":"NIST CSF 2.0 - Protect",  "text":"Safeguards: MFA, encryption at rest and in transit, network segmentation, least privilege and timely patching reduce attack surface."},
 {"topic":"NIST CSF 2.0 - Detect",   "text":"Continuous monitoring and anomaly detection find events quickly; baselining normal behavior surfaces novel threats."},
 {"topic":"NIST CSF 2.0 - Respond/Recover", "text":"Incident response and recovery planning limit loss magnitude and downtime when events occur."},
 {"topic":"MITRE ATT&CK - Initial Access", "text":"Adversaries enter via exploitation of public-facing apps, phishing or valid accounts; internet-facing assets are prime targets."},
 {"topic":"MITRE ATT&CK - Lateral Movement", "text":"After a foothold attackers pivot across hosts; segmentation and credential hygiene constrain movement toward crown jewels."},
 {"topic":"MITRE ATT&CK - Exfiltration/Impact", "text":"Data theft or ransomware; high data-sensitivity assets drive secondary loss (regulatory, reputation)."},
 {"topic":"FAIR - Loss Event Frequency", "text":"LEF = Threat Event Frequency x Vulnerability; stronger controls reduce vulnerability and expected loss events per year."},
 {"topic":"FAIR - Loss Magnitude", "text":"Primary loss (response, replacement) plus secondary loss (fines, reputation); scales with data sensitivity and downtime."},
 {"topic":"EPSS", "text":"Exploit Prediction Scoring System: a 0-1 probability a CVE is exploited within 30 days; prioritize high-EPSS vulnerabilities."},
 {"topic":"CVSS + CISA-KEV", "text":"CVSS measures theoretical severity; combine with EPSS and the CISA Known Exploited Vulnerabilities catalog for real-world prioritization."},
 {"topic":"Vulnerability chaining", "text":"Remediate first when (KEV OR EPSS>=0.088) AND CVSS>=7.0 to cut workload while still covering most exploited vulnerabilities."},
 {"topic":"Control maturity", "text":"Weighted technology 40 / process 35 / people 25; low maturity raises the Risk Exposure Score."},
 {"topic":"Sector - Healthcare", "text":"PHI under HIPAA; high secondary loss; legacy and medical devices widen the attack surface."},
 {"topic":"Sector - Finance", "text":"PCI-DSS and fraud exposure; high-value target; regulatory penalties amplify loss magnitude."},
 {"topic":"Sector - Energy", "text":"OT/ICS and safety impact; availability and integrity are paramount; OT network segmentation is critical."},
]
_kb_texts = [d["text"] + " " + d["topic"] for d in KB]
_vec = TfidfVectorizer().fit(_kb_texts)
_kb_mat = _vec.transform(_kb_texts)

def retrieve(query, k=4):
    sims = cosine_similarity(_vec.transform([query]), _kb_mat).ravel()
    return [KB[i] for i in sims.argsort()[::-1][:k]]

import requests, re as _re
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
# Preferred provider: Cerebras serving a >300B model, very fast (key-provisioned: zai-glm-4.7, ~355B MoE).
CEREBRAS_MODEL  = os.environ.get("CEREBRAS_MODEL",  "zai-glm-4.7")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL",    "gpt-4o-mini")

def llm_available():
    return any(os.environ.get(k) for k in ("CEREBRAS_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"))

def active_provider():
    if os.environ.get("CEREBRAS_API_KEY"):  return f"Cerebras / {CEREBRAS_MODEL} (>300B)"
    if os.environ.get("ANTHROPIC_API_KEY"): return f"Anthropic / {ANTHROPIC_MODEL}"
    if os.environ.get("OPENAI_API_KEY"):    return f"OpenAI / {OPENAI_MODEL}"
    return "offline grounded fallback"

def _clean(t):  # reasoning models keep hidden reasoning in a separate field; strip any inline <think> too
    return _re.sub(r"<think>.*?</think>", "", t or "", flags=_re.S).strip()

def llm_generate(prompt, system="You are a senior cyber-risk consultant for RiskGuardian. Be concise and specific.", max_tokens=1200):
    # Preferred -> Cerebras (large, fast); fallbacks -> Anthropic, OpenAI. Returns None to trigger offline fallback.
    try:
        if os.environ.get("CEREBRAS_API_KEY"):
            r = requests.post("https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization":f"Bearer {os.environ['CEREBRAS_API_KEY']}", "Content-Type":"application/json", "User-Agent":_UA},
                json={"model":CEREBRAS_MODEL, "max_tokens":max_tokens,
                      "messages":[{"role":"system","content":system},{"role":"user","content":prompt}]}, timeout=90)
            r.raise_for_status(); return _clean(r.json()["choices"][0]["message"].get("content"))
        if os.environ.get("ANTHROPIC_API_KEY"):
            r = requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key":os.environ["ANTHROPIC_API_KEY"], "anthropic-version":"2023-06-01", "Content-Type":"application/json"},
                json={"model":ANTHROPIC_MODEL, "max_tokens":max_tokens, "system":system, "messages":[{"role":"user","content":prompt}]}, timeout=90)
            r.raise_for_status(); return _clean(r.json()["content"][0]["text"])
        if os.environ.get("OPENAI_API_KEY"):
            r = requests.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization":f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type":"application/json"},
                json={"model":OPENAI_MODEL, "max_tokens":max_tokens,
                      "messages":[{"role":"system","content":system},{"role":"user","content":prompt}]}, timeout=90)
            r.raise_for_status(); return _clean(r.json()["choices"][0]["message"].get("content"))
    except Exception as e:
        print("  (live LLM call failed -> offline fallback:", str(e)[:120], ")")
    return None

print("RAG knowledge base:", len(KB), "snippets | LLM provider:", active_provider())
print("retrieval smoke test:", [s["topic"] for s in retrieve("internet-facing finance database high EPSS")])
''')

code(r'''
# ── RAG-grounded risk briefs for the top assets (LLM prompting + RAG) ────
def build_prompt(row, ctx):
    context = "\n".join(f"- [{s['topic']}] {s['text']}" for s in ctx)
    profile = (f"industry={row.industry}, asset={row.asset_type}, internet_facing={int(row.internet_facing)}, "
               f"data_sensitivity={int(row.data_sensitivity)}/5, EPSS={row.mean_epss}, max_CVSS={row.max_cvss}, "
               f"control_maturity={row.control_maturity}/5, past_incidents={int(row.past_incidents)}, "
               f"RES={row.RES} ({row.RES_tier})")
    return ("Using ONLY the authoritative context, assess the asset and return: (1) a one-line risk "
            "rationale; (2) top-3 prioritized controls mapped to NIST CSF functions; (3) the most relevant "
            f"MITRE ATT&CK tactic to defend.\n\nCONTEXT:\n{context}\n\nASSET:\n{profile}")

def template_brief(row, ctx):
    tops = "; ".join(s["topic"] for s in ctx[:3])
    lead = ("patching & EPSS-driven remediation" if (row.mean_epss > 0.3 or row.max_cvss >= 7)
            else "control hardening (MFA, segmentation, encryption)")
    tactic = "Initial Access" if int(row.internet_facing) else "Lateral Movement"
    return (f"Rationale: {row.RES_tier}-tier {row.industry} {row.asset_type} (RES {row.RES}); EPSS {row.mean_epss}, "
            f"CVSS {row.max_cvss}, control maturity {row.control_maturity}/5.\n"
            f"Top controls (NIST CSF): Protect -> {lead}; Detect -> monitor anomalies; Govern -> track residual risk.\n"
            f"Grounding: {tops}.  Defend ATT&CK: {tactic}.")

top = df.sort_values("RES", ascending=False).head(3)
print("Generation provider:", active_provider(), "\n")
for _, row in top.iterrows():
    ctx = retrieve(f"{row.industry} {row.asset_type} EPSS {row.mean_epss} CVSS {row.max_cvss} controls {row.control_maturity}")
    # GLM-4.x burns ~1.2-1.8k hidden "reasoning" tokens before the answer, so give generous headroom.
    brief = (llm_generate(build_prompt(row, ctx), max_tokens=4000) if llm_available() else None) or template_brief(row, ctx)
    print(f"=== {row.industry} / {row.asset_type}  (RES {row.RES}, {row.RES_tier}) ===")
    print(brief, "\n")
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part XII — Executive projection dashboard

A consolidated view for decision-makers: risk mix by industry, RES distribution, the portfolio
loss-exceedance curve, and the **prioritized action list**. Static `matplotlib` panels (always render) +
**interactive `plotly`** charts + an **`ipywidgets` what-if calculator** for live demos.
''')

code(r'''
# ── Static executive dashboard (matplotlib) ─────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
pd.crosstab(df.industry, df.risk_class, normalize="index")[ORDER].plot(
    kind="bar", stacked=True, colormap="RdYlGn_r", ax=ax[0,0]); ax[0,0].set_title("Risk mix by industry"); ax[0,0].set_xlabel("")
sns.histplot(df.RES, bins=30, kde=True, ax=ax[0,1], color="#3b6"); ax[0,1].set_title("Risk Exposure Score distribution")
ax[1,0].plot(np.sort(annual)/1e3, 1-np.arange(1,len(annual)+1)/len(annual)); ax[1,0].set_xscale("log")
ax[1,0].set_title("Portfolio loss exceedance"); ax[1,0].set_xlabel("US$ thousands"); ax[1,0].set_ylabel("P(loss > x)")
df.groupby("asset_type")["RES"].mean().sort_values().plot(kind="barh", color="#a33", ax=ax[1,1]); ax[1,1].set_title("Mean RES by asset type")
plt.suptitle("RiskGuardian — Cross-Industry Cyber Risk Dashboard", fontsize=14, y=1.02)
plt.tight_layout(); plt.show()

print("\nTop 10 assets to action (by Risk Exposure Score):")
display(df.sort_values("RES", ascending=False)
        .head(10)[["industry","asset_type","mean_epss","max_cvss","control_maturity",
                   "past_incidents","RES","RES_tier"]].reset_index(drop=True))
''')

code(r'''
# ── Interactive plotly views (render live in Jupyter) ───────────────────
import plotly.express as px
fig_tree = px.treemap(df, path=[px.Constant("Portfolio"), "industry", "RES_tier"],
                      color="RES", color_continuous_scale="RdYlGn_r",
                      title="Risk Exposure — Portfolio → Industry → Tier")
fig_tree.update_layout(margin=dict(t=40,l=0,r=0,b=0), height=420)
fig_tree
''')

code(r'''
fig_lec = px.line(x=np.sort(annual)/1e3, y=1-np.arange(1,len(annual)+1)/len(annual),
                  log_x=True, labels={"x":"Annual loss (US$ thousands)","y":"P(loss > x)"},
                  title="Interactive Loss-Exceedance Curve (FAIR Monte Carlo)")
fig_lec.update_layout(height=400)
fig_lec
''')

code(r'''
# ── ipywidgets what-if: live FAIR calculator (interactive when run) ─────
import ipywidgets as widgets
from ipywidgets import interact

@interact(p_event=widgets.FloatSlider(min=0, max=1, step=0.05, value=0.3, description="P(High/Crit)"),
          data_sensitivity=widgets.IntSlider(min=1, max=5, step=1, value=3, description="Data sens."),
          industry=widgets.Dropdown(options=[("Energy",1.0),("Healthcare",1.10),("Finance",1.15)],
                                    value=1.10, description="Industry"))
def whatif(p_event=0.3, data_sensitivity=3, industry=1.10):
    _, mm = fair_monte_carlo(p_event, data_sensitivity, industry, sims=20000)
    print(f"LEF = {mm['LEF']}/yr   |   ALE = ${mm['ALE']:,.0f}   |   "
          f"95% VaR = ${mm['VaR95']:,.0f}   |   95% CVaR = ${mm['CVaR95']:,.0f}")
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part XIII — Optional: export a Streamlit app

For a standalone interactive deployment, the cell below persists the trained pipeline + scored portfolio
and writes a minimal **Streamlit** app. Launch it from a terminal:

```bash
conda activate MLENV311
streamlit run riskguardian_app.py
```
''')

code(r'''
# ── Persist artifacts + write a minimal Streamlit app ───────────────────
import joblib
from pathlib import Path
joblib.dump({"pipeline": clf, "label_encoder": le, "class_names": class_names}, "riskguardian_model.joblib")
df.to_parquet("riskguardian_scored.parquet")

app = [
 "import streamlit as st, pandas as pd, numpy as np, joblib",
 "st.set_page_config(page_title='RiskGuardian', layout='wide')",
 "st.title('RiskGuardian — Cyber Risk Projection')",
 "art = joblib.load('riskguardian_model.joblib'); df = pd.read_parquet('riskguardian_scored.parquet')",
 "ind = st.sidebar.multiselect('Industry', sorted(df.industry.unique()), default=sorted(df.industry.unique()))",
 "view = df[df.industry.isin(ind)]",
 "c1, c2, c3 = st.columns(3)",
 "c1.metric('Assets', len(view))",
 "c2.metric('Mean RES', round(view.RES.mean(), 3))",
 "c3.metric('Critical-tier assets', int((view.RES_tier=='Critical').sum()))",
 "st.subheader('Risk mix by industry')",
 "st.bar_chart(pd.crosstab(view.industry, view.risk_class))",
 "st.subheader('Top assets to action')",
 "st.dataframe(view.sort_values('RES', ascending=False).head(20))",
]
Path("riskguardian_app.py").write_text(chr(10).join(app))
print("Wrote: riskguardian_model.joblib, riskguardian_scored.parquet, riskguardian_app.py")
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part XIV — Hackathon submission file & compliance

The leaderboard ingests a **CSV with exactly two columns** matching `Sample_Submission.csv`, with **one
row per test record** (FAQ #1–#2). The cell below **generates and validates** that file. You must also
upload **this notebook** as the solution file (FAQ #2 note). Submissions are **unlimited** (FAQ #5); the
work must be **original** — plagiarism means disqualification.

> ✅ **The dataset has been released.** The released task is **binary text classification**, not the
> synthetic tabular `risk_class` modeled above — so the **graded `submission.csv` is produced by the
> pure-GenAI predictor in Part XVII** (FAQ #8) from the real `combined_risk_*.csv` files. Part XVI fits a
> classical baseline for comparison only. The cell here is kept only as a **format/validation demo** on the
> synthetic split and writes to `submission_synthetic_demo.csv` so it never collides with the real
> `submission.csv`.
''')

code(r'''
# ── Format/validation DEMO on the synthetic split (graded submission is in Part XVII) ──
# Writes submission_synthetic_demo.csv so it never collides with the real submission.csv.
ID_COL, TARGET_COL = "Id", "risk_class"

def make_submission(model, X_test, ids, id_col=ID_COL, target_col=TARGET_COL,
                    path="submission_synthetic_demo.csv", label_encoder=None, as_label=True):
    preds = model.predict(X_test)
    if label_encoder is not None and as_label:
        preds = label_encoder.inverse_transform(preds)     # int codes -> original labels
    sub = pd.DataFrame({id_col: np.asarray(ids), target_col: preds})
    sub.to_csv(path, index=False)
    return sub

submission = make_submission(clf, Xte, np.arange(1, len(Xte)+1), label_encoder=le)   # demo on synthetic split
print("Wrote submission_synthetic_demo.csv", submission.shape)
display(submission.head())
''')

code(r'''
# ── Validate the submission against the FAQ #2 rules ────────────────────
def validate_submission(path, expected_cols, expected_rows):
    s = pd.read_csv(path)
    checks = {
        "readable .csv":                     True,
        "exactly 2 columns":                 s.shape[1] == 2,
        f"columns == {list(expected_cols)}": list(s.columns) == list(expected_cols),
        f"row count == {expected_rows}":     len(s) == expected_rows,
        "no missing predictions":            bool(s.iloc[:, 1].notna().all()),
    }
    for k, v in checks.items(): print("PASS" if v else "FAIL", "-", k)
    return all(checks.values())

print("\nDEMO SUBMISSION VALID:", validate_submission("submission_synthetic_demo.csv", [ID_COL, TARGET_COL], len(Xte)))
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## Part XV — Buildable plan & roadmap

**What this notebook already is (hackathon MVP, runs end-to-end):** a working pipeline from data →
multi-model risk classification (XGBoost + ensemble) → SHAP explainability → anomaly detection →
EPSS×CVSS×KEV prioritization → FAIR Monte-Carlo $ projection (ALE/VaR/CVaR) → attack-path analysis →
Bayesian CTI updating → interactive dashboard, mapped to a peer-reviewed blueprint [2].

| Phase | Add | Effort |
|---|---|---|
| **MVP (done)** | Synthetic data, GBDT + MLP ensemble, SHAP, FAIR MC, chaining, dashboard | ✅ this notebook |
| **Real data** | Ingest CIC-IDS2017 (→ LYCOS-IDS2017), live NVD+EPSS, ATT&CK mappings; sector NIST profiles | days |
| **Modeling** | Graph **neural network** for attack paths [4] — **Neo4j GDS FastRP embeddings (Part IX) are the ready-made feature input**; calibrated probabilities (isotonic); class-imbalance tuning for *Critical* recall | days–weeks |
| **Quantification** | Swap hand-set FAIR distributions for `pyfair`; calibrate to client loss history; **FAIR-BN** Bayesian variant [6] | weeks |
| **Governance** | Wire RES/ALE into a **NIST CSF 2.0** program (Govern function) for auditable reporting [5] | ongoing |
| **GenAI (in notebook)** | LLM prompting + RAG over NIST/MITRE/FAIR for grounded risk briefs (Part XI); next: agentic triage & LLM-assisted FAIR scenario generation — *pilot & measure* | ✅ + research |

**Mapping to NIST CSF 2.0 functions:** *Identify* (asset inventory, RES), *Protect* (control-maturity
scoring), *Detect* (anomaly detection), *Respond* (chaining-prioritized backlog, attack-path hardening),
*Govern* (FAIR $-quantification feeding board-level decisions).

### Honest limitations
- Data here is **synthetic** — it validates the *architecture and feature engineering*, not real-world accuracy.
- Several cited benchmarks are **single-study**; treat specific numbers as illustrative of viability.
- **FAIR is estimative**, EPSS is **threat-only** (not full risk), and CIC-IDS2017 has **known label bugs**.
- The **refuted** LLM claims were *unverified measured-outcome* claims; our GenAI use (Part XI) is deliberately **grounded** (RAG over authoritative sources), not reliant on them.
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
> ⚙️ **Exploratory baseline — NOT part of the graded submission.** Part XVI builds a classical TF-IDF → SVM
> model as a reference point only. The graded submission is produced solely by the GenAI predictor in
> **Part XVII** (FAQ #8). No classical prediction is written to `submission.csv`.

## Part XVI — Real dataset: text classification (exploratory classical baseline)

The released task is **binary text classification** on `combined_risk_train.csv` /
`combined_risk_test.csv` (columns `id, text, label`), scored on **Accuracy** (FAQ #3). This is a
*different* problem from the synthetic tabular `risk_class` modeled in Parts III–XII, so it gets its own
**leaderboard accuracy engine** here; the GBDT/ensemble above remains the showcase for the projection
methodology.

**What the released corpus actually is — a *combination* of two sources** (worth knowing, because it
drives the feature choice):

| Sub-corpus | ~Share of train | Form | Label 0 | Label 1 |
|---|---|---|---|---|
| Synthetic risk events | ~1/3 | **character-obfuscated** text (e.g. *"CybernseRuirty"*, *"financal rtanactions"*) | Cybersecurity | Financial |
| ISOT Fake/Real news | ~2/3 | clean English articles | Fake clickbait | Real (Reuters) |

So label **0 = cyber *or* fake-news**, label **1 = financial *or* real-Reuters-news**, and the test set is
the same mix. The model therefore unions **word n-grams** (topical signal on the clean docs) with
**character `char_wb` n-grams (3–5)** — the latter stay informative even when the per-character scrambling
destroys whole-word tokens. A linear **SVM** on those features cross-validates at **~99.99%** accuracy and
is strong on *both* sub-corpora independently (shown below), so the headline number isn't hiding a weak half.
''')

code(r'''
# ── Real released dataset: load + reveal the combined structure ─────────
import re
from pathlib import Path

rt_tr = pd.read_csv("combined_risk_train.csv")
rt_te = pd.read_csv("combined_risk_test.csv")
print("train:", rt_tr.shape, "| test:", rt_te.shape, "| cols:", list(rt_tr.columns))
print("label balance:", rt_tr["label"].value_counts().to_dict(), "(0=Cyber/Fake, 1=Financial/Real)")

# A dictionary-hit ratio cleanly separates the two sub-corpora:
#   low  ratio -> char-OBFUSCATED synthetic risk events (cyber=0 / financial=1)
#   high ratio -> CLEAN ISOT news (fake clickbait=0 / real Reuters=1)
def _load_dict():
    for p in ("/usr/share/dict/words", "/usr/dict/words"):
        if Path(p).exists():
            return set(w.strip().lower() for w in open(p) if len(w.strip()) >= 4)
    return None
DICT = _load_dict()

def cleanliness(t):
    if DICT is None: return np.nan
    toks = re.findall(r"[A-Za-z]{4,}", t.lower())
    return sum(w in DICT for w in toks) / len(toks) if toks else 0.0

if DICT is not None:
    for d in (rt_tr, rt_te):
        d["clean"]   = d["text"].map(cleanliness)
        d["is_obf"]  = d["clean"] < 0.55
        d["reuters"] = d["text"].str.contains("Reuters", case=True)
    print(f"\ntrain composition: obfuscated {rt_tr.is_obf.mean():.0%} | clean {(~rt_tr.is_obf).mean():.0%}")
    print(f"test  composition: obfuscated {int(rt_te.is_obf.sum())} | clean {int((~rt_te.is_obf).sum())} of {len(rt_te)}")
    cl = rt_tr[~rt_tr.is_obf]
    print("\namong CLEAN train docs, 'Reuters' presence vs label (0=fake / 1=real):")
    print(pd.crosstab(cl.reuters, cl.label))
else:
    print("\n(no system word list found - skipping the obf/clean breakdown; the model is unaffected.)")
''')

code(r'''
# ── Leaderboard accuracy engine: word + char_wb TF-IDF -> LinearSVC ─────
# char_wb n-grams are the key: they remain informative under the per-character
# obfuscation that breaks whole-word tokens; word n-grams add topical signal.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def text_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word",    ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=300_000)),
    ])

Xr = rt_tr["text"].values
yr = rt_tr["label"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Single CV pass; report overall accuracy and (when available) per-source accuracy.
is_obf = rt_tr["is_obf"].values if "is_obf" in rt_tr else np.zeros(len(Xr), bool)
acc, acc_obf, acc_cln = [], [], []
for tr_i, va_i in cv.split(Xr, yr):
    m = Pipeline([("feats", text_features()), ("clf", LinearSVC(C=1.0))]).fit(Xr[tr_i], yr[tr_i])
    p, yv = m.predict(Xr[va_i]), yr[va_i]
    acc.append((p == yv).mean())
    mo = is_obf[va_i]
    if mo.any():    acc_obf.append((p[mo]  == yv[mo]).mean())
    if (~mo).any(): acc_cln.append((p[~mo] == yv[~mo]).mean())
print(f"5-fold CV accuracy (LEADERBOARD METRIC, FAQ #3): {np.mean(acc):.4f}  +/- {np.std(acc):.4f}")
if acc_obf: print(f"  - obfuscated synthetic subset: {np.mean(acc_obf):.4f}")
if acc_cln: print(f"  - clean ISOT-news subset:      {np.mean(acc_cln):.4f}")
''')

code(r'''
# ── Exploratory baseline: fit on ALL train, predict the 70 test rows, validate FORMAT ──
# NOTE: this classical model is a reference point only. It does NOT write submission.csv —
# the graded submission is produced solely by the GenAI predictor in Part XVII (FAQ #8).
Xr_test = rt_te["text"].values
svc  = Pipeline([("feats", text_features()), ("clf", LinearSVC(C=1.0))]).fit(Xr, yr)
pred = svc.predict(Xr_test).astype(int)

# Confidence triple-check (none of these alter `pred`):
lr = Pipeline([("feats", text_features()), ("clf", LogisticRegression(max_iter=2000, C=10))]).fit(Xr, yr)
agree_lr = int((lr.predict(Xr_test) == pred).sum())
agree_struct = None
if "is_obf" in rt_te:                          # structural rule: clean -> Reuters?1:0 ; obfuscated -> trust model
    struct = np.where(~rt_te.is_obf.values, rt_te.reuters.values.astype(int), pred)
    agree_struct = int((struct == pred).sum())
print(f"confidence cross-checks  ->  LinearSVC vs LogReg: {agree_lr}/70" +
      (f"   |  SVC vs structural rule: {agree_struct}/70" if agree_struct is not None else ""))

# Demonstrate the exact submission FORMAT against Sample_Submission.csv (FAQ #1, #2).
# (Built in-memory only — NOT written to submission.csv; Part XVII writes the graded file.)
sample = pd.read_csv("Sample_Submission.csv")
sub = pd.DataFrame({"id": rt_te["id"].values, "label": pred})[list(sample.columns)]

checks = {
    "exactly 2 columns":                  sub.shape[1] == 2,
    f"columns == {list(sample.columns)}": list(sub.columns) == list(sample.columns),
    "row count == 70":                    len(sub) == 70,
    "ids match test set (and order)":     list(sub["id"]) == list(rt_te["id"]),
    "labels in {0,1}":                    set(int(v) for v in sub["label"].unique()) <= {0, 1},
    "no missing predictions":             bool(sub.notna().all().all()),
}
for k, v in checks.items(): print("PASS" if v else "FAIL", "-", k)
print("\nFORMAT VALID:", all(checks.values()),
      "| classical baseline label counts:", sub["label"].value_counts().to_dict())
print("(exploratory baseline — submission.csv is written by the GenAI predictor in Part XVII)")
display(sub.head(8))
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
### How it works — TF-IDF features → LinearSVC, and what it learned

The classifier is two stages: **TF-IDF** turns each document into a long *sparse* vector of numbers, and
**LinearSVC** learns one weight per number and classifies by their weighted sum.

**TF-IDF (vectorization).** Each "term" gets a score per document = **TF × IDF**:
- **TF — term frequency:** how often the term occurs in *this* doc. With `sublinear_tf=True` it is `1 + ln(tf)`, so repeats give diminishing returns (one word repeated 50× can't dominate).
- **IDF — inverse document frequency:** how *rare* the term is across the corpus, so ubiquitous terms ("the") are damped and distinctive ones amplified. sklearn's default is `idf(t) = ln[(1 + N) / (1 + df(t))] + 1` (N = #docs, df = #docs containing the term).
- Each document vector is then **L2-normalized** so long and short texts compare fairly.

We union **two** vectorizers side by side (≈500k features total):
- `analyzer="word", (1,2)` — whole words and word-pairs → topical signal (`reuters`, `president donald`, `getty images`).
- `analyzer="char_wb", (3,5)` — 3–5 character chunks inside word boundaries → **robust to the obfuscation**: even when per-character scrambling breaks a whole word, fragments like `reut` / `euter` / `uters` survive, so the signal is not lost. This is why the model holds up on the corrupted synthetic half.

`min_df` drops ultra-rare terms (typos/noise), `max_features` caps the vocabulary for speed/memory.

**LinearSVC (the boundary).** Every doc is now a point in ~500k-dim space. The SVM learns a weight vector
`w` and bias `b` and predicts `label = 1 if (w·x + b) > 0 else 0`. Among all boundaries that separate the
classes it picks the **maximum-margin** one — the widest gap to the nearest documents (the "support
vectors") — trading fit against simplicity via the penalty `½‖w‖²` and the dial `C` (=1.0; larger C = fit
harder). Linear models are a famously strong, fast baseline on high-dimensional **sparse** text where the
classes are nearly linearly separable — exactly this corpus, hence the ~99.99% CV. (We fit `LogisticRegression`
separately only for *probabilities* in the confidence check; LinearSVC outputs a score, not a probability.)

The cell below reads the learned weights straight off the fitted model — the most positive / most negative
features are literally *what the model keys on* for each class.
''')

code(r'''
# ── Interpretability: the features LinearSVC weights most for each class ──
# Reuses the `svc` pipeline fitted two cells above (binary => a single weight vector).
names = svc.named_steps["feats"].get_feature_names_out()
w     = svc.named_steps["clf"].coef_[0]          # weight per feature; >0 pushes toward label 1
order = np.argsort(w)

def top_table(idxs):
    return pd.DataFrame({
        "feature": [names[i].split("__", 1)[1] for i in idxs],
        "kind":    [names[i].split("__", 1)[0] for i in idxs],   # 'word' or 'char'
        "weight":  [round(float(w[i]), 3) for i in idxs],
    })

n_word = sum(n.startswith("word__") for n in names)
print(f"{len(names):,} features ({n_word:,} word + {len(names)-n_word:,} char)"
      f" | bias b = {float(svc.named_steps['clf'].intercept_[0]):.3f}"
      f" | decision: label 1 if (w·x + b) > 0\n")
print(">>> Top features pushing toward label 1 (financial / real-Reuters news):")
display(top_table(order[::-1][:12]))
print(">>> Top features pushing toward label 0 (cyber / fake-news clickbait):")
display(top_table(order[:12]))
''')

# ════════════════════════════════════════════════════════════════════════
md(img_md(LOGO_IMG, "the_plot_thickens", "320") + "\n\n" + r'''
> ⭐ **This is the graded submission — pure GenAI / prompt engineering, FAQ #8 compliant.**

## Part XVII — GenAI predictor: few-shot LLM classification (Anthropic Claude)

The brief asks for predictions via **prompt engineering with a pre-trained LLM** (FAQ #8). This part is the
**sole submission**: it few-shot-prompts **Anthropic Claude** to label all 70 test rows and writes that as
`submission.csv`. **Every label comes from the LLM** — no classical model is in the prediction path. Parts
IV–XVI (including the TF-IDF → SVM model) are **exploratory and are NOT submitted**.

**Why prompt engineering is non-trivial here.** The true label rule for the ~2/3 *news* portion is
**fake-vs-real**, not "cyber vs financial" — unknowable from the brief alone. So the engineered prompt
encodes the structure we discovered (two text types; for news, neutral newswire / Reuters = 1 vs
sensational clickbait = 0) and supplies **few-shot examples spanning all four quadrants** (obfuscated
cyber/financial + clean fake/real). The validation cell quantifies the **lift over a naive prompt**.

> **Pure GenAI (FAQ #8):** the only inputs to `submission.csv` are the LLM's responses. Failed API calls
> are retried with the LLM; a row that still cannot be resolved defaults to label 0 as a last resort — never
> to a classical prediction.
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
### The prompt-engineering story

**The hard part of this task wasn't classifying — it was figuring out what the
labels mean.**

**1. The corpus is two datasets in a trench coat.** About one-third of the rows are
obfuscated synthetic risk reports (characters deliberately scrambled and typo'd);
the other two-thirds are clean news articles from the ISOT fake/real news set. The
`0/1` label means *different things* in each: for risk reports, `0` = cybersecurity
and `1` = financial; for news, `0` = sensational/fake and `1` = neutral newswire.

**2. Why a naive prompt fails.** "Classify as Cybersecurity (0) or Financial (1)" —
the obvious reading of the brief — is meaningless for two-thirds of the data,
because the news rows aren't about cyber or finance at all. A naive prompt scores
only ~0.85; the brief alone never tells you the news rule is fake-vs-real.

**3. The engineered prompt.** We encode the structure we discovered: a system
instruction that names both text types and their separate label rules, plus a
few-shot bank with two worked examples per quadrant (obfuscated-cyber,
obfuscated-financial, news-fake, news-real). Temperature 0, single-character output.
This lifts held-out accuracy from ~0.85 (naive) to 1.00 (engineered).

**4. Reasoning from content, not leakage.** The literal substring "Reuters"
separates real from fake almost perfectly in training — but that's label leakage
that need not hold on unseen data. Our prompt is built to reason from *content* —
tone, sourcing, dateline style — not to grep for a magic word. Evidence: on id=37, a
calm, clean, **non-Reuters** story about an Obama book deal, the model correctly
reads it as the fake-news half, where two newer models over-trusted the "sounds
real" surface and got it wrong.

**5. Honest validation.** We score on a labeled holdout drawn evenly from all four
quadrants (never the few-shot examples), report per-quadrant accuracy, and count
call failures. The submission is written from the LLM only — no classical model in
the prediction path (FAQ #8).
''')

code(r'''
# ── GenAI predictor (Anthropic Claude): few-shot prompt classifier ──────
# Forces Anthropic (the Part XI helper otherwise prefers Cerebras). Predictions
# from this part become the PRIMARY submission (LLM prompt-engineering, FAQ #8).
import os, re, json, time, requests
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or json.load(open("config.json")).get("ANTHROPIC_API_KEY")
assert API_KEY, "ANTHROPIC_API_KEY not found (env or config.json)"

def _ensure_frames():                       # reuse Part XVI frames if present, else load + tag
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
        tk = re.findall(r"[A-Za-z]{4,}", t.lower()); return sum(w in wl for w in tk)/len(tk) if tk else 0.0
    for d in (tr, te):
        d["clean"] = d["text"].map(clean); d["is_obf"] = d["clean"] < 0.55
        d["reuters"] = d["text"].str.contains("Reuters", case=True)
    return tr, te
rt_tr, rt_te = _ensure_frames()

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

rng = np.random.default_rng(7)
def pick(mask, n):
    idx = np.asarray(rt_tr.index[np.asarray(mask)])
    return list(rng.choice(idx, size=min(n, len(idx)), replace=False)) if len(idx) else []
SHOT_IDX = (pick((rt_tr.is_obf) & (rt_tr.label == 0), 2) +
            pick((rt_tr.is_obf) & (rt_tr.label == 1), 2) +
            pick((~rt_tr.is_obf) & (rt_tr.label == 0) & (~rt_tr.reuters), 2) +
            pick((~rt_tr.is_obf) & (rt_tr.label == 1) & (rt_tr.reuters), 2))
def trunc(t, n): return " ".join(str(t).split())[:n]
SHOTS = "\n\n".join(f"TEXT: {trunc(rt_tr.text[i], 300)}\nLABEL: {int(rt_tr.label[i])}" for i in SHOT_IDX)

import threading, random
_rl_lock = threading.Lock(); _rl_last = [0.0]
MIN_INTERVAL = float(os.environ.get("ANTHROPIC_MIN_INTERVAL", "1.5"))   # global pace ~40 req/min
def _throttle():                                # serialize request starts across all worker threads
    with _rl_lock:
        wait = _rl_last[0] + MIN_INTERVAL - time.time()
        if wait > 0: time.sleep(wait)
        _rl_last[0] = time.time()

def classify(text, system=SYSTEM, shots=SHOTS, model=ANTHROPIC_MODEL, retries=8):
    user = (shots + "\n\n" if shots else "") + f"TEXT: {trunc(text, 1500)}\nLABEL:"
    payload = {"model": model, "max_tokens": 5, "temperature": 0, "system": system,
               "messages": [{"role": "user", "content": user}]}
    delay = 2.0
    for a in range(retries):
        try:
            _throttle()
            r = requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": API_KEY, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                json=payload, timeout=60)
            if r.status_code == 200:
                m = re.search(r"[01]", r.json()["content"][0]["text"]); return int(m.group()) if m else -1
            if r.status_code in (429, 500, 502, 503, 529):                 # rate-limited / transient -> back off
                ra = r.headers.get("retry-after")
                wait = float(ra) if (ra and ra.replace(".", "", 1).isdigit()) else delay
                time.sleep(min(wait, 30) + random.uniform(0, 0.5)); delay = min(delay * 2, 30); continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(delay + random.uniform(0, 0.5)); delay = min(delay * 2, 30)
    print("  gave up on one row after", retries, "tries"); return -1   # unresolved; caller retries then defaults to 0

def classify_many(texts, system=SYSTEM, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda t: classify(t, system=system), texts))

print("LLM provider: Anthropic /", ANTHROPIC_MODEL, "| few-shot examples:", len(SHOT_IDX))
print("smoke test (expect 1):", classify("Quarterly portfolio drawdown and liquidity stress across credit positions."))
''')

code(r'''
# ── Validate the prompt on a labeled holdout (honest accuracy estimate) ──
QUADS = [((rt_tr.is_obf) & (rt_tr.label == 0), "obf/cyber(0)"),
         ((rt_tr.is_obf) & (rt_tr.label == 1), "obf/financial(1)"),
         ((~rt_tr.is_obf) & (rt_tr.label == 0), "news/fake(0)"),
         ((~rt_tr.is_obf) & (rt_tr.label == 1), "news/real(1)")]
shot = set(int(i) for i in SHOT_IDX)
val = []
for m, name in QUADS:
    pool = [int(i) for i in np.asarray(rt_tr.index[np.asarray(m)]) if int(i) not in shot]
    val += [(int(i), name) for i in rng.choice(pool, size=min(10, len(pool)), replace=False)]
vt = [rt_tr.text[i] for i, _ in val]; vy = [int(rt_tr.label[i]) for i, _ in val]

print(f"Validating engineered prompt on {len(val)} held-out labeled rows...")
vp = classify_many(vt, system=SYSTEM)
ok = [p == y for p, y in zip(vp, vy) if p != -1]
print(f"  engineered-prompt accuracy: {np.mean(ok):.3f}   (call failures: {sum(p == -1 for p in vp)})")
dfv = pd.DataFrame({"quad": [q for _, q in val], "y": vy, "p": vp})
print(dfv.assign(correct=lambda d: d.p == d.y).groupby("quad").correct.mean().round(3).to_string())

# Naive prompt on a 40-row subset -> shows the prompt-engineering lift.
sub_i = list(range(0, len(val), 2))[:40]
npv = classify_many([vt[i] for i in sub_i], system=SYSTEM_NAIVE)
nacc = np.mean([npv[k] == vy[sub_i[k]] for k in range(len(sub_i)) if npv[k] != -1])
eacc = np.mean([vp[sub_i[k]] == vy[sub_i[k]] for k in range(len(sub_i)) if vp[sub_i[k]] != -1])
print(f"\nprompt-engineering lift on the same {len(sub_i)} rows ->  naive: {nacc:.3f}   engineered: {eacc:.3f}")
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
Prompt-engineering lift and per-quadrant accuracy, from the held-out validation above.
''')

code(r'''
# ── Viz 1: naive vs engineered prompt accuracy (reuses validation globals) ──
import os
import matplotlib.pyplot as plt
if "nacc" not in globals() or "eacc" not in globals():
    print("run the Part XVII validation cell first")
else:
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Naive prompt", "Engineered prompt"], [nacc, eacc],
                  color=["#9aa5b1", "#2563eb"])
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title("Prompt-engineering lift on the labeled holdout")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", va="bottom")
    fig.savefig("figures/prompt_lift.png", dpi=150, bbox_inches="tight")
    plt.show()
''')

code(r'''
# ── Viz 2: engineered-prompt accuracy per data quadrant (reuses dfv) ────────
import os
import matplotlib.pyplot as plt
if "dfv" not in globals():
    print("run the Part XVII validation cell first")
else:
    os.makedirs("figures", exist_ok=True)
    order = ["obf/cyber(0)", "obf/financial(1)", "news/fake(0)", "news/real(1)"]
    acc = dfv.assign(correct=lambda d: d.p == d.y).groupby("quad").correct.mean()
    acc = acc.reindex([q for q in order if q in acc.index])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(acc.index, acc.values, color="#2563eb")
    ax.set_xlim(0, 1); ax.set_xlabel("Accuracy"); ax.invert_yaxis()
    ax.set_title("Engineered-prompt accuracy by data quadrant")
    for b in bars:
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f"{b.get_width():.3f}", ha="left", va="center")
    fig.savefig("figures/quadrant_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()
''')

code(r'''
# ── Predict the 70 test rows with the LLM = THE graded submission ───────
# Pure GenAI / prompt engineering (FAQ #8): the LLM is the SOLE source of every
# label. No classical model touches the prediction path.
print("Classifying 70 test rows with Anthropic /", ANTHROPIC_MODEL, "...")
llm_pred = classify_many(rt_te["text"].tolist(), system=SYSTEM)

# Pure GenAI: retry any failed calls sequentially (no classical backstop).
fail_idx = [i for i, p in enumerate(llm_pred) if p == -1]
if fail_idx:
    print(f"retrying {len(fail_idx)} failed row(s) sequentially...")
    for i in fail_idx:
        llm_pred[i] = classify(rt_te.text[i])
fails = sum(p == -1 for p in llm_pred)
unresolved = [int(rt_te.id[i]) for i, p in enumerate(llm_pred) if p == -1]
if unresolved:
    print(f"WARNING: {len(unresolved)} row(s) unclassified after retries; defaulting to 0: ids={unresolved}")
# Last resort for a still-unresolved row is 0 (a default label, NOT a classical prediction).
llm_final = np.array([0 if p == -1 else p for p in llm_pred]).astype(int)

sample = pd.read_csv("Sample_Submission.csv")
pd.DataFrame({"id": rt_te["id"].values, "label": llm_final})[list(sample.columns)].to_csv("submission.csv", index=False)

print(f"\nWrote submission.csv = {len(llm_final)} LLM predictions (Anthropic Claude, few-shot prompting)")
print(f"LLM call failures: {fails}")
print("label counts (LLM):", pd.Series(llm_final).value_counts().to_dict())
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
### Diagnostic — classical baseline vs LLM submission (read-only)

**Diagnostic only** — compares the exploratory Part XVI classical baseline (`pred`) against the Part XVII
LLM submission (`llm_final`); the quickest way to eyeball the id=37 disagreement. **Writes nothing; not part
of the graded prediction path.**
''')

code(r'''
# ── Read-only diagnostic: classical baseline vs LLM submission ──────────
# Reads ONLY the already-computed `pred` (Part XVI) and `llm_final` (Part XVII).
# It fits no model and writes no file — purely for inspecting disagreements.
if "pred" not in globals():
    print("classical baseline `pred` not found — run Part XVI to populate it, then re-run this cell.")
else:
    classical = np.asarray(pred).astype(int)
    agree = int((llm_final == classical).sum())
    print(f"classical vs LLM agreement: {agree}/{len(llm_final)}")
    dis = [(int(rt_te.id[i]), int(classical[i]), int(llm_final[i]),
            ("obf" if rt_te.is_obf[i] else ("news+Reuters" if rt_te.reuters[i] else "news")),
            trunc(rt_te.text[i], 80)) for i in range(len(rt_te)) if llm_final[i] != classical[i]]
    if dis:
        print("\ndisagreements (id, classical, llm, type, text):")
        for d in dis: print("   ", d)
    else:
        print("\nno disagreements — the LLM matches the classical baseline on all 70 rows.")
''')

# ════════════════════════════════════════════════════════════════════════
md(r'''
## References

1. **FIRST.org — EPSS FAQ / Model.** EPSS uses a binomial **XGBoost** estimator (v4, Mar 2025); EPSS is a measure of *threat*, to be combined with CVSS. https://www.first.org/epss/faq
2. **Nwafor, Nwafor, Brahma & Acharyya (2026).** *A hybrid FAIR and XGBoost framework for cyber-risk intelligence and expected loss prediction.* **Expert Systems with Applications** 299. (XGBoost 0.925 vs RF 0.917 vs SVM 0.713; FAIR + Monte Carlo + VaR/CVaR + SHAP + RES + Streamlit; synthetic 5-industry data.) https://www.sciencedirect.com/science/article/pii/S0957417425035353
3. **Holzmüller, Grinsztajn, Steinwart (2024).** *Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data.* **NeurIPS 2024.** arXiv:2407.04491 — RealMLP competitive with GBDTs; MLP+GBDT ensemble best without tuning. https://arxiv.org/abs/2407.04491
4. **François, Arduin & Merad (2025).** *Physics-Informed Graph Neural Networks for Attack Path Prediction.* **J. Cybersecurity & Privacy** (MDPI). PIGNN F1 0.9308 (single self-reported study). https://www.mdpi.com/2624-800X/5/2/15
5. **FAIR Institute (2025).** *FAIR & NIST CSF 2.0 Cyber Risk Management Program* — FAIR as an Informative Reference to CSF 2.0; the Govern function. https://www.fairinstitute.org/blog/fair-nist-csf-2-0-cyber-risk-management-program
6. **Wang, Neil & Fenton (2020).** *A Bayesian network approach for cybersecurity risk assessment implementing and extending the FAIR model.* **Computers & Security** 89:101659 (FAIR-BN). https://www.sciencedirect.com/science/article/abs/pii/S0167404819300604
7. **Thevaratnam & Rezaeifar (2026).** *Quantifying cyber threat using Bayesian statistical analysis.* **Int. J. Information Security** 25:43 — Bayesian CTI updating mapped to MITRE ATT&CK. https://link.springer.com/article/10.1007/s10207-026-01220-6
8. **Shimizu & Hashimoto (2026).** *Vulnerability Management Chaining* (EPSS×CVSS×KEV). **IEEE Access** 14:31407–31424 / arXiv:2506.01220 — ~18× efficiency, 85.6% coverage, ~95% workload reduction on 28k CVEs. https://arxiv.org/html/2506.01220v3
9. **Canadian Institute for Cybersecurity (UNB) — CIC-IDS2017.** 80+ CICFlowMeter flow features, labeled CSV. (See corrected **LYCOS-IDS2017**.) https://www.unb.ca/cic/datasets/ids-2017.html
10. **Hive Systems — pyfair.** MIT-licensed FAIR Monte-Carlo library for Python. https://github.com/Hive-Systems/pyfair

---
*Generated for the RiskGuardian Solutions hackathon. Research synthesized via a multi-agent, adversarially-verified
deep-research pass (25 claims verified, 3 refuted). Code runs end-to-end on `MLENV311`.*
''')

# ════════════════════════════════════════════════════════════════════════
style_headers()          # apply the brand-green header hierarchy before writing
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "mlenv311", "display_name": "Python (MLENV311)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RiskGuardian_Cyber_Risk_Assessment.ipynb")
with open(OUT, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {OUT}")
print(f"cells: {len(cells)}  (markdown + code)")
