# 3-2-1 Crack Spread Nowcaster ⛽📈

**Goal:** Nowcast *tomorrow’s* direction of the refinery crack spread:
\[
\text{CRACK} = 2 \cdot \text{RBOB} + 1 \cdot \text{ULSD} - 3 \cdot \text{WTI}
\]
using simple, interpretable ML features.

> **Stack:** Python · pandas · scikit-learn · yfinance · matplotlib  
> **Data:** Yahoo Finance continuous front-month futures (`CL=F`, `RB=F`, `HO=F`)  
> **ML Type:** Supervised **binary classification** (time-series → tabular engineered features)

---

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Then open and Run All:
# notebooks/01_crack_spread_nowcaster.ipynb
Results (Out-of-sample test ≥ 2023-01-01)
Accuracy: 0.540

Precision: 0.541

Recall: 0.443

AUC: 0.539

ROC (test)
<img src="notebooks/figures/roc_curve.png" width="520"/>

Diagnostic equity (no costs; index proxy)
<img src="notebooks/figures/equity_curve.png" width="520"/>

This is a signal study for directional nowcasting.
It does not model contract rolls, carry, or execution costs.
Target is up/down, not P&L.

Method (short)
Label: 1 if CRACK(t+1) > CRACK(t), else 0.

Features: 5/10/20-day momentum (% change) & z-scores for CL/RB/HO and the crack; weekday & month.

Model: StandardScaler → LogisticRegression(max_iter=1000)

Validation: strict time split (train ≤ 2022-12-31, test ≥ 2023-01-01). No shuffling, no leakage.

Why it matters
Crack spread approximates refinery gross margin. The signals map to economics:

Gasoline strength (RB momentum) ↑ → crack ↑

Crude strength (CL momentum) ↑ → crack ↓ (feedstock cost squeeze)

Repo structure
markdown
Copy code
notebooks/
  01_crack_spread_nowcaster.ipynb
  figures/
    roc_curve.png
    equity_curve.png
  reports/
    metrics.json
requirements.txt
README.md
LICENSE
.gitignore
Roadmap (next small upgrades)
Baselines: majority, 1-day crack momentum, always-long diagnostic equity.

Threshold tuning: choose decision threshold by F1 / Youden; show precision/recall trade-off.

Walk-forward: monthly expanding-window AUC (more realistic).

Interpretability: top coefficients with market intuition.

Ablation: base vs +ratios (RB/CL, HO/CL) vs +volatility features.

(Optional) Add EIA weekly inventory deltas (lagged).

License
This project is licensed under the MIT License (see LICENSE).
MIT allows anyone to use, copy, modify, and redistribute the code (including commercially), as long as they keep the license notice. It also disclaims warranties and liability.

Built by Praabveer — feedback welcome.

yaml
Copy code

> If your images live at `figures/` (repo root) instead of `notebooks/figures/`, just change:
> ```markdown
> <img src="figures/roc_curve.png" ...>
> <img src="figures/equity_curve.png" ...>
> ```
