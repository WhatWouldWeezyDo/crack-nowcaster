<h1 align="center">⚙️ 3-2-1 Crack Spread Nowcaster ⛽📈</h1>

<p align="center">
  <img src="notebooks/figures/equity_curve.png" width="500"/>
</p>

<p align="center">
  <b>Machine Learning model to nowcast the daily direction of refinery margins (3-2-1 crack spread)</b>  
  <br>
  Built with <code>Python · scikit-learn · pandas · yfinance · matplotlib</code>
</p>

---

## 🧩 Abstract

This project explores a **supervised machine learning approach** to predict the *next-day direction* of the refinery crack spread — a key profitability measure for refineries defined as:

<p align="center">
  <b>CRACK<sub>3:2:1</sub> = 2 × RBOB + 1 × ULSD − 3 × WTI</b>
</p>


The objective is to evaluate whether short-term price momentum and relative strength features derived from energy futures contain predictive information about next-day changes in the spread.

---

## 📘 Methodology

### Data
- Source: **Yahoo Finance** continuous front-month futures  
  - `CL=F` (Crude Oil - WTI)  
  - `RB=F` (Gasoline - RBOB)  
  - `HO=F` (Heating Oil - ULSD)
- Period: 2013–2025  
- Frequency: Daily close prices

### Feature Engineering
| Category | Description |
|-----------|-------------|
| Momentum | 5, 10, 20-day % changes for CL, RB, HO |
| Spread dynamics | Crack % change, z-scores |
| Calendar | Day-of-week and month dummies |

### Model
- **Pipeline:** `StandardScaler` → `LogisticRegression(max_iter=1000)`  
- **Label:** 1 if `CRACK(t+1) > CRACK(t)`, else 0  
- **Validation:** Time-based split — Train ≤ 2022-12-31, Test ≥ 2023-01-01  
- **No data leakage, no shuffling.**

---

## 📊 Results (Out-of-sample Test)

| Metric | Value |
|:-------|:------|
| Accuracy | 0.540 |
| Precision | 0.541 |
| Recall | 0.443 |
| AUC | 0.539 |

### ROC Curve
<p align="center">
  <img src="notebooks/figures/roc_curve.png" width="520"/>
</p>

### Diagnostic Equity Curve (No Costs)
<p align="center">
  <img src="notebooks/figures/equity_curve.png" width="520"/>
</p>

> The diagnostic equity curve serves as a signal proxy (no contract rolls or execution costs).  
> It demonstrates weak but consistent directionality — evidence of minor predictive structure.

---

## 🧠 Interpretation

- **Gasoline strength (RB ↑)** → Crack ↑  
- **Crude strength (CL ↑)** → Crack ↓ (feedstock cost squeeze)  
- Crack spread movement captures refinery margin volatility.

Future improvements may include:
- Baseline comparison (momentum-only, naive persistence)
- Feature importance analysis (coefficients & shapley values)
- Walk-forward AUC evaluation
- Integration of **EIA inventory deltas** and volatility indicators

---
