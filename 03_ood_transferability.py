"""
RQ3: Transferability of decision structures to out-of-distribution data.

RF trained on baseline s0 (seed=42) applied WITHOUT retraining to a
held-out OOD sample (seed=999, noise_sigma=6.0, N=1000).

Outputs:
    results/rq3_transferability.csv
    figures/fig7_rq3_shap_corr.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def generate_data(n=5000, noise_sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "gad7_score":         rng.integers(0,22,n),
        "phq4_score":         rng.integers(0,13,n),
        "nstf_score":         rng.integers(0,30,n),
        "brs_score":          rng.integers(0,30,n),
        "age":                rng.integers(18,65,n),
        "gender":             rng.choice(["male","female","other"], n, p=[0.45,0.45,0.10]),
        "education":          rng.choice(["secondary","vocational","higher","scientific","other"], n,
                                          p=[0.15,0.20,0.45,0.10,0.10]),
        "marital_status":     rng.choice(["single","married","divorced","widowed","civil_union"], n,
                                          p=[0.30,0.40,0.15,0.05,0.10]),
        "therapy_experience": rng.choice(["yes","no"], n, p=[0.35,0.65]),
        "chronic_disease":    rng.choice(["yes","no"], n, p=[0.25,0.75]),
    })
    risk = (0.4*data["gad7_score"] + 0.3*data["phq4_score"] +
            0.2*data["nstf_score"] - 0.3*data["brs_score"] +
            rng.normal(0, noise_sigma, n))
    q1, q2 = np.quantile(risk, [0.33, 0.66])
    data["risk_group"] = ["low" if s<q1 else ("medium" if s<q2 else "high") for s in risk]
    return data


CAT_COLS = ["gender","education","marital_status","therapy_experience","chronic_disease"]
NUM_COLS = ["gad7_score","phq4_score","nstf_score","brs_score","age"]


def preprocess(X_raw, fit_scaler=None):
    Xe = pd.get_dummies(X_raw, columns=CAT_COLS, drop_first=False)
    if fit_scaler is None:
        fit_scaler = MinMaxScaler()
        Xe[NUM_COLS] = fit_scaler.fit_transform(Xe[NUM_COLS])
    else:
        Xe[NUM_COLS] = fit_scaler.transform(Xe[NUM_COLS])
    return Xe, fit_scaler


def get_shap_imp(explainer, X, high_idx):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.abs(sv[high_idx]).mean(axis=0)
    return np.abs(sv[:,:,high_idx]).mean(axis=0)


# ── Train on baseline s0 ───────────────────────────────────────────────────────

print("Training RF on baseline s0 (seed=42) ...")
df0 = generate_data(seed=42)
X0  = df0.drop(columns=["risk_group"])
y0  = df0["risk_group"]
Xtr0, Xte0, ytr0, yte0 = train_test_split(X0, y0, test_size=0.2, random_state=42, stratify=y0)
X0tr_e, sc0 = preprocess(Xtr0)
X0te_e, _   = preprocess(Xte0, sc0)
X0te_e = X0te_e.reindex(columns=X0tr_e.columns, fill_value=0)

rf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
rf.fit(X0tr_e, ytr0)
acc0 = accuracy_score(yte0, rf.predict(X0te_e))
f1_0 = f1_score(yte0, rf.predict(X0te_e), average="macro")
print(f"  Baseline: accuracy={acc0:.3f}  macro-F1={f1_0:.3f}")

exp0     = shap.TreeExplainer(rf)
high_idx = list(rf.classes_).index("high")
imp_s0   = get_shap_imp(exp0, X0te_e, high_idx)

# ── Apply to OOD sample ────────────────────────────────────────────────────────

print("\nApplying to OOD sample (noise_sigma=6.0, seed=999, N=1000) ...")
df_ood = generate_data(n=1000, noise_sigma=6.0, seed=999)
X_ood  = df_ood.drop(columns=["risk_group"])
y_ood  = df_ood["risk_group"]
X_ood_e, _ = preprocess(X_ood, sc0)
X_ood_e = X_ood_e.reindex(columns=X0tr_e.columns, fill_value=0)

acc_ood = accuracy_score(y_ood, rf.predict(X_ood_e))
f1_ood  = f1_score(y_ood, rf.predict(X_ood_e), average="macro")
print(f"  OOD: accuracy={acc_ood:.3f}  macro-F1={f1_ood:.3f}")

imp_ood = get_shap_imp(exp0, X_ood_e, high_idx)
res     = spearmanr(imp_s0, imp_ood)
corr    = float(res.statistic if hasattr(res,"statistic") else res[0])
print(f"  SHAP rank correlation (s0 vs OOD): {corr:.3f}")

pd.DataFrame([{
    "model":         "RandomForest",
    "eval_set":      "ood_seed999_noise6.0",
    "acc_baseline":  acc0,  "f1_baseline":  f1_0,
    "acc_ood":       acc_ood, "f1_ood":     f1_ood,
    "shap_spearman": corr,
}]).to_csv("results/rq3_transferability.csv", index=False)
print("Saved -> results/rq3_transferability.csv")

# Figure 4.7
fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(["RandomForest"], [corr], color="#55A868", edgecolor="white")
ax.text(0, corr+0.02, f"{corr:.3f}", ha="center", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Spearman rank correlation")
ax.set_title("Figure 4.7 - SHAP rank correlation:\nbaseline s0 vs OOD sample")
ax.axhline(0.6, linestyle="--", color="grey", linewidth=0.8, label="reference 0.6")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig7_rq3_shap_corr.png", dpi=150)
plt.close()
print("Saved -> figures/fig7_rq3_shap_corr.png\n\nRQ3 done.")
