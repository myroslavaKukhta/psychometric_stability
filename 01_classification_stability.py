"""
RQ1: Stability of classification accuracy and macro-F1
across 30 independent runs on synthetic psychometric data.

Outputs:
    results/metrics_rq1.csv
    figures/fig1_macroF1_bar.png
    figures/fig2_macroF1_boxplot.png
    figures/fig3_confusion_matrices.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ── Data generation ────────────────────────────────────────────────────────────

def generate_data(n=5000, noise_sigma=3.0, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "gad7_score":         rng.integers(0, 22, n),
        "phq4_score":         rng.integers(0, 13, n),
        "nstf_score":         rng.integers(0, 30, n),
        "brs_score":          rng.integers(0, 30, n),
        "age":                rng.integers(18, 65, n),
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


CAT_COLS    = ["gender","education","marital_status","therapy_experience","chronic_disease"]
NUM_COLS    = ["gad7_score","phq4_score","nstf_score","brs_score","age"]
LABEL_ORD   = ["low","medium","high"]
MODEL_ORDER = ["LogisticRegression","SVM","RandomForest"]
COLORS      = ["#4C72B0","#DD8452","#55A868"]


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "SVM":                SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "RandomForest":       RandomForestClassifier(n_estimators=100, max_depth=6,
                                                      class_weight="balanced", random_state=42),
    }

# ── 30 independent runs ────────────────────────────────────────────────────────

records, last_cms = [], {}
print("Running 30 independent repetitions ...")
for seed in range(30):
    data  = generate_data(seed=100+seed)
    X_raw = data.drop(columns=["risk_group"])
    y     = data["risk_group"]
    X_enc = pd.get_dummies(X_raw, columns=CAT_COLS, drop_first=False)
    Xtr, Xte, ytr, yte = train_test_split(X_enc, y, test_size=0.2, random_state=seed, stratify=y)
    sc = MinMaxScaler()
    Xtr[NUM_COLS] = sc.fit_transform(Xtr[NUM_COLS])
    Xte[NUM_COLS] = sc.transform(Xte[NUM_COLS])
    for mname, model in get_models().items():
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        records.append({"seed":seed, "model":mname,
                        "accuracy":accuracy_score(yte,yp),
                        "macro_f1":f1_score(yte,yp,average="macro")})
        if seed == 29:
            last_cms[mname] = (model, Xte, yte)
    if (seed+1) % 10 == 0:
        print(f"  {seed+1}/30 done")

df = pd.DataFrame(records)
df.to_csv("results/metrics_rq1.csv", index=False)
print("Saved -> results/metrics_rq1.csv\n\nMean macro-F1 (mean +- 95% CI):")
for m in MODEL_ORDER:
    v = df[df["model"]==m]["macro_f1"]
    print(f"  {m}: {v.mean():.3f} +- {1.96*v.std()/30**0.5:.3f}")

# ── Figure 4.1 — bar chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
means = [df[df["model"]==m]["macro_f1"].mean() for m in MODEL_ORDER]
cis   = [1.96*df[df["model"]==m]["macro_f1"].std()/30**0.5 for m in MODEL_ORDER]
bars  = ax.bar(MODEL_ORDER, means, yerr=cis, capsize=5, color=COLORS, edgecolor="white")
for bar, v in zip(bars, means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(0.5, 0.75)
ax.set_ylabel("Macro-F1 (mean +- 95% CI)")
ax.set_title("Figure 4.1 - Macro-F1 mean values and 95% confidence intervals")
plt.tight_layout(); plt.savefig("figures/fig1_macroF1_bar.png", dpi=150); plt.close()
print("Saved -> figures/fig1_macroF1_bar.png")

# ── Figure 4.2 — boxplot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
bp = ax.boxplot([df[df["model"]==m]["macro_f1"].values for m in MODEL_ORDER],
                labels=MODEL_ORDER, patch_artist=True,
                medianprops={"color":"black","linewidth":2})
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_ylabel("Macro-F1")
ax.set_title("Figure 4.2 - Macro-F1 stability over 30 independent runs")
plt.tight_layout(); plt.savefig("figures/fig2_macroF1_boxplot.png", dpi=150); plt.close()
print("Saved -> figures/fig2_macroF1_boxplot.png")

# ── Figure 4.3 — confusion matrices ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, mname in zip(axes, MODEL_ORDER):
    model, Xte, yte = last_cms[mname]
    ConfusionMatrixDisplay.from_estimator(model, Xte, yte,
        display_labels=LABEL_ORD, cmap="Blues", values_format="d",
        ax=ax, colorbar=False)
    ax.set_title(mname, fontsize=10)
fig.suptitle("Figure 4.3 - Confusion matrices for three models", y=1.02)
plt.tight_layout()
plt.savefig("figures/fig3_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> figures/fig3_confusion_matrices.png\n\nRQ1 done.")
