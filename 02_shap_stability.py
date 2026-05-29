"""
RQ2: Stability of SHAP-based global feature importance rankings
across four variation scenarios (s1-s4) vs baseline s0.

Explainers: TreeExplainer (RF), LinearExplainer (LR).
SVM excluded from SHAP stability analysis due to computational constraints.

Outputs:
    results/shap_rank_correlations.csv
    figures/fig4_shap_stability.png
    figures/fig5_shap_summary.png
    figures/fig6_shap_waterfall.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ── Data generation ────────────────────────────────────────────────────────────

def generate_data(n=5000, noise_sigma=3.0, seed=42,
                  rotation=False, cat_shift=False, overlap=False):
    rng = np.random.default_rng(seed)
    gp  = [0.20,0.70,0.10] if cat_shift else [0.45,0.45,0.10]
    data = pd.DataFrame({
        "gad7_score":         rng.integers(0,22,n),
        "phq4_score":         rng.integers(0,13,n),
        "nstf_score":         rng.integers(0,30,n),
        "brs_score":          rng.integers(0,30,n),
        "age":                rng.integers(18,65,n),
        "gender":             rng.choice(["male","female","other"], n, p=gp),
        "education":          rng.choice(["secondary","vocational","higher","scientific","other"], n,
                                          p=[0.15,0.20,0.45,0.10,0.10]),
        "marital_status":     rng.choice(["single","married","divorced","widowed","civil_union"], n,
                                          p=[0.30,0.40,0.15,0.05,0.10]),
        "therapy_experience": rng.choice(["yes","no"], n, p=[0.35,0.65]),
        "chronic_disease":    rng.choice(["yes","no"], n, p=[0.25,0.75]),
    })
    w = np.array([0.4,0.3,0.2,-0.3])
    if rotation:
        rng2 = np.random.RandomState(seed)
        w = rng2.randn(4); w /= np.linalg.norm(w)
    risk = (w[0]*data["gad7_score"] + w[1]*data["phq4_score"] +
            w[2]*data["nstf_score"] + w[3]*data["brs_score"] +
            rng.normal(0, noise_sigma, n))
    q1, q2 = np.quantile(risk, [0.33, 0.66])
    if overlap:
        dw=q2-q1; q1+=0.15*dw; q2-=0.15*dw
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


def rf_importance(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.mean([np.abs(sv[c]).mean(axis=0) for c in range(len(sv))], axis=0)
    return np.abs(sv).mean(axis=(0,2)) if sv.ndim==3 else np.abs(sv).mean(axis=0)


def lr_importance(explainer, X):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    return np.abs(sv).mean(axis=0)


# ── Base RF for Figures 4.5 and 4.6 ──────────────────────────────────────────

print("Training base RF (s0, seed=42) ...")
df0 = generate_data(seed=42)
X0  = df0.drop(columns=["risk_group"])
y0  = df0["risk_group"]
Xtr0, Xte0, ytr0, yte0 = train_test_split(X0, y0, test_size=0.2, random_state=42, stratify=y0)
X0tr_e, sc0 = preprocess(Xtr0)
X0te_e, _   = preprocess(Xte0, sc0)
X0te_e = X0te_e.reindex(columns=X0tr_e.columns, fill_value=0)

rf0 = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
rf0.fit(X0tr_e, ytr0)
exp0     = shap.TreeExplainer(rf0)
high_idx = list(rf0.classes_).index("high")

sv0      = exp0.shap_values(X0te_e)
sv0_high = sv0[high_idx] if isinstance(sv0,list) else sv0[:,:,high_idx]
imp_base = np.abs(sv0_high).mean(axis=0)

# Figure 4.5 — global SHAP summary
plt.figure(figsize=(8, 5))
shap.summary_plot(sv0_high, X0te_e, show=False, max_display=12)
plt.title("Figure 4.5 - Global SHAP contributions (RandomForest, class 'high')")
plt.tight_layout()
plt.savefig("figures/fig5_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> figures/fig5_shap_summary.png")

# Figure 4.6 — local waterfall
sidx = np.where(yte0.values == "high")[0][0]
eo   = exp0(X0te_e.iloc[[sidx]])
if eo.values.ndim == 3:
    vals = eo.values[0,:,high_idx]; base = float(eo.base_values[0,high_idx])
else:
    vals = eo.values[0];            base = float(eo.base_values[0])
shap.plots.waterfall(shap.Explanation(values=vals, base_values=base,
    data=eo.data[0], feature_names=eo.feature_names), show=False)
plt.title("Figure 4.6 - Local SHAP explanation for a high-risk observation")
plt.tight_layout()
plt.savefig("figures/fig6_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> figures/fig6_shap_waterfall.png")

# ── RQ2: SHAP stability across scenarios ──────────────────────────────────────

SCENARIOS = {
    "s0": {},
    "s1": {"noise_sigma": 6.0},
    "s2": {"rotation": True},
    "s3": {"cat_shift": True},
    "s4": {"overlap": True},
}
N_REPEATS, N_SAMPLES = 10, 2000
all_imp = []

print("\nComputing SHAP stability (RF + LR, R=10, N=2000) ...")
for sc_name, sc_kw in SCENARIOS.items():
    print(f"  {sc_name} ...", end=" ", flush=True)
    for seed in range(N_REPEATS):
        df_s = generate_data(n=N_SAMPLES, seed=100+seed, **sc_kw)
        X_s  = df_s.drop(columns=["risk_group"])
        y_s  = df_s["risk_group"]
        Xtr, Xte, ytr, _ = train_test_split(X_s, y_s, test_size=0.2, random_state=seed, stratify=y_s)
        Xtr_e, sc_ = preprocess(Xtr)
        Xte_e, _   = preprocess(Xte, sc_)
        Xte_e = Xte_e.reindex(columns=Xtr_e.columns, fill_value=0)

        rf_ = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
        rf_.fit(Xtr_e, ytr)
        imp_rf = rf_importance(shap.TreeExplainer(rf_), Xte_e)
        all_imp.append({"scenario":sc_name,"seed":seed,"model":"RandomForest","imp":imp_rf})

        lr_ = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr_.fit(Xtr_e, ytr)
        exp_lr = shap.LinearExplainer(lr_, Xtr_e, feature_perturbation="interventional")
        imp_lr = lr_importance(exp_lr, Xte_e)
        all_imp.append({"scenario":sc_name,"seed":seed,"model":"LogisticRegression","imp":imp_lr})
    print("done")

records = []
for entry in all_imp:
    if entry["scenario"] == "s0":
        continue
    s0 = next((e for e in all_imp
               if e["scenario"]=="s0" and e["seed"]==entry["seed"]
               and e["model"]==entry["model"]), None)
    if s0 is None:
        continue
    res  = spearmanr(np.array(s0["imp"]).ravel(), np.array(entry["imp"]).ravel())
    corr = float(res.statistic if hasattr(res,"statistic") else res[0])
    records.append({"scenario":entry["scenario"],"seed":entry["seed"],
                    "model":entry["model"],"spearman":corr})

df_corr = pd.DataFrame(records)
df_corr.to_csv("results/shap_rank_correlations.csv", index=False)
print("\nSaved -> results/shap_rank_correlations.csv\nSpearman correlations vs s0:")
for m in ["LogisticRegression","RandomForest"]:
    for sc in ["s1","s2","s3","s4"]:
        v = df_corr[(df_corr["model"]==m)&(df_corr["scenario"]==sc)]["spearman"]
        print(f"  {m} | {sc}: {v.mean():.3f} +- {v.std():.3f}")

# Figure 4.4 — stability boxplot
SC_ORDER    = ["s1","s2","s3","s4"]
MODEL_ORDER = ["LogisticRegression","RandomForest"]
COLORS_M    = ["#4C72B0","#55A868"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, mname, color in zip(axes, MODEL_ORDER, COLORS_M):
    sub = df_corr[df_corr["model"]==mname]
    bp  = ax.boxplot([sub[sub["scenario"]==s]["spearman"].values for s in SC_ORDER],
                     labels=SC_ORDER, patch_artist=True,
                     medianprops={"color":"black","linewidth":2})
    for patch in bp["boxes"]:
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_title(mname, fontsize=10)
    ax.set_xlabel("Scenario")
    ax.axhline(0, linestyle="--", color="grey", linewidth=0.8)
axes[0].set_ylabel("Spearman rank correlation vs s0")
fig.suptitle("Figure 4.4 - SHAP stability across variation scenarios")
plt.tight_layout()
plt.savefig("figures/fig4_shap_stability.png", dpi=150)
plt.close()
print("Saved -> figures/fig4_shap_stability.png\n\nRQ2 done.")
