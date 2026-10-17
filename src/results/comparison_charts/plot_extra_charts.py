"""Additional charts for MFFV model comparison.

Draws:
  1. Training curves: val_eer per epoch for all 5 train.py runs
  2. Training curves: val_auc per epoch for all 5 train.py runs
  3. Training curves: train_loss per epoch for all 5 train.py runs
  4. Pareto frontier: EER vs #params with dominant front highlighted
  5. Normalized scorecard (radar chart) across 5 metrics

Run: python src/results/comparison_charts/plot_extra_charts.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "results" / "comparison_charts"
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# train.py runs have "history" with val_eer / val_auc / train_loss per epoch
runs = [
    ("ResNet18+ArcFace (50ep, expD)",
     load(RESULTS / "baseline_resnet18_arcface_expD" / "results.json"),
     "#1f77b4"),
    ("ResNet18+ArcFace (30ep)",
     load(RESULTS / "baseline_resnet18_arcface_30ep" / "results.json"),
     "#5fa8d3"),
    ("FDT (30ep)",
     load(RESULTS / "fdt_30ep_arcface" / "fdt" / "results.json"),
     "#d62728"),
    ("GLA-FD (quick, 2ep)",
     load(RESULTS / "glafd_quick" / "gla-fd" / "results.json"),
     "#9467bd"),
    ("ILCNN (10ep)",
     load(RESULTS / "ilcnn_10ep" / "ilcnn" / "results.json"),
     "#2ca02c"),
]

# ResNet18 v1 uses different log schema (training_log.json with 5-epoch stride)
v1_log = load(RESULTS / "fvusm_resnet18_arcface_v1" / "training_log.json")
v1_eval = load(RESULTS / "fvusm_resnet18_arcface_v1" / "eval_results.json")


# -----------------------------------------------------------------
# 1) Val EER curves
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
for name, data, color in runs:
    epochs = [h["epoch"] for h in data["history"]]
    eers = [h["val_eer"] * 100 for h in data["history"]]
    ax.plot(epochs, eers, marker="o", markersize=4, linewidth=1.8,
            color=color, label=name)

# v1 (5-epoch stride)
v1_epochs = [h["epoch"] for h in v1_log]
v1_eers = [h["eer"] * 100 for h in v1_log]
ax.plot(v1_epochs, v1_eers, marker="s", markersize=7, linewidth=2.2,
        color="#ff7f0e", label="ResNet18+ArcFace v1 (train_fvusm)")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation EER (%) - lower is better", fontsize=12)
ax.set_title("Training curves: Validation EER per epoch", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
ax.set_yscale("symlog", linthresh=0.5)

plt.tight_layout()
out_eer_curve = OUT / "compare_04_val_eer_curves.png"
fig.savefig(out_eer_curve, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_eer_curve}")


# -----------------------------------------------------------------
# 2) Val AUC curves
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
for name, data, color in runs:
    epochs = [h["epoch"] for h in data["history"]]
    aucs = [h.get("val_auc") for h in data["history"]]
    # skip runs that did not log val_auc
    pts = [(e, a) for e, a in zip(epochs, aucs) if a is not None]
    if not pts:
        continue
    es, ass = zip(*pts)
    ax.plot(es, ass, marker="o", markersize=4, linewidth=1.8,
            color=color, label=name)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation AUC - higher is better", fontsize=12)
ax.set_title("Training curves: Validation AUC per epoch", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right", fontsize=9, ncol=2, framealpha=0.9)
ax.set_ylim(0.75, 1.0)

plt.tight_layout()
out_auc_curve = OUT / "compare_05_val_auc_curves.png"
fig.savefig(out_auc_curve, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_auc_curve}")


# -----------------------------------------------------------------
# 3) Train loss curves
# -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
for name, data, color in runs:
    epochs = [h["epoch"] for h in data["history"]]
    losses = [h.get("train_loss") for h in data["history"]]
    pts = [(e, l) for e, l in zip(epochs, losses) if l is not None]
    if not pts:
        continue
    es, ls = zip(*pts)
    ax.plot(es, ls, marker="o", markersize=4, linewidth=1.8,
            color=color, label=name)

# v1 loss
v1_loss_epochs = [h["epoch"] for h in v1_log]
v1_losses = [h["loss"] for h in v1_log]
ax.plot(v1_loss_epochs, v1_losses, marker="s", markersize=7, linewidth=2.2,
        color="#ff7f0e", label="ResNet18+ArcFace v1 (train_fvusm)")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Training Loss (ArcFace)", fontsize=12)
ax.set_title("Training curves: Train Loss per epoch", fontsize=14, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
ax.set_yscale("log")

plt.tight_layout()
out_loss_curve = OUT / "compare_06_train_loss_curves.png"
fig.savefig(out_loss_curve, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_loss_curve}")


# -----------------------------------------------------------------
# 4) Pareto frontier (EER vs #params)
# -----------------------------------------------------------------
all_models = [
    ("ResNet18+ArcFace (50ep, expD)", 0.0104, 11.28, "#1f77b4"),
    ("ResNet18+ArcFace (30ep)", 0.0200, 11.28, "#5fa8d3"),
    ("ResNet18+ArcFace v1", v1_eval["eer_percent"] / 100, 11.28, "#ff7f0e"),
    ("FDT (30ep)", 0.1760, 5.49, "#d62728"),
    ("GLA-FD (quick)", 0.0920, 7.59, "#9467bd"),
    ("ILCNN (10ep)", 0.1120, 2.95, "#2ca02c"),
]

fig, ax = plt.subplots(figsize=(11, 7))
labels = [m[0] for m in all_models]
xs = [m[1] for m in all_models]
ys = [m[2] for m in all_models]
colors = [m[3] for m in all_models]

ax.scatter(xs, ys, s=200, c=colors, edgecolor="black", linewidth=0.7, zorder=4)

# Pareto frontier: a point is Pareto-optimal if no other point has both
# smaller params AND smaller EER. EER plotted on log-y already, so compare
# in linear units.
pareto = []
for x, y, lab, c in all_models:
    dominated = any((px <= x and py <= y and (px < x or py < y))
                    for px, py, _, _ in all_models)
    if not dominated:
        pareto.append((x, y, lab, c))
pareto.sort()
px = [p[0] for p in pareto]
py = [p[1] * 100 for p in pareto]  # convert to % for plotting
ax.plot(px, py, "--", color="gray", linewidth=1.5, alpha=0.7, zorder=2,
        label="Pareto frontier")

# highlight pareto points
for x, y, lab, c in pareto:
    ax.scatter(x, y * 100, s=300, facecolor="none", edgecolor=c, linewidth=2.2, zorder=5)

for x, y, lab, c in zip(xs, ys, labels, colors):
    ax.annotate(lab, (x, y * 100), xytext=(9, 6), textcoords="offset points",
                fontsize=8.8, alpha=0.9)

ax.set_xscale("log")
ax.set_xlabel("#Parameters (M) - log scale", fontsize=12)
ax.set_ylabel("EER (%) - lower is better", fontsize=12)
ax.set_title("Pareto Frontier: EER vs Model Size", fontsize=14, fontweight="bold")
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.invert_yaxis()
ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

plt.tight_layout()
out_pareto = OUT / "compare_07_pareto_frontier.png"
fig.savefig(out_pareto, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_pareto}")


# -----------------------------------------------------------------
# 5) Radar chart - normalized scorecard
# -----------------------------------------------------------------
metrics = ["EER", "AUC", "Speed\n(1/epoch, s)", "Params\n(inv M)", "Test\nPairs"]
# Best per metric in [0,1] where 1 = best
data_for_radar = []

# we need: val_eer (lower better), val_auc (higher better), epoch time (lower better),
# params (lower better), test pairs (higher better)
def get_train_time_sec(d):
    # average epoch time from history
    times = [h.get("time_sec", 0) for h in d.get("history", []) if "time_sec" in h]
    return float(np.mean(times)) if times else float("nan")


model_records = [
    ("ResNet18+ArcFace (50ep, expD)", 0.0104, 0.99896,
     get_train_time_sec(runs[0][1]), 11.28, 5000),
    ("ResNet18+ArcFace (30ep)", 0.0200, 0.99779,
     get_train_time_sec(runs[1][1]), 11.28, 2000),
    ("ResNet18+ArcFace v1", v1_eval["eer_percent"] / 100, 1.0,
     float(np.mean([h.get("elapsed_s", 0) for h in v1_log])), 11.28,
     v1_eval["n_test_images"]),
    ("FDT (30ep)", 0.1760, 0.90062,
     get_train_time_sec(runs[2][1]), 5.49, 2000),
    ("GLA-FD (quick)", 0.0920, 0.97718,
     get_train_time_sec(runs[3][1]), 7.59, 500),
    ("ILCNN (10ep)", 0.1120, 0.95436,
     get_train_time_sec(runs[4][1]), 2.95, 500),
]

# Build score per metric, in [0,1] where higher=better
all_eer = [r[1] for r in model_records]
all_auc = [r[2] for r in model_records]
all_time = [r[3] for r in model_records]
all_params = [r[4] for r in model_records]
all_pairs = [r[5] for r in model_records]

eer_min, eer_max = min(all_eer), max(all_eer)
auc_min, auc_max = min(all_auc), max(all_auc)
time_min, time_max = min(all_time), max(all_time)
params_min, params_max = min(all_params), max(all_params)
pairs_min, pairs_max = min(all_pairs), max(all_pairs)


def norm(v, lo, hi, higher_better=True):
    if hi == lo:
        return 0.5
    score = (v - lo) / (hi - lo)
    return score if higher_better else 1.0 - score


scores = []
for eer, auc, t, p, pp in zip(all_eer, all_auc, all_time, all_params, all_pairs):
    s = [
        norm(eer, eer_min, eer_max, higher_better=False),  # EER lower=better
        norm(auc, auc_min, auc_max, higher_better=True),
        norm(t, time_min, time_max, higher_better=False),  # faster=better
        norm(p, params_min, params_max, higher_better=False),  # smaller=better
        norm(pp, pairs_min, pairs_max, higher_better=True),  # more pairs=better
    ]
    scores.append(s)

n_metrics = len(metrics)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(polar=True))
colors_radar = ["#1f77b4", "#5fa8d3", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c"]
for (label, *_rest), s, c in zip(model_records, scores, colors_radar):
    s_plot = s + s[:1]
    ax.plot(angles, s_plot, linewidth=1.8, color=c, label=label)
    ax.fill(angles, s_plot, alpha=0.07, color=c)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
ax.set_title("Normalized scorecard (higher = better on every axis)",
             fontsize=14, fontweight="bold", pad=22)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9, framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
out_radar = OUT / "compare_08_radar_scorecard.png"
fig.savefig(out_radar, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_radar}")

print("\nAll extra charts generated.")