"""Compare models run in MFFV project (FV-USM).

Reads results.json files in src/results/ and draws:
  1. Bar chart EER (lower is better)
  2. Bar chart AUC (higher is better)
  3. Scatter EER vs #params
  4. Summary table

Run: python src/results/comparison_charts/plot_comparison.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]  # src/
RESULTS = ROOT / "results"
OUT = ROOT / "results" / "comparison_charts"
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


expd = load(RESULTS / "baseline_resnet18_arcface_expD" / "results.json")
r30 = load(RESULTS / "baseline_resnet18_arcface_30ep" / "results.json")
v1_eval = load(RESULTS / "fvusm_resnet18_arcface_v1" / "eval_results.json")
v1_log = load(RESULTS / "fvusm_resnet18_arcface_v1" / "training_log.json")
v1_best = min(v1_log, key=lambda r: r["eer"])
fdt = load(RESULTS / "fdt_30ep_arcface" / "fdt" / "results.json")
gla = load(RESULTS / "glafd_quick" / "gla-fd" / "results.json")
ilcnn = load(RESULTS / "ilcnn_10ep" / "ilcnn" / "results.json")


models = [
    dict(name="ResNet18+ArcFace (50ep, expD)", eer=expd["final_test"]["eer"],
         auc=expd["final_test"]["auc"], params=expd["num_params"],
         epochs=expd["epochs"], pairs=expd["final_test"]["num_pairs"],
         kind="ResNet18 baseline"),
    dict(name="ResNet18+ArcFace (30ep)", eer=r30["final_test"]["eer"],
         auc=r30["final_test"]["auc"], params=r30["num_params"],
         epochs=r30["epochs"], pairs=r30["final_test"]["num_pairs"],
         kind="ResNet18 baseline"),
    dict(name="ResNet18+ArcFace v1 (train_fvusm, best@40)",
         eer=v1_eval["eer_percent"] / 100.0, auc=float("nan"),
         params=11280280, epochs=v1_best["epoch"],
         pairs=v1_eval["n_test_images"], kind="ResNet18 baseline"),
    dict(name="FDT (SCUT-BIP-Lab)", eer=fdt["final_test"]["eer"],
         auc=fdt["final_test"]["auc"], params=fdt["num_params"],
         epochs=fdt["epochs"], pairs=fdt["final_test"]["num_pairs"],
         kind="Third-party (Transformer)"),
    dict(name="GLA-FD (quick test)", eer=gla["final_test"]["eer"],
         auc=gla["final_test"]["auc"], params=gla["num_params"],
         epochs=gla["epochs"], pairs=gla["final_test"]["num_pairs"],
         kind="Third-party (ViT)"),
    dict(name="ILCNN (lightweight)", eer=ilcnn["final_test"]["eer"],
         auc=ilcnn["final_test"]["auc"], params=ilcnn["num_params"],
         epochs=ilcnn["epochs"], pairs=ilcnn["final_test"]["num_pairs"],
         kind="Third-party (lightweight CNN)"),
]


palette = {
    "ResNet18 baseline": "#1f77b4",
    "Third-party (Transformer)": "#d62728",
    "Third-party (ViT)": "#9467bd",
    "Third-party (lightweight CNN)": "#2ca02c",
}
colors = [palette[m["kind"]] for m in models]
names = [m["name"] for m in models]


# 1) Bar chart EER + AUC
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

eer_vals = [m["eer"] * 100 for m in models]
bars = axes[0].bar(names, eer_vals, color=colors, edgecolor="black", linewidth=0.6)
axes[0].set_ylabel("EER (%) - lower is better", fontsize=11)
axes[0].set_title("Equal Error Rate (EER) on FV-USM test", fontsize=13, fontweight="bold")
axes[0].set_ylim(0, max(eer_vals) * 1.25)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars, eer_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                 f"{v:.3f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
axes[0].tick_params(axis="x", rotation=20)

auc_plot = [(n, m["auc"], c) for n, m, c in zip(names, models, colors)
            if not np.isnan(m["auc"])]
auc_names = [x[0] for x in auc_plot]
auc_vals = [x[1] for x in auc_plot]
auc_cols = [x[2] for x in auc_plot]

bars2 = axes[1].bar(auc_names, auc_vals, color=auc_cols, edgecolor="black", linewidth=0.6)
axes[1].set_ylabel("AUC - higher is better", fontsize=11)
axes[1].set_title("Area Under ROC Curve (AUC) on FV-USM test", fontsize=13, fontweight="bold")
axes[1].set_ylim(0.85, 1.005)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)
for bar, v in zip(bars2, auc_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                 f"{v:.4f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
axes[1].tick_params(axis="x", rotation=20)

legend_elems = [Patch(facecolor=c, label=k) for k, c in palette.items()]
axes[0].legend(handles=legend_elems, loc="upper right", fontsize=9, framealpha=0.9)

plt.tight_layout()
out_bar = OUT / "compare_01_eer_auc_bars.png"
fig.savefig(out_bar, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_bar}")


# 2) Scatter EER vs Params
fig2, ax2 = plt.subplots(figsize=(11, 6))
params_m = [m["params"] / 1e6 for m in models]
for p, e, c, n in zip(params_m, eer_vals, colors, names):
    ax2.scatter(p, e, s=180, color=c, edgecolor="black", linewidth=0.6, zorder=3)
    ax2.annotate(n, (p, e), xytext=(8, 6),
                 textcoords="offset points", fontsize=8.5, alpha=0.85)

ax2.set_xscale("log")
ax2.set_xlabel("#Parameters (M) - log scale", fontsize=11)
ax2.set_ylabel("EER (%) - lower is better", fontsize=11)
ax2.set_title("Trade-off: EER vs Model Size", fontsize=13, fontweight="bold")
ax2.grid(True, which="both", linestyle="--", alpha=0.4)
ax2.invert_yaxis()

legend_elems2 = [Patch(facecolor=c, label=k) for k, c in palette.items()]
ax2.legend(handles=legend_elems2, loc="lower left", fontsize=9, framealpha=0.9)

plt.tight_layout()
out_scatter = OUT / "compare_02_eer_vs_params.png"
fig2.savefig(out_scatter, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out_scatter}")


# 3) Summary table
fig3, ax3 = plt.subplots(figsize=(14, 3.4))
ax3.axis("off")

table_data = []
for m in models:
    auc_s = "-" if np.isnan(m["auc"]) else f"{m['auc']:.4f}"
    table_data.append([
        m["name"],
        f"{m['eer']*100:.3f}%",
        auc_s,
        f"{m['params']/1e6:.2f} M",
        f"{m['epochs']}",
        f"{m['pairs']:,}",
    ])

headers = ["Model", "EER", "AUC", "Params", "Epochs", "Test pairs/imgs"]
table = ax3.table(cellText=table_data, colLabels=headers,
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

for j in range(len(headers)):
    cell = table[(0, j)]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

best_idx = int(np.argmin([m["eer"] for m in models]))
for j in range(len(headers)):
    table[(best_idx + 1, j)].set_facecolor("#d4edda")

ax3.set_title("Summary of models run in MFFV Research (FV-USM)",
              fontsize=13, fontweight="bold", pad=18)
plt.tight_layout()
out_table = OUT / "compare_03_summary_table.png"
fig3.savefig(out_table, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Saved: {out_table}")


print()
print("=== SUMMARY ===")
print(f"{'Model':<46} {'EER':>10} {'AUC':>10} {'Params':>12}")
print("-" * 80)
for m in models:
    auc_s = "-" if np.isnan(m["auc"]) else f"{m['auc']:.4f}"
    print(f"{m['name']:<46} "
          f"{m['eer']*100:>9.3f}% {auc_s:>10} {m['params']/1e6:>10.2f}M")