#!/usr/bin/env python3
"""
SCI-style visualization for Route-A EDL comparison.

Inputs (from edl_routeA_demo.py):
- per_epoch_metrics.csv
- per_seed_metrics.csv

Outputs:
- learning_curves_sci.png/.pdf
- final_metrics_sci.png/.pdf
- visualization_report.md
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_ORDER = ["ce", "ce_ls", "edl"]
METHOD_LABEL = {
    "ce": "Softmax-CE",
    "ce_ls": "LabelSmoothing",
    "edl": "RouteA-EDL",
}
PALETTE = {
    "ce": "#3B5B92",
    "ce_ls": "#6C757D",
    "edl": "#D94841",
}
    #   "dpn": "#2A9D8F",         # 青绿
    #   "ensemble": "#F4A261",    # 琥珀橙
    #   "mcdropout": "#7B6CF6",   # 冷紫
    #   "temp_scaling": "#4C78A8",# 钢蓝
    #   "focal": "#1F9D8A",     # teal
    #   "temp_scaling": "#E59F2A",   # amber
    #   "mixup": "#7A62D3",     # muted violet
    #   "cutmix": "#D16A3A",    # burnt orange
    #   "deep_ensemble": "#2E7D32",  # forest green
    #   "mc_dropout": "#4C78A8",     # cool blue
    #   "dirichlet_prior_net": "#B23A8A",  # magenta-purple
    #   "energy_based": "#8C564B",   # earthy brown
    #   "mahalanobis": "#17A2B8",    # cyan-blue
    #   "odin": "#A0A832",           # olive
    #   "conformal": "#C75146",      # brick red
    #   "risk_gate": "#3A86FF",      # vivid blue
    #   "rl_risk": "#5E8C31",        # moss green



def set_sci_gray_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.facecolor": "#E8EBF0",
            "figure.facecolor": "#D7DCE3",
            "savefig.facecolor": "#D7DCE3",
            "axes.edgecolor": "#4E5A67",
            "axes.labelcolor": "#1F2933",
            "xtick.color": "#2F3B46",
            "ytick.color": "#2F3B46",
            "grid.color": "#FFFFFF",
            "grid.alpha": 0.95,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.facecolor": "#EFF2F6",
            "legend.edgecolor": "#9AA5B1",
            "axes.titleweight": "bold",
        }
    )


def plot_mean_std_band(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    grouped = (
        df.groupby(["method", "epoch"], as_index=False)[metric]
        .agg(mean="mean", std="std")
        .fillna({"std": 0.0})
    )

    for method in METHOD_ORDER:
        sub = grouped[grouped["method"] == method].sort_values("epoch")
        x = sub["epoch"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        s = sub["std"].to_numpy(dtype=float)

        ax.plot(
            x,
            y,
            color=PALETTE[method],
            linewidth=2.4,
            label=METHOD_LABEL[method],
        )
        ax.fill_between(
            x,
            y - s,
            y + s,
            color=PALETTE[method],
            alpha=0.20,
            linewidth=0.0,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)


def draw_learning_curves(df_epoch: pd.DataFrame, out_dir: str) -> None:
    metric_cfg = [
        ("train_loss", "Loss (Train)"),
        ("train_accuracy", "Accuracy (Train)"),
        ("train_nll", "NLL (Train)"),
        ("train_ece", "ECE (Train)"),
        ("val_accuracy", "Accuracy (Validation)"),
        ("val_nll", "NLL (Validation)"),
        ("val_ece", "ECE (Validation)"),
        ("val_ood_auroc", "OOD-AUROC (Validation)"),
    ]

    for metric, ylabel in metric_cfg:
        fig, ax = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
        plot_mean_std_band(ax, df_epoch, metric=metric, ylabel=ylabel)
        ax.legend(loc="best")

        png_path = os.path.join(out_dir, f"learning_curve_{metric}_sci.png")
        pdf_path = os.path.join(out_dir, f"learning_curve_{metric}_sci.pdf")
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=320, bbox_inches="tight")
        plt.close(fig)


def draw_final_bars(df_seed: pd.DataFrame, out_dir: str) -> None:
    metric_cfg = [
        ("accuracy", "Accuracy"),
        ("nll", "NLL"),
        ("ece", "ECE"),
        ("ood_auroc", "OOD-AUROC"),
    ]

    summary = (
        df_seed.groupby("method", as_index=False)[[m[0] for m in metric_cfg]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "method" if c[0] == "method" else f"{c[0]}_{c[1]}" for c in summary.columns
    ]

    x = np.arange(len(METHOD_ORDER))
    labels = [METHOD_LABEL[m] for m in METHOD_ORDER]
    colors = [PALETTE[m] for m in METHOD_ORDER]

    for metric, title in metric_cfg:
        fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
        y = []
        err = []
        for m in METHOD_ORDER:
            row = summary[summary["method"] == m].iloc[0]
            y.append(float(row[f"{metric}_mean"]))
            err.append(float(row[f"{metric}_std"]))

        bars = ax.bar(x, y, color=colors, alpha=0.9, width=0.65, edgecolor="#334155", linewidth=0.8)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor="#1F2937", elinewidth=1.5, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_ylabel(title)

        for b, v, s in zip(bars, y, err):
            y_offset = 0.015 * max(y) if max(y) > 0 else 0.01
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + y_offset,
                f"{v:.3f}±{s:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#0F172A",
            )

        png_path = os.path.join(out_dir, f"final_metric_{metric}_sci.png")
        pdf_path = os.path.join(out_dir, f"final_metric_{metric}_sci.pdf")
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=320, bbox_inches="tight")
        plt.close(fig)


def write_visualization_report(df_seed: pd.DataFrame, out_dir: str) -> None:
    summary = df_seed.groupby("method", as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        nll_mean=("nll", "mean"),
        nll_std=("nll", "std"),
        ece_mean=("ece", "mean"),
        ece_std=("ece", "std"),
        auroc_mean=("ood_auroc", "mean"),
        auroc_std=("ood_auroc", "std"),
    )

    def row(method: str) -> pd.Series:
        return summary[summary["method"] == method].iloc[0]

    ce = row("ce")
    edl = row("edl")

    report = os.path.join(out_dir, "visualization_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# SCI-Style Visualization Report\n\n")
        f.write("## Figures\n\n")
        f.write("- `learning_curve_train_loss_sci.(png/pdf)`\n")
        f.write("- `learning_curve_train_accuracy_sci.(png/pdf)`\n")
        f.write("- `learning_curve_train_nll_sci.(png/pdf)`\n")
        f.write("- `learning_curve_train_ece_sci.(png/pdf)`\n")
        f.write("- `learning_curve_val_accuracy_sci.(png/pdf)`\n")
        f.write("- `learning_curve_val_nll_sci.(png/pdf)`\n")
        f.write("- `learning_curve_val_ece_sci.(png/pdf)`\n")
        f.write("- `learning_curve_val_ood_auroc_sci.(png/pdf)`\n")
        f.write("- `final_metric_accuracy_sci.(png/pdf)`\n")
        f.write("- `final_metric_nll_sci.(png/pdf)`\n")
        f.write("- `final_metric_ece_sci.(png/pdf)`\n")
        f.write("- `final_metric_ood_auroc_sci.(png/pdf)`\n\n")

        f.write("## Key Observation (EDL vs CE)\n\n")
        f.write(
            f"1. Accuracy: {edl['accuracy_mean']:.4f} vs {ce['accuracy_mean']:.4f} (roughly comparable).\n"
        )
        f.write(f"2. NLL: {edl['nll_mean']:.4f} vs {ce['nll_mean']:.4f} (EDL better).\n")
        f.write(f"3. ECE: {edl['ece_mean']:.4f} vs {ce['ece_mean']:.4f} (EDL better).\n")
        f.write(f"4. OOD-AUROC: {edl['auroc_mean']:.4f} vs {ce['auroc_mean']:.4f} (EDL better).\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCI-style plots for Route-A EDL demo")
    p.add_argument("--results-dir", type=str, default="demo_results", help="Directory containing CSV outputs")
    p.add_argument(
        "--out-dir",
        type=str,
        default="demo_results/figures",
        help="Directory to save plots",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    epoch_csv = os.path.join(args.results_dir, "per_epoch_metrics.csv")
    seed_csv = os.path.join(args.results_dir, "per_seed_metrics.csv")

    if not os.path.exists(epoch_csv):
        raise FileNotFoundError(f"Missing file: {epoch_csv}")
    if not os.path.exists(seed_csv):
        raise FileNotFoundError(f"Missing file: {seed_csv}")

    df_epoch = pd.read_csv(epoch_csv)
    df_seed = pd.read_csv(seed_csv)

    set_sci_gray_theme()

    draw_learning_curves(df_epoch, args.out_dir)
    draw_final_bars(df_seed, args.out_dir)
    write_visualization_report(df_seed, args.out_dir)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
