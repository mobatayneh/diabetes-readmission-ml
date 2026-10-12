"""Aggregate-only diagnostics for frozen experiments; no patient record export."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from src.evaluation import metrics

LABELS = {"logistic": "Logistic Regression", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
COLORS = {"logistic": "#2864A5", "lightgbm": "#147D68", "xgboost": "#B85827"}


def error_analysis(test, predictions, thresholds, primary):
    rows = []
    for name in primary:
        y, p = test.readmitted_binary.to_numpy(), np.asarray(predictions[name])
        for column in ("race", "gender", "age", "discharge_disposition_id"):
            values = test[column].fillna("Unknown").astype(str).to_numpy()
            for group in sorted(set(values)):
                mask = values == group
                n, positives = int(mask.sum()), int(y[mask].sum())
                eligible = n >= 50 and positives >= 10 and n - positives >= 10
                row = {"model": name, "analysis": "subgroup", "variable": column, "group": group,
                       "n": n, "positives": positives, "metrics_suppressed": not eligible}
                if eligible:
                    row.update(metrics(y[mask], p[mask], thresholds[name]))
                rows.append(row)
        pred = p >= thresholds[name]
        for label, mask in (("false_negative", (y == 1) & ~pred), ("true_positive", (y == 1) & pred),
                            ("false_positive", (y == 0) & pred), ("true_negative", (y == 0) & ~pred)):
            for feature in ("number_inpatient", "number_emergency", "num_medications", "time_in_hospital"):
                rows.append({"model": name, "analysis": "error_profile", "variable": feature, "group": label,
                    "n": int(mask.sum()), "feature_median": float(test.loc[mask, feature].median()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def render_results(y, predictions, thresholds, output: Path, primary):
    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    def finish(fig, filename):
        fig.savefig(destination / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for name in primary:
        root = name.removesuffix("_tuned")
        fpr, tpr, _ = roc_curve(y, predictions[name])
        precision, recall, _ = precision_recall_curve(y, predictions[name])
        axes[0].plot(fpr, tpr, color=COLORS[root], label=LABELS[root])
        axes[1].plot(recall, precision, color=COLORS[root], label=LABELS[root])
    axes[0].plot([0, 1], [0, 1], "--", color="#929292", linewidth=1)
    axes[1].axhline(np.mean(y), ls="--", color="#929292", label=f"Prevalence {np.mean(y):.1%}")
    axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC · same test patients")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall · same test patients")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Controlled benchmark | tuned models | 10,499 patients" if len(y) == 10499 else "Smoke check | not portfolio results")
    finish(fig, "roc_pr.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    for ax, name in zip(axes, primary):
        root = name.removesuffix("_tuned")
        cm = confusion_matrix(y, predictions[name] >= thresholds[name], labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i,j] > cm.max() / 2 else "#172A3A", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
               xlabel="Predicted", ylabel="Observed", title=f"{LABELS[root]}\nValidation threshold {thresholds[name]:.2f}")
    fig.suptitle("Held-out confusion matrices | one observation per patient")
    finish(fig, "confusion_matrices.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, name in zip(axes, primary):
        root = name.removesuffix("_tuned")
        for variant, style in (("tuned", "--"), ("calibrated", "-")):
            observed, predicted = calibration_curve(y, predictions[root + "_" + variant], n_bins=10, strategy="quantile")
            ax.plot(predicted, observed, style, marker="o", ms=4, color=COLORS[root], label=variant)
        ax.plot([0, 1], [0, 1], ":", color="#999999")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=LABELS[root], xlabel="Mean predicted probability", ylabel="Observed fraction positive")
        ax.legend(frameon=False)
    fig.suptitle("Calibration diagnostics | 10 quantile bins | test data")
    finish(fig, "calibration.png")

    curves = pd.read_csv(output / "validation_threshold_curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), constrained_layout=True)
    for ax, name in zip(axes, primary):
        part = curves.loc[curves.model == name]
        for key, color in (("precision", "#2864A5"), ("recall", "#147D68"), ("f1", "#B85827")):
            ax.plot(part.threshold, part[key], label=key, color=color)
        ax.axvline(thresholds[name], color="#888", ls=":")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=LABELS[name.removesuffix("_tuned")], xlabel="Threshold", ylabel="Validation score")
        ax.legend(frameon=False)
    fig.suptitle("Threshold selection uses validation patients only")
    finish(fig, "validation_thresholds.png")

    ci = pd.read_csv(output / "confidence_intervals.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for ax, key, title in zip(axes, ("roc_auc", "average_precision", "f1"), ("ROC-AUC", "Average precision", "F1")):
        for i, name in enumerate(primary):
            row = ci.loc[(ci.model == name) & (ci.metric == key)].iloc[0]
            ax.errorbar(row.estimate, i, xerr=[[row.estimate-row.low], [row.high-row.estimate]], fmt="o", capsize=4, color=COLORS[name.removesuffix("_tuned")])
        ax.set(yticks=range(3), yticklabels=[LABELS[n.removesuffix("_tuned")] for n in primary], xlim=(0, 1), title=title)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=.15)
    fig.suptitle("Tuned models | pointwise 95% patient-bootstrap intervals")
    finish(fig, "metric_intervals.png")

    imp = pd.read_csv(output / "importance.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    for ax, name in zip(axes, primary):
        root = name.removesuffix("_tuned")
        part = imp.loc[imp.model == root].nlargest(7, "mean_ap_decrease").sort_values("mean_ap_decrease")
        ax.barh(part.feature, part.mean_ap_decrease, color=COLORS[root], xerr=part.sd_across_permutations, capsize=2)
        ax.axvline(0, color="#888", lw=1)
        ax.set(title=LABELS[root], xlabel="Decrease in validation average precision")
    fig.suptitle("Permutation importance | associations, not causal effects | bars: permutation SD")
    finish(fig, "permutation_importance.png")
