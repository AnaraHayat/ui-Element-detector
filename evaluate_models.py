"""Evaluate the trained classifiers from the test outputs (the *_test.csv / *_test.npz files).

For every model it computes: accuracy, precision, recall, F1 (macro + weighted), Cohen's kappa,
15x15 confusion matrix, one-vs-rest ROC curves + AUC, and a bootstrap 95% CI (resampling whole
images). Across models it runs pairwise McNemar tests.

Usage (folder that holds convnext_tiny_gt_test.csv, convnext_tiny_gt_test.npz, ...):
    python evaluate_models.py --dir /kaggle/working --source gt  --out eval_gt
    python evaluate_models.py --dir /kaggle/working --source det --out eval_det

Needs: numpy pandas scikit-learn scipy matplotlib
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

CLASSES = ["Layout", "Text", "Image", "Button", "Toolbar", "ProgressBar", "List", "WebView",
           "Tab", "Drawer", "Ad", "Card", "ScrollView", "Checkbox", "ViewPager"]
K = len(CLASSES)


def find_probs(npz_path: Path, n_rows: int):
    """Return the (n_rows, 15) probability array from the .npz, whatever its key is called."""
    if not npz_path.exists():
        return None
    z = np.load(npz_path)
    for key in z.files:
        a = z[key]
        if a.ndim == 2 and a.shape == (n_rows, K):
            return a.astype(np.float32)
    print(f"  ! no ({n_rows}, {K}) array in {npz_path.name}; keys = {z.files}. ROC skipped.")
    return None


def looks_valid(df: pd.DataFrame) -> bool:
    """Reject files where predictions were never written (confidence exactly 0)."""
    zero = (df["pred_conf"] == 0).mean()
    if zero > 0.5 or df["pred"].nunique() < 3:
        print(f"  ! INVALID: {zero:.0%} of rows have pred_conf == 0 and only "
              f"{df['pred'].nunique()} distinct predicted classes. Fix the test script first.")
        return False
    return True


def macro_f1_from_cm(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    p = tp / np.maximum(cm.sum(0), 1)
    r = tp / np.maximum(cm.sum(1), 1)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(f.mean())


def bootstrap_ci(y, pred, images, n_boot=1000, seed=42):
    """95% CI for accuracy and macro-F1, resampling images (crops of one image are correlated)."""
    _, img_idx = np.unique(images, return_inverse=True)
    n_img = img_idx.max() + 1
    per_img = np.zeros((n_img, K * K), dtype=np.float64)
    np.add.at(per_img, (img_idx, y * K + pred), 1)
    rng = np.random.default_rng(seed)
    acc, f1 = [], []
    for _ in range(n_boot):
        cm = per_img[rng.integers(0, n_img, n_img)].sum(0).reshape(K, K)
        acc.append(np.trace(cm) / cm.sum())
        f1.append(macro_f1_from_cm(cm))
    return (np.percentile(acc, [2.5, 97.5]), np.percentile(f1, [2.5, 97.5]))


def plot_confusion(cm, title, path):
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K), CLASSES, rotation=60, ha="right")
    ax.set_yticks(range(K), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(K):
        for j in range(K):
            if norm[i, j] >= 0.05:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="share of true class")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y, probs, title, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = {}
    for k in range(K):
        pos = (y == k)
        if pos.sum() == 0 or pos.all():
            continue
        fpr, tpr, _ = roc_curve(pos, probs[:, k])
        aucs[CLASSES[k]] = roc_auc_score(pos, probs[:, k])
        ax.plot(fpr, tpr, lw=1, label=f"{CLASSES[k]} ({aucs[CLASSES[k]]:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{title}\nmacro AUC = {np.mean(list(aucs.values())):.3f}")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return aucs


def mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    if b + c == 0:
        return b, c, 1.0
    if b + c < 25:
        return b, c, float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(stats.chi2.sf(chi2, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="folder with <model>_<source>_test.csv/.npz")
    ap.add_argument("--source", default="gt", choices=["gt", "det"])
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    src, out = Path(args.dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob(f"*_{args.source}_test.csv"))
    if not files:
        sys.exit(f"no *_{args.source}_test.csv files in {src}")

    rows, correct_by_model, truth_ref = [], {}, None
    for f in files:
        model = f.name.replace(f"_{args.source}_test.csv", "")
        print(f"\n{model}")
        df = pd.read_csv(f)
        if not looks_valid(df):
            continue
        probs = find_probs(f.with_suffix(".npz"), len(df))
        keep = (df["truth"] >= 0).to_numpy()
        d = df[keep]
        y, pred = d["truth"].to_numpy(), d["pred"].to_numpy()
        pr = probs[keep] if probs is not None else None
        print(f"  {len(d):,} scored crops ({(~keep).sum():,} unmatched dropped)")

        cm = confusion_matrix(y, pred, labels=range(K))
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / f"{model}_confusion.csv")
        plot_confusion(cm, f"{model} ({args.source}) - row-normalised", out / f"{model}_confusion.png")
        rep = classification_report(y, pred, labels=range(K), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(out / f"{model}_per_class.csv")

        macro_auc = np.nan
        if pr is not None:
            aucs = plot_roc(y, pr, f"{model} ({args.source})", out / f"{model}_roc.png")
            pd.Series(aucs).to_csv(out / f"{model}_auc_per_class.csv", header=["auc"])
            macro_auc = float(np.mean(list(aucs.values())))

        acc_ci, f1_ci = bootstrap_ci(y, pred, d["image"].to_numpy(), args.n_boot)
        rows.append({
            "model": model, "crops": len(d),
            "accuracy": accuracy_score(y, pred), "acc_ci_low": acc_ci[0], "acc_ci_high": acc_ci[1],
            "balanced_acc": balanced_accuracy_score(y, pred),
            "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
            "f1_ci_low": f1_ci[0], "f1_ci_high": f1_ci[1],
            "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
            "kappa": cohen_kappa_score(y, pred), "auc_macro": macro_auc,
        })
        correct_by_model[model] = (y == pred)
        if truth_ref is None:
            truth_ref = y

    if not rows:
        sys.exit("\nnothing evaluated - every file was rejected as invalid.")
    summary = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    summary.to_csv(out / "summary.csv", index=False)

    mc = []
    for a, b in itertools.combinations(correct_by_model, 2):
        ca, cb = correct_by_model[a], correct_by_model[b]
        if len(ca) != len(cb):
            continue
        n01, n10, p = mcnemar(ca, cb)
        mc.append({"model_A": a, "model_B": b, "A_right_B_wrong": n01, "A_wrong_B_right": n10,
                   "p_value": p})
    if mc:
        m = pd.DataFrame(mc)
        m["p_bonferroni"] = (m["p_value"] * len(m)).clip(upper=1.0)
        m["significant_0.05"] = m["p_bonferroni"] < 0.05
        m.to_csv(out / "mcnemar.csv", index=False)
        print("\nMcNemar (pairwise):\n", m.round(4).to_string(index=False))

    pd.set_option("display.width", 200)
    print("\nSUMMARY\n", summary.round(4).to_string(index=False))
    print(f"\nall files written to {out}/")


if __name__ == "__main__":
    main()
