import csv
import os
import re
import sys
from datetime import datetime
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path(os.environ.get("VIEW_RESULTS_INPUT_DIR", THIS_DIR / "newest_results"))
OUTPUT_ROOT = Path(os.environ.get("VIEW_RESULTS_OUTPUT_ROOT", THIS_DIR / "view_results_outputs"))
MODALITY_SHAPES = [(50, 50), (100, 100)]
SUMMARY_FIELDNAMES = [
    "epochs",
    "seed",
    "lr",
    "poly_degree",
    "burnin_ratio",
    "saved_r2_train_mean",
    "saved_r2_test_mean",
    "mean_param_r2_train",
    "mean_param_r2_test",
    "M1 Accuracy",
    "M1 TPR",
    "M1 FPR",
    "M1 FDR",
    "M2 Accuracy",
    "M2 TPR",
    "M2 FPR",
    "M2 FDR",
]
FILE_RE = re.compile(
    r"results_(?P<test_name>.+)_epochs_(?P<epochs>\d+)_seed_(?P<seed>\d+)_"
    r"lr_(?P<lr_label>[^_]+)_polydegree_(?P<poly_degree>\d+)\.npz$"
)


def parse_lr(label):
    return float(label.replace("p", ".").replace("m", "-"))


def format_float_for_path(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def burnin_label(value):
    return str(float(value)).replace(".", "p")


def result_label(row):
    return (
        f"{row['test_name']}_epochs_{row['epochs']}_seed_{row['seed']}_"
        f"lr_{format_float_for_path(row['lr'])}_polydegree_{row['poly_degree']}"
    )


def group_label(group_rows):
    row = group_rows[0]
    return (
        f"{row['test_name']}_epochs_{row['epochs']}_"
        f"lr_{format_float_for_path(row['lr'])}_polydegree_{row['poly_degree']}"
    )


def list_result_files(result_dir):
    rows = []
    for path in sorted(result_dir.glob("results_*_epochs_*_seed_*_lr_*_polydegree_*.npz")):
        match = FILE_RE.match(path.name)
        if match is None:
            print(f"Skipping unrecognized result file name: {path.name}")
            continue
        rows.append(
            {
                "test_name": match.group("test_name"),
                "epochs": int(match.group("epochs")),
                "seed": int(match.group("seed")),
                "lr": parse_lr(match.group("lr_label")),
                "poly_degree": int(match.group("poly_degree")),
                "path": path,
            }
        )
    rows.sort(key=lambda row: (row["test_name"], row["lr"], row["poly_degree"], row["seed"]))
    return rows


def load_bundle(row):
    return np.load(row["path"], allow_pickle=True)


def get_results_by_burnin(bundle):
    return bundle["results_by_burnin"].item()


def get_result(bundle, burnin_ratio):
    return get_results_by_burnin(bundle)[str(float(burnin_ratio))]


def infer_shape(values, modality_idx):
    size = int(np.asarray(values).size)
    if modality_idx < len(MODALITY_SHAPES):
        shape = MODALITY_SHAPES[modality_idx]
        if shape[0] * shape[1] == size:
            return shape

    side = int(np.sqrt(size))
    if side * side == size:
        return (side, side)

    return (1, size)


def progress_epoch_axis(values, epochs):
    values = np.asarray(values)
    if len(values) == 0:
        return np.array([])
    return np.linspace(1, epochs, len(values))


def save_figure(fig, output_dir, name):
    path = output_dir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


def write_summary_csv(rows, summary_path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        summary_path.write_text("")
    print(f"Saved summary CSV: {summary_path}")


def empty_selection_metrics():
    return {
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tp": np.nan,
        "accuracy": np.nan,
        "true_positive_rate": np.nan,
        "false_positive_rate": np.nan,
        "false_discovery_rate": np.nan,
    }


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def selection_metrics_from_confusion_matrix(confusion_matrix_values):
    if confusion_matrix_values is None:
        return empty_selection_metrics()

    cm = np.asarray(confusion_matrix_values)
    if cm.shape != (2, 2):
        return empty_selection_metrics()

    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    total = tn + fp + fn + tp
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": safe_divide(tp + tn, total),
        "true_positive_rate": safe_divide(tp, tp + fn),
        "false_positive_rate": safe_divide(fp, fp + tn),
        "false_discovery_rate": safe_divide(fp, fp + tp),
    }


def add_modality_selection_metrics(summary_row, result, modality_idx):
    metric_prefix = f"M{modality_idx + 1}"
    confusion_matrices = result.get("confusion_matrix_list", [])
    if modality_idx < len(confusion_matrices):
        metrics = selection_metrics_from_confusion_matrix(confusion_matrices[modality_idx])
    else:
        metrics = empty_selection_metrics()

    metric_labels = {
        "accuracy": "Accuracy",
        "true_positive_rate": "TPR",
        "false_positive_rate": "FPR",
        "false_discovery_rate": "FDR",
    }
    for metric_name, metric_label in metric_labels.items():
        summary_row[f"{metric_prefix} {metric_label}"] = metrics[metric_name]


def summarize_rows(result_files):
    rows = []
    for row in result_files:
        with load_bundle(row) as bundle:
            for burnin_ratio in bundle["burnin_ratios"]:
                result = get_result(bundle, burnin_ratio)
                selected_counts = [int(np.asarray(mask).sum()) for mask in result["mask_list"]]
                mean_parameter_r2 = result["mean_parameter_r2"]
                summary_row = {
                    "test_name": row["test_name"],
                    "epochs": row["epochs"],
                    "seed": row["seed"],
                    "lr": row["lr"],
                    "poly_degree": row["poly_degree"],
                    "burnin_ratio": float(burnin_ratio),
                    "source_file": row["path"].name,
                    "saved_r2_train_mean": float(np.nanmean(result["saved_r2_train"])),
                    "saved_r2_test_mean": float(np.nanmean(result["saved_r2_test"])),
                    "mean_param_r2_train": float(mean_parameter_r2["r2_train"]),
                    "mean_param_r2_test": float(mean_parameter_r2["r2_test"]),
                    "selected_modality_1": selected_counts[0] if len(selected_counts) > 0 else np.nan,
                    "selected_modality_2": selected_counts[1] if len(selected_counts) > 1 else np.nan,
                }
                add_modality_selection_metrics(summary_row, result, modality_idx=0)
                add_modality_selection_metrics(summary_row, result, modality_idx=1)
                rows.append(summary_row)
    return rows


def summarize_results(result_files, output_dir):
    rows = summarize_rows(result_files)
    summary_path = output_dir / "results_summary.csv"
    write_summary_csv(rows, summary_path)
    return summary_path


def sorted_group_rows(group_rows):
    return sorted(group_rows, key=lambda row: row["seed"])


def grid_shape(n_items, ncols=5):
    return max(1, ceil(n_items / ncols)), ncols


def plot_r2_grid(group_rows, output_dir, which, ncols=5):
    group_rows = sorted_group_rows(group_rows)
    nrows, ncols = grid_shape(len(group_rows), ncols=ncols)
    values_key = f"r2_{which}"
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, row in zip(axes, group_rows):
        with load_bundle(row) as bundle:
            if values_key not in bundle.files:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_title(f"seed={row['seed']}", fontsize=9)
                ax.axis("off")
                continue
            values = np.asarray(bundle[values_key], dtype=float)

        ax.plot(progress_epoch_axis(values, row["epochs"]), values, linewidth=1.0)
        ax.set_title(f"seed={row['seed']}", fontsize=9)
        ax.grid(alpha=0.25)

    for ax in axes[len(group_rows) :]:
        ax.axis("off")

    label = group_label(group_rows)
    fig.suptitle(f"{label}, {which} R2 across seeds", fontsize=12)
    save_figure(fig, output_dir, f"{label}_{which}_r2_grid.png")
    plt.close(fig)


def plot_selection_grid(group_rows, output_dir, burnin_ratio, modality_idx, ncols=5):
    group_rows = sorted_group_rows(group_rows)
    nrows, ncols = grid_shape(len(group_rows), ncols=ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, row in zip(axes, group_rows):
        with load_bundle(row) as bundle:
            result = get_result(bundle, burnin_ratio)
            mask = np.asarray(result["mask_list"][modality_idx])
        shape = infer_shape(mask, modality_idx)
        ax.imshow(mask.reshape(shape), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"seed={row['seed']}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(group_rows) :]:
        ax.axis("off")

    label = group_label(group_rows)
    fig.suptitle(f"{label}, modality={modality_idx + 1}, selected regions, burnin={burnin_ratio}", fontsize=12)
    save_figure(
        fig,
        output_dir,
        f"{label}_burnin_{burnin_label(burnin_ratio)}_modality_{modality_idx + 1}_selection_grid.png",
    )
    plt.close(fig)


def plot_aggregate_selection(group_rows, output_dir, burnin_ratio, modality_idx):
    masks = []
    for row in sorted_group_rows(group_rows):
        with load_bundle(row) as bundle:
            result = get_result(bundle, burnin_ratio)
            masks.append(np.asarray(result["mask_list"][modality_idx], dtype=float))
    if not masks:
        return

    freq = np.mean(np.stack(masks, axis=0), axis=0)
    shape = infer_shape(freq, modality_idx)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(freq.reshape(shape), cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"Modality {modality_idx + 1}: aggregate region selection")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    label = group_label(group_rows)
    fig.suptitle(f"{label}, burnin={burnin_ratio}, n_seeds={len(group_rows)}", fontsize=12)
    save_figure(
        fig,
        output_dir,
        f"{label}_burnin_{burnin_label(burnin_ratio)}_modality_{modality_idx + 1}_aggregate_selection.png",
    )
    plt.close(fig)


def export_all(result_files, output_dir):
    groups = {}
    for row in result_files:
        key = (row["test_name"], row["epochs"], row["lr"], row["poly_degree"])
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        plot_r2_grid(group_rows, output_dir, which="train")
        plot_r2_grid(group_rows, output_dir, which="test")

        with load_bundle(group_rows[0]) as bundle:
            burnin_ratios = [float(x) for x in bundle["burnin_ratios"]]

        for burnin_ratio in burnin_ratios:
            with load_bundle(group_rows[0]) as bundle:
                first_result = get_result(bundle, burnin_ratio)
                n_modalities = len(first_result["mask_list"])

            for modality_idx in range(n_modalities):
                plot_selection_grid(group_rows, output_dir, burnin_ratio, modality_idx)
                plot_aggregate_selection(group_rows, output_dir, burnin_ratio, modality_idx)

    summarize_results(result_files, output_dir)


def export_csv_only(input_dir=INPUT_DIR, output_root=OUTPUT_ROOT):
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f"view_results_csv_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing newest results folder: {input_dir}")

    result_files = list_result_files(input_dir)
    print(f"Input folder: {input_dir.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Found {len(result_files)} result files")

    if not result_files:
        print("No files found in newest_results; no CSV saved.")
        return None

    summary_path = summarize_results(result_files, output_dir)
    print(f"Saved CSV only under: {output_dir}")
    return summary_path


def main():
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"view_results_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing newest results folder: {INPUT_DIR}")

    result_files = list_result_files(INPUT_DIR)
    print(f"Input folder: {INPUT_DIR.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Found {len(result_files)} result files")

    if not result_files:
        print("No files found in newest_results; no CSV or figures saved.")
        return

    export_all(result_files, output_dir)
    print(f"Saved CSV and figures under: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"view_results export failed: {exc}", file=sys.stderr)
        raise
