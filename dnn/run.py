import csv
import os
import sys
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DNN_classes_4_11 import Experiment


DEFAULT_SEEDS = range(1, 21)
DEFAULT_TRUE_R2_VALUES = (0.8, 0.5, 0.3)
DEFAULT_EPOCHS = 600
DEFAULT_OUTPUT_DIR = THIS_DIR / "results"
RUN_CONFIG = {
    "sample_size": 2000,
    "val_ratio": 0.5,
    "in_feature_list": [50, 100],
    "lr": 1e-3,
}
SUMMARY_FIELDNAMES = [
    "true_r2",
    "seed",
    "epochs",
    "sample_size",
    "val_ratio",
    "in_feature_list",
    "lr",
    "r2_train",
    "r2_test",
]


def format_float_for_path(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def result_path(output_dir, true_r2, seed, epochs, lr):
    return (
        output_dir
        / f"dnn_results_r2_{format_float_for_path(true_r2)}_epochs_{epochs}_seed_{seed}_lr_{format_float_for_path(lr)}.npz"
    )


def run_label(true_r2, epochs, lr):
    return f"dnn_r2_{format_float_for_path(true_r2)}_epochs_{epochs}_lr_{format_float_for_path(lr)}"


def progress_epoch_axis(values, epochs):
    values = np.asarray(values)
    if len(values) == 0:
        return np.array([])
    return np.linspace(1, epochs, len(values))


def run_one(seed, true_r2, epochs, output_dir):
    print(f"Starting DNN true_r2={true_r2}, seed={seed}, epochs={epochs}")
    experiment = Experiment()
    trainer = experiment.run_whole_experiment(
        seed=seed,
        sample_size=RUN_CONFIG["sample_size"],
        true_r2=true_r2,
        val_ratio=RUN_CONFIG["val_ratio"],
        in_feature_list=RUN_CONFIG["in_feature_list"],
        lr=RUN_CONFIG["lr"],
        epochs=epochs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_path(output_dir, true_r2, seed, epochs, RUN_CONFIG["lr"])
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            true_r2=np.array(true_r2),
            seed=np.array(seed),
            epochs=np.array(epochs),
            sample_size=np.array(RUN_CONFIG["sample_size"]),
            val_ratio=np.array(RUN_CONFIG["val_ratio"]),
            in_feature_list=np.array(RUN_CONFIG["in_feature_list"], dtype=int),
            lr=np.array(RUN_CONFIG["lr"]),
            r2_train=np.array(trainer.r2_train, dtype=float),
            r2_test=np.array(trainer.r2_val, dtype=float),
            loss_train=np.array(trainer.loss_train, dtype=float),
            loss_test=np.array(trainer.loss_val, dtype=float),
        )
    os.replace(temp_path, output_path)
    print(f"Saved {output_path}")
    return output_path


def result_files(output_dir):
    r2_order = {value: idx for idx, value in enumerate(DEFAULT_TRUE_R2_VALUES)}
    rows = []
    for path in sorted(output_dir.glob("dnn_results_r2_*_epochs_*_seed_*_lr_*.npz")):
        try:
            with np.load(path, allow_pickle=True) as bundle:
                rows.append(
                    {
                        "path": path,
                        "true_r2": float(bundle["true_r2"]),
                        "seed": int(bundle["seed"]),
                        "epochs": int(bundle["epochs"]),
                        "sample_size": int(bundle["sample_size"]),
                        "val_ratio": float(bundle["val_ratio"]),
                        "in_feature_list": [int(x) for x in bundle["in_feature_list"]],
                        "lr": float(bundle["lr"]),
                    }
                )
        except Exception as exc:
            print(f"Skipping unreadable result file {path}: {exc}")

    rows.sort(
        key=lambda row: (
            r2_order.get(row["true_r2"], len(r2_order)),
            row["seed"],
        )
    )
    return rows


def summarize_rows(files):
    rows = []
    for row in files:
        with np.load(row["path"], allow_pickle=True) as bundle:
            r2_train = np.asarray(bundle["r2_train"], dtype=float)
            r2_test = np.asarray(bundle["r2_test"], dtype=float)

        rows.append(
            {
                "true_r2": row["true_r2"],
                "seed": row["seed"],
                "epochs": row["epochs"],
                "sample_size": row["sample_size"],
                "val_ratio": row["val_ratio"],
                "in_feature_list": str(row["in_feature_list"]),
                "lr": row["lr"],
                "r2_train": float(r2_train[-1]) if len(r2_train) else np.nan,
                "r2_test": float(r2_test[-1]) if len(r2_test) else np.nan,
            }
        )
    return rows


def write_summary_csv(files, output_dir):
    output_path = output_dir / "dnn_results_summary.csv"
    rows = summarize_rows(files)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, output_path)
    print(f"Saved {output_path}")
    return output_path


def grid_shape(n_items, ncols=5):
    if n_items <= 20 and ncols == 5:
        return 4, 5
    return max(1, ceil(n_items / ncols)), ncols


def save_figure(fig, output_dir, filename):
    output_path = output_dir / filename
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.png")
    fig.savefig(temp_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    os.replace(temp_path, output_path)
    print(f"Saved {output_path}")


def plot_r2_grid(group_rows, output_dir, which, ncols=5):
    group_rows = sorted(group_rows, key=lambda row: row["seed"])
    nrows, ncols = grid_shape(len(group_rows), ncols=ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    key = f"r2_{which}"

    for ax, row in zip(axes, group_rows):
        with np.load(row["path"], allow_pickle=True) as bundle:
            values = np.asarray(bundle[key], dtype=float)
        ax.plot(progress_epoch_axis(values, row["epochs"]), values, linewidth=1.0)
        ax.set_title(f"seed={row['seed']}", fontsize=9)
        ax.grid(alpha=0.25)

    for ax in axes[len(group_rows) :]:
        ax.axis("off")

    first = group_rows[0]
    label = run_label(first["true_r2"], first["epochs"], first["lr"])
    fig.suptitle(f"{label}, {which} R2", fontsize=12)
    save_figure(fig, output_dir, f"{label}_{which}_r2_grid.png")


def export_results(output_dir):
    files = result_files(output_dir)
    if not files:
        print(f"No DNN result files found in {output_dir}; skipping CSV/plots.")
        return

    write_summary_csv(files, output_dir)

    groups = {}
    for row in files:
        groups.setdefault((row["true_r2"], row["epochs"], row["lr"]), []).append(row)

    for key in DEFAULT_TRUE_R2_VALUES:
        matching_groups = [rows for group_key, rows in groups.items() if group_key[0] == key]
        for group_rows in matching_groups:
            plot_r2_grid(group_rows, output_dir, which="train")
            plot_r2_grid(group_rows, output_dir, which="test")


def seeds_to_run():
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return list(DEFAULT_SEEDS)

    seed = int(task_id)
    if seed not in DEFAULT_SEEDS:
        raise ValueError(f"SLURM_ARRAY_TASK_ID must be 1-20, got {seed}")
    return [seed]


def main():
    output_dir = Path(os.environ.get("DNN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)).expanduser().resolve()
    epochs = int(os.environ.get("DNN_EPOCHS", DEFAULT_EPOCHS))

    saved_paths = []
    for seed in seeds_to_run():
        for true_r2 in DEFAULT_TRUE_R2_VALUES:
            saved_paths.append(run_one(seed=seed, true_r2=true_r2, epochs=epochs, output_dir=output_dir))

    export_results(output_dir)

    print("Completed DNN runs.")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
