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
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BNN_STGP_unified_nn_classes import Experiment


DEFAULT_SEEDS = range(1, 21)
DEFAULT_TRUE_R2_VALUES = (0.3, 0.5, 0.8)
DEFAULT_A_FOR_SIGMA_SQUARED = 1000
DEFAULT_CR_CW_PRIOR_VALUES = (0.01, 1.0)
DEFAULT_EPOCHS = 600
DEFAULT_BURNIN_RATIOS = (0.9,)
DEFAULT_OUTPUT_DIR = THIS_DIR / "results"
MODALITY_SHAPES = [(50, 50), (100, 100)]
RUN_CONFIG = {
    "sample_size": 2000,
    "data_type": "simulation",
    "gamma": 0.1,
    "in_feature_list": [50, 100],
    "hidden_unit_list": [4],
    "step_gamma": 0.5,
    "step_decay_epoch": 100,
    "lr": 5e-6,
    "poly_degree": 5,
}
SUMMARY_FIELDNAMES = [
    "true_r2",
    "a_for_sigma_squared",
    "a_cr",
    "b_cr",
    "a_cw",
    "b_cw",
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


def format_float_for_path(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def burnin_label(value):
    return str(float(value)).replace(".", "p")


def result_path(output_dir, true_r2, a_for_sigma_squared, cr_cw_prior_value, seed, epochs, lr, poly_degree):
    return (
        output_dir
        / (
            f"results_r2_{format_float_for_path(true_r2)}_a_sigma_{format_float_for_path(a_for_sigma_squared)}_"
            f"cr_cw_{format_float_for_path(cr_cw_prior_value)}_epochs_{epochs}_"
            f"seed_{seed}_lr_{format_float_for_path(lr)}_polydegree_{poly_degree}.npz"
        )
    )


def run_label(row):
    return (
        f"r2_{format_float_for_path(row['true_r2'])}_a_sigma_{format_float_for_path(row['a_for_sigma_squared'])}_"
        f"cr_cw_{format_float_for_path(row['a_cr'])}_epochs_{row['epochs']}_"
        f"lr_{format_float_for_path(row['lr'])}_polydegree_{row['poly_degree']}"
    )


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else np.nan


def selection_metrics(confusion_matrix_values):
    cm = np.asarray(confusion_matrix_values)
    if cm.shape != (2, 2):
        return np.nan, np.nan, np.nan, np.nan

    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    accuracy = safe_divide(tp + tn, tn + fp + fn + tp)
    tpr = safe_divide(tp, tp + fn)
    fpr = safe_divide(fp, fp + tn)
    fdr = safe_divide(fp, fp + tp)
    return accuracy, tpr, fpr, fdr


def infer_shape(values, modality_idx):
    size = int(np.asarray(values).size)
    if modality_idx < len(MODALITY_SHAPES):
        shape = MODALITY_SHAPES[modality_idx]
        if shape[0] * shape[1] == size:
            return shape

    side = int(np.sqrt(size))
    if side * side == size:
        return side, side

    return 1, size


def progress_epoch_axis(values, epochs):
    values = np.asarray(values)
    if len(values) == 0:
        return np.array([])
    return np.linspace(1, epochs, len(values))


def get_results_by_burnin(bundle):
    return bundle["results_by_burnin"].item()


def get_result(bundle, burnin_ratio):
    return get_results_by_burnin(bundle)[str(float(burnin_ratio))]


def prior_lists(cr_cw_prior_value):
    return [cr_cw_prior_value, cr_cw_prior_value]


def run_one(seed, true_r2, a_for_sigma_squared, cr_cw_prior_value, epochs, burnin_ratios, output_dir):
    a_for_eigen_cr_list = prior_lists(cr_cw_prior_value)
    b_for_eigen_cr_list = prior_lists(cr_cw_prior_value)
    a_for_eigen_cw_list = prior_lists(cr_cw_prior_value)
    b_for_eigen_cw_list = prior_lists(cr_cw_prior_value)

    print(
        f"Starting true_r2={true_r2}, a_for_sigma_squared={a_for_sigma_squared}, "
        f"a_cr=b_cr=a_cw=b_cw={cr_cw_prior_value}, seed={seed}, epochs={epochs}"
    )

    experiment = Experiment()
    trainer = experiment.run_whole_experiment(
        sample_size=RUN_CONFIG["sample_size"],
        data_type=RUN_CONFIG["data_type"],
        seed=seed,
        true_r2=true_r2,
        gamma=RUN_CONFIG["gamma"],
        in_feature_list=RUN_CONFIG["in_feature_list"],
        a_for_eigen_cr_list=a_for_eigen_cr_list,
        b_for_eigen_cr_list=b_for_eigen_cr_list,
        a_for_eigen_cw_list=a_for_eigen_cw_list,
        b_for_eigen_cw_list=b_for_eigen_cw_list,
        a_for_sigma_squared=a_for_sigma_squared,
        hidden_unit_list=RUN_CONFIG["hidden_unit_list"],
        step_gamma=RUN_CONFIG["step_gamma"],
        step_decay_epoch=RUN_CONFIG["step_decay_epoch"],
        lr=RUN_CONFIG["lr"],
        epochs=epochs,
        poly_degree=RUN_CONFIG["poly_degree"],
    )

    results_by_burnin = {}
    for burnin_ratio in burnin_ratios:
        print(
            f"Processing true_r2={true_r2}, a_for_sigma_squared={a_for_sigma_squared}, "
            f"a_cr=b_cr=a_cw=b_cw={cr_cw_prior_value}, seed={seed}, burnin_ratio={burnin_ratio}"
        )
        results_by_burnin[str(float(burnin_ratio))] = experiment.process_results(
            trainer,
            burnin_ratio=burnin_ratio,
            gamma=RUN_CONFIG["gamma"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_path(
        output_dir=output_dir,
        true_r2=true_r2,
        a_for_sigma_squared=a_for_sigma_squared,
        cr_cw_prior_value=cr_cw_prior_value,
        seed=seed,
        epochs=epochs,
        lr=RUN_CONFIG["lr"],
        poly_degree=RUN_CONFIG["poly_degree"],
    )
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            true_r2=np.array(true_r2),
            a_for_sigma_squared=np.array(a_for_sigma_squared),
            a_cr=np.array(cr_cw_prior_value),
            b_cr=np.array(cr_cw_prior_value),
            a_cw=np.array(cr_cw_prior_value),
            b_cw=np.array(cr_cw_prior_value),
            a_for_eigen_cr_list=np.array(a_for_eigen_cr_list, dtype=float),
            b_for_eigen_cr_list=np.array(b_for_eigen_cr_list, dtype=float),
            a_for_eigen_cw_list=np.array(a_for_eigen_cw_list, dtype=float),
            b_for_eigen_cw_list=np.array(b_for_eigen_cw_list, dtype=float),
            seed=np.array(seed),
            epochs=np.array(epochs),
            lr=np.array(RUN_CONFIG["lr"]),
            poly_degree=np.array(RUN_CONFIG["poly_degree"]),
            burnin_ratios=np.array(burnin_ratios, dtype=float),
            results_by_burnin=np.array(results_by_burnin, dtype=object),
            r2_train=np.array(trainer.net.r2_train, dtype=float),
            r2_test=np.array(trainer.net.r2_test, dtype=float),
            loss_train=np.array(trainer.loss_train, dtype=float),
            loss_val=np.array(trainer.loss_val, dtype=float),
            accu_train=np.array(trainer.accu_train, dtype=float),
            accu_val=np.array(trainer.accu_val, dtype=float),
        )
    os.replace(temp_path, output_path)

    print(f"Saved {output_path}")
    return output_path


def result_files(output_dir):
    r2_order = {value: idx for idx, value in enumerate(DEFAULT_TRUE_R2_VALUES)}
    files = []
    for path in sorted(output_dir.glob("results_r2_*_a_sigma_*_cr_cw_*_epochs_*_seed_*_lr_*_polydegree_*.npz")):
        with np.load(path, allow_pickle=True) as bundle:
            files.append(
                {
                    "path": path,
                    "true_r2": float(bundle["true_r2"]),
                    "a_for_sigma_squared": float(bundle["a_for_sigma_squared"]),
                    "a_cr": float(bundle["a_cr"]),
                    "b_cr": float(bundle["b_cr"]),
                    "a_cw": float(bundle["a_cw"]),
                    "b_cw": float(bundle["b_cw"]),
                    "epochs": int(bundle["epochs"]),
                    "seed": int(bundle["seed"]),
                    "lr": float(bundle["lr"]),
                    "poly_degree": int(bundle["poly_degree"]),
                }
            )
    files.sort(
        key=lambda row: (
            row["a_cr"],
            r2_order.get(row["true_r2"], len(r2_order)),
            row["lr"],
            row["poly_degree"],
            row["seed"],
        )
    )
    return files


def summarize_file(row):
    rows = []
    with np.load(row["path"], allow_pickle=True) as bundle:
        for burnin_ratio in bundle["burnin_ratios"]:
            result = get_result(bundle, burnin_ratio)
            mean_parameter_r2 = result["mean_parameter_r2"]
            summary_row = {
                "true_r2": row["true_r2"],
                "a_for_sigma_squared": row["a_for_sigma_squared"],
                "a_cr": row["a_cr"],
                "b_cr": row["b_cr"],
                "a_cw": row["a_cw"],
                "b_cw": row["b_cw"],
                "epochs": row["epochs"],
                "seed": row["seed"],
                "lr": row["lr"],
                "poly_degree": row["poly_degree"],
                "burnin_ratio": float(burnin_ratio),
                "saved_r2_train_mean": float(np.nanmean(result["saved_r2_train"])),
                "saved_r2_test_mean": float(np.nanmean(result["saved_r2_test"])),
                "mean_param_r2_train": float(mean_parameter_r2["r2_train"]),
                "mean_param_r2_test": float(mean_parameter_r2["r2_test"]),
            }

            confusion_matrices = result.get("confusion_matrix_list", [])
            for modality_idx, prefix in enumerate(("M1", "M2")):
                if modality_idx < len(confusion_matrices):
                    accuracy, tpr, fpr, fdr = selection_metrics(confusion_matrices[modality_idx])
                else:
                    accuracy, tpr, fpr, fdr = np.nan, np.nan, np.nan, np.nan
                summary_row[f"{prefix} Accuracy"] = accuracy
                summary_row[f"{prefix} TPR"] = tpr
                summary_row[f"{prefix} FPR"] = fpr
                summary_row[f"{prefix} FDR"] = fdr

            rows.append(summary_row)
    return rows


def write_summary_csv(rows, output_dir):
    output_path = output_dir / "results_summary.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_path}")
    return output_path


def save_figure(fig, output_dir, filename):
    path = output_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def grid_shape(n_items, ncols=5):
    return max(1, ceil(n_items / ncols)), ncols


def plot_r2_grid(group_rows, output_dir, which, ncols=5):
    group_rows = sorted(group_rows, key=lambda row: row["seed"])
    nrows, ncols = grid_shape(len(group_rows), ncols=ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    values_key = f"r2_{which}"

    for ax, row in zip(axes, group_rows):
        with np.load(row["path"], allow_pickle=True) as bundle:
            values = np.asarray(bundle[values_key], dtype=float)
        ax.plot(progress_epoch_axis(values, row["epochs"]), values, linewidth=1.0)
        ax.set_title(f"seed={row['seed']}", fontsize=9)
        ax.grid(alpha=0.25)

    for ax in axes[len(group_rows) :]:
        ax.axis("off")

    label = run_label(group_rows[0])
    fig.suptitle(f"{label}, {which} R2 across seeds", fontsize=12)
    save_figure(fig, output_dir, f"{label}_{which}_r2_grid.png")


def plot_selection_grid(group_rows, output_dir, burnin_ratio, modality_idx, ncols=5):
    group_rows = sorted(group_rows, key=lambda row: row["seed"])
    nrows, ncols = grid_shape(len(group_rows), ncols=ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, row in zip(axes, group_rows):
        with np.load(row["path"], allow_pickle=True) as bundle:
            result = get_result(bundle, burnin_ratio)
            mask = np.asarray(result["mask_list"][modality_idx])
        ax.imshow(mask.reshape(infer_shape(mask, modality_idx)), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"seed={row['seed']}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(group_rows) :]:
        ax.axis("off")

    label = run_label(group_rows[0])
    save_figure(
        fig,
        output_dir,
        f"{label}_burnin_{burnin_label(burnin_ratio)}_M{modality_idx + 1}_selection_grid.png",
    )


def plot_aggregate_selection(group_rows, output_dir, burnin_ratio, modality_idx):
    masks = []
    for row in sorted(group_rows, key=lambda item: item["seed"]):
        with np.load(row["path"], allow_pickle=True) as bundle:
            result = get_result(bundle, burnin_ratio)
            masks.append(np.asarray(result["mask_list"][modality_idx], dtype=float))
    if not masks:
        return

    freq = np.mean(np.stack(masks, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(freq.reshape(infer_shape(freq, modality_idx)), cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"Modality {modality_idx + 1}: aggregate region selection")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    label = run_label(group_rows[0])
    save_figure(
        fig,
        output_dir,
        f"{label}_burnin_{burnin_label(burnin_ratio)}_M{modality_idx + 1}_aggregate_selection.png",
    )


def export_results(output_dir):
    files = result_files(output_dir)
    if not files:
        print(f"No result files found in {output_dir}; skipping CSV/plots.")
        return

    summary_rows = []
    for row in files:
        summary_rows.extend(summarize_file(row))
    write_summary_csv(summary_rows, output_dir)

    groups = {}
    for row in files:
        key = (
            row["true_r2"],
            row["a_for_sigma_squared"],
            row["a_cr"],
            row["b_cr"],
            row["a_cw"],
            row["b_cw"],
            row["epochs"],
            row["lr"],
            row["poly_degree"],
        )
        groups.setdefault(key, []).append(row)

    for group_rows in groups.values():
        plot_r2_grid(group_rows, output_dir, which="train")
        plot_r2_grid(group_rows, output_dir, which="test")

        with np.load(group_rows[0]["path"], allow_pickle=True) as bundle:
            burnin_ratios = [float(value) for value in bundle["burnin_ratios"]]
            n_modalities = len(get_result(bundle, burnin_ratios[0])["mask_list"])

        for burnin_ratio in burnin_ratios:
            for modality_idx in range(n_modalities):
                plot_selection_grid(group_rows, output_dir, burnin_ratio, modality_idx)
                plot_aggregate_selection(group_rows, output_dir, burnin_ratio, modality_idx)


def run_configs_to_run():
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return [
            (seed, true_r2, cr_cw_prior_value)
            for cr_cw_prior_value in DEFAULT_CR_CW_PRIOR_VALUES
            for true_r2 in DEFAULT_TRUE_R2_VALUES
            for seed in DEFAULT_SEEDS
        ]

    task_index = int(task_id) - 1
    total_tasks = len(DEFAULT_SEEDS) * len(DEFAULT_TRUE_R2_VALUES) * len(DEFAULT_CR_CW_PRIOR_VALUES)
    if not 0 <= task_index < total_tasks:
        raise ValueError(f"SLURM_ARRAY_TASK_ID must be 1-{total_tasks}, got {task_id}")

    prior_index, remainder = divmod(task_index, len(DEFAULT_TRUE_R2_VALUES) * len(DEFAULT_SEEDS))
    r2_index, seed_index = divmod(remainder, len(DEFAULT_SEEDS))
    seed = list(DEFAULT_SEEDS)[seed_index]
    true_r2 = DEFAULT_TRUE_R2_VALUES[r2_index]
    cr_cw_prior_value = DEFAULT_CR_CW_PRIOR_VALUES[prior_index]
    return [(seed, true_r2, cr_cw_prior_value)]


def main():
    output_dir = Path(os.environ.get("BNN_STGP_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)).expanduser().resolve()
    epochs = int(os.environ.get("BNN_STGP_EPOCHS", DEFAULT_EPOCHS))
    a_for_sigma_squared = float(os.environ.get("BNN_STGP_A_FOR_SIGMA_SQUARED", DEFAULT_A_FOR_SIGMA_SQUARED))
    burnin_ratios = tuple(
        float(value)
        for value in os.environ.get(
            "BNN_STGP_BURNIN_RATIOS",
            ",".join(str(value) for value in DEFAULT_BURNIN_RATIOS),
        ).split(",")
    )

    saved_paths = []
    for seed, true_r2, cr_cw_prior_value in run_configs_to_run():
        saved_paths.append(
            run_one(
                seed=seed,
                true_r2=true_r2,
                a_for_sigma_squared=a_for_sigma_squared,
                cr_cw_prior_value=cr_cw_prior_value,
                epochs=epochs,
                burnin_ratios=burnin_ratios,
                output_dir=output_dir,
            )
        )

    export_results(output_dir)

    print("Completed runs.")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
