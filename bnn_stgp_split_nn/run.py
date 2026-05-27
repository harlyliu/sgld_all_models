import os
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from BNN_STGP_split_nn import Experiment


DEFAULT_SEEDS = range(1, 21)
DEFAULT_EPOCHS = 600
DEFAULT_BURNIN_RATIOS = (0.9,)
DEFAULT_OUTPUT_DIR = THIS_DIR / "newest_results"
TEST_CONFIGS = [
    {"test_name": "r2_sweep_r2_0p8", "true_r2": 0.8, "lr": 5e-6, "poly_degree": 5},
    {"test_name": "r2_sweep_r2_0p5", "true_r2": 0.5, "lr": 5e-6, "poly_degree": 5},
    {"test_name": "r2_sweep_r2_0p3", "true_r2": 0.3, "lr": 5e-6, "poly_degree": 5},
]


def format_float_for_path(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_one(seed, epochs, burnin_ratios, output_dir, test_name, true_r2, lr, poly_degree):
    print(
        f"Starting test={test_name}, seed={seed}, epochs={epochs}, "
        f"true_r2={true_r2}, lr={lr}, poly_degree={poly_degree}"
    )

    experiment = Experiment()
    trainer = experiment.run_whole_experiment(
        sample_size=2000,
        data_type="simulation",
        seed=seed,
        true_r2=true_r2,
        gamma=0.1,
        in_feature_list=[50, 100],
        a_for_eigen_cr_list=[0.0001, 0.0001],
        b_for_eigen_cr_list=[50, 50],
        a_for_eigen_cw_list=[0.0001, 0.0001],
        b_for_eigen_cw_list=[50, 50],
        hidden_unit_list=[4],
        step_gamma=0.5,
        step_decay_epoch=100,
        lr=lr,
        epochs=epochs,
        poly_degree=poly_degree,
    )

    results_by_burnin = {}
    for burnin_ratio in burnin_ratios:
        print(
            f"Processing test={test_name}, seed={seed}, epochs={epochs}, "
            f"true_r2={true_r2}, lr={lr}, poly_degree={poly_degree}, "
            f"burnin_ratio={burnin_ratio}"
        )
        results_by_burnin[str(burnin_ratio)] = experiment.process_results(
            trainer,
            burnin_ratio=burnin_ratio,
            gamma=0.1,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    lr_label = format_float_for_path(lr)
    output_path = (
        output_dir
        / f"results_{test_name}_epochs_{epochs}_seed_{seed}_lr_{lr_label}_polydegree_{poly_degree}.npz"
    )
    np.savez_compressed(
        output_path,
        test_name=np.array(test_name),
        seed=np.array(seed),
        epochs=np.array(epochs),
        true_r2=np.array(true_r2),
        lr=np.array(lr),
        poly_degree=np.array(poly_degree),
        burnin_ratios=np.array(burnin_ratios, dtype=float),
        results_by_burnin=np.array(results_by_burnin, dtype=object),
        r2_train=np.array(trainer.net.r2_train, dtype=float),
        r2_test=np.array(trainer.net.r2_test, dtype=float),
        loss_train=np.array(trainer.loss_train, dtype=float),
        loss_val=np.array(trainer.loss_val, dtype=float),
        accu_train=np.array(trainer.accu_train, dtype=float),
        accu_val=np.array(trainer.accu_val, dtype=float),
    )

    print(f"Saved {output_path}")
    return output_path


def seeds_to_run():
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None:
        return list(DEFAULT_SEEDS)

    seed = int(task_id)
    if seed not in DEFAULT_SEEDS:
        raise ValueError(f"SLURM_ARRAY_TASK_ID must be 1-20, got {seed}")
    return [seed]


def main():
    output_dir = DEFAULT_OUTPUT_DIR.expanduser().resolve()

    saved_paths = []
    for seed in seeds_to_run():
        for config in TEST_CONFIGS:
            saved_paths.append(
                run_one(
                    seed=seed,
                    epochs=DEFAULT_EPOCHS,
                    burnin_ratios=DEFAULT_BURNIN_RATIOS,
                    output_dir=output_dir,
                    **config,
                )
            )

    print("Completed all runs.")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
