import argparse
import gc
import importlib.util
import logging
import os
import time
from enum import Enum
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
import yaml
from memory_profiler import memory_usage  # type: ignore

use_submission = importlib.util.find_spec("submission") is not None
if use_submission:
    from submission import denoising_score_matching_objective, score_matching_objective


# Set up logging
logging.basicConfig(level=logging.WARNING)


class RunGoal(Enum):
    """Enumeration for run tracking goals."""

    TIME_TRACK = "time_track"
    MEMORY_TRACK = "memory_track"


def measure_peak_memory_usage(func: Callable, *args: Tuple, **kwargs: Dict) -> float:
    """Measures peak memory usage during function execution.

    Args:
        func (Callable): The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        float: The maximum memory usage in MiB.
    """
    mem_usage = memory_usage((func, args, kwargs))
    return max(mem_usage)


def run_track(run_goal: RunGoal, func: Callable, *args: Tuple, **kwargs: Dict) -> float:
    """Runs a function with specified tracking goals.

    Args:
        run_goal (RunGoal): The goal for tracking (TIME_TRACK or MEMORY_TRACK).
        func (Callable): The function to execute.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        float: The result of the function execution, either time or memory usage.

    Raises:
        ValueError: If an invalid run goal is provided.
    """
    if run_goal not in RunGoal:
        logging.error(f"Invalid run goal: {run_goal}")
        raise ValueError(f"Invalid run goal: {run_goal}")

    try:
        if run_goal == RunGoal.TIME_TRACK:
            start_time = time.time()
            func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            return elapsed_time
        elif run_goal == RunGoal.MEMORY_TRACK:
            return measure_peak_memory_usage(func, *args, **kwargs)
    except Exception as e:
        logging.exception("Error during %s: %s", run_goal, str(e))
        raise


# Run Experiments
def run_experiment(
    dimensions: List[int],
    batch_sizes: List[int],
    objective_func: Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor],
    runs: Dict[str, int],
    output_file: str,
    experiment_name: str,
    return_results: bool = False,
) -> Union[None, Dict]:
    """Runs the specified experiment and collects results.

    Args:
        dimensions (List[int]): List of dimensions to test.
        batch_sizes (List[int]): List of batch sizes to test.
        objective_func (Callable[[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]): The objective function to evaluate.
        runs (Dict[str, int]): Dictionary of run types and their counts.
        output_file (str): Path to save the output results.
        experiment_name (str): Name of the experiment.
    """
    results = {dim: {"times": [], "memory_usages": []} for dim in dimensions}

    for run in runs.keys():
        for dim in dimensions:
            for batch_size in batch_sizes:

                x = torch.randn(batch_size, dim * dim, requires_grad=True)
                theta = {
                    "mean": torch.zeros(dim * dim),
                    "log_var": torch.zeros(dim * dim),
                }

                gc.collect()

                elapsed_time = run_track(RunGoal[run.upper()], objective_func, x, theta)

                if run == "time_track":
                    print(
                        f"Time for Dimension: {dim}, Batch Size: {batch_size}, time elapsed: {elapsed_time:.4f}s"
                    )
                    results[dim]["times"].append(elapsed_time)
                elif run == "memory_track":
                    print(
                        f"Memory for Dimension: {dim}, Batch Size: {batch_size}, Memory: {elapsed_time} MiB"
                    )
                    results[dim]["memory_usages"].append(elapsed_time)

    if return_results:
        return results
    else:
        plot_results(
            results,
            dimensions,
            batch_sizes,
            output_file,
            experiment_name=experiment_name,
        )


# Plot Results
def plot_results(
    results: Dict[int, Dict[str, List[float]]],
    dimensions: List[int],
    batch_sizes: List[int],
    output_file: str,
    experiment_name: str = "denoising",
) -> None:
    """Plots the results of the experiments.

    Args:
        results (Dict[int, Dict[str, List[float]]]): Collected results from the experiments.
        dimensions (List[int]): List of dimensions tested.
        batch_sizes (List[int]): List of batch sizes tested.
        output_file (str): Path to save the plot.
        experiment_name (str): Name of the experiment for labeling.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for dim in dimensions:
        bz = batch_sizes
        plt.plot(bz, results[dim]["times"], marker="o", label=f"{dim}x{dim}")
    plt.xlabel("Batch Size")
    plt.ylabel("Time (seconds)")
    plt.title("Time vs Batch Size")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for dim in dimensions:
        bz = batch_sizes
        plt.plot(bz, results[dim]["memory_usages"], marker="o", label=f"{dim}x{dim}")
    plt.xlabel("Batch Size")
    plt.ylabel("Memory Usage (MiB)")
    plt.title("Memory Usage vs Batch Size")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Score Matching experiments.")
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="If set, run the denoising score matching experiment.",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    config_path = os.path.join(os.path.dirname(__file__), "score_matching_config.yml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)

    if args.denoise:
        print("Running Denoising Score Matching...")
        run_experiment(
            config["dimensions"],
            config["batch_sizes"],
            denoising_score_matching_objective,
            config["runs"],
            os.path.join(config["output_dir"], "denoising_results.png"),
            experiment_name="denoising",
        )
    else:
        print("Running Score Matching...")
        run_experiment(
            config["dimensions"],
            config["batch_sizes"],
            score_matching_objective,
            config["runs"],
            os.path.join(config["output_dir"], "score_matching_results.png"),
            experiment_name="exact",
        )
