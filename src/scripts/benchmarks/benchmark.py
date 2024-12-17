import argparse
import gc
import os
from typing import List, Tuple, Union
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader

from datasets.loaders import IMAGE_DATASETS
from models import PC
from scripts.logger import Logger
from scripts.utils import (
    retrieve_tboard_runs,
    set_global_seed,
    setup_data_loaders,
    setup_model,
)
from utilities import PCS_MODELS

parser = argparse.ArgumentParser(description="Benchmarking script")
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument(
    "dataset", type=str, choices=IMAGE_DATASETS, help="The image dataset"
)
parser.add_argument("model", type=str, choices=PCS_MODELS, help="The PC to benchmark")
parser.add_argument(
    "--data-path", type=str, default="datasets", help="The data sets directory"
)
parser.add_argument(
    "--num-iterations",
    type=int,
    default=50,
    help="The number of iterations",
)
parser.add_argument(
    "--burnin-iterations",
    type=int,
    default=5,
    help="Burnin iterations (additional to --num-iterations)",
)
parser.add_argument("--device", type=str, default="cuda", help="The device id")
parser.add_argument("--batch-size", type=int, default=512, help="The batch size")
parser.add_argument(
    "--complex",
    action="store_true",
    default=False,
    help="Whether to use complex parameters",
)
parser.add_argument(
    "--num-units",
    type=str,
    default="32 64 128 256",
    help="A numbers of units in each layer to benchmark, separated by space",
)
parser.add_argument(
    "--mono-num-units",
    type=int,
    default=8,
    help="The number of units in monotonic PCs, for ExpSOS models only",
)
parser.add_argument(
    "--min-bubble-radius", type=float, default=20.0, help="Bubble sizes minimum"
)
parser.add_argument(
    "--scale-bubble-radius", type=float, default=1.0, help="Bubble sizes scaler"
)
parser.add_argument(
    "--exp-bubble-radius",
    type=float,
    default=2.0,
    help="The exponent for computing the bubble sizes",
)
parser.add_argument(
    "--backprop",
    action="store_true",
    default=False,
    help="Whether to benchmark also backpropagation",
)
parser.add_argument("--metric", type=str, default="bpd", help="The test metric to log")
parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")


def run_benchmark(
    model: PC,
    data_loader: DataLoader,
    *,
    device: torch.device,
    num_iterations: int,
    burnin_iterations: int = 1,
    backprop: bool = False,
    partition_function_only: bool = False,
) -> tuple[list[float], list[float]]:
    def infinite_dataloader() -> Iterator[list[Tensor] | tuple[Tensor] | Tensor]:
        while True:
            yield from data_loader

    model = model.to(device)

    if backprop:
        # Setup losses and a dummy optimizer (only used to free gradient tensors)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = None

    elapsed_times = list()
    gpu_memory_peaks = list()
    for i, batch in enumerate(infinite_dataloader()):
        if i == num_iterations + burnin_iterations:
            break
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        # Run GC manually and then disable it
        gc.collect()
        gc.disable()
        # Reset peak memory usage statistics
        torch.cuda.reset_peak_memory_stats(device)
        # torch.cuda.synchronize(device)  # Synchronize CUDA operations
        batch = batch.to(device)
        # torch.cuda.synchronize(device)  # Make sure the batch is already loaded (do not take into account this!)
        # start_time = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))
        if partition_function_only:
            lls = model.log_partition()
        else:
            if len(batch.shape) < 3:
                batch = batch.unsqueeze(dim=1)
            lls = model.log_likelihood(batch)
        if backprop:
            loss = -lls.mean()
            loss.backward(retain_graph=False)  # Free the autodiff graph
        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)  # Synchronize CUDA Kernels before measuring time
        # end_time = time.perf_counter()
        gpu_memory_peaks.append(torch.cuda.max_memory_allocated(device))

        if backprop:
            assert optimizer is not None
            optimizer.zero_grad()  # Free gradients tensors
        gc.enable()  # Enable GC again
        gc.collect()  # Manual GC
        # elapsed_times.append(end_time - start_time)
        elapsed_times.append(start.elapsed_time(end) * 1e-3)

    # Discard burnin iterations and compute averages
    elapsed_times = elapsed_times[burnin_iterations:]
    gpu_memory_peaks = gpu_memory_peaks[burnin_iterations:]
    return elapsed_times, gpu_memory_peaks


if __name__ == "__main__":
    args = parser.parse_args()
    set_global_seed(args.seed)
    num_units_ls = list(map(int, args.num_units.split()))

    metric = "Best/Test/" + args.metric
    df = retrieve_tboard_runs(os.path.join(args.tboard_path, args.dataset), metric)
    df = df[df["model"] == args.model]
    if "SOS" in args.model:
        df = df[
            (
                (df["exp_alias"] == "complex")
                if args.complex
                else (df["exp_alias"] == "real")
            )
        ]

    logger = Logger("benchmark", verbose=True)
    metadata, (data_loader, _, _) = setup_data_loaders(
        args.dataset,
        args.data_path,
        logger,
        batch_size=args.batch_size,
        num_workers=12,
        drop_last=True,
    )
    device = torch.device(args.device)

    benchmark_results = []
    for num_units in num_units_ls:
        if args.model == "ExpSOS":
            conditioned_df = df[
                (df["num_units"] == num_units)
                & (df["num_input_units"] == num_units)
                & (df["mono_num_units"] == args.mono_num_units)
                & (df["mono_num_input_units"] == args.mono_num_units)
            ]
        else:
            conditioned_df = df[
                (df["num_units"] == num_units) & (df["num_input_units"] == num_units)
            ]
        metric_value = conditioned_df[metric].mean()

        model = setup_model(
            args.model,
            metadata,
            logger,
            region_graph="qt",
            num_components=1,
            num_units=num_units,
            mono_num_units=args.mono_num_units,
            mono_clamp=True if args.model in ["MPC", "ExpSOS"] else False,
            complex=args.complex,
            seed=args.seed,
        )

        try:
            elapsed_times, gpu_memory_peaks = run_benchmark(
                model,
                data_loader,
                device=device,
                num_iterations=args.num_iterations,
                burnin_iterations=args.burnin_iterations,
                backprop=args.backprop,
                partition_function_only=False,
            )
            mu_time = np.mean(elapsed_times)
            peak_gpu_memory = np.max(gpu_memory_peaks)
        except torch.cuda.OutOfMemoryError:
            mu_time, peak_gpu_memory = np.nan, np.nan
        del model

        benchmark_results.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "exp_alias": (
                    ("complex" if args.complex else "real")
                    if "SOS" in args.model
                    else ""
                ),
                "time": mu_time,
                "gpu_memory": peak_gpu_memory,
                "num_components": 1,
                "num_units": num_units,
                metric: metric_value,
            }
        )

    path = os.path.join("benchmarks", args.dataset)
    os.makedirs(path, exist_ok=True)
    filename = "-".join(
        [args.model]
        + ((["complex"] if args.complex else ["real"]) if "SOS" in args.model else [])
    )
    filepath = os.path.join(path, f"{filename}.csv")
    pd.DataFrame.from_dict(benchmark_results).to_csv(filepath)
