import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics as a swarm plot based on number of squares",
)
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="avg_ll", help="The metric to plot")
parser.add_argument("--models", default="MonotonicPC;BornPC", help="The models")
# parser.add_argument("--xlabels", default="Number of components;Number of squares", help="The x-axis labels for each model")


def format_metric(m: str) -> str:
    if m == "avg_ll":
        return "Average LL"
    elif m == "bpd":
        return "Bits per dimension"
    elif m == "ppl":
        return "Perplexity"
    assert False


def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]
    return df


def format_model(m: str) -> str:
    if m == "MonotonicPC":
        return r"$+_{\mathsf{sd}}$"
    elif m == "BornPC":
        return r"$\Sigma_{\mathsf{sd}}^2$"
    assert False


def format_dataset(d: str) -> str:
    return {
        "power": "Power",
        "gas": "Gas",
        "hepmass": "Hepmass",
        "miniboone": "MiniBoonE",
        "bsds300": "BSDS300",
    }[d]


if __name__ == "__main__":
    args = parser.parse_args()
    metric = "Best/Test/" + args.metric
    models = args.models.split(";")
    df = retrieve_tboard_runs(args.tboard_path, metric)
    df = df[df["dataset"] == args.dataset]
    df = df[df["model"].isin(models)]
    df = df.sort_values("model", ascending=False)
    df["model"] = df["model"].apply(format_model)
    df["num_replicas"] = df["num_replicas"].astype(int)
    num_sum_parameters = df["num_sum_params"].tolist()
    rel_num_sum_parameters = (
        np.max(num_sum_parameters) - np.min(num_sum_parameters)
    ) / np.min(num_sum_parameters)
    print(f"Relative diff. num. of parameters: {rel_num_sum_parameters * 100:.1f}")
    num_rows = 1
    num_cols = 1

    setup_tueplots(num_rows, num_cols, rel_width=0.415, hw_ratio=0.8)
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    g = sb.swarmplot(
        df,
        x="num_replicas",
        y=metric,
        hue="model",
        ax=ax,
        dodge=True,
        alpha=0.55,
        marker="x",
        linewidth=1,
        legend="brief",
    )
    g.get_legend().set_title(None)
    sb.boxplot(
        df,
        x="num_replicas",
        y=metric,
        hue="model",
        ax=ax,
        dodge=False,
        fill=False,
        gap=0.25,
        whiskerprops={"visible": False},
        showfliers=False,
        showbox=False,
        showcaps=False,
        legend=False,
        zorder=999,
    )
    sb.move_legend(ax, handlelength=1.0, handletextpad=0.5, loc="best")
    ax.set_xlabel(r"Num. of $\mathrm{MPCs}$ / $\mathrm{NPC}^2\mathrm{s}$", fontsize=12)
    ax.set_ylabel(format_metric(args.metric), fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_title(format_dataset(args.dataset), fontsize=12)

    path = os.path.join("figures", "num-of-squares")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f"{args.dataset}.pdf"))
