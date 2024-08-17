import argparse
import os

import matplotlib.ticker as tck
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib import rcParams

from graphics.utils import setup_tueplots
from scripts.plots.utils import format_dataset, format_metric, preprocess_dataframe

parser = argparse.ArgumentParser(
    description="Plot benchmark results",
)
parser.add_argument(
    "csvdir", type=str, help="The directory containing CSV benchmark results"
)
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="bpd", help="The metric to plot")
parser.add_argument(
    "--legend", action="store_trure", default=False, help="Whether to show the legend"
)


def from_bytes_to_gib(bytes: int) -> float:
    return bytes / (1024.0 * 1024.0 * 1024.0)


if __name__ == "__main__":
    args = parser.parse_args()
    metric = (
        ("Best/Train/" + args.metric) if args.train else ("Best/Test/" + args.metric)
    )

    dfs = []
    for filename in os.listdir(args.csvdir):
        filepath = os.path.join(args.csvdir, filename)
        df = pd.read_csv(args.csv)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df = preprocess_dataframe(df)

    num_rows = 1
    num_cols = 1
    setup_tueplots(
        num_rows,
        num_cols,
        rel_width=0.5,
        hw_ratio=0.8,
    )
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    sb.scatterplot(
        df,
        x="time",
        y=metric,
        hue="gpu_memory",
        palette="model_id",
        legend=args.legend,
        alpha=0.8,
        ax=ax,
    )

    if args.ylabel:
        formatted_metric = format_metric(args.metric)
        if args.ylabel_horizontal:
            ax.annotate(
                formatted_metric,
                fontsize=9,
                xy=(0, 1.1),
                xytext=(-0.5 * rcParams["xtick.major.pad"], 1),
                ha="right",
                va="top",
                xycoords="axes fraction",
                textcoords="offset points",
            )
            ax.set_ylabel("")
        else:
            ax.set_ylabel(formatted_metric)
    else:
        ax.set_ylabel("")
    ax.set_title(format_dataset(args.dataset))
    if args.legend:
        if args.move_legend_outside:
            sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="")
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
    ax.grid(linestyle="--", which="major", alpha=0.3, linewidth=0.5)
    ax.grid(linestyle="--", which="minor", alpha=0.3, linewidth=0.3)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

    path = os.path.join("figures", "benchmarks")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}.pdf"
    plt.savefig(os.path.join(path, filename))
