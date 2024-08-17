import argparse
import os
from typing import Union

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
    "csvpath", type=str, help="The directory containing CSV benchmark results"
)
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="bpd", help="The metric to plot")
parser.add_argument(
    "--xlabel",
    action="store_true",
    default=False,
    help="Whether to show the x-axis label",
)
parser.add_argument(
    "--ylabel",
    action="store_true",
    default=False,
    help="Whether to show the y-axis label",
)
parser.add_argument(
    "--ylabel-horizontal",
    action="store_true",
    default=False,
    help="Whether to rotate the y-axis label horizontally",
)
parser.add_argument(
    "--legend", action="store_true", default=False, help="Whether to show the legend"
)
parser.add_argument(
    "--move-legend-outside",
    action="store_true",
    default=False,
    help="Whether to move the legend outside",
)


def from_bytes_to_gib(bytes: Union[int, float]) -> float:
    return bytes / (1024.0 * 1024.0 * 1024.0)


if __name__ == "__main__":
    args = parser.parse_args()
    metric = "Best/Test/" + args.metric

    dfs = []
    for filename in os.listdir(os.path.join(args.csvpath, args.dataset)):
        filepath = os.path.join(args.csvpath, args.dataset, filename)
        df = pd.read_csv(filepath)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df = preprocess_dataframe(df)
    df["gpu_memory"] = df.apply(
        lambda row: from_bytes_to_gib(row.gpu_memory),
        axis=1,
    )

    num_rows = 1
    num_cols = 1
    setup_tueplots(
        num_rows,
        num_cols,
        rel_width=0.55,
        hw_ratio=0.67,
    )
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    sb.scatterplot(
        df,
        x="time",
        y=metric,
        size="gpu_memory",
        hue="model_id",
        legend="brief" if args.legend else False,
        alpha=0.7,
        sizes=(36, 512),
        ax=ax,
    )
    sb.scatterplot(
        df,
        x="time",
        y=metric,
        hue="model_id",
        style="model_id",
        s=8,
        legend=False,
        alpha=0.7,
        ax=ax,
    )
    ax.margins(x=0.1, y=0.16)
    ax.set_xscale("log")
    ax.set_xlabel("")
    if args.xlabel:
        ax.annotate(
            r"time\,(s) / optimization step",
            fontsize=9,
            xy=(1.05, 0),
            xytext=(1, -1.5 * rcParams["xtick.major.pad"]),
            ha="left",
            va="top",
            xycoords="axes fraction",
            textcoords="offset points",
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
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 11:
                del handles[10]
                del labels[10]
            if len(handles) > 8:
                del handles[8]
                del labels[8]
            del handles[5]
            del labels[5]
            del handles[0]
            del labels[0]

            ax.legend(
                loc="upper left",
                handles=handles,
                labels=labels,
                bbox_to_anchor=(1, 1),
                fontsize=8,
                title=r"Class \quad Memory (GiB)",
                ncols=2,
                handleheight=3.2,
                alignment="left",
            )
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
