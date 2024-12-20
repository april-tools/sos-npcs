import argparse
import os

import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib import rcParams

from graphics.utils import setup_tueplots
from scripts.plots.utils import (
    format_dataset,
    format_metric,
    format_model,
    preprocess_dataframe,
)
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics as a swarm plot based on number of squares",
)
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="avg_ll", help="The metric to plot")
parser.add_argument("--models", default="MPC;SOS;SOS", help="The models")
parser.add_argument(
    "--train",
    action="store_true",
    default=False,
    help="Whether to show the metric on the training data",
)
parser.add_argument(
    "--ylabel",
    action="store_true",
    default=False,
    help="Whether to show the y-axis label",
)
parser.add_argument(
    "--ylabel-detailed",
    action="store_true",
    default=False,
    help="Whether to show a more detailed y-axis label",
)
parser.add_argument(
    "--ylabel-horizontal",
    action="store_true",
    default=False,
    help="Whether to rotate the y-axis label horizontally",
)


if __name__ == "__main__":
    args = parser.parse_args()
    metric = (
        ("Best/Train/" + args.metric) if args.train else ("Best/Test/" + args.metric)
    )
    models = args.models.split(";")
    df = retrieve_tboard_runs(os.path.join(args.tboard_path, args.dataset), metric)
    df = df[df["dataset"] == args.dataset]
    df = df[df["model"].isin(models)]
    df = df[df["num_components"] == 1]
    df = preprocess_dataframe(df)

    num_rows = 1
    num_cols = 1
    setup_tueplots(num_rows, num_cols, rel_width=0.375, hw_ratio=0.84)
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    g = sb.boxplot(
        df,
        x="model_id",
        y=metric,
        hue="model_id",
        width=0.65,
        linewidth=0.5,
        fliersize=3.0,
        whis=2.0,
        ax=ax,
    )
    ax.set_xlabel("")
    if args.ylabel:
        formatted_metric = format_metric(
            args.metric, train=args.train if args.ylabel_detailed else None
        )
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
    ax.grid(linestyle="--", which="major", alpha=0.3, linewidth=0.5)
    # ax.grid(linestyle="--", which="minor", alpha=0.3, linewidth=0.3)
    ax.set_title(format_dataset(args.dataset))

    path = os.path.join("figures", "complex-squared-npcs")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}-train.pdf" if args.train else f"{args.dataset}-test.pdf"
    plt.savefig(os.path.join(path, filename))
