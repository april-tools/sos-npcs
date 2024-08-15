import argparse
import os

import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib import rcParams

from graphics.utils import setup_tueplots
from scripts.plots.utils import format_metric, format_dataset, preprocess_dataframe
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics as a swarm plot based on number of squares",
)
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="avg_ll", help="The metric to plot")
parser.add_argument("--models", default="MPC;SOS", help="The models")
parser.add_argument(
    "--plot-single-squares", action='store_true', default=False,
    help="Whether to also show single squared PC"
)
parser.add_argument(
    "--train",
    action="store_true",
    default=False,
    help="Whether to show the metric on the training data",
)
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
parser.add_argument(
    "--legend", action="store_true", default=False, help="Whether to show the legend"
)
parser.add_argument(
    "--move-legend-outside",
    action="store_true",
    default=False,
    help="Whether to move the legend outside",
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
    if not args.plot_single_squares:
        df = df[df["num_components"] > 1]
    df["num_components"] = df["num_components"].astype(int)
    df = preprocess_dataframe(df, merge_model_ids=args.plot_single_squares)

    num_rows = 1
    num_cols = 1
    rel_width = 0.45
    hw_ratio = 0.775 if args.plot_single_squares else 0.7
    setup_tueplots(num_rows, num_cols, rel_width=rel_width, hw_ratio=hw_ratio)
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    g = sb.boxplot(
        df,
        x="num_components",
        y=metric,
        hue="model_id",
        width=0.65,
        whis=2.0,
        linewidth=0.4,
        medianprops={"linewidth": 0.5},
        boxprops={"alpha": 0.8},
        fliersize=2.5,
        flierprops={"marker": "x", "alpha": 0.7},
        ax=ax,
        legend=args.legend,
    )
    # sb.move_legend(ax, handlelength=1.0, handletextpad=0.5, loc="best")
    if args.legend:
        if args.move_legend_outside and not args.plot_single_squares:
            sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="")
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
    if args.xlabel:
        ax.set_xlabel(r"Num. of components")
    else:
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
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(linestyle="--", which="major", alpha=0.3, linewidth=0.5)
    # ax.grid(linestyle="--", which="minor", alpha=0.3, linewidth=0.3)
    ax.set_title(format_dataset(args.dataset))

    path = os.path.join("figures", "sum-of-squares")
    os.makedirs(path, exist_ok=True)
    if args.train:
        flags = 'train'
    else:
        flags = 'test'
    if args.plot_single_squares:
        flags = f'{flags}-compact'
    filename = f"{args.dataset}-{flags}.pdf"
    plt.savefig(os.path.join(path, filename))
