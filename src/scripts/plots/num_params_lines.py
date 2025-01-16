import argparse
import os

import matplotlib.ticker as tck
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib import rcParams

from graphics.utils import setup_tueplots
from scripts.plots.utils import format_dataset, format_metric, preprocess_dataframe
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics by number of parameters from Tensorboard files",
)
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="avg_ll", help="The metric considered")
parser.add_argument("--models", default="MPC;SOS;ExpSOS", help="The models")
parser.add_argument(
    "--sum-params-only",
    action="store_true",
    default=False,
    help="Whether to show the number of parameters of sum units only",
)
parser.add_argument(
    "--select-best-run",
    action="store_true",
    default=False,
    help="Whether to select the best run",
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
    train_metric = "Best/Train/" + args.metric
    valid_metric = "Best/Valid/" + args.metric
    test_metric = "Best/Test/" + args.metric
    metrics = [train_metric, valid_metric, test_metric]
    metric = train_metric if args.train else test_metric
    models = args.models.split(";")
    df = retrieve_tboard_runs(os.path.join(args.tboard_path, args.dataset), metrics)
    df = df[df["dataset"] == args.dataset]
    df = df[df["model"].isin(models)]
    group_by_cols = [
        "seed",
        "dataset",
        "model",
        "exp_alias",
        "num_components",
        "num_units",
        "num_input_units",
        "mono_num_units",
        "mono_num_input_units",
        "learning_rate",
    ]
    cols_to_keep = group_by_cols + metrics + ["num_sum_params", "num_params"]
    df = df.drop(df.columns.difference(cols_to_keep), axis=1)
    if args.select_best_run:
        df = df.sort_values(valid_metric, ascending=False)
        df: pd.DataFrame = df.groupby(group_by_cols).first()
        df.reset_index(inplace=True)
    df = preprocess_dataframe(df)

    num_rows = 1
    num_cols = 1
    setup_tueplots(
        num_rows,
        num_cols,
        rel_width=0.375,
        hw_ratio=0.8,
        tight_layout=False,
        constrained_layout=False,
    )
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    num_params = "num_sum_params" if args.sum_params_only else "num_params"
    g = sb.lineplot(
        df,
        x=num_params,
        y=metric,
        hue="model_id",
        style="model_id",
        errorbar=("ci", 100),
        linewidth=1.2,
        markers=True,
        dashes=False,
        legend=args.legend,
        alpha=0.75,
        ax=ax,
    )
    ax.margins(0.1)
    ax.set_xscale("log")
    # ax.set_xlabel(
    #     "Num. of sum parameters" if args.sum_params_only else "Num. of parameters"
    # )
    ax.set_xlabel("")
    if args.xlabel:
        ax.annotate(
            r"\# sum params" if args.sum_params_only else r"\# params",
            xy=(1.05, 0),
            xytext=(1, -1.5 * rcParams["xtick.major.pad"]),
            ha="left",
            va="top",
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=11
        )
    if args.ylabel:
        formatted_metric = format_metric(
            args.metric, train=args.train if args.ylabel_detailed else None
        )
        if args.ylabel_horizontal:
            ax.annotate(
                formatted_metric,
                xy=(0.0, 1.0),
                #xytext=(-0.5 * rcParams["xtick.major.pad"], 1),
                fontsize=11,
                ha="center",
                va="bottom",
                xycoords="axes fraction",
                #textcoords="offset points",
            )
            ax.set_ylabel("")
        else:
            ax.set_ylabel(formatted_metric)
    else:
        ax.set_ylabel("")
    ax.set_title(format_dataset(args.dataset))
    if args.legend:
        if args.move_legend_outside:
            sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="", handlelength=1.0, handletextpad=0.5)
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, handlelength=1.0, handletextpad=0.5)
    ax.grid(linestyle="--", which="major", alpha=0.3, linewidth=0.5)
    ax.grid(linestyle="--", which="minor", alpha=0.3, linewidth=0.3)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

    path = os.path.join("figures", "num-params")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}-train.pdf" if args.train else f"{args.dataset}-test.pdf"
    plt.savefig(os.path.join(path, filename))
