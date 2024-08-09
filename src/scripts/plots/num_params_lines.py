import argparse
import os
from typing import Optional

import matplotlib.ticker as tck
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib import rcParams

from graphics.utils import setup_tueplots
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics by number of sum unit parameters from Tensorboard files",
)
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="avg_ll", help="The metric considered")
parser.add_argument("--models", default="MPC;SOS", help="The models")
parser.add_argument(
    "--sum-params-only",
    action="store_true",
    default=False,
    help="Whether to show the number of parameters of sum units only",
)
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
parser.add_argument(
    "--legend", action="store_true", default=False, help="Whether to show the legend"
)
parser.add_argument(
    "--move-legend-outside",
    action="store_true",
    default=False,
    help="Whether to move the legend outside",
)


def format_metric(m: str, train: Optional[bool] = None) -> str:
    if m == "avg_ll":
        m = "LL"
    elif m == "bpd":
        m = "BPD"
    elif m == "ppl":
        m = "PPL"
    else:
        assert False
    if train is None:
        return m
    if train:
        return f"{m} [train]"
    return f"{m} [test]"


def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]
    return df


def format_model(m: str, exp_alias: str, num_components: int) -> str:
    if m == "MPC":
        return r"$+_{\mathsf{sd}}$"
    elif m == "SOS":
        if "real" in exp_alias:
            if num_components > 1:
                return r"$\Sigma_{\mathsf{cmp}}^2 (\mathbb{R})$"
            return r"$\pm^2 (\mathbb{R})$"
        elif "complex" in exp_alias:
            if num_components > 1:
                return r"$\Sigma_{\mathsf{cmp}}^2 (\mathbb{C})$"
            return r"$\pm^2 (\mathbb{C})$"
    elif m == "ExpSOS":
        if "real" in exp_alias:
            return r"$+_{\mathsf{sd}}\!\cdot\!\pm^2 (\mathbb{R})$"
        elif "complex" in exp_alias:
            return r"$+_{\mathsf{sd}}\!\cdot\!\pm^2 (\mathbb{C})$"
    assert False


def format_model_order(m: str, exp_alias: str, num_components: int) -> (int, int):
    if m == "MPC":
        return 0, 0
    elif m == "SOS":
        if "real" in exp_alias:
            return 1, num_components
        elif "complex" in exp_alias:
            return 2, num_components
    elif m == "ExpSOS":
        if "real" in exp_alias:
            return 3, 0
        elif "complex" in exp_alias:
            return 4, 0
    assert False


def format_dataset(d: str) -> str:
    return {
        "power": "Power",
        "gas": "Gas",
        "hepmass": "Hepmass",
        "miniboone": "MiniBoonE",
        "bsds300": "BSDS300",
        "MNIST": "MNIST",
        "FashionMNIST": "Fashion-MNIST",
        "CIFAR10": "CIFAR-10",
    }[d]


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
    df = df.sort_values(valid_metric, ascending=False)
    df: pd.DataFrame = df.groupby(group_by_cols).first()
    df.reset_index(inplace=True)

    df["model_id"] = df.apply(
        lambda row: format_model(row.model, row.exp_alias, row.num_components), axis=1
    )
    df["model_order"] = df.apply(
        lambda row: format_model_order(row.model, row.exp_alias, row.num_components),
        axis=1,
    )
    df = df.sort_values(by="model_order", ascending=True)

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
        linewidth=1.2,
        markers=True,
        dashes=False,
        legend=args.legend,
        alpha=0.75,
        ax=ax,
    )
    ax.margins(0.1)
    ax.set_xscale("log")
    ax.set_xlabel(
        "Num. of sum parameters" if args.sum_params_only else "Num. of parameters"
    )
    if args.ylabel:
        formatted_metric = format_metric(
            args.metric, train=args.train if args.ylabel_detailed else None
        )
        if args.ylabel_horizontal:
            ax.annotate(
                formatted_metric,
                fontsize=9,
                xy=(0, 1.08),
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

    path = os.path.join("figures", "num-params")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}-train.pdf" if args.train else f"{args.dataset}-test.pdf"
    plt.savefig(os.path.join(path, filename))
