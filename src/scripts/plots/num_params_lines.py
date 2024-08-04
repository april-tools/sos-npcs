import argparse
import os

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots
from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics by number of sum unit parameters from Tensorboard files",
)
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="avg_ll", help="The metric considered")
parser.add_argument("--models", default="SOS;ExpSOS", help="The models")
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
    "--legend", action="store_true", default=False, help="Whether to show the legend"
)


def format_metric(m: str, train: bool = False) -> str:
    if m == "avg_ll":
        m = "Average LL"
    elif m == "bpd":
        m = "Bits per dimension"
    elif m == "ppl":
        m = "Perplexity"
    else:
        assert False
    if train:
        m = f"{m} [train]"
    else:
        m = f"{m} [test]"
    return m


def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]
    return df


def format_model(m: str, exp_alias: str) -> str:
    if m == "MPC":
        return r"$+_{\mathsf{sd}}$"
    elif m == "SOS":
        if "real" in exp_alias:
            return r"$\Sigma_{\mathsf{cmp}}^2 (\mathbb{R})$"
        elif "complex" in exp_alias:
            return r"$\Sigma_{\mathsf{cmp}}^2 (\mathbb{C})$"
    elif m == "ExpSOS":
        if "real" in exp_alias:
            return r"$+_{\mathsf{sd}}\!\cdot\!\pm^2 (\mathbb{R})$"
        elif "complex" in exp_alias:
            return r"$+_{\mathsf{sd}}\!\cdot\!\pm^2 (\mathbb{C})$"
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
        "mono_num_units",
        "mono_num_input_units",
    ]
    cols_to_keep = (
        group_by_cols + metrics + ["learning_rate", "num_sum_params", "num_params"]
    )
    df = df.drop(df.columns.difference(cols_to_keep), axis=1)
    df = df.sort_values(by=valid_metric, ascending=False)
    df: pd.DataFrame = df.groupby(group_by_cols).first()
    df.reset_index(inplace=True)

    df["model_id"] = df.apply(
        lambda row: format_model(row.model, row.exp_alias), axis=1
    )
    df = df.sort_values("model_id", ascending=False)
    num_rows = 1
    num_cols = 1

    setup_tueplots(
        num_rows,
        num_cols,
        rel_width=0.4,
        hw_ratio=0.8,
        tight_layout=False,
        constrained_layout=False,
    )
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    g = sb.lineplot(
        df,
        x="num_sum_params",
        y=metric,
        hue="model_id",
        style="model_id",
        linewidth=1.2,
        markers=True,
        legend=args.legend,
        alpha=0.75,
        ax=ax,
    )
    ax.margins(0.1)
    ax.set_xscale("log")
    ax.set_xlabel("Num of sum param")
    if args.ylabel:
        ax.set_ylabel(format_metric(args.metric, train=args.train))
    else:
        ax.set_ylabel("")
    ax.set_title(format_dataset(args.dataset))
    if args.legend:
        sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="")

    path = os.path.join("figures", "num-params")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}-train.pdf" if args.train else f"{args.dataset}-test.pdf"
    plt.savefig(os.path.join(path, filename))
