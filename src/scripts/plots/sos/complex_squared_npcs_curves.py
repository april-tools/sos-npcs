import argparse
import os
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots
from scripts.utils import retrieve_tboard_runs


parser = argparse.ArgumentParser(
    description="Plot metrics as a swarm plot based on number of squares",
)
parser.add_argument("checkpoint_path", type=str, help="The checkpoints path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--metric", default="avg_ll", help="The metric to plot")
parser.add_argument("--models", default="MPC;SOS;SOS", help="The models")
parser.add_argument("--train", action='store_true', default=False, help="Whether to show the metric on the training data")
parser.add_argument("--ylabel", action='store_true', default=False, help="Whether to show the y-axis label")


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
        if exp_alias == 'real':
            return r"$\pm^2 (\mathbb{R})$"
        elif exp_alias == 'complex':
            return r"$\pm^2 (\mathbb{C})$"
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
    metric = ("Best/Train/" + args.metric) if args.train else ("Best/Test/" + args.metric)
    models = args.models.split(";")
    df = retrieve_tboard_runs(os.path.join(args.checkpoint_path, args.dataset), metric)
    df = df[df["dataset"] == args.dataset]
    df = df[df["model"].isin(models)]
    df = df.sort_values("model", ascending=True)
    df.to_csv('hepmass.csv')
    df["model_id"] = df.apply(
        lambda row: format_model(row.model, row.exp_alias),
        axis=1
    )
    num_rows = 1
    num_cols = 1

    setup_tueplots(num_rows, num_cols, rel_width=0.4, hw_ratio=0.8)
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    g = sb.boxplot(
        df,
        x="model_id",
        y=metric,
        hue="model_id",
        ax=ax
    )
    ax.set_xlabel('')
    if args.ylabel:
        ax.set_ylabel(format_metric(args.metric, train=args.train))
    else:
        ax.set_ylabel('')
    ax.set_title(format_dataset(args.dataset))

    path = os.path.join("figures", "complex-squared-npcs")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}-train.pdf" if args.train else f"{args.dataset}-test.pdf"
    plt.savefig(os.path.join(path, filename))
