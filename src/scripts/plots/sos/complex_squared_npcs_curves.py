import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt

from graphics.utils import setup_tueplots

parser = argparse.ArgumentParser(
    description="Plot metrics as a swarm plot based on number of squares",
)
parser.add_argument("checkpoint_path", type=str, help="The checkpoints path")
parser.add_argument("dataset", type=str, help="Dataset name")
parser.add_argument("--models", default="MPC;SOS;SOS", help="The models")
parser.add_argument("--learning-rate", type=float, default=0.001, help="The learning rate")
parser.add_argument(
    "--ylabel",
    action="store_true",
    default=False,
    help="Whether to show the y-axis label",
)
parser.add_argument(
    "--max-epochs", type=int, default=1000, help="The maximum number of epochs to show"
)
parser.add_argument(
    "--legend", action='store_true', default=False, help="Whether to show the legend"
)
parser.add_argument(
    "--title", action='store_true', default=False, help="Whether to show the title"
)



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
        if exp_alias == "real":
            return r"$\pm^2 (\mathbb{R})$"
        elif exp_alias == "complex":
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
    models = args.models.split(";")
    curves_info = []
    for model in models:
        checkpoint_path = os.path.join(args.checkpoint_path, args.dataset, model)
        for path in os.listdir(checkpoint_path):
            sub_path = os.path.join(checkpoint_path, path)
            sub_files = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
            if all(os.path.isdir(f) for f in sub_files):
                for f in sub_files:
                    exp_alias = f.split("/")[-2]
                    for g in os.listdir(f):
                        if "scalars-" in g:
                            curves_info.append((model, exp_alias, os.path.join(f, g)))
            else:
                for g in sub_files:
                    if "scalars-" in g:
                        curves_info.append((model, "", g))

    raw_curves_data = {}
    training_num_epochs = defaultdict(list)
    for model, exp_alias, curve_file in curves_info:
        if f'LR{args.learning_rate}' not in curve_file:
            continue
        trial_id = "-".join(curve_file.split("scalars-")[1].split(".")[:2])
        if model not in raw_curves_data:
            raw_curves_data[model] = {}
        with open(curve_file, "r") as fp:
            curve_data = json.load(fp)
            if exp_alias not in raw_curves_data[model]:
                raw_curves_data[model][exp_alias] = {}
            raw_curves_data[model][exp_alias][trial_id] = curve_data["Loss"]
            training_num_epochs[(model, exp_alias)].append(curve_data["Loss"][-1][1])
    max_training_num_epochs = {k: max(vs) for (k, vs) in training_num_epochs.items()}

    ys_data, xs_data, hues_data = zip(*[
        (*zip(*curve), format_model(model, exp_alias))
        for model, model_data in raw_curves_data.items()
        for exp_alias, exp_alias_data in model_data.items()
        for trial_id, curve in exp_alias_data.items()
    ])
    xs_data = [[float(x) for x in xs[:args.max_epochs]] for xs in xs_data]
    ys_data = [ys[:args.max_epochs] for ys in ys_data]
    cs_data_map = {r"$+_{\mathsf{sd}}$": f"C0", r"$\pm^2 (\mathbb{R})$": f"C1", r"$\pm^2 (\mathbb{C})$": f"C2"}
    cs_data = [cs_data_map[hue] for hue in hues_data]

    num_rows = 1
    num_cols = 1
    setup_tueplots(
        num_rows, num_cols,
        rel_width=0.5675 if args.legend else 0.4,
        hw_ratio=0.57 if args.legend else 0.8
    )
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    for x, y, c, hue in zip(xs_data, ys_data, cs_data, hues_data):
        x = x[:args.max_epochs]
        y = y[:args.max_epochs]
        g = sb.lineplot(
            x=x,
            y=y,
            ax=ax,
            linewidth=0.8,
            legend=False,
            label=hue,
            color=c,
            alpha=0.5
        )
    if args.ylabel:
        ax.set_ylabel("NLL")
    if args.title:
        ax.set_title(format_dataset(args.dataset))
    if args.legend:
        handles, labels = ax.get_legend_handles_labels()
        temp = {k: v for k, v in zip(labels, handles)}
        ax.legend(temp.values(), temp.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    path = os.path.join("figures", "complex-squared-npcs")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}-lr{args.learning_rate}-curves.pdf"
    plt.savefig(os.path.join(path, filename))
