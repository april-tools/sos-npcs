import argparse
import json
import os
from collections import defaultdict

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
parser.add_argument("--ylabel", action='store_true', default=False, help="Whether to show the y-axis label")
parser.add_argument("--max-epochs", type=int, default=1000, help="The maximum number of epochs to show")


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
    models = args.models.split(";")
    curves_info = []
    for model in models:
        checkpoint_path = os.path.join(args.checkpoint_path, args.dataset, model)
        for path in os.listdir(checkpoint_path):
            sub_path = os.path.join(checkpoint_path, path)
            sub_files = [os.path.join(sub_path, f) for f in os.listdir(sub_path)]
            if all(os.path.isdir(f) for f in sub_files):
                for f in sub_files:
                    exp_alias = f.split('/')[-2]
                    for g in os.listdir(f):
                        if 'scalars-' in g:
                            curves_info.append((model, exp_alias, os.path.join(f, g)))
            else:
                for g in sub_files:
                    if 'scalars-' in g:
                        curves_info.append((model, '', g))

    raw_curves_data = {}
    training_num_epochs = defaultdict(list)
    for model, exp_alias, curve_file in curves_info:
        trial_id = '-'.join(curve_file.split('scalars-')[1].split('.')[:2])
        if model not in raw_curves_data:
            raw_curves_data[model] = {}
        with open(curve_file, 'r') as fp:
            curve_data = json.load(fp)
            if exp_alias not in raw_curves_data[model]:
                raw_curves_data[model][exp_alias] = {}
            raw_curves_data[model][exp_alias][trial_id] = curve_data['Loss']
            training_num_epochs[(model, exp_alias)].append(curve_data['Loss'][-1][1])
    max_training_num_epochs = {k: max(vs) for (k, vs) in training_num_epochs.items()}

    curves_data = {'model': [], 'exp_alias': [], 'trial_id': [], 'step': [], 'loss': []}
    for model, exp_alias, curve_file in curves_info:
        trial_id = '-'.join(curve_file.split('scalars-')[1].split('.')[:2])
        loss_step_values = raw_curves_data[model][exp_alias][trial_id]
        for loss, step in loss_step_values:
            curves_data['model'].append(model)
            curves_data['exp_alias'].append(exp_alias)
            curves_data['trial_id'].append(trial_id)
            curves_data['step'].append(step)
            curves_data['loss'].append(loss)
        last_loss = loss_step_values[-1][0]
        for j in range(len(loss_step_values) + 1, max_training_num_epochs[(model, exp_alias)] + 1):
            curves_data['model'].append(model)
            curves_data['exp_alias'].append(exp_alias)
            curves_data['trial_id'].append(trial_id)
            curves_data['step'].append(j)
            curves_data['loss'].append(last_loss)

    df = pd.DataFrame.from_dict(curves_data)
    df = df.sort_values(["model", "exp_alias"], ascending=[True, False])
    df["model_id"] = df.apply(
        lambda row: format_model(row.model, row.exp_alias),
        axis=1
    )
    df.drop("model", axis=1, inplace=True)
    df.drop("exp_alias", axis=1, inplace=True)
    df = df[df.step <= args.max_epochs]

    num_rows = 1
    num_cols = 1
    setup_tueplots(num_rows, num_cols, rel_width=0.4, hw_ratio=0.8)
    fig, ax = plt.subplots(num_rows, num_cols, squeeze=True, sharey=True)
    g = sb.lineplot(
        df,
        x="step",
        y="loss",
        hue="model_id",
        ax=ax,
        legend=False
    )
    ax.set_ylabel('NLL')
    ax.set_title(format_dataset(args.dataset))

    path = os.path.join("figures", "complex-squared-npcs")
    os.makedirs(path, exist_ok=True)
    filename = f"{args.dataset}-curves.pdf"
    plt.savefig(os.path.join(path, filename))
