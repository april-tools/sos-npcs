import argparse

import pandas as pd

from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Find the best learning rate from Tensorboard files",
)
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument("filename", type=str, help="The output filename")
parser.add_argument("--metric", default="avg_ll", help="The metric considered")


if __name__ == '__main__':
    args = parser.parse_args()
    train_metric = "Best/Train/" + args.metric
    valid_metric = "Best/Valid/" + args.metric
    test_metric = "Best/Test/" + args.metric
    metrics = [train_metric, valid_metric, test_metric]
    df = retrieve_tboard_runs(args.tboard_path, metrics)
    #df = df[df['seed'] == 123]
    group_by_cols = ['dataset', 'model', 'exp_alias', 'num_components']
    cols_to_keep = group_by_cols + metrics + ['learning_rate']
    df = df.drop(df.columns.difference(cols_to_keep), axis=1)
    df = df.sort_values(valid_metric, ascending=False)
    df: pd.DataFrame = df.groupby(group_by_cols).first()
    df.to_csv(f'{args.filename}-diffsed.csv')
