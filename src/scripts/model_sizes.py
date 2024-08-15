import argparse


import numpy as np

from scripts.utils import retrieve_tboard_runs

parser = argparse.ArgumentParser(
    description="Plot metrics as a swarm plot based on number of squares",
)
parser.add_argument("tboard_path", type=str, help="The Tensorboard runs path")
parser.add_argument("--metric", default="avg_ll", help="The metric to plot")

if __name__ == "__main__":
    args = parser.parse_args()
    metric = "Best/Test/" + args.metric
    df = retrieve_tboard_runs(args.tboard_path, metric)
    # df = df.groupby(by=["dataset", "model", "exp_alias", "num_components"]).agg(
    #     {"num_sum_params": [np.min, np.max]}
    # )
    # df.to_csv("model_sizes.csv")
    df = df.groupby(by=["dataset"]).agg(
        {"num_sum_params": [np.min, np.max]}
    )
    print(df)
