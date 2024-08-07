import itertools
import os
import random
import subprocess
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.preprocessing import StandardScaler
from tbparse import SummaryReader
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from zuko.flows import MAF, NICE, NSF, Flow

from datasets.loaders import (
    BINARY_DATASETS,
    CONTINUOUS_DATASETS,
    IMAGE_DATASETS,
    LANGUAGE_DATASETS,
    SMALL_UCI_DATASETS,
    load_artificial_dataset,
    load_binary_dataset,
    load_continuous_dataset,
    load_image_dataset,
    load_language_dataset,
    load_small_uci_dataset,
)
from graphics.distributions import (
    kde_samples_hmap,
    plot_bivariate_discrete_samples_hmap,
)
from models import MPC, PC, SOS, ExpSOS
from scripts.logger import Logger
from utilities import (
    FLOW_MODELS,
    MODELS,
    PCS_MODELS,
    REGION_GRAPHS,
    retrieve_default_dtype,
)

WANDB_KEY_FILE = "wandb_api.key"  # Put your wandb api key in this file, first line


def drop_na(
    df: pd.DataFrame, drop_cols: List[str], verbose: bool = True
) -> pd.DataFrame:
    N = len(df)

    for c in drop_cols:
        if verbose:
            print(
                f"Dropping {len(df[pd.isna(df[c])])} runs that do not contain values for {c}"
            )
        df = df[~pd.isna(df[c])]

    if verbose:
        print(f"Dropped {N - len(df)} out of {N} rows.")

    return df


def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        # If v is a list, filter out rows with values NOT in the list
        if isinstance(v, list):
            df = df[df[k].isin(v)]
        else:
            if isinstance(v, bool):
                v = float(v)
            df = df[df[k] == v]
    return df


def unroll_hparams(hparams: dict) -> List[dict]:
    """
    :param hparams: dictionary with hyperparameter names as keys and hyperparam value domain as list

    Returns
    """
    unroll_hparams = [dict()]
    for k in hparams:
        vs = hparams[k]
        new_unroll_hparams = list()
        for v in vs:
            for hp in unroll_hparams:
                new_hp = hp.copy()
                new_hp[k] = v
                new_unroll_hparams.append(new_hp)
        unroll_hparams = new_unroll_hparams
    return unroll_hparams


def format_model(m: str) -> str:
    if m == "MPC":
        return r"$+$"
    elif m == "SOS":
        return r"$\Sigma^2$"
    assert False


def set_global_seed(seed: int, is_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    torch._dynamo.config.cache_size_limit = 24


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8")


def retrieve_wandb_runs(
    project_names: Union[str, List[str]], verbose: bool = True
) -> pd.DataFrame:
    """
    Returns all wandb runs from project name(s) specified

    :param project_names: The wandb user or team name and the project name as a string e.g "user12/project34"
    :param verbose: Bool for printing messages about processing
    """
    api = wandb.Api(api_key=open(WANDB_KEY_FILE, "r").readline())

    if isinstance(project_names, str):
        project_names = [project_names]

    # Project is specified by <entity/project-name>
    runs = []
    for project_name in project_names:
        runs += api.runs(project_name)

    if verbose:
        print(f"Loaded {len(runs)} from wandb project(s): {','.join(project_names)}")

    # summary_list, config_list, name_list = [], [], []
    run_dicts = []
    for run in runs:
        run_dict = dict()
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        # summary_list.append(run.summary._json_dict)
        run_dict.update(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        # config_list.append(config)

        run_dict.update(config)

        # .name is the human-readable name of the run.
        # name_list.append(run.name)
        run_dict.update({"name": run.name})
        run_dicts.append(run_dict)

    runs_df = pd.DataFrame(run_dicts)

    return runs_df


def retrieve_tboard_runs(
    tboard_path: str, metrics: Union[str, List[str]], ignore_diverged=False
) -> pd.DataFrame:
    reader = SummaryReader(tboard_path, pivot=True, extra_columns={"dir_name"})
    df_hparams = reader.hparams
    df_scalars = reader.scalars

    if not isinstance(metrics, list):
        metrics = [metrics]

    print(f"Number of retrieved experiments: {len(df_hparams)}")
    # Throw out rows with no result for the metric
    for m in metrics:
        df_scalars = df_scalars[~pd.isna(df_scalars[m])]

    assert len(df_hparams) == len(df_scalars), "Number of runs and results is different"
    if ignore_diverged:
        n_diverged = int(np.sum(df_scalars["diverged"]))
        print(f"Found {n_diverged} diverged runs. Ignoring...")
        df_scalars = df_scalars[df_scalars["diverged"] == False]
    df = df_hparams.merge(df_scalars, on="dir_name", sort=True).drop("dir_name", axis=1)

    return df


def retrieve_tboard_df(tboard_path: str) -> pd.DataFrame:
    reader = SummaryReader(tboard_path, pivot=True, extra_columns={"dir_name"})
    df_hparams = reader.hparams
    df_scalars = reader.scalars

    df_scalars = df_scalars[~pd.isna(df_scalars["Best/Test/avg_ll"])]

    # df_scalars = df_scalars.dropna(axis=1).drop('step', axis=1)
    df = df_hparams.merge(df_scalars, on="dir_name", sort=True).drop("dir_name", axis=1)

    print(len(df_hparams))
    return df


def retrieve_tboard_images(tboard_path: str) -> pd.DataFrame:
    reader = SummaryReader(tboard_path, pivot=False, extra_columns={"dir_name"})
    df_images = reader.images
    return df_images


def bits_per_dimension(average_ll: float, num_variables: int) -> float:
    return -average_ll / (num_variables * np.log(2.0))


def perplexity(average_ll: float, num_variables: int) -> float:
    return np.exp(-average_ll / num_variables)


def build_run_id(args):
    rs = list()
    if args.complex:
        rs.append(args.model + "-C")
    else:
        rs.append(args.model)
    if args.model in PCS_MODELS:
        if args.region_graph_sd:
            rs.append(f"RG{args.region_graph}-sd")
        else:
            rs.append(f"RG{args.region_graph}")
        rs.append(f"R{args.num_components}")
    rs.append(f"K{args.num_units}")
    if args.num_input_units > 0:
        rs.append(f"KI{args.num_input_units}")
    if args.model == "ExpSOS":
        rs.append(f"MK{args.mono_num_units}")
        if args.mono_num_input_units > 0:
            rs.append(f"MKI{args.mono_num_input_units}")
    rs.append(f"O{args.optimizer}")
    rs.append(f"LR{args.learning_rate}")
    rs.append(f"BS{args.batch_size}")
    if args.model in PCS_MODELS:
        if args.splines:
            num_input_units = (
                args.num_input_units if args.num_input_units > 0 else args.num_units
            )
            num_knots = num_input_units - args.spline_order - 1
            rs.append(f"SO{args.spline_order}_SK{num_knots}")
    if args.weight_decay > 0.0:
        rs.append(f"WD{args.weight_decay}")
    return "_".join(rs)


@torch.no_grad()
def evaluate_model_log_likelihood(
    model: Union[PC, Flow], dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    lls = list()
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(device)
        if isinstance(model, PC):
            batch = batch.unsqueeze(dim=1)
            log_probs = model.log_likelihood(batch)
        else:
            log_probs = model().log_prob(batch)
        if len(log_probs.shape) > 1:
            log_probs.squeeze(dim=1)
        lls.extend(log_probs.tolist())
    return np.mean(lls).item(), np.std(lls).item()


def setup_experiment_path(
    root: str, dataset: str, model_name: str, alias: str = "", trial_id: str = ""
):
    return os.path.join(root, dataset, model_name, alias, trial_id)


def setup_data_loaders(
    dataset: str,
    path: str,
    logger: Logger,
    batch_size: int = 128,
    num_workers: int = 0,
    num_samples: int = 1000,
    standardize: bool = False,
    discretize: bool = False,
    discretize_unique: bool = False,
    discretize_bins: int = 32,
    shuffle_bins: bool = False,
) -> Tuple[dict, Tuple[DataLoader, DataLoader, DataLoader]]:
    logger.info(f"Loading dataset '{dataset}' ...")

    numpy_dtype = retrieve_default_dtype(numpy=True)
    metadata = dict()
    # Load the dataset
    small_uci_dataset = dataset in SMALL_UCI_DATASETS
    binary_dataset = dataset in BINARY_DATASETS
    image_dataset = dataset in IMAGE_DATASETS
    continuous_dataset = dataset in CONTINUOUS_DATASETS
    language_dataset = dataset in LANGUAGE_DATASETS
    if small_uci_dataset:
        train_data, valid_data, test_data = load_small_uci_dataset(
            dataset, path=path, seed=123
        )
        metadata["image_shape"] = None
        metadata["num_variables"] = train_data.shape[1]
        metadata["hmap"] = None
        metadata["type"] = "categorical"
        max_state_value = max(np.max(train_data), np.max(valid_data), np.max(test_data))
        metadata["interval"] = (0, max_state_value)
        metadata["domains"] = None
    elif image_dataset:
        (
            image_shape,
            (train_data, valid_data, test_data),
        ) = load_image_dataset(dataset, path=path)
        metadata["image_shape"] = image_shape
        metadata["num_variables"] = np.prod(image_shape).item()
        metadata["hmap"] = None
        metadata["type"] = "image"
        metadata["interval"] = (0, 255)
        metadata["domains"] = None
        train_data = TensorDataset(train_data)
        valid_data = TensorDataset(valid_data)
        test_data = TensorDataset(test_data)
    elif binary_dataset:
        sep = ","
        if dataset == "binarized_mnist":
            sep = " "
        train_data, valid_data, test_data = load_binary_dataset(
            dataset, path=path, sep=sep
        )
        metadata["num_variables"] = train_data.shape[1]
        metadata["hmap"] = None
        metadata["domains"] = None
        if dataset == "binarized_mnist":
            metadata["image_shape"] = (1, 28, 28)
            metadata["type"] = "image"
            metadata["interval"] = (0, 1)
        else:
            metadata["image_shape"] = None
            metadata["type"] = "binary"
            metadata["interval"] = (0, 1)
    elif continuous_dataset:
        train_data, valid_data, test_data = load_continuous_dataset(
            dataset, path=path, dtype=numpy_dtype
        )
        train_valid_data = np.concatenate([train_data, valid_data], axis=0)
        data_min = np.min(train_valid_data, axis=0)
        data_max = np.max(train_valid_data, axis=0)
        drange = np.abs(data_max - data_min)
        data_min, data_max = (data_min - drange * 0.05), (data_max + drange * 0.05)
        metadata["image_shape"] = None
        metadata["num_variables"] = train_data.shape[1]
        metadata["hmap"] = None
        metadata["type"] = "continuous"
        metadata["interval"] = (np.min(data_min), np.max(data_max))
        metadata["domains"] = [(data_min[i], data_max[i]) for i in range(len(data_min))]
        train_data = TensorDataset(torch.tensor(train_data))
        valid_data = TensorDataset(torch.tensor(valid_data))
        test_data = TensorDataset(torch.tensor(test_data))
    elif language_dataset:
        train_data, valid_data, test_data = load_language_dataset(
            dataset, path=path, seed=123
        )
        seq_length = train_data.shape[1]
        metadata["image_shape"] = None
        metadata["num_variables"] = seq_length
        metadata["hmap"] = None
        metadata["type"] = "language"
        metadata["interval"] = (
            torch.min(train_data).item(),
            torch.max(train_data).item(),
        )
        metadata["domains"] = None
        train_data = TensorDataset(train_data)
        valid_data = TensorDataset(valid_data)
        test_data = TensorDataset(test_data)
    else:
        train_data, valid_data, test_data = load_artificial_dataset(
            dataset,
            num_samples=num_samples,
            discretize=discretize,
            discretize_unique=discretize_unique,
            discretize_bins=discretize_bins,
            shuffle_bins=shuffle_bins,
            dtype=retrieve_default_dtype(numpy=True),
        )
        metadata["image_shape"] = None
        metadata["num_variables"] = 2
        if discretize:
            metadata["type"] = "categorical"
            metadata["interval"] = (0, discretize_bins - 1)
            metadata["domains"] = [(0, discretize_bins - 1), (0, discretize_bins - 1)]
            metadata["hmap"] = plot_bivariate_discrete_samples_hmap(
                train_data, xlim=metadata["domains"][0], ylim=metadata["domains"][1]
            )
        else:
            if standardize:
                scaler = StandardScaler()
                scaler.fit(train_data)
                train_data = scaler.transform(train_data)
                valid_data = scaler.transform(valid_data)
                test_data = scaler.transform(test_data)
            train_valid_data = np.concatenate([train_data, valid_data], axis=0)
            data_min = np.min(train_valid_data, axis=0)
            data_max = np.max(train_valid_data, axis=0)
            drange = np.abs(data_max - data_min)
            data_min, data_max = (data_min - drange * 0.05), (data_max + drange * 0.05)
            metadata["type"] = "artificial"
            metadata["interval"] = (np.min(data_min), np.max(data_max))
            metadata["domains"] = [
                (data_min[i], data_max[i]) for i in range(len(data_min))
            ]
            metadata["hmap"] = kde_samples_hmap(
                train_data, xlim=metadata["domains"][0], ylim=metadata["domains"][1]
            )
    train_dataloader = DataLoader(
        train_data, batch_size, num_workers=num_workers, shuffle=True
    )
    valid_dataloader = DataLoader(valid_data, batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size, num_workers=num_workers)
    return metadata, (train_dataloader, valid_dataloader, test_dataloader)


def setup_model(
    model_name: str,
    dataset_metadata: dict,
    logger: Logger,
    region_graph: str = "rnd",
    structured_decomposable: bool = False,
    num_components: int = 1,
    num_units: int = 2,
    num_input_units: int = -1,
    mono_num_units: int = 2,
    mono_num_input_units: int = -1,
    complex: bool = False,
    splines: bool = False,
    spline_order: int = 2,
    seed: int = 123,
) -> Union[PC, Flow]:
    logger.info(f"Building model '{model_name}' ...")

    if complex and model_name not in ["SOS", "ExpSOS"]:
        raise ValueError("--complex can only be used with SOS or ExpSOS circuits")
    assert model_name in MODELS
    if splines:
        raise NotImplementedError()
    dataset_type = dataset_metadata["type"]
    num_variables = dataset_metadata["num_variables"]

    if model_name in FLOW_MODELS:
        return setup_flow_model(model_name, dataset_type, num_variables, num_units)

    assert region_graph in REGION_GRAPHS

    interval = dataset_metadata["interval"]
    if dataset_type in ["image", "categorical", "language", "binary"]:
        if model_name == "MPC":
            input_layer = "categorical"
            input_layer_kwargs = dict(num_categories=interval[1] + 1)
        else:
            input_layer = "embedding"
            input_layer_kwargs = dict(num_states=interval[1] + 1)
    else:
        input_layer = "gaussian"
        input_layer_kwargs = dict()
    image_shape = dataset_metadata["image_shape"] if dataset_type == "image" else None

    if model_name == "MPC":
        model = MPC(
            num_variables,
            image_shape=image_shape,
            num_input_units=num_input_units,
            num_sum_units=num_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            num_components=num_components,
            region_graph=region_graph,
            structured_decomposable=structured_decomposable,
            seed=seed,
        )
        return model

    if model_name == "SOS":
        model = SOS(
            num_variables,
            image_shape=image_shape,
            num_input_units=num_input_units,
            num_sum_units=num_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            num_squares=num_components,
            region_graph=region_graph,
            structured_decomposable=structured_decomposable,
            complex=complex,
            seed=seed,
        )
        return model

    if model_name == "ExpSOS":
        model = ExpSOS(
            num_variables,
            image_shape=image_shape,
            num_input_units=num_input_units,
            num_sum_units=num_units,
            mono_num_input_units=mono_num_input_units,
            mono_num_sum_units=mono_num_units,
            input_layer=input_layer,
            input_layer_kwargs=input_layer_kwargs,
            region_graph=region_graph,
            structured_decomposable=structured_decomposable,
            complex=complex,
            seed=seed,
        )
        return model

    raise ValueError(f"Unknown model called {model_name}")


def setup_flow_model(
    model_name: str,
    dataset_type: str,
    num_variables: int,
    num_units: int,
) -> Flow:
    assert model_name in FLOW_MODELS
    if model_name == "NICE":
        if dataset_type not in ["continuous", "artificial"]:
            raise ValueError("NICE is not supported for the requested data set")
        model = NICE(
            features=num_variables,
            transforms=10,
            hidden_features=(num_units, num_variables),
        )
        return model
    elif model_name == "MAF":
        if dataset_type not in ["continuous", "artificial"]:
            raise ValueError("MAF is not supported for the requested data set")
        model = MAF(
            features=num_variables,
            transforms=10,
            hidden_features=(num_units, num_variables),
        )
        return model
    elif model_name == "NSF":
        if dataset_type not in ["continuous", "artificial"]:
            raise ValueError("MAF is not supported for the requested data set")
        model = NSF(
            features=num_variables,
            transforms=10,
            hidden_features=(num_units, num_variables),
            bins=8,
        )
        return model
    raise NotImplementedError()


def num_parameters(
    model: Union[PC, nn.Module], requires_grad: bool = True, sum_only: bool = False
) -> int:
    if isinstance(model, PC):
        if sum_only:
            return model.num_sum_params(requires_grad)
        return model.num_params(requires_grad)
    assert not sum_only
    params = model.parameters()
    if requires_grad:
        params = filter(lambda p: p.requires_grad, params)
    return sum(p.numel() for p in params)
