from typing import Optional

import pandas as pd


def format_model(
    m: str,
    exp_alias: str,
    num_components: Optional[int] = None,
    merge_model_ids: bool = False,
) -> str:
    if m == "MPC":
        return r"$+_{\mathsf{sd}}$"
    elif m == "SOS":
        if "real" in exp_alias:
            if merge_model_ids:
                return r"$\pm^2_{\mathbb{R}} \ \ \Sigma_{\mathsf{cmp},\mathbb{R}}^2$"
            if num_components is not None and num_components > 1:
                return r"$\Sigma_{\mathsf{cmp},\mathbb{R}}^2$"
            return r"$\pm^2_{\mathbb{R}}$"
        elif "complex" in exp_alias:
            if merge_model_ids:
                return r"$\pm^2_{\mathbb{C}} \ \ \Sigma_{\mathsf{cmp},\mathbb{C}}^2$"
            if num_components is not None and num_components > 1:
                return r"$\Sigma_{\mathsf{cmp},\mathbb{C}}^2$"
            return r"$\pm^2_{\mathbb{C}}$"
    elif m == "ExpSOS":
        if "real" in exp_alias:
            return r"$+_{\mathsf{sd}}\!\cdot\!\pm^2_{\mathbb{R}}$"
        elif "complex" in exp_alias:
            return r"$+_{\mathsf{sd}}\!\cdot\!\pm^2_{\mathbb{C}}$"
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
        "CelebA": "CelebA",
    }[d]


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


def filter_dataframe(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in filter_dict.items():
        if isinstance(v, bool):
            v = float(v)
        df = df[df[k] == v]
    return df


def preprocess_dataframe(
    df: pd.DataFrame, merge_model_ids: bool = False
) -> pd.DataFrame:
    df = df.copy()
    df["model_id"] = df.apply(
        lambda row: format_model(
            row.model,
            row.exp_alias,
            row.num_components,
            merge_model_ids=merge_model_ids,
        ),
        axis=1,
    )
    df["model_order"] = df.apply(
        lambda row: format_model_order(row.model, row.exp_alias, row.num_components),
        axis=1,
    )
    df.sort_values(by="model_order", ascending=True, inplace=True)
    return df


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
