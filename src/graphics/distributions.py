import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity

from graphics.utils import matplotlib_buffer_to_image, setup_tueplots
from models import PC
from utilities import retrieve_default_dtype


def plot_bivariate_samples_hmap(
    data: np.ndarray,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zm: float = 0.0,
    nbins: int = 256,
) -> np.ndarray:
    setup_tueplots(1, 1, hw_ratio=1.0)
    if xlim is None:
        xlim = data[:, 0].min(), data[:, 0].max()
    if ylim is None:
        ylim = data[:, 1].min(), data[:, 1].max()
    zm_xamount = np.abs(xlim[1] - xlim[0])
    zm_yamount = np.abs(ylim[1] - ylim[0])
    xlim = (xlim[0] - zm * zm_xamount), (xlim[1] + zm * zm_xamount)
    ylim = (ylim[0] - zm * zm_yamount), (ylim[1] + zm * zm_yamount)
    fig, ax = plt.subplots()
    xi, yi = np.mgrid[xlim[0] : xlim[1] : nbins * 1j, ylim[0] : ylim[1] : nbins * 1j]
    zi, _, _ = np.histogram2d(
        data[:, 0],
        data[:, 1],
        bins=nbins,
        range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
        density=True,
    )
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", vmin=0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    return matplotlib_buffer_to_image(fig)


def plot_bivariate_discrete_samples_hmap(
    data: np.ndarray,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[int, int] | None = None,
) -> np.ndarray:
    if xlim is None:
        xlim = np.min(data[:, 0]), np.max(data[:, 0])
    if ylim is None:
        ylim = np.min(data[:, 0]), np.max(data[:, 1])
    setup_tueplots(1, 1, hw_ratio=1.0)
    fig, ax = plt.subplots()
    zi, xedges, yedges = np.histogram2d(
        data[:, 0],
        data[:, 1],
        bins=[xlim[1] - xlim[0] + 1, ylim[1] - ylim[0] + 1],
        range=[[xlim[0], xlim[1] + 1], [ylim[0], ylim[1] + 1]],
        density=True,
    )
    yi, xi = np.meshgrid(xedges[:-1], yedges[:-1])
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading="auto", vmin=0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    return matplotlib_buffer_to_image(fig)


def discrete_samples_hmap(
    data: np.ndarray,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[int, int] | None = None,
) -> np.ndarray:
    if xlim is None:
        xlim = np.min(data[:, 0]), np.max(data[:, 0])
    if ylim is None:
        ylim = np.min(data[:, 0]), np.max(data[:, 1])
    zi, xedges, yedges = np.histogram2d(
        data[:, 0],
        data[:, 1],
        bins=[xlim[1] - xlim[0] + 1, ylim[1] - ylim[0] + 1],
        range=[[xlim[0], xlim[1] + 1], [ylim[0], ylim[1] + 1]],
        density=True,
    )
    yi, xi = np.meshgrid(xedges[:-1], yedges[:-1])
    return zi.reshape(xi.shape)


def kde_samples_hmap(
    data: np.ndarray,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    zm: float = 0.0,
    nbins: int = 256,
    *,
    bandwidth: float = 0.2
) -> np.ndarray:
    if xlim is None:
        xlim = data[:, 0].min(), data[:, 0].max()
    if ylim is None:
        ylim = data[:, 1].min(), data[:, 1].max()
    zm_xamount = np.abs(xlim[1] - xlim[0])
    zm_yamount = np.abs(ylim[1] - ylim[0])
    xlim = (xlim[0] - zm * zm_xamount), (xlim[1] + zm * zm_xamount)
    ylim = (ylim[0] - zm * zm_yamount), (ylim[1] + zm * zm_yamount)
    xi, yi = np.mgrid[xlim[0] : xlim[1] : nbins * 1j, ylim[0] : ylim[1] : nbins * 1j]
    xy = np.stack([xi.flatten(), yi.flatten()], axis=1).astype(
        retrieve_default_dtype(numpy=True), copy=False
    )
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(data)
    zi = np.exp(kde.score_samples(xy)).reshape(xi.shape)
    return zi


@torch.no_grad()
def bivariate_pdf_heatmap(
    model: PC,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zm: float = 0.0,
    nbins: int = 256,
    device: str | torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = "cpu"
    zm_xamount = np.abs(xlim[1] - xlim[0])
    zm_yamount = np.abs(ylim[1] - ylim[0])
    xlim = (xlim[0] - zm * zm_xamount), (xlim[1] + zm * zm_xamount)
    ylim = (ylim[0] - zm * zm_yamount), (ylim[1] + zm * zm_yamount)
    xi, yi = np.mgrid[xlim[0] : xlim[1] : nbins * 1j, ylim[0] : ylim[1] : nbins * 1j]
    xy = np.stack([xi.flatten(), yi.flatten()], axis=1).astype(
        retrieve_default_dtype(numpy=True), copy=False
    )
    if model.num_variables < 1:
        raise ValueError("A PC defined over two variables is required")
    elif model.num_variables == 2:
        xy = torch.from_numpy(xy).unsqueeze(dim=1).to(device)
        zi = model.log_likelihood(xy)
    else:
        raise NotImplementedError()
    zi = torch.exp(zi).cpu().numpy().reshape(xi.shape)
    return zi


@torch.no_grad()
def bivariate_pmf_heatmap(
    model: PC,
    xlim: tuple[int, int],
    ylim: tuple[int, int],
    device: str | torch.device | None = None,
) -> np.ndarray:
    xi, yi = np.mgrid[xlim[0] : xlim[1] + 1, ylim[0] : ylim[1] + 1]
    xy = np.stack([xi.flatten(), yi.flatten()], axis=1)
    if device is None:
        device = "cpu"
    xy = torch.from_numpy(xy).to(device)
    zi = model.log_likelihood(xy)
    zi = torch.exp(zi).cpu().numpy().reshape(xi.shape)
    return zi
