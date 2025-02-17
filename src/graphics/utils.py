import io

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from PIL import Image as pillow
from tueplots import figsizes, fonts, fontsizes


def setup_tueplots(
    nrows: int,
    ncols: int,
    rel_width: float = 1.0,
    hw_ratio: float | None = None,
    default_smaller: int = -2,
    use_tex: bool = True,
    tight_layout=False,
    constrained_layout=False,
    **kwargs
):
    if use_tex:
        font_config = fonts.neurips2024_tex(family="serif")
    else:
        font_config = fonts.neurips2024(family="serif")
    if hw_ratio is not None:
        kwargs["height_to_width_ratio"] = hw_ratio
    size = figsizes.neurips2024(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        tight_layout=tight_layout,
        constrained_layout=constrained_layout,
        **kwargs
    )
    fontsize_config = fontsizes.neurips2024(default_smaller=default_smaller)
    rc_params = {
        **font_config,
        **size,
        **fontsize_config,
    }
    rc_params.update({"text.latex.preamble": r"\usepackage{amsfonts}"})
    plt.rcParams.update(rc_params)
    # plt.rcParams.update({
    #    "axes.prop_cycle": plt.cycler(
    #        color=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
    #               "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]
    #    ),
    #    "patch.facecolor": "#0173B2"
    # })
    sb.color_palette("colorblind")


def array_to_image(
    array: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
) -> np.ndarray:
    assert len(array.shape) == 2
    xi, yi = np.mgrid[range(array.shape[0]), range(array.shape[1])]
    setup_tueplots(1, 1, hw_ratio=1.0)
    fig, ax = plt.subplots()
    cmap = "turbo" if colorbar else "jet"
    pcm = ax.pcolormesh(xi, yi, array, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if colorbar:
        plt.colorbar(pcm)
    ax.set_xticks([])
    ax.set_yticks([])
    return matplotlib_buffer_to_image(fig)


def matplotlib_buffer_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buffer_to_image(buf)


def buffer_to_image(buf: io.BytesIO) -> np.ndarray:
    with pillow.open(buf, formats=["png"]) as fp:
        return np.array(fp, dtype=np.uint8).transpose([2, 0, 1])
