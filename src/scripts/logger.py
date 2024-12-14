import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import wandb
from PIL import Image as pillow
from torch.utils.tensorboard import SummaryWriter

from graphics.distributions import bivariate_pdf_heatmap, bivariate_pmf_heatmap
from models import PC


class Logger:
    def __init__(
        self,
        trail_id: str,
        verbose: bool,
        *,
        checkpoint_path: str | None = None,
        tboard_path: str | None = None,
        wandb_path: str | None = None,
        wandb_kwargs: dict[str, Any] | None = None,
    ):
        self.trial_id = trail_id
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self._tboard_writer: SummaryWriter | None = None

        if tboard_path:
            self._setup_tboard(tboard_path)
        if wandb_path:
            if wandb_kwargs is None:
                wandb_kwargs = dict()
            self._setup_wandb(wandb_path, **wandb_kwargs)

        self._best_distribution = None
        self._logged_scalars: dict[str, list[tuple[float, int | None]]] = defaultdict(
            list
        )
        self._logged_distributions = list()
        self._logged_wcoords = list()

    @property
    def has_graphical_endpoint(self) -> bool:
        return self._tboard_writer is not None or wandb.run

    def info(self, m: str):
        if self.verbose:
            print(m)

    def _setup_tboard(self, path: str):
        self._tboard_writer = SummaryWriter(log_dir=path)

    def _setup_wandb(
        self,
        path: str,
        project: str | None = None,
        config: dict[str, Any] | None = None,
        group: str | None = None,
        name: str | None = None,
        online: bool = True,
    ):
        if wandb.run is None:
            wandb.init(
                project=project,
                name=name,
                dir=path,
                group=group,
                config=config,
                mode="online" if online else "offline",
            )

    def log_scalar(self, tag: str, value: float, step: int | None = None):
        if self._tboard_writer is not None:
            self._tboard_writer.add_scalar(tag, value, global_step=step)
        if wandb.run:
            wandb.log({tag: value}, step=step)
        self._logged_scalars[tag].append((value, step))

    def log_image(
        self,
        tag: str,
        value: np.ndarray | torch.Tensor,
        step: int | None = None,
        dataformats: str = "CHW",
    ):
        if self._tboard_writer is not None:
            self._tboard_writer.add_image(
                tag, value, global_step=step, dataformats=dataformats
            )
        if wandb.run:
            if isinstance(value, torch.Tensor):
                value = value.permute(1, 2, 0)
            else:
                value = value.transpose([1, 2, 0])
            image = wandb.Image(value)
            wandb.log({tag: image}, step=step)

    def log_hparams(
        self,
        hparam_dict: dict[str, Any],
        metric_dict: dict[str, Any],
        hparam_domain_discrete: dict[str, list[Any]] | None = None,
        run_name: str | None = None,
    ):
        if self._tboard_writer is not None:
            self._tboard_writer.add_hparams(
                hparam_dict,
                metric_dict,
                hparam_domain_discrete=hparam_domain_discrete,
                run_name=run_name,
            )
        if wandb.run:
            wandb.run.summary.update(metric_dict)

    def log_best_distribution(
        self,
        model: PC,
        discretized: bool,
        lim: tuple[
            tuple[float | int, float | int],
            tuple[float | int, float | int],
        ],
        device: str | torch.device | None = None,
    ):
        xlim, ylim = lim
        if discretized:
            dist_hmap = bivariate_pmf_heatmap(model, xlim, ylim, device=device)
        else:
            dist_hmap = bivariate_pdf_heatmap(model, xlim, ylim, device=device)
        self._best_distribution = dist_hmap.astype(np.float32, copy=False)

    def log_step_distribution(
        self,
        model: PC,
        discretized: bool,
        lim: tuple[
            tuple[float | int, float | int],
            tuple[float | int, float | int],
        ],
        device: str | torch.device | None = None,
    ):
        xlim, ylim = lim
        if discretized:
            dist_hmap = bivariate_pmf_heatmap(model, xlim, ylim, device=device)
        else:
            dist_hmap = bivariate_pdf_heatmap(model, xlim, ylim, device=device)
        self._logged_distributions.append(dist_hmap.astype(np.float32, copy=False))

    def close(self):
        if self._logged_distributions:
            self.save_array(self._best_distribution, f"distbest-{self.trial_id}.npy")
            self.save_array(
                np.stack(self._logged_distributions, axis=0),
                f"diststeps-{self.trial_id}.npy",
            )
        if self._logged_wcoords:
            self.save_array(
                np.stack(self._logged_wcoords, axis=0), f"wcoords-{self.trial_id}.npy"
            )
        if self._tboard_writer is not None:
            self._tboard_writer.close()
        if wandb.run:
            wandb.finish(quiet=True)
        self.save_dict(self._logged_scalars, f"scalars-{self.trial_id}.json")

    def save_checkpoint(self, data: dict[str, Any], filepath: str):
        if self.checkpoint_path:
            torch.save(data, os.path.join(self.checkpoint_path, filepath))

    def save_image(self, data: np.ndarray, filepath: str):
        if self.checkpoint_path:
            pillow.fromarray(data.transpose([1, 2, 0])).save(
                os.path.join(self.checkpoint_path, filepath)
            )

    def save_array(self, array: np.ndarray, filepath: str):
        if self.checkpoint_path:
            np.save(os.path.join(self.checkpoint_path, filepath), array)

    def save_dict(self, data: dict, filepath: str):
        with open(os.path.join(self.checkpoint_path, filepath), "w") as fp:
            json.dump(data, fp)

    def load_array(self, filepath: str) -> np.ndarray | None:
        if self.checkpoint_path:
            try:
                array = np.load(os.path.join(self.checkpoint_path, filepath))
            except OSError:
                return None
            return array
        return None
