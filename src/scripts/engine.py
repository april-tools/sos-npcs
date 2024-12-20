import os
import time
from argparse import Namespace
from copy import copy
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from zuko.flows import Flow

from datasets.loaders import ALL_DATASETS
from models import PC
from optimization.optimizers import OPTIMIZERS_NAMES, setup_optimizer
from optimization.schedulers import ReduceLROnPlateau
from scripts.logger import Logger
from scripts.utils import (
    bits_per_dimension,
    build_run_id,
    evaluate_model_log_likelihood,
    get_git_revision_hash,
    num_parameters,
    perplexity,
    set_global_seed,
    setup_data_loaders,
    setup_experiment_path,
    setup_model,
)
from utilities import MODELS, REGION_GRAPHS


class Engine:
    def __init__(self, args: Namespace):
        self.args = args
        self._device = torch.device(args.device)
        self._git_rev_hash = get_git_revision_hash()
        self._trial_unique_id = str(round(time.time(), 6))

        set_global_seed(args.seed)
        torch.set_default_dtype(
            torch.float32 if args.dtype == "float32" else torch.float64
        )

        # Creating experiment directories
        kwargs = dict()
        run_id = build_run_id(args)
        run_group = f"{args.dataset}-{args.model}"
        if args.exp_alias:
            run_group = f"{run_group}-{args.exp_alias}"
        exp_path = setup_experiment_path(
            args.dataset, args.model, args.exp_alias, run_id
        )
        kwargs["checkpoint_path"] = os.path.join(args.checkpoint_path, exp_path)
        os.makedirs(kwargs["checkpoint_path"], exist_ok=True)
        kwargs["tboard_path"] = os.path.join(args.tboard_path, exp_path)
        os.makedirs(kwargs["tboard_path"], exist_ok=True)
        if args.wandb_path:
            kwargs["wandb_path"] = args.wandb_path
            kwargs["wandb_kwargs"] = {
                "project": args.wandb_project,
                "name": run_id,
                "group": run_group,
                "config": self.hparams,
            }
            os.makedirs(kwargs["wandb_path"], exist_ok=True)

        self.logger = Logger(self._trial_unique_id, args.verbose, **kwargs)
        self.metadata: dict[str, Any] = dict()

        self.dataloaders: dict[str, DataLoader | None] = {
            "train": None,
            "valid": None,
            "test": None,
        }

        self.model: PC | Flow | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._log_distribution = self.args.log_distribution

    def shutdown(self):
        self.logger.close()

    @property
    def hparams(self) -> dict[str, Any]:
        return {
            "seed": self.args.seed,
            "dataset": self.args.dataset,
            "discretize": self.args.discretize,
            "discretize_bins": self.args.discretize_bins,
            "model": self.args.model,
            "complex": self.args.complex,
            "region_graph": self.args.region_graph,
            "region_graph_sd": self.args.region_graph_sd,
            "num_units": self.args.num_units,
            "num_input_units": self.args.num_input_units,
            "mono_num_units": self.args.mono_num_units,
            "mono_num_input_units": self.args.mono_num_input_units,
            "num_components": self.args.num_components,
            "optimizer": self.args.optimizer,
            "mono_clamp": self.args.mono_clamp,
            "learning_rate": self.args.learning_rate,
            "batch_size": self.args.batch_size,
            "weight_decay": self.args.weight_decay,
            "exp_alias": self.args.exp_alias,
            "git_rev_hash": self._git_rev_hash,
            "load_checkpoint": self.args.load_checkpoint,
            "checkpoint_hparams": self.args.checkpoint_hparams,
        }

    def _eval_step(
        self,
        epoch_idx: int,
        metrics: dict[str, float],
        train_avg_ll: float | None = None,
    ) -> dict[str, float]:
        # Log metrics based on training data
        if train_avg_ll is not None:
            if self.metadata["type"] in ["image", "categorical", "language"]:
                train_bpd = bits_per_dimension(
                    train_avg_ll, self.metadata["num_variables"]
                )
            else:
                train_bpd = np.inf
            if self.metadata["type"] == "language":
                train_ppl = perplexity(train_avg_ll, self.metadata["num_variables"])
            else:
                train_ppl = np.inf
        else:
            train_avg_ll = -np.inf
            train_bpd = np.inf
            train_ppl = np.inf
        if self.metadata["type"] in ["image", "categorical"]:
            best_loss = metrics["best_train_bpd"]
            cur_loss = train_bpd
        elif self.metadata["type"] == "language":
            best_loss = metrics["best_train_ppl"]
            cur_loss = train_ppl
        else:
            best_loss = -metrics["best_train_avg_ll"]
            cur_loss = -train_avg_ll
        best_train_found = (best_loss == np.inf) or (
            cur_loss < best_loss - self.args.patience_threshold * np.abs(best_loss)
        )
        if best_train_found:
            metrics["best_train_epoch"] = epoch_idx
            metrics["best_train_avg_ll"] = train_avg_ll
            metrics["best_train_bpd"] = train_bpd
            metrics["best_train_ppl"] = train_ppl

        # Log metrics based on validation and test data
        valid_avg_ll, valid_std_ll = evaluate_model_log_likelihood(
            self.model, self.dataloaders["valid"], self._device
        )

        self.logger.info(
            f"[{self.args.dataset}] Epoch {epoch_idx}, Valid ll: {valid_avg_ll:.03f}"
        )
        self.logger.log_scalar("Valid/ll", valid_avg_ll, step=epoch_idx)
        if self.metadata["type"] in ["image", "categorical", "language"]:
            valid_bpd = bits_per_dimension(valid_avg_ll, self.metadata["num_variables"])
            self.logger.info(
                f"[{self.args.dataset}] Epoch {epoch_idx}, Valid bpd: {valid_bpd:.03f}"
            )
            self.logger.log_scalar("Valid/bpd", valid_bpd, step=epoch_idx)
        else:
            valid_bpd = np.inf
        if self.metadata["type"] == "language":
            valid_ppl = perplexity(valid_avg_ll, self.metadata["num_variables"])
            self.logger.info(
                f"[{self.args.dataset}] Epoch {epoch_idx}, Valid ppl: {valid_ppl:.03f}"
            )
            self.logger.log_scalar("Valid/ppl", valid_ppl, step=epoch_idx)
        else:
            valid_ppl = np.inf
        if self.metadata["type"] in ["image", "categorical"]:
            best_loss = metrics["best_valid_bpd"]
            cur_loss = valid_bpd
        elif self.metadata["type"] == "language":
            best_loss = metrics["best_valid_ppl"]
            cur_loss = valid_ppl
        else:
            best_loss = -metrics["best_valid_avg_ll"]
            cur_loss = -valid_avg_ll
        if self.scheduler is not None:  # Update scheduler based on validation data
            if self.args.step_lr_decay:
                if epoch_idx <= 2 * self.args.step_size_lr_decay:
                    self.scheduler.step()
            else:
                self.scheduler.step(cur_loss)
        best_valid_found = (best_loss == np.inf) or (
            cur_loss
            < best_loss
            - self.args.patience_threshold * np.maximum(5.0, np.abs(best_loss))
        )
        if best_valid_found:
            test_avg_ll, test_std_ll = evaluate_model_log_likelihood(
                self.model, self.dataloaders["test"], self._device
            )
            self.logger.info(
                f"[{self.args.dataset}] Epoch {epoch_idx}, Test ll: {test_avg_ll:.03f}"
            )
            self.logger.log_scalar("Test/avg_ll", test_avg_ll, step=epoch_idx)
            test_bpd = np.inf
            if self.metadata["type"] in ["image", "categorical", "language"]:
                test_bpd = bits_per_dimension(
                    test_avg_ll, self.metadata["num_variables"]
                )
                self.logger.info(
                    f"[{self.args.dataset}] Epoch {epoch_idx}, Test bpd: {test_bpd:.03f}"
                )
                self.logger.log_scalar("Test/bpd", test_bpd, step=epoch_idx)
            test_ppl = np.inf
            if self.metadata["type"] == "language":
                test_ppl = perplexity(test_avg_ll, self.metadata["num_variables"])
                self.logger.info(
                    f"[{self.args.dataset}] Epoch {epoch_idx}, Test ppl: {test_ppl:.03f}"
                )
                self.logger.log_scalar("Test/ppl", test_ppl, step=epoch_idx)
            if self._log_distribution:
                self.logger.log_best_distribution(
                    self.model,
                    self.args.discretize,
                    lim=self.metadata["domains"],
                    device=self._device,
                )
            metrics["best_valid_epoch"] = epoch_idx
            metrics["best_valid_avg_ll"] = valid_avg_ll
            metrics["best_valid_std_ll"] = valid_std_ll
            metrics["best_valid_bpd"] = valid_bpd
            metrics["best_valid_ppl"] = valid_ppl
            metrics["test_avg_ll"] = test_avg_ll
            metrics["test_std_ll"] = test_std_ll
            metrics["test_bpd"] = test_bpd
            metrics["test_ppl"] = test_ppl

        should_checkpoint = isinstance(self.model, PC) and (
            (self.args.early_stop_loss and best_train_found)
            or (not self.args.early_stop_loss and best_valid_found)
        )
        if self.args.save_checkpoint and should_checkpoint:
            self.logger.save_checkpoint(
                {
                    "hparams": self.hparams,
                    "weights": self.model.state_dict(),
                    "opt": self.optimizer.state_dict(),
                    "best_train": {
                        "epoch": metrics["best_train_epoch"],
                        "avg_ll": metrics["best_train_avg_ll"],
                        "bpd": metrics["best_train_bpd"],
                        "ppl": metrics["best_train_ppl"],
                    },
                    "best_valid": {
                        "epoch": metrics["best_valid_epoch"],
                        "avg_ll": metrics["best_valid_avg_ll"],
                        "std_ll": metrics["best_valid_std_ll"],
                        "bpd": metrics["best_valid_bpd"],
                        "ppl": metrics["best_valid_ppl"],
                    },
                    "test": {
                        "epoch": metrics["best_valid_epoch"],
                        "avg_ll": metrics["test_avg_ll"],
                        "std_ll": metrics["test_std_ll"],
                        "bpd": metrics["test_bpd"],
                        "ppl": metrics["test_ppl"],
                    },
                },
                f"checkpoint-{self._trial_unique_id}.pt",
            )
        return metrics

    def run(self):
        # Setup the data loaders
        metadata, (
            train_dataloader,
            valid_dataloader,
            test_dataloader,
        ) = setup_data_loaders(
            self.args.dataset,
            self.args.data_path,
            logger=self.logger,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            num_samples=self.args.num_samples,
            standardize=self.args.standardize,
            discretize_unique=self.args.discretize_unique,
            discretize=self.args.discretize,
            discretize_bins=self.args.discretize_bins,
            shuffle_bins=self.args.shuffle_bins,
        )
        self.metadata = metadata
        self.dataloaders["train"] = train_dataloader
        self.dataloaders["valid"] = valid_dataloader
        self.dataloaders["test"] = test_dataloader
        self._log_distribution &= self.metadata["type"] == "artificial" or (
            self.metadata["type"] == "categorical"
            and self.metadata["num_variables"] == 2
        )
        self.logger.info(f"Number of variables: {self.metadata['num_variables']}")

        # Initialize the model
        self.model = setup_model(
            self.args.model,
            self.metadata,
            logger=self.logger,
            region_graph=self.args.region_graph,
            structured_decomposable=self.args.region_graph_sd,
            num_components=self.args.num_components,
            num_units=self.args.num_units,
            num_input_units=self.args.num_input_units,
            mono_num_units=self.args.mono_num_units,
            mono_num_input_units=self.args.mono_num_input_units,
            mono_clamp=self.args.mono_clamp,
            complex=self.args.complex,
            seed=self.args.seed,
        )

        # Instantiate the optimizer
        self.optimizer = setup_optimizer(
            self.model.parameters(),
            self.args.optimizer,
            self.args.learning_rate,
            self.args.decay1,
            self.args.decay2,
            self.args.momentum,
            self.args.weight_decay,
        )

        # Initialize the dictionary containing metrics
        metrics = {
            "best_train_epoch": -1,
            "best_train_avg_ll": -np.inf,
            "best_train_bpd": np.inf,
            "best_train_ppl": np.inf,
            "best_valid_epoch": -1,
            "best_valid_avg_ll": -np.inf,
            "best_valid_std_ll": np.nan,
            "best_valid_bpd": np.inf,
            "best_valid_ppl": np.inf,
            "test_avg_ll": -np.inf,
            "test_std_ll": np.nan,
            "test_bpd": np.inf,
            "test_ppl": np.inf,
        }

        # Load the model checkpoint, if required
        if self.args.load_checkpoint:
            if self.args.load_checkpoint_path:
                checkpoint_path = self.args.load_checkpoint_path
            else:
                checkpoint_path = self.args.checkpoint_path

            # If alternate checkpoint hparams given, replace values in hparams from CL
            checkpoint_args = copy(self.args)
            for hp in self.args.checkpoint_hparams.split(";"):
                if hp:
                    hp_name, hp_value = hp.split("=")
                    checkpoint_args.__setattr__(hp_name.replace("-", "_"), hp_value)

            checkpoint_run_id = build_run_id(checkpoint_args)
            checkpoint_exp_path = setup_experiment_path(
                checkpoint_args.dataset,
                checkpoint_args.model,
                checkpoint_args.exp_alias,
                checkpoint_run_id,
            )

            # Loading the model
            checkpoint_filepath = os.path.join(
                checkpoint_path, checkpoint_exp_path, "model.pt"
            )
            state_dict = torch.load(checkpoint_filepath, map_location="cpu")
            self.model.load_state_dict(state_dict["weights"])
            self.model.to(self._device)
            del state_dict

            self.logger.info(f"Checkpoint loaded from {checkpoint_filepath}")
            metrics = self._eval_step(0, metrics)

        # Log something
        self.logger.info(f"Model architecture:\n{self.model}")
        num_params = num_parameters(self.model)
        self.logger.info(f"Number of parameters: {num_params}")
        if isinstance(self.model, PC):
            num_sum_params = num_parameters(self.model, sum_only=True)
            self.logger.info(f"Number of parameters of sum units: {num_sum_params}")
        else:
            num_sum_params = np.nan
        self.model.to(self._device)

        # Instantiate the LR scheduler, if any
        if self.args.reduce_lr_plateau:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=self.args.reduce_lr_patience,
                threshold=self.args.patience_threshold,
                threshold_mode="rel",
                min_lr=1e-7,
            )
        elif self.args.step_lr_decay:
            self.scheduler = StepLR(
                self.optimizer,
                self.args.step_size_lr_decay,
                gamma=self.args.amount_lr_decay,
                verbose=True,
            )

        if self._log_distribution:
            self.logger.save_array(self.metadata["hmap"], "gt.npy")
            self.logger.log_step_distribution(
                self.model,
                self.args.discretize,
                lim=self.metadata["domains"],
                device=self._device,
            )

        # The train loop
        diverged = False
        opt_counter = 0
        for epoch_idx in range(1, self.args.num_epochs + 1):
            self.model.train()
            running_average_loss = 0.0
            running_training_samples = 0
            for batch_idx, batch in enumerate(train_dataloader):
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self._device)
                if isinstance(self.model, PC):
                    if len(batch.shape) < 3:
                        batch = batch.unsqueeze(dim=1)
                    lls = self.model.log_likelihood(batch)
                else:
                    lls = self.model().log_prob(batch)
                loss = -torch.mean(lls)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss = loss.item()
                running_average_loss += loss * len(batch)
                running_training_samples += len(batch)
                if not np.isfinite(loss):
                    self.logger.info(f"[{self.args.dataset}] Loss is not finite")
                    diverged = True
                    break
                if (
                    opt_counter
                    % (
                        max(1, int(1e-1 * self.args.log_frequency))
                        if epoch_idx == 1
                        else (
                            max(1, int(2e-1 * self.args.log_frequency))
                            if epoch_idx == 2
                            else self.args.log_frequency
                        )
                    )
                    == 0
                ):
                    if self._log_distribution:
                        self.logger.log_step_distribution(
                            self.model,
                            self.args.discretize,
                            lim=self.metadata["domains"],
                            device=self._device,
                        )
                opt_counter += 1
            if diverged:
                self.logger.info(f"Diverged, exiting ...")
                break
            running_average_loss /= running_training_samples
            self.logger.info(
                f"[{self.args.dataset}] Epoch {epoch_idx}, Training loss: {running_average_loss:.03f}"
            )
            self.logger.log_scalar("Loss", running_average_loss, step=epoch_idx)
            metrics = self._eval_step(
                epoch_idx,
                metrics,
                train_avg_ll=-running_average_loss,
            )
            best_epoch = (
                metrics["best_train_epoch"]
                if self.args.early_stop_loss
                else metrics["best_valid_epoch"]
            )
            if epoch_idx - best_epoch + 1 > self.args.early_stop_patience:
                self.logger.info(f"Early stopping ...")
                break

        self.logger.log_hparams(
            self.hparams,
            {
                "Best/Train/epoch": metrics["best_train_epoch"],
                "Best/Train/avg_ll": metrics["best_train_avg_ll"],
                "Best/Train/bpd": metrics["best_train_bpd"],
                "Best/Train/ppl": metrics["best_train_ppl"],
                "Best/Valid/epoch": metrics["best_valid_epoch"],
                "Best/Valid/avg_ll": metrics["best_valid_avg_ll"],
                "Best/Valid/std_ll": metrics["best_valid_std_ll"],
                "Best/Valid/bpd": metrics["best_valid_bpd"],
                "Best/Valid/ppl": metrics["best_valid_ppl"],
                "Best/Test/avg_ll": metrics["test_avg_ll"],
                "Best/Test/std_ll": metrics["test_std_ll"],
                "Best/Test/bpd": metrics["test_bpd"],
                "Best/Test/ppl": metrics["test_ppl"],
                "num_params": num_params,
                "num_sum_params": num_sum_params,
                "diverged": diverged,
            },
            run_name=self._trial_unique_id,
            hparam_domain_discrete={
                "dataset": ALL_DATASETS,
                "model": MODELS,
                "optimizer": OPTIMIZERS_NAMES,
                "region_graph": REGION_GRAPHS,
                "region_graph_sd": [False, True],
                "mono_clamp": [False, True],
                "complex": [False, True],
            },
        )
