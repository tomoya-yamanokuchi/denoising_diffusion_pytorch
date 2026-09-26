"""
I2SB conditional image trainer with product-aware periodic sampling logs.

Per-product logging layout:
samples/
  <product>/
    static/
      target.png
      cond.png
      mask.png
      x1.png
    pred/
      step_005000.png
      step_010000.png
      ...

The fixed evaluation target/condition/mask/X1 are saved only once per run.
Only predictions are saved at every checkpoint.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from denoising_diffusion_pytorch.models.helpers import (
    cycle,
    divisible_by,
    exists,
)
from denoising_diffusion_pytorch.version import __version__


class Trainer:
    def __init__(
        self,
        model,
        diffusion,
        dataset,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=2000,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        num_workers=8,
        num_samples=4,
        samples_per_product=1,
        sample_nfe=20,
        calculate_fid=False,
        save_best_and_latest_only=False,
        **unused_kwargs,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else "no",
        )

        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset

        self.channels = model.channels
        self.image_size = dataset.image_size

        self.batch_size = int(train_batch_size)
        self.gradient_accumulate_every = int(gradient_accumulate_every)
        self.train_num_steps = int(train_num_steps)
        self.save_and_sample_every = int(save_and_sample_every)
        self.max_grad_norm = float(max_grad_norm)

        self.num_samples = int(num_samples)
        self.samples_per_product = int(samples_per_product)
        self.sample_nfe = int(sample_nfe)

        if self.batch_size <= 0:
            raise ValueError("train_batch_size must be positive.")
        if self.gradient_accumulate_every <= 0:
            raise ValueError("gradient_accumulate_every must be positive.")
        if self.train_num_steps <= 0:
            raise ValueError("train_num_steps must be positive.")
        if self.samples_per_product <= 0:
            raise ValueError("samples_per_product must be positive.")
        if self.sample_nfe <= 0:
            raise ValueError("sample_nfe must be positive.")
        if len(dataset) < 2:
            raise ValueError(
                f"I2SB training requires at least 2 samples, got {len(dataset)}."
            )

        self.calculate_fid = calculate_fid
        self.save_best_and_latest_only = save_best_and_latest_only

        if self.calculate_fid:
            self.accelerator.print(
                "[I2SB] calculate_fid=True was provided, but FID evaluation "
                "is not connected in this trainer."
            )

        if unused_kwargs and self.accelerator.is_main_process:
            self.accelerator.print(
                "[I2SB] Unused trainer config keys:",
                sorted(unused_kwargs.keys()),
            )

        data_samples = len(dataset)
        train_size = int(data_samples * 0.9)
        val_size = data_samples - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
        )

        self.log_samples = self._build_fixed_log_samples(val_dataset)

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        val_dl = DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )

        self.opt = Adam(
            self.model.parameters(),
            lr=train_lr,
            betas=adam_betas,
        )

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.sw_dir = self.results_folder / "sw_dir"
        self.sw_dir.mkdir(parents=True, exist_ok=True)

        self.samples_dir = self.results_folder / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.step = 0

        if self.accelerator.is_main_process:
            self.ema = EMA(
                self.model,
                beta=ema_decay,
                update_every=ema_update_every,
            )
            self.ema.to(self.device)

        self.model, self.opt, train_dl, val_dl = self.accelerator.prepare(
            self.model,
            self.opt,
            train_dl,
            val_dl,
        )

        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)

        if self.accelerator.is_main_process:
            self._print_log_sample_summary()

    @property
    def device(self):
        return self.accelerator.device

    def _build_fixed_log_samples(self, val_dataset):
        if not hasattr(self.dataset, "samples"):
            raise AttributeError(
                "Product-aware I2SB logging requires dataset.samples metadata."
            )

        product_order = [
            product["name"]
            for product in getattr(self.dataset, "products", [])
        ]

        if len(product_order) == 0:
            product_order = sorted(
                {
                    sample["product_name"]
                    for sample in self.dataset.samples
                }
            )

        candidates = defaultdict(list)

        for dataset_idx in val_dataset.indices:
            product_name = self.dataset.samples[
                dataset_idx
            ]["product_name"]

            if len(candidates[product_name]) < self.samples_per_product:
                candidates[product_name].append(dataset_idx)

        missing = [
            product_name
            for product_name in product_order
            if len(candidates[product_name]) < self.samples_per_product
        ]

        if missing:
            raise RuntimeError(
                "Validation split does not contain enough fixed logging "
                f"samples for: {missing}."
            )

        log_samples = []

        for product_name in product_order:
            for sample_id, dataset_idx in enumerate(
                candidates[product_name]
            ):
                item = self.dataset[dataset_idx]

                log_samples.append(
                    {
                        "product_name": product_name,
                        "sample_id": sample_id,
                        "dataset_idx": dataset_idx,
                        "x0": item["x0"].detach().cpu().clone(),
                        "x1": item["x1"].detach().cpu().clone(),
                        "cond": item["cond"].detach().cpu().clone(),
                        "mask": item["mask"].detach().cpu().clone(),
                        "x0_path": item.get("x0_path", ""),
                        "x1_path": item.get("x1_path", ""),
                    }
                )

        return log_samples

    def _print_log_sample_summary(self):
        self.accelerator.print(
            "[I2SB] fixed product-aware validation samples:"
        )

        for sample in self.log_samples:
            self.accelerator.print(
                "  "
                f"{sample['product_name']} "
                f"sample={sample['sample_id']} "
                f"dataset_idx={sample['dataset_idx']} "
                f"x0={sample['x0_path']}"
            )

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
            "version": __version__,
            "method": "i2sb",
        }

        torch.save(
            data,
            str(self.results_folder / f"model-{milestone}.pt"),
        )

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        checkpoint_path = self.results_folder / f"model-{milestone}.pt"

        data = torch.load(
            str(checkpoint_path),
            map_location=device,
        )

        model = accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])

        if accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if (
            exists(accelerator.scaler)
            and exists(data.get("scaler", None))
        ):
            accelerator.scaler.load_state_dict(data["scaler"])

        if "version" in data:
            accelerator.print(
                f"loading from version {data['version']}"
            )

    def compute_loss(
        self,
        x0,
        x1,
        cond,
        mask,
        step=None,
        ot_ode=False,
    ):
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 shape mismatch: {x0.shape} vs {x1.shape}"
            )

        if cond.shape != x0.shape:
            raise ValueError(
                f"cond and x0 shape mismatch: {cond.shape} vs {x0.shape}"
            )

        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(
                "mask must have shape [B, 1, H, W], "
                f"got {tuple(mask.shape)}"
            )

        batch_size = x0.shape[0]

        if step is None:
            step = torch.randint(
                low=0,
                high=len(self.diffusion.betas),
                size=(batch_size,),
                device=x0.device,
                dtype=torch.long,
            )
        else:
            step = step.to(
                device=x0.device,
                dtype=torch.long,
            )

        xt = self.diffusion.q_sample(
            step=step,
            x0=x0,
            x1=x1,
            ot_ode=ot_ode,
        )

        label = self.diffusion.compute_label(
            step=step,
            x0=x0,
            xt=xt,
        )

        pred = self.model(
            xt,
            step,
            None,
            cond,
        )

        if pred.shape != label.shape:
            raise RuntimeError(
                "Network prediction and I2SB target must have identical "
                f"shapes, got pred={tuple(pred.shape)}, "
                f"label={tuple(label.shape)}."
            )

        pred_masked = mask * pred
        label_masked = mask * label

        loss = F.mse_loss(
            pred_masked,
            label_masked,
        )

        diagnostics = {
            "step": step.detach(),
            "xt": xt.detach(),
            "label": label.detach(),
            "pred": pred.detach(),
        }

        return loss, diagnostics

    def _make_sampling_steps(self):
        interval = len(self.diffusion.betas)

        steps = np.linspace(
            0,
            interval - 1,
            self.sample_nfe + 1,
            dtype=int,
        )

        steps = np.unique(steps)

        if steps[0] != 0:
            steps = np.insert(steps, 0, 0)

        if steps[-1] != interval - 1:
            steps = np.append(steps, interval - 1)

        return steps.tolist()

    @staticmethod
    def _to_01(x):
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    @staticmethod
    def _mask_to_rgb(mask):
        if mask.shape[1] == 1:
            return mask.repeat(1, 3, 1, 1)
        return mask

    @torch.no_grad()
    def _sample_one_log_item(
        self,
        item,
        steps,
    ):
        x0 = item["x0"].unsqueeze(0).to(self.device)
        x1 = item["x1"].unsqueeze(0).to(self.device)
        cond = item["cond"].unsqueeze(0).to(self.device)
        mask = item["mask"].unsqueeze(0).to(self.device)

        def pred_x0_fn(
            xt,
            step_value,
        ):
            step = torch.full(
                (xt.shape[0],),
                int(step_value),
                device=self.device,
                dtype=torch.long,
            )

            net_out = self.ema.ema_model(
                xt,
                step,
                None,
                cond,
            )

            return self.diffusion.compute_pred_x0(
                step=step,
                xt=xt,
                net_out=net_out,
                clip_denoise=True,
            )

        xs, _ = self.diffusion.ddpm_sampling(
            steps=steps,
            pred_x0_fn=pred_x0_fn,
            x1=x1,
            cond=cond,
            mask=mask,
            ot_ode=False,
            log_steps=[0],
            verbose=False,
        )

        recon = xs[:, 0].to(self.device)

        recon = (
            (1.0 - mask) * cond
            + mask * recon
        )

        return {
            "product_name": item["product_name"],
            "sample_id": item["sample_id"],
            "pred": recon.detach().cpu(),
            "mask": mask.detach().cpu(),
            "target": x0.detach().cpu(),
            "cond": cond.detach().cpu(),
            "x1": x1.detach().cpu(),
        }

    @torch.no_grad()
    def sample_and_log(self, milestone):
        if not self.accelerator.is_main_process:
            return

        self.accelerator.print(
            "[I2SB] start product-aware "
            f"eval process: {milestone}"
        )

        self.ema.ema_model.eval()

        steps = self._make_sampling_steps()

        results = [
            self._sample_one_log_item(
                item,
                steps,
            )
            for item in self.log_samples
        ]

        # Combined files for compatibility.
        all_pred = torch.cat(
            [r["pred"] for r in results],
            dim=0,
        )
        all_mask = torch.cat(
            [r["mask"] for r in results],
            dim=0,
        )
        all_target = torch.cat(
            [r["target"] for r in results],
            dim=0,
        )
        all_cond = torch.cat(
            [r["cond"] for r in results],
            dim=0,
        )
        all_x1 = torch.cat(
            [r["x1"] for r in results],
            dim=0,
        )

        pred_vis = self._to_01(all_pred)
        target_vis = self._to_01(all_target)
        cond_vis = self._to_01(all_cond)
        x1_vis = self._to_01(all_x1)
        mask_vis = self._mask_to_rgb(all_mask).clamp(0.0, 1.0)

        nrow = max(1, self.samples_per_product)

        utils.save_image(
            pred_vis,
            str(self.results_folder / f"sample-{milestone}_pred.png"),
            nrow=nrow,
        )
        utils.save_image(
            mask_vis,
            str(self.results_folder / f"sample-{milestone}_mask.png"),
            nrow=nrow,
        )
        utils.save_image(
            target_vis,
            str(self.results_folder / f"sample-{milestone}_target.png"),
            nrow=nrow,
        )
        utils.save_image(
            cond_vis,
            str(self.results_folder / f"sample-{milestone}_cond.png"),
            nrow=nrow,
        )
        utils.save_image(
            x1_vis,
            str(self.results_folder / f"sample-{milestone}_x1.png"),
            nrow=nrow,
        )

        grouped = defaultdict(list)

        for result in results:
            grouped[result["product_name"]].append(result)

        step_name = f"step_{milestone:06d}.png"

        for product_name, product_results in grouped.items():
            product_pred = torch.cat(
                [r["pred"] for r in product_results],
                dim=0,
            )
            product_mask = torch.cat(
                [r["mask"] for r in product_results],
                dim=0,
            )
            product_target = torch.cat(
                [r["target"] for r in product_results],
                dim=0,
            )
            product_cond = torch.cat(
                [r["cond"] for r in product_results],
                dim=0,
            )
            product_x1 = torch.cat(
                [r["x1"] for r in product_results],
                dim=0,
            )

            product_pred_vis = self._to_01(product_pred)
            product_target_vis = self._to_01(product_target)
            product_cond_vis = self._to_01(product_cond)
            product_x1_vis = self._to_01(product_x1)
            product_mask_vis = (
                self._mask_to_rgb(product_mask)
                .clamp(0.0, 1.0)
            )

            product_nrow = max(
                1,
                len(product_results),
            )

            product_dir = self.samples_dir / product_name
            static_dir = product_dir / "static"
            pred_dir = product_dir / "pred"

            static_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            pred_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Fixed items: save only once per run.
            static_items = {
                "target.png": product_target_vis,
                "cond.png": product_cond_vis,
                "mask.png": product_mask_vis,
                "x1.png": product_x1_vis,
            }

            for filename, image_tensor in static_items.items():
                path = static_dir / filename

                if not path.exists():
                    utils.save_image(
                        image_tensor,
                        str(path),
                        nrow=product_nrow,
                    )

            # Dynamic item: save at every checkpoint.
            utils.save_image(
                product_pred_vis,
                str(pred_dir / step_name),
                nrow=product_nrow,
            )

            # Product-specific TensorBoard groups.
            self.writer.add_images(
                f"i2sb/{product_name}/pred",
                product_pred_vis,
                self.step,
                dataformats="NCHW",
            )
            self.writer.add_images(
                f"i2sb/{product_name}/mask",
                product_mask_vis,
                self.step,
                dataformats="NCHW",
            )
            self.writer.add_images(
                f"i2sb/{product_name}/target",
                product_target_vis,
                self.step,
                dataformats="NCHW",
            )
            self.writer.add_images(
                f"i2sb/{product_name}/condition",
                product_cond_vis,
                self.step,
                dataformats="NCHW",
            )
            self.writer.add_images(
                f"i2sb/{product_name}/x1",
                product_x1_vis,
                self.step,
                dataformats="NCHW",
            )

        # Combined TensorBoard overview.
        self.writer.add_images(
            "i2sb/all_products/pred",
            pred_vis,
            self.step,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "i2sb/all_products/mask",
            mask_vis,
            self.step,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "i2sb/all_products/target",
            target_vis,
            self.step,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "i2sb/all_products/condition",
            cond_vis,
            self.step,
            dataformats="NCHW",
        )
        self.writer.add_images(
            "i2sb/all_products/x1",
            x1_vis,
            self.step,
            dataformats="NCHW",
        )

        self.ema.ema_model.train()

        self.accelerator.print(
            "[I2SB] product-aware samples saved under "
            f"{self.samples_dir}"
        )

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.writer = SummaryWriter(
            log_dir=self.sw_dir
        )

        self.model.train()

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(
                    self.gradient_accumulate_every
                ):
                    data = next(self.train_dl)

                    x0 = data["x0"].to(
                        device,
                        non_blocking=True,
                    )
                    x1 = data["x1"].to(
                        device,
                        non_blocking=True,
                    )
                    cond = data["cond"].to(
                        device,
                        non_blocking=True,
                    )
                    mask = data["mask"].to(
                        device,
                        non_blocking=True,
                    )

                    with accelerator.autocast():
                        loss, _ = self.compute_loss(
                            x0=x0,
                            x1=x1,
                            cond=cond,
                            mask=mask,
                            ot_ode=False,
                        )

                        loss = (
                            loss
                            / self.gradient_accumulate_every
                        )

                        total_loss += loss.item()

                    accelerator.backward(loss)

                accelerator.wait_for_everyone()

                accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update()

                    self.writer.add_scalar(
                        "Train_loss",
                        total_loss,
                        self.step,
                    )

                    if (
                        self.step != 0
                        and divisible_by(
                            self.step,
                            self.save_and_sample_every,
                        )
                    ):
                        milestone = self.step

                        self.sample_and_log(milestone)
                        self.save(milestone)

                pbar.set_description(
                    f"loss: {total_loss:.4f}"
                )
                pbar.update(1)

        self.writer.close()

        accelerator.print(
            "I2SB training complete"
        )
