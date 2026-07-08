# trainer.py

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Any
import time
import json

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from losses import LossConfig, compute_total_loss
from counterfactual import CounterfactualConfig, counterfactual_gate_loss

@dataclass
class TrainerConfig:
    output_dir: str = "./Runs/NGSIM_CTEGNN/debug"

    max_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0

    device: str = "cuda"

    # Logging
    log_every: int = 50
    histogram_every: int = 200

    # Validation / checkpoint
    val_every: int = 1
    save_every: int = 1

    # Optional debugging limits
    max_train_batches_per_epoch: Optional[int] = None
    max_val_batches: Optional[int] = 200

    # Counterfactual switch
    use_cf: bool = False
    lambda_cf: float = 1e-4
    cf_start_epoch: int = 20
    cf_every: int = 5

    # AMP
    use_amp: bool = False

    # Checkpoint resume
    resume_path: Optional[str] = None


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    moved = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v

    return moved


def tensor_to_float(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


class AverageMeter:
    def __init__(self):
        self.data = {}

    def update(self, values: Dict[str, Any], n: int = 1) -> None:
        for k, v in values.items():
            if not torch.is_tensor(v):
                continue

            if v.numel() != 1:
                continue

            value = float(v.detach().cpu())

            if k not in self.data:
                self.data[k] = {
                    "sum": 0.0,
                    "count": 0,
                }

            self.data[k]["sum"] += value * n
            self.data[k]["count"] += n

    def average(self) -> Dict[str, float]:
        out = {}

        for k, item in self.data.items():
            if item["count"] > 0:
                out[k] = item["sum"] / item["count"]

        return out


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_cfg: LossConfig,
        trainer_cfg: TrainerConfig,
        cf_cfg: Optional[CounterfactualConfig] = None,
        model_cfg: Optional[Any] = None,
        dataset_cfg: Optional[Any] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_cfg = loss_cfg
        self.trainer_cfg = trainer_cfg
        self.cf_cfg = cf_cfg
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg

        if trainer_cfg.device == "cuda" and not torch.cuda.is_available():
            print("[Trainer] CUDA requested but not available. Using CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(trainer_cfg.device)

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(trainer_cfg.lr),
            weight_decay=float(trainer_cfg.weight_decay),
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(trainer_cfg.use_amp and self.device.type == "cuda")
        )

        self.output_dir = Path(trainer_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_ade = float("inf")

        self._save_config_json()

        if trainer_cfg.resume_path is not None and len(trainer_cfg.resume_path) > 0:
            self.load_checkpoint(trainer_cfg.resume_path)

    def _save_config_json(self) -> None:
        config = {
            "trainer_cfg": asdict(self.trainer_cfg),
            "loss_cfg": asdict(self.loss_cfg),
            "cf_cfg": asdict(self.cf_cfg) if self.cf_cfg is not None else None,
            "model_cfg": asdict(self.model_cfg) if self.model_cfg is not None else None,
            "dataset_cfg": asdict(self.dataset_cfg) if self.dataset_cfg is not None else None,
        }

        path = self.output_dir / "config.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def should_use_cf(self, epoch: int, batch_idx: int) -> bool:
        if not self.trainer_cfg.use_cf:
            return False

        if self.cf_cfg is None:
            return False

        if epoch < int(self.trainer_cfg.cf_start_epoch):
            return False

        cf_every = int(self.trainer_cfg.cf_every)
        if cf_every <= 0:
            return False

        # 每个 epoch 内固定频率触发。
        # batch_idx=0, cf_every, 2*cf_every, ...
        return (batch_idx % cf_every) == 0

    def train(self) -> None:
        print("[Trainer] Start training")
        print("[Trainer] output_dir:", self.output_dir)
        print("[Trainer] device:", self.device)
        print("[Trainer] use_cf:", self.trainer_cfg.use_cf)

        for epoch in range(self.start_epoch, int(self.trainer_cfg.max_epochs)):
            epoch_start = time.time()

            train_avg = self.train_one_epoch(epoch)

            print(
                f"[Epoch {epoch:03d}] train done | "
                f"total={train_avg.get('total_effective', 0.0):.6f} | "
                f"traj={train_avg.get('traj', 0.0):.6f} | "
                f"cte={train_avg.get('cte', 0.0):.6f} | "
                f"l0={train_avg.get('l0', 0.0):.6f} | "
                f"ade_t={train_avg.get('ade_target', 0.0):.6f} | "
                f"open={train_avg.get('open_prob_mean', 0.0):.6f} | "
                f"time={time.time() - epoch_start:.1f}s"
            )

            if (epoch + 1) % int(self.trainer_cfg.val_every) == 0:
                val_avg = self.validate(epoch)

                val_ade = val_avg.get("ade_target", float("inf"))

                print(
                    f"[Epoch {epoch:03d}] val done | "
                    f"total={val_avg.get('total', 0.0):.6f} | "
                    f"traj={val_avg.get('traj', 0.0):.6f} | "
                    f"cte={val_avg.get('cte', 0.0):.6f} | "
                    f"l0={val_avg.get('l0', 0.0):.6f} | "
                    f"ade_t={val_ade:.6f} | "
                    f"rmse={val_avg.get('rmse_all', 0.0):.6f}"
                )

                if val_ade < self.best_val_ade:
                    self.best_val_ade = val_ade
                    self.save_checkpoint(epoch, name="checkpoint_best.pt")

            if (epoch + 1) % int(self.trainer_cfg.save_every) == 0:
                self.save_checkpoint(epoch, name="checkpoint_latest.pt")
                self.save_checkpoint(epoch, name=f"checkpoint_epoch_{epoch:03d}.pt")

        self.writer.close()
        print("[Trainer] Training finished")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = AverageMeter()

        for batch_idx, batch in enumerate(self.train_loader):
            if (
                self.trainer_cfg.max_train_batches_per_epoch is not None
                and batch_idx >= int(self.trainer_cfg.max_train_batches_per_epoch)
            ):
                break

            batch = move_batch_to_device(batch, self.device)
            batch_size = batch["history"].shape[0]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=bool(self.trainer_cfg.use_amp and self.device.type == "cuda")
            ):
                out = self.model(batch)
                loss_dict = compute_total_loss(out, batch, self.loss_cfg)

                total_loss = loss_dict["total"]
                cf_dict = None

                if self.should_use_cf(epoch,batch_idx):
                    cf_dict = counterfactual_gate_loss(
                        model=self.model,
                        batch=batch,
                        out_full=out,
                        cfg=self.cf_cfg,
                    )
                    total_loss = total_loss + float(self.trainer_cfg.lambda_cf) * cf_dict["cf"]

            if not torch.isfinite(total_loss):
                print(f"[Trainer] Non-finite loss at step {self.global_step}. Skipping batch.")
                continue

            self.scaler.scale(total_loss).backward()

            if self.trainer_cfg.grad_clip_norm is not None and self.trainer_cfg.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=float(self.trainer_cfg.grad_clip_norm),
                )
            else:
                grad_norm = torch.tensor(0.0, device=self.device)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            log_dict = dict(loss_dict)
            log_dict["total_effective"] = total_loss.detach()
            log_dict["grad_norm"] = grad_norm.detach()

            if cf_dict is not None:
                log_dict["cf"] = cf_dict["cf"].detach()
                log_dict["cf_num_edges"] = cf_dict["cf_num_edges"].detach()
                log_dict["cf_mean_degradation"] = cf_dict["cf_mean_degradation"].detach()
                log_dict["cf_max_degradation"] = cf_dict["cf_max_degradation"].detach()
                log_dict["cf_mean_label"] = cf_dict["cf_mean_label"].detach()
                log_dict["cf_mean_open_prob"] = cf_dict["cf_mean_open_prob"].detach()
            else:
                log_dict["cf"] = torch.tensor(0.0, device=self.device)
                log_dict["cf_num_edges"] = torch.tensor(0.0, device=self.device)
                log_dict["cf_mean_degradation"] = torch.tensor(0.0, device=self.device)
                log_dict["cf_max_degradation"] = torch.tensor(0.0, device=self.device)
                log_dict["cf_mean_label"] = torch.tensor(0.0, device=self.device)
                log_dict["cf_mean_open_prob"] = torch.tensor(0.0, device=self.device)

            meter.update(log_dict, n=batch_size)

            if self.global_step % int(self.trainer_cfg.log_every) == 0:
                self.log_train_scalars(log_dict)
                self.print_train_line(epoch, batch_idx, log_dict)

            if self.global_step % int(self.trainer_cfg.histogram_every) == 0:
                self.log_histograms(out, prefix="train_hist")

            self.global_step += 1

        return meter.average()

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        meter = AverageMeter()

        for batch_idx, batch in enumerate(self.val_loader):
            if (
                self.trainer_cfg.max_val_batches is not None
                and batch_idx >= int(self.trainer_cfg.max_val_batches)
            ):
                break

            batch = move_batch_to_device(batch, self.device)
            batch_size = batch["history"].shape[0]

            out = self.model(
                batch,
                deterministic_gate=True,
            )

            loss_dict = compute_total_loss(out, batch, self.loss_cfg)
            meter.update(loss_dict, n=batch_size)

        avg = meter.average()

        for k, v in avg.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)

        return avg

    def format_optional_metric(
            self,
            log_dict: Dict[str, torch.Tensor],
            key: str,
            name: str,
    ) -> str:
        if key not in log_dict:
            return f"{name}=NA"

        value = tensor_to_float(log_dict[key])
        return f"{name}={value:.4f}"

    def print_train_line(
            self,
            epoch: int,
            batch_idx: int,
            log_dict: Dict[str, torch.Tensor],
    ) -> None:
        rmse_str = " | ".join(
            [
                self.format_optional_metric(log_dict, "rmse_target_1s", "R1"),
                self.format_optional_metric(log_dict, "rmse_target_2s", "R2"),
                self.format_optional_metric(log_dict, "rmse_target_3s", "R3"),
                self.format_optional_metric(log_dict, "rmse_target_4s", "R4"),
                self.format_optional_metric(log_dict, "rmse_target_5s", "R5"),
                self.format_optional_metric(log_dict, "ade_target_5s", "ADE5"),
            ]
        )

        print(
            f"[train] epoch={epoch:03d} | "
            f"batch={batch_idx:05d} | "
            f"step={self.global_step:07d} | "
            f"total={tensor_to_float(log_dict['total_effective']):.6f} | "
            f"traj={tensor_to_float(log_dict['traj']):.6f} | "
            f"cte={tensor_to_float(log_dict['cte']):.6f} | "
            f"l0={tensor_to_float(log_dict['l0']):.6f} | "
            f"cf={tensor_to_float(log_dict['cf']):.6f} | "
            f"ade_t={tensor_to_float(log_dict['ade_target']):.6f} | "
            f"{rmse_str} | "
            f"open={tensor_to_float(log_dict['open_prob_mean']):.4f} | "
            f"beta={tensor_to_float(log_dict['beta_mean']):.4f} | "
            f"cf_deg={tensor_to_float(log_dict['cf_mean_degradation']):.6f}"
        )

    def log_train_scalars(self, log_dict: Dict[str, torch.Tensor]) -> None:
        scalar_keys = [
            "total_effective",
            "total",
            "traj",
            "pos",
            "vel",
            "cte",
            "l0",
            "cf",
            "ade_all",
            "ade_target",
            "rmse_all",
            "rmse_target_1s",
            "rmse_target_2s",
            "rmse_target_3s",
            "rmse_target_4s",
            "rmse_target_5s",
            "ade_target_5s",
            "cte_groups",
            "cte_valid_edges",
            "open_prob_mean",
            "beta_mean",
            "gate_mean",
            "alpha_mean",
            "grad_norm",
            "cf_num_edges",
            "cf_mean_degradation",
            "cf_max_degradation",
            "cf_mean_label",
            "cf_mean_open_prob",
        ]

        for key in scalar_keys:
            if key in log_dict and torch.is_tensor(log_dict[key]):
                self.writer.add_scalar(
                    f"train/{key}",
                    float(log_dict[key].detach().cpu()),
                    self.global_step,
                )

    def log_histograms(
        self,
        out: Dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        edge_mask = out["edge_mask"]             # [B,N,N]
        open_probs = out["open_probs"]           # [B,T,N,N]
        b, t, n, _ = open_probs.shape

        edge_mask_t = edge_mask.unsqueeze(1).expand(b, t, n, n)

        def add_hist(name: str, tensor: torch.Tensor, mask: Optional[torch.Tensor] = None):
            if tensor is None:
                return

            with torch.no_grad():
                if mask is not None:
                    values = tensor[mask]
                else:
                    values = tensor.reshape(-1)

                if values.numel() == 0:
                    return

                values = values.detach().float().cpu()
                self.writer.add_histogram(
                    f"{prefix}/{name}",
                    values,
                    self.global_step,
                )

        add_hist("open_probs", out.get("open_probs"), edge_mask_t)
        add_hist("betas", out.get("betas"), edge_mask_t)
        add_hist("gates", out.get("gates"), edge_mask_t)
        add_hist("alphas", out.get("alphas"), edge_mask_t)
        add_hist("scores", out.get("scores"), edge_mask_t)
        add_hist("gate_logits", out.get("gate_logits"), edge_mask_t)

        if "cte_teacher" in out:
            add_hist("cte_teacher", out["cte_teacher"], edge_mask_t)

        if "cte_valid" in out:
            valid_ratio = out["cte_valid"].float().mean()
            self.writer.add_scalar(
                f"{prefix}/cte_valid_ratio",
                float(valid_ratio.detach().cpu()),
                self.global_step,
            )

    def save_checkpoint(self, epoch: int, name: str = "checkpoint_latest.pt") -> None:
        path = self.ckpt_dir / name

        ckpt = {
            "epoch": int(epoch),
            "global_step": int(self.global_step),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_val_ade": float(self.best_val_ade),
            "trainer_cfg": asdict(self.trainer_cfg),
            "loss_cfg": asdict(self.loss_cfg),
            "cf_cfg": asdict(self.cf_cfg) if self.cf_cfg is not None else None,
            "model_cfg": asdict(self.model_cfg) if self.model_cfg is not None else None,
            "dataset_cfg": asdict(self.dataset_cfg) if self.dataset_cfg is not None else None,
        }

        torch.save(ckpt, path)
        print(f"[Trainer] Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])

        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_val_ade = float(ckpt.get("best_val_ade", float("inf")))

        print(f"[Trainer] Loaded checkpoint: {ckpt_path}")
        print(f"[Trainer] start_epoch={self.start_epoch}, global_step={self.global_step}")