# eval_tgsim.py

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import csv
import json
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from tgsim_dataset import TGSIMCTEDataset, TGSIMDatasetConfig, simple_collate_fn
from dynamic_cte_gnn import DynamicCTEGNN, DynamicCTEGNNConfig
from losses import LossConfig, compute_total_loss


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class TensorCollector:
    """
    Collect a bounded number of valid edge-level values for histogram / npz saving.
    """

    def __init__(self, max_values_per_key: int = 200000):
        self.max_values_per_key = int(max_values_per_key)
        self.storage: Dict[str, List[torch.Tensor]] = {}

    def add(
        self,
        key: str,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
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

            current = 0
            if key in self.storage:
                current = sum(x.numel() for x in self.storage[key])

            remain = self.max_values_per_key - current
            if remain <= 0:
                return

            if values.numel() > remain:
                idx = torch.randperm(values.numel())[:remain]
                values = values[idx]

            self.storage.setdefault(key, []).append(values)

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self.storage:
            return None

        if len(self.storage[key]) == 0:
            return None

        return torch.cat(self.storage[key], dim=0)

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        out = {}

        for key in self.storage.keys():
            values = self.get(key)
            if values is not None:
                out[key] = values.numpy()

        return out


def build_subset(dataset, subset_size: Optional[int], seed: int):
    if subset_size is None:
        return dataset

    subset_size = int(subset_size)

    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False).tolist()
    return Subset(dataset, indices)


def load_checkpoint_configs(
    checkpoint_path: str,
) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model_cfg_dict = ckpt.get("model_cfg", None)
    loss_cfg_dict = ckpt.get("loss_cfg", None)
    dataset_cfg_dict = ckpt.get("dataset_cfg", None)

    if model_cfg_dict is None:
        print("[eval] checkpoint has no model_cfg. Using default DynamicCTEGNNConfig.")
        model_cfg = DynamicCTEGNNConfig()
    else:
        model_cfg = DynamicCTEGNNConfig(**model_cfg_dict)

    if loss_cfg_dict is None:
        print("[eval] checkpoint has no loss_cfg. Using default LossConfig.")
        loss_cfg = LossConfig()
    else:
        loss_cfg = LossConfig(**loss_cfg_dict)

    if dataset_cfg_dict is None:
        print("[eval] checkpoint has no dataset_cfg. Using default TGSIMDatasetConfig.")
        dataset_cfg = TGSIMDatasetConfig()
    else:
        dataset_cfg = TGSIMDatasetConfig(**dataset_cfg_dict)

    return {
        "ckpt": ckpt,
        "model_cfg": model_cfg,
        "loss_cfg": loss_cfg,
        "dataset_cfg": dataset_cfg,
    }


def build_val_loader(
    dataset_cfg: TGSIMDatasetConfig,
    batch_size: int,
    num_workers: int,
    val_subset_size: Optional[int],
    seed: int,
):
    val_cfg = TGSIMDatasetConfig(
        window_path=dataset_cfg.window_path,
        cte_table_path=dataset_cfg.cte_table_path,
        split="val",
        automation_level=dataset_cfg.automation_level,
        top_k_neighbors=dataset_cfg.top_k_neighbors,
        max_ttc=dataset_cfg.max_ttc,
        allow_invalid_lane_edges=dataset_cfg.allow_invalid_lane_edges,
        num_vehicle_types=dataset_cfg.num_vehicle_types,
        padding_type_index=dataset_cfg.padding_type_index,
        verbose=False,
    )

    val_dataset = TGSIMCTEDataset(val_cfg)

    print("\n[val dataset]")
    val_dataset.summary(diagnostics_samples=1000)

    if val_subset_size is not None and int(val_subset_size) > 0:
        print(f"[eval] using val subset: {val_subset_size}")
        val_dataset = build_subset(
            val_dataset,
            subset_size=int(val_subset_size),
            seed=seed,
        )

    pin_memory = torch.cuda.is_available()

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=simple_collate_fn,
        persistent_workers=(num_workers > 0),
    )

    return val_loader


def make_no_interaction_drop_mask(
    batch: Dict[str, torch.Tensor],
    pred_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Drop all candidate edges for all rollout steps.

    This uses the existing edge_drop_mask interface and does not require
    modifying DynamicCTEGNN.
    """
    edge_mask = batch["edge_mask"].to(device).bool()  # [B,N,N]
    b, n, _ = edge_mask.shape

    drop_mask = edge_mask.unsqueeze(1).expand(b, pred_steps, n, n).clone()
    return drop_mask


def run_model_by_mode(
    model: DynamicCTEGNN,
    batch: Dict[str, torch.Tensor],
    mode: str,
    pred_steps: int,
    device: torch.device,
):
    """
    Modes:
        normal:
            learned Hard Concrete graph, deterministic gate.

        no_interaction:
            all betas are removed through edge_drop_mask.

        all_open:
            force use_hard_concrete_gate=False temporarily, so all candidate
            edges are open and beta=alpha.
    """
    old_use_hard_concrete = bool(model.cfg.use_hard_concrete_gate)

    try:
        if mode == "normal":
            out = model(
                batch,
                pred_steps=pred_steps,
                deterministic_gate=True,
                edge_drop_mask=None,
            )

        elif mode == "no_interaction":
            drop_mask = make_no_interaction_drop_mask(
                batch=batch,
                pred_steps=pred_steps,
                device=device,
            )

            out = model(
                batch,
                pred_steps=pred_steps,
                deterministic_gate=True,
                edge_drop_mask=drop_mask,
            )

        elif mode == "all_open":
            model.cfg.use_hard_concrete_gate = False

            out = model(
                batch,
                pred_steps=pred_steps,
                deterministic_gate=True,
                edge_drop_mask=None,
            )

        else:
            raise ValueError(
                f"Unknown mode: {mode}. "
                "Expected 'normal', 'no_interaction', or 'all_open'."
            )

    finally:
        model.cfg.use_hard_concrete_gate = old_use_hard_concrete

    return out


@torch.no_grad()
def evaluate_mode(
    model: DynamicCTEGNN,
    val_loader,
    loss_cfg: LossConfig,
    mode: str,
    pred_steps: int,
    device: torch.device,
    max_batches: Optional[int],
    writer: Optional[SummaryWriter],
    collector_max_values: int,
) -> Dict[str, float]:
    model.eval()

    meter = AverageMeter()
    collector = TensorCollector(max_values_per_key=collector_max_values)

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break

        batch = move_batch_to_device(batch, device)
        batch_size = batch["history"].shape[0]

        out = run_model_by_mode(
            model=model,
            batch=batch,
            mode=mode,
            pred_steps=pred_steps,
            device=device,
        )

        loss_dict = compute_total_loss(out, batch, loss_cfg)
        meter.update(loss_dict, n=batch_size)

        edge_mask = out["edge_mask"]  # [B,N,N]
        b, t, n, _ = out["open_probs"].shape
        edge_mask_t = edge_mask.unsqueeze(1).expand(b, t, n, n)

        collector.add("open_probs", out.get("open_probs"), edge_mask_t)
        collector.add("betas", out.get("betas"), edge_mask_t)
        collector.add("alphas", out.get("alphas"), edge_mask_t)
        collector.add("gates", out.get("gates"), edge_mask_t)
        collector.add("scores", out.get("scores"), edge_mask_t)
        collector.add("gate_logits", out.get("gate_logits"), edge_mask_t)
        collector.add("cte_teacher", out.get("cte_teacher"), edge_mask_t)

        if (batch_idx + 1) % 50 == 0:
            avg_now = meter.average()
            print(
                f"[eval:{mode}] batch={batch_idx + 1:04d} | "
                f"ade_t={avg_now.get('ade_target', 0.0):.6f} | "
                f"rmse={avg_now.get('rmse_all', 0.0):.6f} | "
                f"R1={avg_now.get('rmse_target_1s', 0.0):.4f} | "
                f"R5={avg_now.get('rmse_target_5s', 0.0):.4f} | "
                f"ADE5={avg_now.get('ade_target_5s', 0.0):.4f}"
            )

    avg = meter.average()

    if writer is not None:
        for k, v in avg.items():
            writer.add_scalar(f"{mode}/{k}", v, 0)

        for key, values_np in collector.to_numpy_dict().items():
            if values_np.size > 0:
                writer.add_histogram(
                    f"{mode}_hist/{key}",
                    torch.from_numpy(values_np),
                    0,
                )

    avg["_hist"] = collector.to_numpy_dict()
    return avg


def save_metrics_csv(
    metrics_by_mode: Dict[str, Dict[str, Any]],
    path: Path,
) -> None:
    all_keys = set()

    for mode, metrics in metrics_by_mode.items():
        for k, v in metrics.items():
            if k == "_hist":
                continue
            all_keys.add(k)

    ordered_keys = ["mode"] + sorted(all_keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()

        for mode, metrics in metrics_by_mode.items():
            row = {"mode": mode}

            for k in ordered_keys:
                if k == "mode":
                    continue
                row[k] = metrics.get(k, "")

            writer.writerow(row)

    print(f"[eval] saved metrics csv: {path}")


def save_hist_npz(
    metrics_by_mode: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    for mode, metrics in metrics_by_mode.items():
        hist = metrics.get("_hist", None)
        if hist is None:
            continue

        if len(hist) == 0:
            continue

        path = output_dir / f"hist_{mode}.npz"
        np.savez_compressed(path, **hist)
        print(f"[eval] saved hist npz: {path}")


def print_comparison(metrics_by_mode: Dict[str, Dict[str, Any]]) -> None:
    keys = [
        "ade_target",
        "ade_target_5s",
        "rmse_all",
        "rmse_target_1s",
        "rmse_target_2s",
        "rmse_target_3s",
        "rmse_target_4s",
        "rmse_target_5s",
        "open_prob_mean",
        "beta_mean",
        "gate_mean",
        "alpha_mean",
    ]

    print("\n========== Interaction Ablation Comparison ==========")

    header = "metric".ljust(22)
    for mode in metrics_by_mode.keys():
        header += f" | {mode:>16}"
    print(header)
    print("-" * len(header))

    for key in keys:
        row = key.ljust(22)

        for mode, metrics in metrics_by_mode.items():
            value = metrics.get(key, None)
            if value is None:
                row += f" | {'NA':>16}"
            else:
                row += f" | {float(value):16.6f}"

        print(row)

    print("====================================================\n")

    if "normal" in metrics_by_mode and "no_interaction" in metrics_by_mode:
        normal = metrics_by_mode["normal"].get("ade_target", None)
        no_int = metrics_by_mode["no_interaction"].get("ade_target", None)

        if normal is not None and no_int is not None:
            delta = float(no_int) - float(normal)
            rel = delta / (float(normal) + 1e-8)

            print(
                f"[decision] no_interaction - normal ADE_target = "
                f"{delta:.6f} ({rel * 100:.2f}%)"
            )

            if rel > 0.05:
                print("[decision] Interaction has clear predictive contribution.")
            elif rel > 0.02:
                print("[decision] Interaction has weak but visible contribution.")
            elif abs(rel) <= 0.02:
                print("[decision] Interaction contribution is small; be careful with AI4S discovery.")
            else:
                print("[decision] No-interaction is better; message passing may be hurting prediction.")

    if "normal" in metrics_by_mode and "all_open" in metrics_by_mode:
        normal = metrics_by_mode["normal"].get("ade_target", None)
        all_open = metrics_by_mode["all_open"].get("ade_target", None)

        if normal is not None and all_open is not None:
            delta = float(all_open) - float(normal)
            rel = delta / (float(normal) + 1e-8)

            print(
                f"[decision] all_open - normal ADE_target = "
                f"{delta:.6f} ({rel * 100:.2f}%)"
            )

            if rel < -0.02:
                print("[decision] All-open is better; current gate/L0 may be too sparse.")
            elif rel > 0.02:
                print("[decision] Sparse learned graph is better than dense all-open.")
            else:
                print("[decision] Normal and all-open are similar.")


def main() -> None:
    seed = 42
    seed_everything(seed)

    # =========================
    # Edit these in PyCharm
    # =========================

    checkpoint_path = "./Runs/TGSIM_CTEGNN/your_tgsim_run/checkpoints/checkpoint_best.pt"

    output_dir = "./Runs/TGSIM_CTEGNN/eval_interaction_ablation_tgsim"

    batch_size = 16
    num_workers = 0
    val_subset_size = 10000       # set None for full val
    max_batches = None            # set e.g. 100 for quick eval
    pred_steps_override = None    # set None to use checkpoint model_cfg.pred_steps

    modes = [
        "normal",
        "no_interaction",
        "all_open",
    ]

    collector_max_values = 200000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # =========================
    # Load checkpoint and configs
    # =========================

    ckpt_data = load_checkpoint_configs(checkpoint_path)
    ckpt = ckpt_data["ckpt"]
    model_cfg = ckpt_data["model_cfg"]
    loss_cfg = ckpt_data["loss_cfg"]
    dataset_cfg = ckpt_data["dataset_cfg"]

    if pred_steps_override is not None:
        model_cfg.pred_steps = int(pred_steps_override)

    pred_steps = int(model_cfg.pred_steps)

    print("[eval] checkpoint:", checkpoint_path)
    print("[eval] device:", device)
    print("[eval] pred_steps:", pred_steps)
    print("[eval] model_cfg:", asdict(model_cfg))
    print("[eval] loss_cfg:", asdict(loss_cfg))
    print("[eval] dataset_cfg:", asdict(dataset_cfg))

    # =========================
    # Build data / model
    # =========================

    val_loader = build_val_loader(
        dataset_cfg=dataset_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        val_subset_size=val_subset_size,
        seed=seed,
    )

    model = DynamicCTEGNN(model_cfg).to(device)

    if "model" not in ckpt:
        raise KeyError("Checkpoint does not contain key 'model'.")

    model.load_state_dict(ckpt["model"])
    model.eval()

    # Save config snapshot.
    config_snapshot = {
        "checkpoint_path": checkpoint_path,
        "output_dir": str(output_dir),
        "batch_size": batch_size,
        "val_subset_size": val_subset_size,
        "max_batches": max_batches,
        "pred_steps": pred_steps,
        "modes": modes,
        "model_cfg": asdict(model_cfg),
        "loss_cfg": asdict(loss_cfg),
        "dataset_cfg": asdict(dataset_cfg),
    }

    with open(output_dir / "eval_config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # =========================
    # Evaluate modes
    # =========================

    metrics_by_mode: Dict[str, Dict[str, Any]] = {}

    for mode in modes:
        print(f"\n[eval] running mode: {mode}")

        metrics = evaluate_mode(
            model=model,
            val_loader=val_loader,
            loss_cfg=loss_cfg,
            mode=mode,
            pred_steps=pred_steps,
            device=device,
            max_batches=max_batches,
            writer=writer,
            collector_max_values=collector_max_values,
        )

        metrics_by_mode[mode] = metrics

        print(f"[eval:{mode}] summary:")
        for k, v in metrics.items():
            if k == "_hist":
                continue
            print(f"  {k}: {v}")

    writer.close()

    # =========================
    # Save / print comparison
    # =========================

    save_metrics_csv(
        metrics_by_mode=metrics_by_mode,
        path=output_dir / "metrics_interaction_ablation.csv",
    )

    save_hist_npz(
        metrics_by_mode=metrics_by_mode,
        output_dir=output_dir,
    )

    print_comparison(metrics_by_mode)

    print("[eval] done.")


if __name__ == "__main__":
    main()