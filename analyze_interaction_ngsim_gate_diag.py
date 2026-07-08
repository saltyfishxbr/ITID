# analyze_interactions_ngsim.py

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import csv
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ngsim_dataset import NGSIMCTEDataset, NGSIMDatasetConfig, simple_collate_fn
from dynamic_cte_gnn import DynamicCTEGNN, DynamicCTEGNNConfig
from losses import LossConfig


IDX_X = 0
IDX_Y = 1
IDX_VX = 2
IDX_VY = 3
IDX_SPEED = 4
IDX_ACC = 5

LANE_SAME_FRONT = 0
LANE_SAME_BACK = 1
LANE_ADJACENT = 2

TYPE_NAMES_NGSIM = {
    0: "motorcycle",
    1: "small_vehicle",
    2: "large_vehicle",
    3: "padding",
}

LANE_REL_NAMES = {
    0: "same_lane_front",
    1: "same_lane_back",
    2: "adjacent_lane",
    -1: "invalid",
}


@dataclass
class AnalysisConfig:
    checkpoint_path: str = "./Runs/NGSIM_CTEGNN/ngsim_ctegnn_no_cf_20260615_181812/checkpoints/checkpoint_best.pt"
    output_dir: str = "./Runs/NGSIM_CTEGNN/interaction_analysis_ngsim"

    split: str = "val"
    batch_size: int = 16
    num_workers: int = 0
    val_subset_size: Optional[int] = 2000
    max_batches: Optional[int] = None
    pred_steps_override: Optional[int] = None

    seed: int = 42
    device: str = "cuda"

    # Active edge definition.
    active_open_threshold: float = 0.5
    strong_beta_quantile: float = 0.90

    # Gate / open-probability diagnostics.
    # active_edge_ratio = mean(open_prob > active_open_threshold), within edge_mask.
    # hesitation_ratio   = mean(hesitation_low <= open_prob <= hesitation_high), within edge_mask.
    # gate_activation_rate = mean(gate > gate_active_threshold), within edge_mask.
    hesitation_low: float = 0.2
    hesitation_high: float = 0.8
    gate_active_threshold: float = 0.5
    max_gate_diag_values_per_group: int = 500_000

    # Export control.
    write_edge_csv: bool = True
    write_node_role_csv: bool = True

    # To avoid huge CSV.
    edge_export_min_open_prob: float = 0.5
    max_edge_rows: int = 2_000_000

    # Spearman sample cap per group.
    max_spearman_values_per_group: int = 200_000

    # Perturbation propagation.
    perturbation_lags: Tuple[int, ...] = (5, 10, 15, 20, 30)
    perturbation_eps_mode: str = "abs_delta_speed"  # "abs_acc" or "abs_delta_speed"
    downstream_mode: str = "same_lane_following"   # "same_lane_following" or "all_outgoing"

    # Physical binning.
    d_abs_bins: Tuple[float, ...] = (0.0, 10.0, 20.0, 40.0, 80.0, 1e9)
    dv_bins: Tuple[float, ...] = (-1e9, -5.0, -2.0, 2.0, 5.0, 1e9)
    ttc_bins: Tuple[float, ...] = (0.0, 2.0, 4.0, 8.0, 15.0, 20.0, 1e9)
    density_bins: Tuple[float, ...] = (-1e9, 0.025, 0.06, 1e9)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def safe_make_config(cls, cfg_dict: Optional[Dict[str, Any]]):
    if cfg_dict is None:
        return cls()

    valid_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_names}
    return cls(**filtered)


def load_checkpoint_and_configs(cfg: AnalysisConfig):
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")

    model_cfg = safe_make_config(DynamicCTEGNNConfig, ckpt.get("model_cfg", None))
    loss_cfg = safe_make_config(LossConfig, ckpt.get("loss_cfg", None))
    dataset_cfg = safe_make_config(NGSIMDatasetConfig, ckpt.get("dataset_cfg", None))

    if cfg.pred_steps_override is not None:
        model_cfg.pred_steps = int(cfg.pred_steps_override)

    return ckpt, model_cfg, loss_cfg, dataset_cfg


def build_subset(dataset, subset_size: Optional[int], seed: int):
    if subset_size is None:
        return dataset

    subset_size = int(subset_size)
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False).tolist()
    return Subset(dataset, indices)


def build_loader(dataset_cfg: NGSIMDatasetConfig, cfg: AnalysisConfig):
    ds_cfg = NGSIMDatasetConfig(
        window_path=dataset_cfg.window_path,
        cte_table_path=dataset_cfg.cte_table_path,
        split=cfg.split,
        top_k_neighbors=dataset_cfg.top_k_neighbors,
        max_ttc=dataset_cfg.max_ttc,
        allow_invalid_lane_edges=dataset_cfg.allow_invalid_lane_edges,
        num_vehicle_types=dataset_cfg.num_vehicle_types,
        padding_type_index=dataset_cfg.padding_type_index,
        verbose=False,
    )

    dataset = NGSIMCTEDataset(ds_cfg)
    dataset.summary(diagnostics_samples=1000)

    dataset = build_subset(dataset, cfg.val_subset_size, cfg.seed)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=simple_collate_fn,
        persistent_workers=(cfg.num_workers > 0),
    )

    return loader


def type_name(t: int) -> str:
    return TYPE_NAMES_NGSIM.get(int(t), f"unknown_{int(t)}")


def lane_name(x: int) -> str:
    return LANE_REL_NAMES.get(int(x), f"lane_rel_{int(x)}")


def bin_name(value: float, bins: Tuple[float, ...], prefix: str) -> str:
    v = float(value)
    bins_arr = list(bins)

    for k in range(len(bins_arr) - 1):
        lo = bins_arr[k]
        hi = bins_arr[k + 1]

        if lo <= v < hi:
            lo_s = "-inf" if lo < -1e8 else f"{lo:g}"
            hi_s = "inf" if hi > 1e8 else f"{hi:g}"
            return f"{prefix}[{lo_s},{hi_s})"

    return f"{prefix}_out_of_range"


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """
    Average-rank implementation for Spearman correlation.
    """
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 5:
        return float("nan")

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")

    rx = rankdata_average(x)
    ry = rankdata_average(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return float("nan")

    return float((rx * ry).sum() / denom)


class GroupMeanStats:
    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, group_key: str, values: Dict[str, float]) -> None:
        for k, v in values.items():
            if not np.isfinite(v):
                continue
            key = (group_key, k)
            self.sum[key] += float(v)
            self.count[key] += 1

    def to_rows(self, group_name: str) -> List[Dict[str, Any]]:
        groups = sorted({g for (g, _) in self.sum.keys()})
        metrics = sorted({m for (_, m) in self.sum.keys()})

        rows = []
        for g in groups:
            row = {group_name: g}
            for m in metrics:
                key = (g, m)
                if self.count[key] > 0:
                    row[m] = self.sum[key] / self.count[key]
                else:
                    row[m] = ""
            rows.append(row)

        return rows


class EdgeGateDiagnostics:
    """
    Summarize whether the learned graph is sparse / decisive.

    All statistics are computed only on valid candidate edges inside edge_mask.

    active_edge_ratio:
        mean(open_prob > active_threshold)

    open_prob_median:
        median(open_prob)

    open_prob_hesitation_ratio:
        mean(hesitation_low <= open_prob <= hesitation_high)

    gate_activation_rate:
        mean(gate > gate_threshold)
        In analysis we usually call model(..., deterministic_gate=True), so gate is
        the deterministic Hard-Concrete decision. If deterministic_gate=False is
        used later, this becomes the sampled gate activation rate.
    """

    def __init__(
        self,
        active_threshold: float = 0.5,
        hesitation_low: float = 0.2,
        hesitation_high: float = 0.8,
        gate_threshold: float = 0.5,
        max_values_per_group: int = 500_000,
    ):
        self.active_threshold = float(active_threshold)
        self.hesitation_low = float(hesitation_low)
        self.hesitation_high = float(hesitation_high)
        self.gate_threshold = float(gate_threshold)
        self.max_values_per_group = int(max_values_per_group)

        self.count = defaultdict(int)
        self.active_sum = defaultdict(float)
        self.hesitation_sum = defaultdict(float)
        self.gate_active_sum = defaultdict(float)
        self.gate_sum = defaultdict(float)
        self.open_prob_sum = defaultdict(float)

        # For medians / quantiles. This is capped per group to avoid huge memory.
        self.open_values = defaultdict(list)

    def update(self, group_key: str, open_prob: float, gate: float) -> None:
        op = float(open_prob)
        ga = float(gate)
        if not np.isfinite(op) or not np.isfinite(ga):
            return

        self.count[group_key] += 1
        self.open_prob_sum[group_key] += op
        self.active_sum[group_key] += 1.0 if op > self.active_threshold else 0.0
        self.hesitation_sum[group_key] += 1.0 if (self.hesitation_low <= op <= self.hesitation_high) else 0.0
        self.gate_active_sum[group_key] += 1.0 if ga > self.gate_threshold else 0.0
        self.gate_sum[group_key] += ga

        vals = self.open_values[group_key]
        if len(vals) < self.max_values_per_group:
            vals.append(op)

    def to_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for g in sorted(self.count.keys()):
            n = int(self.count[g])
            if n <= 0:
                continue

            vals = np.asarray(self.open_values.get(g, []), dtype=np.float64)
            if vals.size > 0:
                median = float(np.median(vals))
                q25 = float(np.quantile(vals, 0.25))
                q75 = float(np.quantile(vals, 0.75))
                sampled_values = int(vals.size)
            else:
                median = float("nan")
                q25 = float("nan")
                q75 = float("nan")
                sampled_values = 0

            rows.append(
                {
                    "group": g,
                    "num_valid_edges": n,
                    "num_values_for_quantile": sampled_values,
                    "open_prob_mean": self.open_prob_sum[g] / n,
                    "open_prob_median": median,
                    "open_prob_q25": q25,
                    "open_prob_q75": q75,
                    "active_edge_ratio": self.active_sum[g] / n,
                    "open_prob_hesitation_ratio": self.hesitation_sum[g] / n,
                    "gate_mean": self.gate_sum[g] / n,
                    "gate_activation_rate": self.gate_active_sum[g] / n,
                    "active_threshold": self.active_threshold,
                    "hesitation_low": self.hesitation_low,
                    "hesitation_high": self.hesitation_high,
                    "gate_threshold": self.gate_threshold,
                }
            )
        return rows


class SpearmanCollector:
    def __init__(self, max_values_per_group: int):
        self.max_values_per_group = int(max_values_per_group)
        self.data = defaultdict(lambda: defaultdict(list))
        self.count = defaultdict(int)

    def add(self, group_kind: str, group_value: str, metric_name: str, x: float, y: float) -> None:
        group = (group_kind, group_value, metric_name)
        if self.count[group] >= self.max_values_per_group:
            return

        if not np.isfinite(x) or not np.isfinite(y):
            return

        self.data[group]["x"].append(float(x))
        self.data[group]["y"].append(float(y))
        self.count[group] += 1

    def to_rows(self) -> List[Dict[str, Any]]:
        rows = []

        for (group_kind, group_value, metric_name), xy in sorted(self.data.items()):
            x = np.asarray(xy["x"], dtype=np.float64)
            y = np.asarray(xy["y"], dtype=np.float64)

            rows.append(
                {
                    "group_kind": group_kind,
                    "group": group_value,
                    "metric": metric_name,
                    "n": int(len(x)),
                    "spearman": spearman_corr(x, y),
                }
            )

        return rows


class MotifCounter:
    def __init__(self):
        self.counts = defaultdict(int)

    def update(self, group_key: str, motif: str) -> None:
        self.counts[(group_key, motif)] += 1

    def to_rows(self) -> List[Dict[str, Any]]:
        groups = sorted({g for (g, _) in self.counts.keys()})
        motifs = ["none", "u_to_v_only", "v_to_u_only", "mutual"]

        rows = []
        for g in groups:
            total = sum(self.counts[(g, m)] for m in motifs)
            row = {"group": g, "total_pairs": total}
            for m in motifs:
                c = self.counts[(g, m)]
                row[f"{m}_count"] = c
                row[f"{m}_ratio"] = c / total if total > 0 else 0.0
            rows.append(row)

        return rows


class AmplificationStats:
    def __init__(self):
        self.sum_pemit = defaultdict(float)
        self.sum_rdown = defaultdict(float)
        self.sum_amp = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, group_key: str, lag: int, pemit: float, rdown: float, amp: float) -> None:
        key = (group_key, int(lag))
        if not (np.isfinite(pemit) and np.isfinite(rdown) and np.isfinite(amp)):
            return

        self.sum_pemit[key] += float(pemit)
        self.sum_rdown[key] += float(rdown)
        self.sum_amp[key] += float(amp)
        self.count[key] += 1

    def to_rows(self) -> List[Dict[str, Any]]:
        rows = []
        for (group_key, lag), n in sorted(self.count.items()):
            rows.append(
                {
                    "group": group_key,
                    "lag_step": lag,
                    "lag_sec": lag * 0.1,
                    "n": n,
                    "mean_pemit": self.sum_pemit[(group_key, lag)] / n,
                    "mean_rdown": self.sum_rdown[(group_key, lag)] / n,
                    "mean_amplification": self.sum_amp[(group_key, lag)] / n,
                }
            )
        return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if len(rows) == 0:
        print(f"[write_csv] no rows for {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(set().union(*[r.keys() for r in rows]))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[write_csv] saved: {path} ({len(rows)} rows)")


def open_csv_writer(path: Path, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def compute_edge_state_for_graph(batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Align edge features with graph step.

    At rollout step t, the graph was computed from the state before predicting pred_states[:, t].
    Therefore:
        graph_state[0] = history last state
        graph_state[t] = pred_states[:, t-1] for t > 0
    """
    state0 = batch["history"][:, :, -1, :]              # [B,N,6]
    pred = out["pred_states"]                           # [B,T,N,6]

    if pred.shape[1] == 1:
        return state0.unsqueeze(1)

    graph_state = torch.cat(
        [
            state0.unsqueeze(1),
            pred[:, :-1],
        ],
        dim=1,
    )

    return graph_state


def pairwise_physical_features(
    graph_state: torch.Tensor,
    edge_lane_relation: torch.Tensor,
    density0: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    graph_state:
        [B,T,N,6]
    edge_lane_relation:
        [B,N,N]
    density0:
        [B]
    """
    b, t, n, _ = graph_state.shape

    x = graph_state[..., IDX_X]
    y = graph_state[..., IDX_Y]
    speed = graph_state[..., IDX_SPEED]

    x_s = x.unsqueeze(3)
    x_r = x.unsqueeze(2)

    y_s = y.unsqueeze(3)
    y_r = y.unsqueeze(2)

    sp_s = speed.unsqueeze(3)
    sp_r = speed.unsqueeze(2)

    delta_x = x_s - x_r
    delta_y = y_s - y_r
    delta_speed = sp_s - sp_r

    closing_speed = sp_r - sp_s
    max_ttc = 20.0

    ttc = torch.full_like(delta_y, fill_value=max_ttc)
    valid_ttc = (delta_y > 0.0) & (closing_speed > 1e-3)
    ttc_value = delta_y / torch.clamp(closing_speed, min=1e-3)
    ttc_value = torch.clamp(ttc_value, min=0.0, max=max_ttc)
    ttc = torch.where(valid_ttc, ttc_value, ttc)

    rho = density0.view(b, 1, 1, 1).expand(b, t, n, n)

    lane_rel_t = edge_lane_relation.unsqueeze(1).expand(b, t, n, n)

    return {
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_speed": delta_speed,
        "ttc": ttc,
        "rho": rho,
        "lane_rel": lane_rel_t,
    }


def compute_pi(beta: torch.Tensor, edge_mask: torch.Tensor, lambda_self: float) -> torch.Tensor:
    """
    beta:
        [B,T,N,N]
    edge_mask:
        [B,N,N]
    """
    b, t, n, _ = beta.shape
    edge_mask_t = edge_mask.unsqueeze(1).expand(b, t, n, n)

    beta = beta * edge_mask_t.to(beta.dtype)
    incoming_sum = beta.sum(dim=2, keepdim=True)  # sum over source
    pi = beta / (float(lambda_self) + incoming_sum + 1e-8)
    pi = pi * edge_mask_t.to(beta.dtype)

    return pi


def entropy_from_weights(weights: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    """
    weights are nonnegative. Normalize along dim, then entropy.
    """
    total = weights.sum(dim=dim, keepdim=True)
    p = weights / (total + eps)
    h = -(p * torch.log(p + eps)).sum(dim=dim)
    return h


def get_gt_future_epsilon(batch: Dict[str, torch.Tensor], cfg: AnalysisConfig, pred_steps: int) -> torch.Tensor:
    """
    Return epsilon:
        [B,T,N]

    Use GT future for motion perturbation.
    """
    future = batch["future"].permute(0, 2, 1, 3)  # [B,T,N,6]
    t = min(pred_steps, future.shape[1])
    future = future[:, :t]

    if cfg.perturbation_eps_mode == "abs_acc":
        eps = torch.abs(future[..., IDX_ACC])
        return eps

    if cfg.perturbation_eps_mode == "abs_delta_speed":
        speed = future[..., IDX_SPEED]
        speed0 = batch["history"][:, :, -1, IDX_SPEED].unsqueeze(1)  # [B,1,N]
        speed_prev = torch.cat([speed0, speed[:, :-1]], dim=1)
        eps = torch.abs(speed - speed_prev)
        return eps

    raise ValueError(f"Unknown perturbation_eps_mode={cfg.perturbation_eps_mode}")


def active_motif(active_uv: bool, active_vu: bool) -> str:
    if active_uv and active_vu:
        return "mutual"
    if active_uv and not active_vu:
        return "u_to_v_only"
    if (not active_uv) and active_vu:
        return "v_to_u_only"
    return "none"


def main() -> None:
    cfg = AnalysisConfig()

    seed_everything(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt, model_cfg, loss_cfg, dataset_cfg = load_checkpoint_and_configs(cfg)

    pred_steps = int(model_cfg.pred_steps)
    if cfg.pred_steps_override is not None:
        pred_steps = int(cfg.pred_steps_override)
        model_cfg.pred_steps = pred_steps

    print("[analysis] checkpoint:", cfg.checkpoint_path)
    print("[analysis] output_dir:", output_dir)
    print("[analysis] device:", device)
    print("[analysis] pred_steps:", pred_steps)
    print("[analysis] model_cfg:", asdict(model_cfg))
    print("[analysis] dataset_cfg:", asdict(dataset_cfg))

    with open(output_dir / "analysis_config.json", "w") as f:
        json.dump(
            {
                "analysis_cfg": asdict(cfg),
                "model_cfg": asdict(model_cfg),
                "loss_cfg": asdict(loss_cfg),
                "dataset_cfg": asdict(dataset_cfg),
            },
            f,
            indent=2,
        )

    loader = build_loader(dataset_cfg, cfg)

    model = DynamicCTEGNN(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    edge_fields = [
        "global_row_id",
        "batch_global_index",
        "sample_index",
        "center_frame",
        "time_step",
        "time_sec",
        "source_idx",
        "receiver_idx",
        "source_vehicle_id",
        "receiver_vehicle_id",
        "source_type",
        "receiver_type",
        "type_pair",
        "lane_relation",
        "density",
        "density_bin",
        "delta_x",
        "delta_y",
        "abs_delta_y",
        "delta_speed",
        "ttc",
        "d_abs_bin",
        "dv_bin",
        "ttc_bin",
        "open_prob",
        "gate",
        "alpha",
        "beta",
        "pi",
        "cte_teacher",
        "active_open",
    ]

    node_fields = [
        "global_row_id",
        "batch_global_index",
        "sample_index",
        "center_frame",
        "node_idx",
        "vehicle_id",
        "node_type",
        "Iin",
        "Iout",
        "deltaI",
        "Hin",
        "Hout",
        "mean_eps",
        "mean_Pemit",
    ]

    edge_file = None
    edge_writer = None
    node_file = None
    node_writer = None

    if cfg.write_edge_csv:
        edge_file, edge_writer = open_csv_writer(output_dir / "edges_val.csv", edge_fields)

    if cfg.write_node_role_csv:
        node_file, node_writer = open_csv_writer(output_dir / "nodes_role_val.csv", node_fields)

    physical_summary = GroupMeanStats()
    gate_diagnostics = EdgeGateDiagnostics(
        active_threshold=cfg.active_open_threshold,
        hesitation_low=cfg.hesitation_low,
        hesitation_high=cfg.hesitation_high,
        gate_threshold=cfg.gate_active_threshold,
        max_values_per_group=cfg.max_gate_diag_values_per_group,
    )
    role_summary = GroupMeanStats()
    spearman = SpearmanCollector(max_values_per_group=cfg.max_spearman_values_per_group)
    motif_counter = MotifCounter()
    amp_stats = AmplificationStats()

    edge_rows_written = 0
    node_rows_written = 0
    global_edge_row_id = 0
    global_node_row_id = 0
    batch_global_index = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if cfg.max_batches is not None and batch_idx >= int(cfg.max_batches):
                break

            batch = move_batch_to_device(batch, device)

            out = model(
                batch,
                pred_steps=pred_steps,
                deterministic_gate=True,
                edge_drop_mask=None,
            )

            graph_state = compute_edge_state_for_graph(batch, out)
            phys = pairwise_physical_features(
                graph_state=graph_state,
                edge_lane_relation=batch["edge_lane_relation"],
                density0=batch["density0"],
            )

            edge_mask = out["edge_mask"].bool()               # [B,N,N]
            node_mask = out["node_mask"].bool()               # [B,N]
            edge_mask_t = edge_mask.unsqueeze(1).expand_as(out["betas"])

            beta = out["betas"] * edge_mask_t.to(out["betas"].dtype)
            open_prob = out["open_probs"] * edge_mask_t.to(out["open_probs"].dtype)
            gate = out["gates"] * edge_mask_t.to(out["gates"].dtype)
            alpha = out["alphas"] * edge_mask_t.to(out["alphas"].dtype)
            cte_teacher = out["cte_teacher"] * edge_mask_t.to(out["cte_teacher"].dtype)

            pi = compute_pi(
                beta=beta,
                edge_mask=edge_mask,
                lambda_self=float(model_cfg.lambda_self),
            )

            active_open = open_prob >= float(cfg.active_open_threshold)

            bsz, t_steps, n, _ = beta.shape

            epsilon_gt = get_gt_future_epsilon(batch, cfg, pred_steps=t_steps)  # [B,T,N]

            Iin_t = beta.sum(dim=2)    # incoming to receiver v: [B,T,N]
            Iout_t = beta.sum(dim=3)   # outgoing from source u: [B,T,N]

            Iin = Iin_t.sum(dim=1)     # [B,N]
            Iout = Iout_t.sum(dim=1)   # [B,N]
            deltaI = Iin - Iout

            Hin_t = entropy_from_weights(beta, dim=2)  # incoming distribution over source: [B,T,N]
            Hout_t = entropy_from_weights(beta, dim=3) # outgoing distribution over receiver: [B,T,N]

            Hin = Hin_t.mean(dim=1)
            Hout = Hout_t.mean(dim=1)

            Pemit_t = epsilon_gt * Iout_t
            Pemit_mean = Pemit_t.mean(dim=1)
            eps_mean = epsilon_gt.mean(dim=1)

            # =========================
            # Node role CSV and summary
            # =========================

            node_type_np = batch["node_type"].detach().cpu().numpy()
            vehicle_ids_np = batch["vehicle_ids"].detach().cpu().numpy()
            sample_index_np = batch["sample_index"].detach().cpu().numpy()
            center_frame_np = batch["center_frame"].detach().cpu().numpy()

            Iin_np = Iin.detach().cpu().numpy()
            Iout_np = Iout.detach().cpu().numpy()
            deltaI_np = deltaI.detach().cpu().numpy()
            Hin_np = Hin.detach().cpu().numpy()
            Hout_np = Hout.detach().cpu().numpy()
            eps_mean_np = eps_mean.detach().cpu().numpy()
            Pemit_mean_np = Pemit_mean.detach().cpu().numpy()
            node_mask_np = node_mask.detach().cpu().numpy()

            for bi in range(bsz):
                for vi in range(n):
                    if not node_mask_np[bi, vi]:
                        continue

                    nt = int(node_type_np[bi, vi])
                    nt_name = type_name(nt)

                    role_values = {
                        "Iin": float(Iin_np[bi, vi]),
                        "Iout": float(Iout_np[bi, vi]),
                        "deltaI": float(deltaI_np[bi, vi]),
                        "Hin": float(Hin_np[bi, vi]),
                        "Hout": float(Hout_np[bi, vi]),
                        "mean_eps": float(eps_mean_np[bi, vi]),
                        "mean_Pemit": float(Pemit_mean_np[bi, vi]),
                    }

                    role_summary.update(f"type={nt_name}", role_values)

                    if node_writer is not None:
                        node_writer.writerow(
                            {
                                "global_row_id": global_node_row_id,
                                "batch_global_index": batch_global_index,
                                "sample_index": int(sample_index_np[bi]),
                                "center_frame": int(center_frame_np[bi]),
                                "node_idx": vi,
                                "vehicle_id": int(vehicle_ids_np[bi, vi]),
                                "node_type": nt_name,
                                **role_values,
                            }
                        )
                        global_node_row_id += 1
                        node_rows_written += 1

            # =========================
            # Edge-level arrays to CPU
            # =========================

            beta_np = beta.detach().cpu().numpy()
            pi_np = pi.detach().cpu().numpy()
            open_np = open_prob.detach().cpu().numpy()
            gate_np = gate.detach().cpu().numpy()
            alpha_np = alpha.detach().cpu().numpy()
            cte_np = cte_teacher.detach().cpu().numpy()
            active_np = active_open.detach().cpu().numpy()
            edge_mask_np = edge_mask.detach().cpu().numpy()

            delta_x_np = phys["delta_x"].detach().cpu().numpy()
            delta_y_np = phys["delta_y"].detach().cpu().numpy()
            delta_speed_np = phys["delta_speed"].detach().cpu().numpy()
            ttc_np = phys["ttc"].detach().cpu().numpy()
            rho_np = phys["rho"].detach().cpu().numpy()
            lane_rel_np = phys["lane_rel"].detach().cpu().numpy()

            # =========================
            # Edge summary, physical law, CSV
            # =========================

            for bi in range(bsz):
                for ti in range(t_steps):
                    time_sec = (ti + 1) * 0.1

                    for u in range(n):
                        if not node_mask_np[bi, u]:
                            continue

                        for v in range(n):
                            if not node_mask_np[bi, v]:
                                continue
                            if u == v:
                                continue
                            if not edge_mask_np[bi, u, v]:
                                continue

                            src_type = type_name(int(node_type_np[bi, u]))
                            rec_type = type_name(int(node_type_np[bi, v]))
                            type_pair = f"{src_type}->{rec_type}"

                            lr_int = int(lane_rel_np[bi, ti, u, v])
                            lr_name = lane_name(lr_int)

                            dy = float(delta_y_np[bi, ti, u, v])
                            dx = float(delta_x_np[bi, ti, u, v])
                            abs_dy = abs(dy)
                            ds = float(delta_speed_np[bi, ti, u, v])
                            ttc = float(ttc_np[bi, ti, u, v])
                            rho = float(rho_np[bi, ti, u, v])

                            d_bin = bin_name(abs_dy, cfg.d_abs_bins, "abs_d")
                            dv_bin = bin_name(ds, cfg.dv_bins, "dv")
                            ttc_bin = bin_name(ttc, cfg.ttc_bins, "ttc")
                            rho_bin = bin_name(rho, cfg.density_bins, "rho")

                            op = float(open_np[bi, ti, u, v])
                            ga = float(gate_np[bi, ti, u, v])
                            al = float(alpha_np[bi, ti, u, v])
                            be = float(beta_np[bi, ti, u, v])
                            pii = float(pi_np[bi, ti, u, v])
                            cte = float(cte_np[bi, ti, u, v])
                            act = bool(active_np[bi, ti, u, v])

                            values = {
                                "open_prob": op,
                                "gate": ga,
                                "alpha": al,
                                "beta": be,
                                "pi": pii,
                                "cte_teacher": cte,
                                "abs_delta_y": abs_dy,
                                "delta_speed": ds,
                                "ttc": ttc,
                                "active_open": 1.0 if act else 0.0,
                            }

                            physical_summary.update(f"type_pair={type_pair}", values)
                            physical_summary.update(f"lane_relation={lr_name}", values)
                            physical_summary.update(f"density_bin={rho_bin}", values)
                            physical_summary.update(f"time_step={ti + 1}", values)
                            physical_summary.update(f"type_pair={type_pair}|lane={lr_name}", values)

                            # Gate/open-probability decisiveness diagnostics.
                            # These are useful to report whether the graph is sparse,
                            # whether open_prob is polarized, and whether gates are active.
                            for diag_group in [
                                "all",
                                f"type_pair={type_pair}",
                                f"lane_relation={lr_name}",
                                f"density_bin={rho_bin}",
                                f"time_step={ti + 1}",
                                f"type_pair={type_pair}|lane={lr_name}",
                            ]:
                                gate_diagnostics.update(diag_group, open_prob=op, gate=ga)

                            for group_kind, group_value in [
                                ("type_pair", type_pair),
                                ("lane_relation", lr_name),
                                ("density_bin", rho_bin),
                            ]:
                                spearman.add(group_kind, group_value, "open_prob_vs_abs_delta_y", abs_dy, op)
                                spearman.add(group_kind, group_value, "beta_vs_abs_delta_y", abs_dy, be)
                                spearman.add(group_kind, group_value, "pi_vs_abs_delta_y", abs_dy, pii)

                                spearman.add(group_kind, group_value, "open_prob_vs_delta_speed", ds, op)
                                spearman.add(group_kind, group_value, "beta_vs_delta_speed", ds, be)
                                spearman.add(group_kind, group_value, "pi_vs_delta_speed", ds, pii)

                                spearman.add(group_kind, group_value, "open_prob_vs_ttc", ttc, op)
                                spearman.add(group_kind, group_value, "beta_vs_ttc", ttc, be)
                                spearman.add(group_kind, group_value, "pi_vs_ttc", ttc, pii)

                            if (
                                edge_writer is not None
                                and edge_rows_written < cfg.max_edge_rows
                                and op >= float(cfg.edge_export_min_open_prob)
                            ):
                                edge_writer.writerow(
                                    {
                                        "global_row_id": global_edge_row_id,
                                        "batch_global_index": batch_global_index,
                                        "sample_index": int(sample_index_np[bi]),
                                        "center_frame": int(center_frame_np[bi]),
                                        "time_step": ti + 1,
                                        "time_sec": time_sec,
                                        "source_idx": u,
                                        "receiver_idx": v,
                                        "source_vehicle_id": int(vehicle_ids_np[bi, u]),
                                        "receiver_vehicle_id": int(vehicle_ids_np[bi, v]),
                                        "source_type": src_type,
                                        "receiver_type": rec_type,
                                        "type_pair": type_pair,
                                        "lane_relation": lr_name,
                                        "density": rho,
                                        "density_bin": rho_bin,
                                        "delta_x": dx,
                                        "delta_y": dy,
                                        "abs_delta_y": abs_dy,
                                        "delta_speed": ds,
                                        "ttc": ttc,
                                        "d_abs_bin": d_bin,
                                        "dv_bin": dv_bin,
                                        "ttc_bin": ttc_bin,
                                        "open_prob": op,
                                        "gate": ga,
                                        "alpha": al,
                                        "beta": be,
                                        "pi": pii,
                                        "cte_teacher": cte,
                                        "active_open": int(act),
                                    }
                                )
                                edge_rows_written += 1
                                global_edge_row_id += 1

            # =========================
            # Motif summary
            # =========================

            for bi in range(bsz):
                for ti in range(t_steps):
                    time_group = f"time_step={ti + 1}"
                    for u in range(n):
                        if not node_mask_np[bi, u]:
                            continue
                        for v in range(u + 1, n):
                            if not node_mask_np[bi, v]:
                                continue
                            if not edge_mask_np[bi, u, v] or not edge_mask_np[bi, v, u]:
                                continue

                            active_uv = bool(active_np[bi, ti, u, v])
                            active_vu = bool(active_np[bi, ti, v, u])
                            motif = active_motif(active_uv, active_vu)

                            u_type = type_name(int(node_type_np[bi, u]))
                            v_type = type_name(int(node_type_np[bi, v]))
                            type_pair_undirected = "|".join(sorted([u_type, v_type]))

                            lr_uv = lane_name(int(lane_rel_np[bi, ti, u, v]))

                            motif_counter.update("all", motif)
                            motif_counter.update(time_group, motif)
                            motif_counter.update(f"type_pair_undirected={type_pair_undirected}", motif)
                            motif_counter.update(f"lane_relation_uv={lr_uv}", motif)

            # =========================
            # Perturbation amplification
            # =========================

            epsilon_np = epsilon_gt.detach().cpu().numpy()
            Pemit_t_np = Pemit_t.detach().cpu().numpy()

            for bi in range(bsz):
                for ti in range(t_steps):
                    for u in range(n):
                        if not node_mask_np[bi, u]:
                            continue

                        src_type = type_name(int(node_type_np[bi, u]))
                        pemit = float(Pemit_t_np[bi, ti, u])

                        if pemit <= 1e-12:
                            continue

                        for lag in cfg.perturbation_lags:
                            tj = ti + int(lag)
                            if tj >= t_steps:
                                continue

                            rdown = 0.0

                            for v in range(n):
                                if u == v:
                                    continue
                                if not node_mask_np[bi, v]:
                                    continue
                                if not edge_mask_np[bi, u, v]:
                                    continue

                                if cfg.downstream_mode == "same_lane_following":
                                    # source u is ahead of receiver v in same lane.
                                    if int(lane_rel_np[bi, ti, u, v]) != LANE_SAME_FRONT:
                                        continue

                                elif cfg.downstream_mode == "all_outgoing":
                                    pass
                                else:
                                    raise ValueError(f"Unknown downstream_mode={cfg.downstream_mode}")

                                rdown += float(beta_np[bi, tj, u, v]) * float(epsilon_np[bi, tj, v])

                            amp = rdown / (pemit + 1e-8)

                            amp_stats.update(
                                group_key=f"source_type={src_type}",
                                lag=int(lag),
                                pemit=pemit,
                                rdown=rdown,
                                amp=amp,
                            )

                            amp_stats.update(
                                group_key="all",
                                lag=int(lag),
                                pemit=pemit,
                                rdown=rdown,
                                amp=amp,
                            )

            batch_global_index += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"[analysis] batch={batch_idx + 1:04d} | "
                    f"edge_rows={edge_rows_written} | "
                    f"node_rows={node_rows_written}"
                )

    if edge_file is not None:
        edge_file.close()

    if node_file is not None:
        node_file.close()

    # =========================
    # Save summaries
    # =========================

    write_csv(
        output_dir / "physical_summary_by_group.csv",
        physical_summary.to_rows(group_name="group"),
    )

    write_csv(
        output_dir / "gate_open_diagnostics_by_group.csv",
        gate_diagnostics.to_rows(),
    )

    write_csv(
        output_dir / "physical_spearman_by_group.csv",
        spearman.to_rows(),
    )

    write_csv(
        output_dir / "role_index_by_type.csv",
        role_summary.to_rows(group_name="type_group"),
    )

    write_csv(
        output_dir / "motif_summary.csv",
        motif_counter.to_rows(),
    )

    write_csv(
        output_dir / "perturbation_amplification_by_lag.csv",
        amp_stats.to_rows(),
    )

    print("\n[analysis done]")
    print("Saved to:", output_dir)
    print("Main files:")
    print("  edges_val.csv")
    print("  nodes_role_val.csv")
    print("  physical_summary_by_group.csv")
    print("  gate_open_diagnostics_by_group.csv")
    print("  physical_spearman_by_group.csv")
    print("  role_index_by_type.csv")
    print("  motif_summary.csv")
    print("  perturbation_amplification_by_lag.csv")


if __name__ == "__main__":
    main()