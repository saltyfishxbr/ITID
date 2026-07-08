# losses.py

from dataclasses import dataclass
from typing import Dict, Optional,Tuple

import torch
import torch.nn.functional as F


IDX_X = 0
IDX_Y = 1
IDX_VX = 2
IDX_VY = 3
IDX_SPEED = 4
IDX_ACC = 5


@dataclass
class LossConfig:
    # Total weights
    lambda_traj: float = 1.0
    lambda_cte: float = 0.05
    lambda_l0: float = 1e-3

    # Trajectory loss
    lambda_vel: float = 0.2
    neighbor_weight: float = 0.5
    target_weight: float = 1.0
    smooth_l1_beta: float = 1.0

    metric_fps: int=10
    metric_seconds: Tuple[int,...]=(1,2,3,4,5)

    # CTE KL loss
    cte_temperature: float = 0.2
    cte_use_valid_only: bool = True
    cte_use_significant_only: bool = False
    cte_require_positive: bool = True
    cte_positive_eps: float = 1e-8
    cte_min_edges: int = 2
    cte_invalid_score: float = -1e9

    # Numerical
    eps: float = 1e-8


def masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype, device=value.device)
    return (value * mask).sum() / (mask.sum() + eps)

def target_horizon_metrics(
    pos_pred: torch.Tensor,
    pos_gt: torch.Tensor,
    node_mask: torch.Tensor,
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    """
    Target-only horizon metrics.

    pos_pred:
        [B, T, N, 2]

    pos_gt:
        [B, T, N, 2]

    node_mask:
        [B, N]

    RMSE@ks:
        sqrt(mean_b ||p_hat_{b,k} - p_{b,k}||_2^2)

    ADE@5s:
        mean_{b,t<=5s} ||p_hat_{b,t} - p_{b,t}||_2
    """
    metrics = {}

    b, t, n, _ = pos_pred.shape
    target_valid = node_mask[:, 0].bool()  # [B]

    for sec in cfg.metric_seconds:
        step = int(round(float(sec) * float(cfg.metric_fps))) - 1

        if step < 0 or step >= t:
            continue

        err2 = ((pos_pred[:, step, 0, :] - pos_gt[:, step, 0, :]) ** 2).sum(dim=-1)
        rmse = torch.sqrt(masked_mean(err2, target_valid, cfg.eps) + cfg.eps)

        metrics[f"rmse_target_{sec}s"] = rmse.detach()

    # Explicit 5s ADE. Only log if prediction horizon covers 5s.
    step_5s = int(round(5.0 * float(cfg.metric_fps)))

    if t >= step_5s:
        pred_5s = pos_pred[:, :step_5s, 0, :]  # [B, 50, 2]
        gt_5s = pos_gt[:, :step_5s, 0, :]

        l2 = torch.sqrt(((pred_5s - gt_5s) ** 2).sum(dim=-1) + cfg.eps)  # [B,50]

        valid_5s = target_valid.unsqueeze(1).expand_as(l2)
        ade_5s = masked_mean(l2, valid_5s, cfg.eps)

        metrics["ade_target_5s"] = ade_5s.detach()

    return metrics

def trajectory_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    """
    Deterministic trajectory loss.

    pred_states:
        [B, T, N, 6]

    future:
        [B, N, T, 6]

    We supervise:
        position: x, y
        velocity: vx, vy

    Target node gets larger weight.
    Neighbor nodes are also supervised to make all receivers learnable.
    """
    pred = out["pred_states"]                         # [B, T, N, 6]
    gt = batch["future"].permute(0, 2, 1, 3)          # [B, T, N, 6]

    node_mask = batch["node_mask"].bool()             # [B, N]

    b, t_pred, n, d = pred.shape
    t_gt = gt.shape[1]
    t = min(t_pred, t_gt)

    pred = pred[:, :t]
    gt = gt[:, :t]

    node_mask_t = node_mask.unsqueeze(1).expand(b, t, n)

    node_weight = torch.full(
        size=(b, n),
        fill_value=float(cfg.neighbor_weight),
        dtype=pred.dtype,
        device=pred.device,
    )
    node_weight[:, 0] = float(cfg.target_weight)
    node_weight = node_weight * node_mask.to(pred.dtype)
    node_weight_t = node_weight.unsqueeze(1).expand(b, t, n)

    valid_weight = node_weight_t * node_mask_t.to(pred.dtype)

    pos_pred = pred[..., [IDX_X, IDX_Y]]
    pos_gt = gt[..., [IDX_X, IDX_Y]]

    vel_pred = pred[..., [IDX_VX, IDX_VY]]
    vel_gt = gt[..., [IDX_VX, IDX_VY]]

    pos_loss_raw = F.smooth_l1_loss(
        pos_pred,
        pos_gt,
        reduction="none",
        beta=float(cfg.smooth_l1_beta),
    ).sum(dim=-1)  # [B, T, N]

    vel_loss_raw = F.smooth_l1_loss(
        vel_pred,
        vel_gt,
        reduction="none",
        beta=float(cfg.smooth_l1_beta),
    ).sum(dim=-1)  # [B, T, N]

    pos_loss = (pos_loss_raw * valid_weight).sum() / (valid_weight.sum() + cfg.eps)
    vel_loss = (vel_loss_raw * valid_weight).sum() / (valid_weight.sum() + cfg.eps)

    traj = pos_loss + float(cfg.lambda_vel) * vel_loss

    # Metrics for logging only.
    pos_l2 = torch.sqrt(((pos_pred - pos_gt) ** 2).sum(dim=-1) + cfg.eps)  # [B,T,N]
    ade_all = (pos_l2 * valid_weight).sum() / (valid_weight.sum() + cfg.eps)

    target_mask = node_mask_t[:, :, 0]
    ade_target = masked_mean(pos_l2[:, :, 0], target_mask, cfg.eps)

    rmse_all = torch.sqrt(
        (((pos_pred - pos_gt) ** 2).sum(dim=-1) * valid_weight).sum()
        / (valid_weight.sum() + cfg.eps)
        + cfg.eps
    )
    horizon_metrics = target_horizon_metrics(
        pos_pred=pos_pred,
        pos_gt=pos_gt,
        node_mask=node_mask,
        cfg=cfg,
    )

    result = {
        "traj": traj,
        "pos": pos_loss,
        "vel": vel_loss,
        "ade_all": ade_all.detach(),
        "ade_target": ade_target.detach(),
        "rmse_all": rmse_all.detach(),
    }

    result.update(horizon_metrics)
    return result


def _masked_log_softmax(
    scores: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    invalid_value: float,
) -> torch.Tensor:
    scores = scores.masked_fill(~mask, invalid_value)
    return F.log_softmax(scores, dim=dim)


def _masked_softmax(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    invalid_value: float,
) -> torch.Tensor:
    values = values.masked_fill(~mask, invalid_value)
    return F.softmax(values, dim=dim)


def cte_kl_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    """
    Incoming CTE supervision.

    For each receiver v, source dimension u is normalized:

        q_{u->v} = softmax_u(score_{u->v})
        r_{u->v} = softmax_u(CTE_{u->v} / T_c)

    KL(r || q) is computed only for valid receiver groups.
    """
    scores = out["scores"]                 # [B, T, N, N], source -> receiver
    teacher = out["cte_teacher"]           # [B, T, N, N]
    edge_mask = out["edge_mask"]           # [B, N, N]
    node_mask = out["node_mask"]           # [B, N]

    b, t, n, _ = scores.shape

    edge_mask_t = edge_mask.unsqueeze(1).expand(b, t, n, n)

    if cfg.cte_use_valid_only and "cte_valid" in out:
        valid = out["cte_valid"].bool() & edge_mask_t
    else:
        valid = edge_mask_t

    if cfg.cte_use_significant_only and "cte_significant" in out:
        valid = valid & out["cte_significant"].bool()

    # Need valid incoming edges for each receiver.
    # Source dimension is dim=2.
    valid_count = valid.sum(dim=2)  # [B, T, receiver N]

    group_mask = valid_count >= int(cfg.cte_min_edges)

    if cfg.cte_require_positive:
        positive = ((teacher > float(cfg.cte_positive_eps)) & valid).any(dim=2)
        group_mask = group_mask & positive

    # Receiver itself must be a valid node.
    receiver_mask = node_mask.unsqueeze(1).expand(b, t, n)
    group_mask = group_mask & receiver_mask

    if group_mask.sum() == 0:
        zero = scores.sum() * 0.0
        return {
            "cte": zero,
            "cte_groups": torch.tensor(0.0, device=scores.device),
            "cte_valid_edges": valid.sum().detach().float(),
        }

    log_q = _masked_log_softmax(
        scores=scores,
        mask=valid,
        dim=2,
        invalid_value=float(cfg.cte_invalid_score),
    )

    teacher_scaled = teacher / float(cfg.cte_temperature)

    r = _masked_softmax(
        values=teacher_scaled,
        mask=valid,
        dim=2,
        invalid_value=float(cfg.cte_invalid_score),
    )

    # KL(r || q) = sum r * (log r - log q)
    log_r = torch.log(r.clamp_min(cfg.eps))

    kl_per_edge = r * (log_r - log_q)
    kl_per_group = kl_per_edge.sum(dim=2)  # [B, T, receiver N]

    cte_loss = masked_mean(
        kl_per_group,
        group_mask,
        eps=cfg.eps,
    )

    return {
        "cte": cte_loss,
        "cte_groups": group_mask.sum().detach().float(),
        "cte_valid_edges": valid.sum().detach().float(),
    }


def l0_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    """
    L0 loss on Hard Concrete open probabilities.

    open_probs:
        [B, T, N, N]
    """
    open_probs = out["open_probs"]
    edge_mask = out["edge_mask"]  # [B,N,N]

    b, t, n, _ = open_probs.shape
    edge_mask_t = edge_mask.unsqueeze(1).expand(b, t, n, n)

    l0 = masked_mean(
        open_probs,
        edge_mask_t,
        eps=cfg.eps,
    )

    beta_mean = masked_mean(
        out["betas"],
        edge_mask_t,
        eps=cfg.eps,
    ).detach()

    gate_mean = masked_mean(
        out["gates"],
        edge_mask_t,
        eps=cfg.eps,
    ).detach()

    alpha_mean = masked_mean(
        out["alphas"],
        edge_mask_t,
        eps=cfg.eps,
    ).detach()

    return {
        "l0": l0,
        "open_prob_mean": l0.detach(),
        "beta_mean": beta_mean,
        "gate_mean": gate_mean,
        "alpha_mean": alpha_mean,
    }


def compute_total_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: LossConfig,
) -> Dict[str, torch.Tensor]:
    traj_dict = trajectory_loss(out, batch, cfg)
    cte_dict = cte_kl_loss(out, batch, cfg)
    l0_dict = l0_loss(out, batch, cfg)

    total = (
        float(cfg.lambda_traj) * traj_dict["traj"]
        + float(cfg.lambda_cte) * cte_dict["cte"]
        + float(cfg.lambda_l0) * l0_dict["l0"]
    )

    loss_dict = {
        "total": total,

        "traj": traj_dict["traj"],
        "pos": traj_dict["pos"],
        "vel": traj_dict["vel"],

        "cte": cte_dict["cte"],
        "l0": l0_dict["l0"],

        # logging values
        "ade_all": traj_dict["ade_all"],
        "ade_target": traj_dict["ade_target"],
        "rmse_all": traj_dict["rmse_all"],

        "cte_groups": cte_dict["cte_groups"],
        "cte_valid_edges": cte_dict["cte_valid_edges"],

        "open_prob_mean": l0_dict["open_prob_mean"],
        "beta_mean": l0_dict["beta_mean"],
        "gate_mean": l0_dict["gate_mean"],
        "alpha_mean": l0_dict["alpha_mean"],
    }

    for sec in cfg.metric_seconds:
        key = f"rmse_target_{sec}s"
        if key in traj_dict:
            loss_dict[key] = traj_dict[key]

    if "ade_target_5s" in traj_dict:
        loss_dict["ade_target_5s"] = traj_dict["ade_target_5s"]

    return loss_dict


def main() -> None:
    """
    Smoke test using dataset + model.
    """

    from ngsim_dataset import NGSIMCTEDataset, NGSIMDatasetConfig, simple_collate_fn

    from dynamic_cte_gnn import DynamicCTEGNN, DynamicCTEGNNConfig

    from torch.utils.data import DataLoader

    dataset_cfg = NGSIMDatasetConfig(
        split="train",
        top_k_neighbors=8,
    )
    dataset = NGSIMCTEDataset(dataset_cfg)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate_fn,
    )

    batch = next(iter(loader))

    model_cfg = DynamicCTEGNNConfig(
        pred_steps=10,
        cte_table_path=dataset_cfg.cte_table_path,
    )
    model = DynamicCTEGNN(model_cfg)

    out = model(batch)

    loss_cfg = LossConfig()
    loss_dict = compute_total_loss(out, batch, loss_cfg)

    print("[loss smoke test]")
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            print(k, float(v.detach().cpu()))
        else:
            print(k, v)

    loss_dict["total"].backward()
    print("[loss smoke test] backward ok")


if __name__ == "__main__":
    main()