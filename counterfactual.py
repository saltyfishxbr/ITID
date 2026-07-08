# counterfactual.py

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F


IDX_X = 0
IDX_Y = 1


@dataclass
class CounterfactualConfig:
    """
    Lightweight counterfactual supervision.

    Main idea:
        1. Select edges where expected effective interaction beta and
           CTE teacher distribution disagree the most.
        2. Only evaluate them at a few key forecast horizons.
        3. Use deterministic full baseline and deterministic edge-deleted
           rollout to compute receiver-specific degradation.
        4. Use degradation as pseudo label to supervise aggregated open probability.
    """

    # Python indices for 1s,2s,3s,4s,5s under 10Hz.
    # If pred_steps is smaller, invalid indices are ignored.
    cf_step_indices: Tuple[int, ...] = (9, 19, 29, 39, 49)

    # Number of disagreement edges selected globally per batch.
    top_k_edges: int = 16

    # Disagreement uses incoming receiver-normalized expected beta and CTE teacher.
    cte_temperature: float = 0.2

    # Only consider edges with valid CTE teacher.
    use_cte_valid_only: bool = True
    require_positive_teacher_group: bool = True
    min_edges_per_receiver: int = 2

    # Filters to avoid selecting edges that are effectively closed already.
    min_open_prob: float = 0.2
    min_expected_beta: float = 1e-3

    # Drop mode.
    # If True, selected edge is removed from selected time step to the end.
    # If False, only removed at that single step.
    drop_from_step_to_end: bool = True

    # Counterfactual rollout mode.
    deterministic_gate_for_cf: bool = True
    no_grad_cf_rollout: bool = True

    # Temporarily switch model.eval() during deterministic CF rollouts
    # to avoid dropout randomness.
    eval_mode_for_cf: bool = True

    # CF label construction.
    label_temperature: float = 1.0
    min_degradation_for_positive: float = 1e-6

    # BCE supervision target.
    # Recommended: supervise the segment-level mean open probability.
    # This matches segment-level degradation.
    supervise_segment_mean: bool = True

    eps: float = 1e-8


def _masked_softmax(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    invalid_value: float = -1e9,
) -> torch.Tensor:
    values = values.masked_fill(~mask, invalid_value)
    return F.softmax(values, dim=dim)


def compute_incoming_disagreement(
    out_full: Dict[str, torch.Tensor],
    cfg: CounterfactualConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute incoming disagreement:

        D_{u->v} = | beta_expected_norm_{u->v}
                     - CTE_softmax_{u->v} |

    where:
        beta_expected = open_prob * alpha

    Tensor convention:
        [B, T, N, N]
        dim=2 is source u
        dim=3 is receiver v
    """
    open_probs = out_full["open_probs"]          # [B,T,N,N]
    alphas = out_full["alphas"]                  # [B,T,N,N]
    teacher = out_full["cte_teacher"]            # [B,T,N,N]
    edge_mask = out_full["edge_mask"]            # [B,N,N]

    expected_beta = open_probs * alphas          # [B,T,N,N]

    b, t, n, _ = expected_beta.shape
    edge_mask_t = edge_mask.unsqueeze(1).expand(b, t, n, n)

    if cfg.use_cte_valid_only and "cte_valid" in out_full:
        valid = out_full["cte_valid"].bool() & edge_mask_t
    else:
        valid = edge_mask_t

    if cfg.min_open_prob > 0:
        valid = valid & (open_probs >= float(cfg.min_open_prob))

    if cfg.min_expected_beta > 0:
        valid = valid & (expected_beta >= float(cfg.min_expected_beta))

    # Need enough valid incoming edges for each receiver.
    valid_count = valid.sum(dim=2)  # [B,T,receiver]
    group_mask = valid_count >= int(cfg.min_edges_per_receiver)

    if cfg.require_positive_teacher_group:
        positive = ((teacher > cfg.eps) & valid).any(dim=2)
        group_mask = group_mask & positive

    valid = valid & group_mask.unsqueeze(2)

    beta_valid = expected_beta * valid.to(expected_beta.dtype)
    beta_sum = beta_valid.sum(dim=2, keepdim=True)
    beta_norm = beta_valid / (beta_sum + float(cfg.eps))

    teacher_scaled = teacher / float(cfg.cte_temperature)
    teacher_prob = _masked_softmax(
        values=teacher_scaled,
        mask=valid,
        dim=2,
        invalid_value=-1e9,
    )

    disagreement = torch.abs(beta_norm - teacher_prob)
    disagreement = disagreement * valid.to(disagreement.dtype)

    return disagreement, valid


def select_top_disagreement_edges(
    out_full: Dict[str, torch.Tensor],
    cfg: CounterfactualConfig,
) -> List[Tuple[int, int, int, int]]:
    """
    Select top-K edges by CTE-beta disagreement.

    Returns:
        list of (batch_index, time_index, source_index, receiver_index)
    """
    disagreement, valid = compute_incoming_disagreement(out_full, cfg)

    b, t, n, _ = disagreement.shape
    device = disagreement.device

    step_mask = torch.zeros((t,), dtype=torch.bool, device=device)
    for idx in cfg.cf_step_indices:
        if 0 <= int(idx) < t:
            step_mask[int(idx)] = True

    if step_mask.sum() == 0:
        return []

    valid = valid & step_mask.view(1, t, 1, 1)

    candidate_mask = valid & (disagreement > float(cfg.eps))
    if candidate_mask.sum() == 0:
        return []

    score = disagreement.masked_fill(~candidate_mask, -1.0)
    flat = score.reshape(-1)

    num_valid = int(candidate_mask.sum().detach().cpu())
    k = min(int(cfg.top_k_edges), num_valid)

    _, indices = torch.topk(flat, k=k, largest=True)

    selected: List[Tuple[int, int, int, int]] = []

    for idx in indices.detach().cpu().tolist():
        b_idx = idx // (t * n * n)
        rem = idx % (t * n * n)

        time_idx = rem // (n * n)
        rem = rem % (n * n)

        source_idx = rem // n
        receiver_idx = rem % n

        selected.append(
            (
                int(b_idx),
                int(time_idx),
                int(source_idx),
                int(receiver_idx),
            )
        )

    return selected


def build_edge_drop_mask(
    batch_size: int,
    pred_steps: int,
    num_nodes: int,
    selected_edge: Tuple[int, int, int, int],
    cfg: CounterfactualConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Build edge_drop_mask for one selected edge.

    Shape:
        [B,T,N,N]

    If drop_from_step_to_end=True:
        remove edge u->v from t_e to the end.
    """
    b_idx, t_idx, u_idx, v_idx = selected_edge

    mask = torch.zeros(
        (batch_size, pred_steps, num_nodes, num_nodes),
        dtype=torch.bool,
        device=device,
    )

    if cfg.drop_from_step_to_end:
        mask[b_idx, t_idx:, u_idx, v_idx] = True
    else:
        mask[b_idx, t_idx, u_idx, v_idx] = True

    return mask


def receiver_position_error_from_step(
    pred_states: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    b_idx: int,
    receiver_idx: int,
    start_t: int,
    eps: float,
) -> torch.Tensor:
    """
    Receiver-specific ADE from start_t to prediction end.

    pred_states:
        [B,T,N,6]

    batch["future"]:
        [B,N,T,6]
    """
    gt = batch["future"].permute(0, 2, 1, 3)  # [B,T,N,6]

    t_pred = pred_states.shape[1]
    t_gt = gt.shape[1]
    t_end = min(t_pred, t_gt)

    if start_t >= t_end:
        return pred_states.sum() * 0.0

    pred_pos = pred_states[b_idx, start_t:t_end, receiver_idx, [IDX_X, IDX_Y]]
    gt_pos = gt[b_idx, start_t:t_end, receiver_idx, [IDX_X, IDX_Y]]

    l2 = torch.sqrt(((pred_pos - gt_pos) ** 2).sum(dim=-1) + eps)
    return l2.mean()


def build_cf_labels_from_degradation(
    degradation: torch.Tensor,
    cfg: CounterfactualConfig,
) -> torch.Tensor:
    """
    Convert nonnegative degradation values into soft labels.

    degradation:
        C_e = max(0, error_cf - error_base)

    If all degradation values are tiny, labels are zeros.
    Otherwise use robust median/MAD normalization.
    """
    degradation = torch.clamp(degradation, min=0.0)

    if degradation.numel() == 0:
        return degradation

    # This CPU scalar check is low-frequency, once per CF loss call.
    if float(degradation.max().detach().cpu()) <= float(cfg.min_degradation_for_positive):
        return torch.zeros_like(degradation)

    median = degradation.median()
    mad = torch.median(torch.abs(degradation - median))

    # This CPU scalar check is low-frequency, once per CF loss call.
    if float(mad.detach().cpu()) <= float(cfg.eps):
        std = degradation.std(unbiased=False)
        scale = torch.clamp(std, min=float(cfg.eps))
    else:
        scale = torch.clamp(mad, min=float(cfg.eps))

    normalized = (degradation - median) / scale
    labels = torch.sigmoid(normalized / float(cfg.label_temperature))

    return labels.detach()


def _run_model_for_cf(
    model,
    batch: Dict[str, torch.Tensor],
    pred_steps: int,
    deterministic_gate: bool,
    edge_drop_mask: torch.Tensor = None,
    cfg: CounterfactualConfig = None,
) -> Dict[str, torch.Tensor]:
    """
    Run model for CF baseline/deleted rollout.

    Optionally switches model.eval() temporarily to remove dropout randomness
    during deterministic CF label generation.
    """
    if cfg is None:
        return model(
            batch,
            pred_steps=pred_steps,
            deterministic_gate=deterministic_gate,
            edge_drop_mask=edge_drop_mask,
        )

    was_training = model.training

    if cfg.eval_mode_for_cf:
        model.eval()

    try:
        out = model(
            batch,
            pred_steps=pred_steps,
            deterministic_gate=deterministic_gate,
            edge_drop_mask=edge_drop_mask,
        )
    finally:
        if cfg.eval_mode_for_cf and was_training:
            model.train()

    return out


def get_segment_open_probability(
    open_probs: torch.Tensor,
    b_idx: int,
    t_idx: int,
    u_idx: int,
    v_idx: int,
    cfg: CounterfactualConfig,
) -> torch.Tensor:
    """
    Return a segment-level open probability scalar.

    This fixes the semantic mismatch:
        segment-level degradation should supervise segment-level open probability,
        not each time step independently with the same scalar label.
    """
    if cfg.supervise_segment_mean:
        p = open_probs[b_idx, t_idx:, u_idx, v_idx].clamp(cfg.eps, 1.0 - cfg.eps).mean()
    else:
        p = open_probs[b_idx, t_idx, u_idx, v_idx].clamp(cfg.eps, 1.0 - cfg.eps)

    return p


def counterfactual_gate_loss(
    model,
    batch: Dict[str, torch.Tensor],
    out_full: Dict[str, torch.Tensor],
    cfg: CounterfactualConfig,
) -> Dict[str, torch.Tensor]:
    """
    Lightweight counterfactual gate loss.

    Normal training forward:
        out_full = model(batch)

    CF label generation:
        1. Select top disagreement edges from out_full using expected_beta = p_open * alpha.
        2. Run deterministic full baseline with no_grad:
              out_base = model(batch, deterministic_gate=True)
        3. For each selected edge, run deterministic deleted-edge rollout:
              out_cf = model(batch, deterministic_gate=True, edge_drop_mask=...)
        4. Compute receiver-specific degradation:
              C_e = max(0, error_cf - error_base)
        5. Convert C_e to soft labels.
        6. Supervise segment-level mean open probability from out_full.

    Gradients:
        - No gradient through out_base/out_cf.
        - Gradient flows through out_full["open_probs"].
    """
    pred_full = out_full["pred_states"]
    open_probs = out_full["open_probs"]

    b, t, n, _ = pred_full.shape
    device = pred_full.device

    selected_edges = select_top_disagreement_edges(out_full, cfg)

    if len(selected_edges) == 0:
        zero = open_probs.sum() * 0.0
        return {
            "cf": zero,
            "cf_num_edges": torch.tensor(0.0, device=device),
            "cf_mean_degradation": torch.tensor(0.0, device=device),
            "cf_max_degradation": torch.tensor(0.0, device=device),
            "cf_mean_label": torch.tensor(0.0, device=device),
            "cf_mean_open_prob": torch.tensor(0.0, device=device),
        }

    # Deterministic full baseline for clean degradation comparison.
    if cfg.no_grad_cf_rollout:
        with torch.no_grad():
            out_base = _run_model_for_cf(
                model=model,
                batch=batch,
                pred_steps=t,
                deterministic_gate=cfg.deterministic_gate_for_cf,
                edge_drop_mask=None,
                cfg=cfg,
            )
    else:
        out_base = _run_model_for_cf(
            model=model,
            batch=batch,
            pred_steps=t,
            deterministic_gate=cfg.deterministic_gate_for_cf,
            edge_drop_mask=None,
            cfg=cfg,
        )

    base_errors = []
    cf_errors = []
    p_segments = []

    for edge in selected_edges:
        b_idx, t_idx, u_idx, v_idx = edge

        base_err = receiver_position_error_from_step(
            pred_states=out_base["pred_states"],
            batch=batch,
            b_idx=b_idx,
            receiver_idx=v_idx,
            start_t=t_idx,
            eps=cfg.eps,
        )

        drop_mask = build_edge_drop_mask(
            batch_size=b,
            pred_steps=t,
            num_nodes=n,
            selected_edge=edge,
            cfg=cfg,
            device=device,
        )

        if cfg.no_grad_cf_rollout:
            with torch.no_grad():
                out_cf = _run_model_for_cf(
                    model=model,
                    batch=batch,
                    pred_steps=t,
                    deterministic_gate=cfg.deterministic_gate_for_cf,
                    edge_drop_mask=drop_mask,
                    cfg=cfg,
                )
        else:
            out_cf = _run_model_for_cf(
                model=model,
                batch=batch,
                pred_steps=t,
                deterministic_gate=cfg.deterministic_gate_for_cf,
                edge_drop_mask=drop_mask,
                cfg=cfg,
            )

        cf_err = receiver_position_error_from_step(
            pred_states=out_cf["pred_states"],
            batch=batch,
            b_idx=b_idx,
            receiver_idx=v_idx,
            start_t=t_idx,
            eps=cfg.eps,
        )

        p_segment = get_segment_open_probability(
            open_probs=open_probs,
            b_idx=b_idx,
            t_idx=t_idx,
            u_idx=u_idx,
            v_idx=v_idx,
            cfg=cfg,
        )

        base_errors.append(base_err.detach())
        cf_errors.append(cf_err.detach())
        p_segments.append(p_segment)

    base_errors_t = torch.stack(base_errors)
    cf_errors_t = torch.stack(cf_errors)

    degradation = torch.clamp(cf_errors_t - base_errors_t, min=0.0)
    labels = build_cf_labels_from_degradation(degradation, cfg)

    p_segments_t = torch.stack(p_segments).clamp(cfg.eps, 1.0 - cfg.eps)

    # label is already detached and on the same device.
    cf_loss = F.binary_cross_entropy(
        p_segments_t,
        labels,
        reduction="mean",
    )

    return {
        "cf": cf_loss,
        "cf_num_edges": torch.tensor(float(len(selected_edges)), device=device),
        "cf_mean_degradation": degradation.mean().detach(),
        "cf_max_degradation": degradation.max().detach(),
        "cf_mean_label": labels.mean().detach(),
        "cf_mean_open_prob": p_segments_t.detach().mean(),
    }


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def probe_counterfactual_signal(
    num_batches: int = 50,
    batch_size: int = 4,
    pred_steps: int = 50,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """
    Run CF signal probe on real data for multiple batches.

    Recommended usage:
        - First train a trajectory warm-up model.
        - Save checkpoint.
        - Set checkpoint_path to that checkpoint.
        - Run this probe before enabling CF loss in training.

    If checkpoint_path is None, this probes a randomly initialized model,
    which is much less informative.
    """

    from ngsim_dataset import NGSIMCTEDataset, NGSIMDatasetConfig, simple_collate_fn

    from dynamic_cte_gnn import DynamicCTEGNN, DynamicCTEGNNConfig

    from torch.utils.data import DataLoader

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    dataset_cfg = NGSIMDatasetConfig(
        split="train",
        top_k_neighbors=8,
    )
    dataset = NGSIMCTEDataset(dataset_cfg)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate_fn,
    )

    model_cfg = DynamicCTEGNNConfig(
        pred_steps=pred_steps,
        cte_table_path=dataset_cfg.cte_table_path,
    )
    model = DynamicCTEGNN(model_cfg).to(device_obj)

    if checkpoint_path is not None and len(checkpoint_path) > 0:
        ckpt = torch.load(checkpoint_path, map_location=device_obj)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"[probe] loaded checkpoint: {checkpoint_path}")
    else:
        print("[probe] no checkpoint provided. Probing a randomly initialized model.")

    cfg = CounterfactualConfig(
        cf_step_indices=(9, 19, 29, 39, 49),
        top_k_edges=16,
        min_open_prob=0.2,
        min_expected_beta=1e-3,
        supervise_segment_mean=True,
    )

    model.eval()

    all_num_edges = []
    all_mean_deg = []
    all_max_deg = []
    all_mean_label = []
    all_mean_p = []

    checked = 0

    for batch_idx, batch in enumerate(loader):
        if checked >= num_batches:
            break

        batch = move_batch_to_device(batch, device_obj)

        with torch.no_grad():
            out_full = model(
                batch,
                pred_steps=pred_steps,
                deterministic_gate=True,
            )

        cf_dict = counterfactual_gate_loss(
            model=model,
            batch=batch,
            out_full=out_full,
            cfg=cfg,
        )

        num_edges = float(cf_dict["cf_num_edges"].detach().cpu())
        mean_deg = float(cf_dict["cf_mean_degradation"].detach().cpu())
        max_deg = float(cf_dict["cf_max_degradation"].detach().cpu())
        mean_label = float(cf_dict["cf_mean_label"].detach().cpu())
        mean_p = float(cf_dict["cf_mean_open_prob"].detach().cpu())

        all_num_edges.append(num_edges)
        all_mean_deg.append(mean_deg)
        all_max_deg.append(max_deg)
        all_mean_label.append(mean_label)
        all_mean_p.append(mean_p)

        print(
            f"[probe batch {checked + 1:03d}] "
            f"num_edges={num_edges:.1f} | "
            f"mean_deg={mean_deg:.6f} | "
            f"max_deg={max_deg:.6f} | "
            f"mean_label={mean_label:.6f} | "
            f"mean_p={mean_p:.6f}"
        )

        checked += 1

    if checked == 0:
        print("[probe] no batch checked.")
        return

    def summarize(name: str, values: List[float]) -> None:
        x = torch.tensor(values, dtype=torch.float32)
        print(
            f"[probe summary] {name}: "
            f"mean={float(x.mean()):.6f}, "
            f"std={float(x.std(unbiased=False)):.6f}, "
            f"min={float(x.min()):.6f}, "
            f"max={float(x.max()):.6f}"
        )

    print("\n[probe summary]")
    summarize("cf_num_edges", all_num_edges)
    summarize("cf_mean_degradation", all_mean_deg)
    summarize("cf_max_degradation", all_max_deg)
    summarize("cf_mean_label", all_mean_label)
    summarize("cf_mean_open_prob", all_mean_p)

    nonzero_batches = sum(v > 1e-6 for v in all_max_deg)
    print(
        f"[probe decision hint] nonzero max degradation batches: "
        f"{nonzero_batches}/{checked}"
    )

    if nonzero_batches == 0:
        print(
            "[probe decision hint] CF signal looks absent. "
            "Do not enable CF loss yet."
        )
    elif nonzero_batches < checked * 0.3:
        print(
            "[probe decision hint] CF signal is sparse. "
            "Enable only with small weight or later after warm-up."
        )
    else:
        print(
            "[probe decision hint] CF signal exists. "
            "You can consider enabling CF with a small lambda_cf."
        )


def main() -> None:
    """
    PyCharm entry.

    For meaningful CF signal, set checkpoint_path to a warm-up trained model.
    """
    probe_counterfactual_signal(
        num_batches=50,
        batch_size=4,
        pred_steps=50,
        checkpoint_path=None,
        device=None,
    )


if __name__ == "__main__":
    main()