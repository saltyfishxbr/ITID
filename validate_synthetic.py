# validate_synthetic_fixed.py
#
# Validate synthetic recovery against known directed ground truth.
#
# This version is self-contained: it does not depend on analyze_interaction_ngsim.py
# helper imports, so file-name differences will not break validation.
#
# PyCharm usage:
#   Edit SynthValConfig below, then run this file.

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Set
from collections import defaultdict
import csv
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ngsim_dataset import NGSIMCTEDataset, NGSIMDatasetConfig, simple_collate_fn
from dynamic_cte_gnn import DynamicCTEGNN, DynamicCTEGNNConfig


@dataclass
class SynthValConfig:
    checkpoint_path: str = "./Runs/NGSIM_CTEGNN/synth_ctegnn_no_cf_20260629_155256/checkpoints/checkpoint_best.pt"
    synth_window_path: str = "./ProcessedDataset/SYNTH/synth_windows_compact.pkl"
    synth_cte_table_path: str = "./ProcessedDataset/SYNTH/synth_cte_table.pkl"
    gt_path: str = "./ProcessedDataset/SYNTH/synth_gt.json"
    output_dir: str = "./Runs/NGSIM_CTEGNN/synthetic_validation"

    split: str = "val"
    fallback_to_train_if_empty: bool = True

    batch_size: int = 16
    num_workers: int = 0
    max_batches: Optional[int] = None
    device: str = "cuda"
    seed: int = 42

    active_open_threshold: float = 0.5
    cte_spurious_threshold: float = 1e-9

    # beta | expected_beta | open_prob | pi | expected_pi
    strength_key: str = "expected_beta"
    time_reduce: str = "mean"

    # Optional: require a pair to be observed at least this many times across windows.
    min_pair_observations: int = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def safe_make_config(cls, cfg_dict: Optional[Dict[str, Any]]):
    if cfg_dict is None:
        return cls()
    valid = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid}
    return cls(**filtered)


def compute_pi(beta: torch.Tensor, edge_mask: torch.Tensor, lambda_self: float) -> torch.Tensor:
    """
    beta shape: [B,T,N_source,N_receiver]
    edge_mask shape: [B,N_source,N_receiver]

    Receiver-normalized influence:
        pi[u,v] = beta[u,v] / (lambda_self + sum_p beta[p,v])
    """
    b, t, n, _ = beta.shape
    mask_t = edge_mask.unsqueeze(1).expand(b, t, n, n).to(beta.dtype)
    beta = beta * mask_t
    incoming_sum = beta.sum(dim=2, keepdim=True)
    pi = beta / (float(lambda_self) + incoming_sum + 1e-8)
    return pi * mask_t


def rankdata_average(x: np.ndarray) -> np.ndarray:
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
    rx = rankdata_average(x) - np.mean(rankdata_average(x))
    ry = rankdata_average(y) - np.mean(rankdata_average(y))
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = rankdata_average(allv)
    rank_pos = ranks[: len(pos)].sum()
    auc = (rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    yt = y_true[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / max(yt.sum(), 1e-9)
    ap = 0.0
    prev_recall = 0.0
    for k in range(len(yt)):
        if yt[k] == 1:
            ap += precision[k] * (recall[k] - prev_recall)
            prev_recall = recall[k]
    return float(ap)


def load_checkpoint_model(cfg: SynthValConfig):
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")

    model_cfg = safe_make_config(DynamicCTEGNNConfig, ckpt.get("model_cfg", None))
    dataset_cfg = safe_make_config(NGSIMDatasetConfig, ckpt.get("dataset_cfg", None))

    # Important: model dynamic CTE lookup must use the synthetic-world table.
    model_cfg.cte_table_path = cfg.synth_cte_table_path

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = DynamicCTEGNN(model_cfg).to(device)

    state_dict = ckpt.get("model", None)
    if state_dict is None:
        state_dict = ckpt.get("model_state_dict", None)
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'model' or 'model_state_dict'.")

    model.load_state_dict(state_dict)
    model.eval()
    return model, model_cfg, dataset_cfg, device


def make_dataset_cfg(cfg: SynthValConfig, dataset_cfg: NGSIMDatasetConfig, split: str) -> NGSIMDatasetConfig:
    return NGSIMDatasetConfig(
        window_path=cfg.synth_window_path,
        cte_table_path=cfg.synth_cte_table_path,
        split=split,
        top_k_neighbors=dataset_cfg.top_k_neighbors,
        max_ttc=dataset_cfg.max_ttc,
        allow_invalid_lane_edges=dataset_cfg.allow_invalid_lane_edges,
        num_vehicle_types=dataset_cfg.num_vehicle_types,
        padding_type_index=dataset_cfg.padding_type_index,
        verbose=False,
    )


def build_loader(cfg: SynthValConfig, dataset_cfg: NGSIMDatasetConfig):
    ds_cfg = make_dataset_cfg(cfg, dataset_cfg, cfg.split)
    dataset = NGSIMCTEDataset(ds_cfg)

    if len(dataset) == 0 and cfg.fallback_to_train_if_empty and cfg.split != "train":
        print(f"[warning] split={cfg.split} is empty. Falling back to split=train.")
        ds_cfg = make_dataset_cfg(cfg, dataset_cfg, "train")
        dataset = NGSIMCTEDataset(ds_cfg)

    print("\n[synthetic validation dataset]")
    print("split:", ds_cfg.split)
    print("num samples:", len(dataset))
    dataset.summary(diagnostics_samples=500)

    if len(dataset) == 0:
        raise RuntimeError("Synthetic validation dataset is empty. Check synth windows splits.")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
        collate_fn=simple_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
    )
    return loader, ds_cfg


def parse_edge_key(k: str) -> Tuple[int, int]:
    s, r = k.split("->")
    return int(s), int(r)


def load_ground_truth(gt_path: str):
    with open(gt_path, "r") as f:
        gt = json.load(f)

    gt_edges: Set[Tuple[int, int]] = set(parse_edge_key(k) for k in gt.get("gt_edges", {}).keys())
    gt_strength: Dict[Tuple[int, int], float] = {
        parse_edge_key(k): float(v) for k, v in gt.get("gt_strength", {}).items()
    }
    vclass = {int(k): int(v) for k, v in gt.get("vclass", {}).items()}
    confounded_false_edges: Set[Tuple[int, int]] = set(
        parse_edge_key(k) for k in gt.get("confounded_false_edges", {}).keys()
    )

    return gt, gt_edges, gt_strength, vclass, confounded_false_edges


def infer_confounded_false_pairs(gt_edges: Set[Tuple[int, int]], candidate_pairs: Set[Tuple[int, int]]):
    leaders_of = defaultdict(set)
    for s, r in gt_edges:
        leaders_of[r].add(s)

    out = []
    for i, j in candidate_pairs:
        if (i, j) in gt_edges:
            continue
        if len(leaders_of[i] & leaders_of[j]) > 0:
            out.append((i, j))
    return set(out)


def aggregate_dict_list(d: Dict[Tuple[int, int], List[float]]) -> Dict[Tuple[int, int], float]:
    return {k: float(np.mean(v)) for k, v in d.items() if len(v) > 0}


def main() -> None:
    cfg = SynthValConfig()
    seed_everything(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "synthetic_validation_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    model, model_cfg, dataset_cfg, device = load_checkpoint_model(cfg)
    loader, used_ds_cfg = build_loader(cfg, dataset_cfg)
    gt_raw, gt_edges, gt_strength, vclass, confounded_false_edges_from_gt = load_ground_truth(cfg.gt_path)

    print("\n[checkpoint]")
    print("checkpoint_path:", cfg.checkpoint_path)
    print("model pred_steps:", model_cfg.pred_steps)
    print("model cte_table_path:", model_cfg.cte_table_path)
    print("dataset top_k:", used_ds_cfg.top_k_neighbors)
    print("GT edges:", len(gt_edges))
    print("GT strength edges:", len(gt_strength))
    print("GT explicit confounded false edges:", len(confounded_false_edges_from_gt))
    print("Convention: out['betas'][b,t,u,v] = source node u -> receiver node v")

    lambda_self = float(getattr(model_cfg, "lambda_self", 1.0))

    edge_openprob = defaultdict(list)
    edge_beta = defaultdict(list)
    edge_expected_beta = defaultdict(list)
    edge_pi = defaultdict(list)
    edge_expected_pi = defaultdict(list)
    edge_cte = defaultdict(list)
    edge_obs_count = defaultdict(int)
    candidate_pairs: Set[Tuple[int, int]] = set()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if cfg.max_batches is not None and batch_idx >= int(cfg.max_batches):
                break

            batch = move_batch_to_device(batch, device)
            out = model(batch, deterministic_gate=True)

            betas = out["betas"]              # [B,T,N,N]
            open_probs = out["open_probs"]    # [B,T,N,N]
            alphas = out.get("alphas", None)
            edge_mask = out["edge_mask"].bool()
            cte_teacher = out.get("cte_teacher", None)
            pi = compute_pi(betas, edge_mask, lambda_self)

            if alphas is not None:
                expected_betas = open_probs * alphas
                expected_pi = compute_pi(expected_betas, edge_mask, lambda_self)
            else:
                expected_betas = betas
                expected_pi = pi

            if cfg.time_reduce != "mean":
                raise ValueError(f"Unsupported time_reduce={cfg.time_reduce}")

            beta_red = betas.mean(dim=1).detach().cpu().numpy()
            op_red = open_probs.mean(dim=1).detach().cpu().numpy()
            exp_beta_red = expected_betas.mean(dim=1).detach().cpu().numpy()
            pi_red = pi.mean(dim=1).detach().cpu().numpy()
            exp_pi_red = expected_pi.mean(dim=1).detach().cpu().numpy()
            if cte_teacher is not None:
                cte_red = cte_teacher.mean(dim=1).detach().cpu().numpy()
            else:
                cte_red = None

            vids = batch["vehicle_ids"].detach().cpu().numpy()
            node_mask = batch["node_mask"].detach().cpu().numpy().astype(bool)
            em = edge_mask.detach().cpu().numpy().astype(bool)

            b, n, _ = em.shape
            for bi in range(b):
                for u in range(n):
                    if not node_mask[bi, u]:
                        continue
                    for v in range(n):
                        if u == v or not node_mask[bi, v]:
                            continue
                        if not em[bi, u, v]:
                            continue

                        s_vid = int(vids[bi, u])
                        r_vid = int(vids[bi, v])
                        if s_vid <= 0 or r_vid <= 0:
                            continue

                        key = (s_vid, r_vid)
                        candidate_pairs.add(key)
                        edge_obs_count[key] += 1
                        edge_openprob[key].append(float(op_red[bi, u, v]))
                        edge_beta[key].append(float(beta_red[bi, u, v]))
                        edge_expected_beta[key].append(float(exp_beta_red[bi, u, v]))
                        edge_pi[key].append(float(pi_red[bi, u, v]))
                        edge_expected_pi[key].append(float(exp_pi_red[bi, u, v]))
                        if cte_red is not None:
                            edge_cte[key].append(float(cte_red[bi, u, v]))

            if (batch_idx + 1) % 50 == 0:
                print(f"[validate] batch={batch_idx + 1:04d} candidate_pairs={len(candidate_pairs)}")

    # Filter very rarely observed candidate pairs if requested.
    if cfg.min_pair_observations > 1:
        candidate_pairs = {p for p in candidate_pairs if edge_obs_count[p] >= cfg.min_pair_observations}

    op_pair = aggregate_dict_list(edge_openprob)
    beta_pair = aggregate_dict_list(edge_beta)
    exp_beta_pair = aggregate_dict_list(edge_expected_beta)
    pi_pair = aggregate_dict_list(edge_pi)
    exp_pi_pair = aggregate_dict_list(edge_expected_pi)
    cte_pair = aggregate_dict_list(edge_cte)

    pairs = sorted(candidate_pairs)
    y_true = np.array([1 if p in gt_edges else 0 for p in pairs], dtype=np.int32)
    y_score = np.array([op_pair.get(p, 0.0) for p in pairs], dtype=np.float64)

    auc = roc_auc(y_true, y_score)
    aupr = average_precision(y_true, y_score)

    y_pred = (y_score >= float(cfg.active_open_threshold)).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)

    gt_in_candidates = sum(1 for e in gt_edges if e in candidate_pairs)
    gt_recall_ceiling = gt_in_candidates / (len(gt_edges) + 1e-9)

    strength_dict = {
        "beta": beta_pair,
        "expected_beta": exp_beta_pair,
        "open_prob": op_pair,
        "pi": pi_pair,
        "expected_pi": exp_pi_pair,
    }[cfg.strength_key]

    common_strength = [p for p in gt_strength.keys() if p in strength_dict]
    if len(common_strength) >= 5:
        gt_vec = np.array([gt_strength[p] for p in common_strength], dtype=np.float64)
        model_vec = np.array([strength_dict[p] for p in common_strength], dtype=np.float64)
        strength_spearman = spearman_corr(gt_vec, model_vec)
    else:
        strength_spearman = float("nan")

    asym_total = 0
    asym_correct = 0
    asym_values = []
    for s, r in gt_edges:
        if (r, s) in gt_edges:
            continue
        fwd = beta_pair.get((s, r), None)
        if fwd is None:
            continue
        rev = beta_pair.get((r, s), 0.0)
        asym_total += 1
        asym_values.append(float(fwd - rev))
        if fwd > rev:
            asym_correct += 1
    asym_acc = asym_correct / (asym_total + 1e-9)

    if len(confounded_false_edges_from_gt) > 0:
        confounded_false_pairs = confounded_false_edges_from_gt & candidate_pairs
    else:
        confounded_false_pairs = infer_confounded_false_pairs(gt_edges, candidate_pairs)

    def spurious_rate(pair_set: Set[Tuple[int, int]], score_dict: Dict[Tuple[int, int], float], thresh: float):
        pair_list = sorted(pair_set)
        if len(pair_list) == 0:
            return float("nan"), 0, float("nan")
        scores = np.array([score_dict.get(p, 0.0) for p in pair_list], dtype=np.float64)
        return float((scores >= thresh).mean()), len(pair_list), float(np.mean(scores))

    cte_spurious, n_conf, cte_conf_mean = spurious_rate(
        confounded_false_pairs, cte_pair, cfg.cte_spurious_threshold
    )
    model_spurious, _, op_conf_mean = spurious_rate(
        confounded_false_pairs, op_pair, cfg.active_open_threshold
    )
    beta_conf_mean = float(np.mean([beta_pair.get(p, 0.0) for p in confounded_false_pairs])) if n_conf > 0 else float("nan")

    true_edge_op_mean = float(np.mean([op_pair.get(p, 0.0) for p in gt_edges if p in op_pair])) if gt_in_candidates > 0 else float("nan")
    false_edge_op_mean = float(np.mean([op_pair.get(p, 0.0) for p in pairs if p not in gt_edges])) if len(pairs) > gt_in_candidates else float("nan")

    results = {
        "scenario": gt_raw.get("scenario", "unknown"),
        "checkpoint_path": cfg.checkpoint_path,
        "synth_window_path": cfg.synth_window_path,
        "synth_cte_table_path": cfg.synth_cte_table_path,
        "split_used": used_ds_cfg.split,
        "scenario_gt_edges": len(gt_edges),
        "candidate_pairs": len(candidate_pairs),
        "gt_edges_reachable_as_candidate": gt_in_candidates,
        "gt_recall_ceiling": gt_recall_ceiling,
        "existence_AUC": auc,
        "existence_AUPRC": aupr,
        f"existence_precision@{cfg.active_open_threshold}": precision,
        f"existence_recall@{cfg.active_open_threshold}": recall,
        f"existence_F1@{cfg.active_open_threshold}": f1,
        "true_edge_mean_open_prob": true_edge_op_mean,
        "false_edge_mean_open_prob": false_edge_op_mean,
        "strength_key": cfg.strength_key,
        "strength_edges_scored": len(common_strength),
        "strength_spearman_vs_gt": strength_spearman,
        "asymmetry_pairs": asym_total,
        "asymmetry_direction_accuracy": asym_acc,
        "mean_beta_forward_minus_reverse_on_asym_gt": float(np.mean(asym_values)) if len(asym_values) > 0 else float("nan"),
        "confounded_false_pairs_explicit_or_inferred": n_conf,
        "spurious_rate_pure_CTE": cte_spurious,
        "spurious_rate_our_model": model_spurious,
        "mean_cte_on_confounded_false": cte_conf_mean,
        "mean_open_prob_on_confounded_false": op_conf_mean,
        "mean_beta_on_confounded_false": beta_conf_mean,
    }

    print("\n================ SYNTHETIC VALIDATION ================")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print("=====================================================")

    with open(out_dir / "synthetic_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_dir / 'synthetic_validation_results.json'}")

    conf_set = set(confounded_false_pairs)
    with open(out_dir / "synthetic_edge_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_vid",
                "receiver_vid",
                "source_vclass",
                "receiver_vclass",
                "gt_edge",
                "gt_strength",
                "observations",
                "open_prob",
                "beta",
                "expected_beta",
                "pi",
                "expected_pi",
                "cte_teacher",
                "is_confounded_false",
            ]
        )
        for p in sorted(candidate_pairs):
            writer.writerow(
                [
                    p[0],
                    p[1],
                    vclass.get(p[0], ""),
                    vclass.get(p[1], ""),
                    1 if p in gt_edges else 0,
                    gt_strength.get(p, ""),
                    edge_obs_count.get(p, 0),
                    op_pair.get(p, ""),
                    beta_pair.get(p, ""),
                    exp_beta_pair.get(p, ""),
                    pi_pair.get(p, ""),
                    exp_pi_pair.get(p, ""),
                    cte_pair.get(p, ""),
                    1 if p in conf_set else 0,
                ]
            )
    print(f"[saved] {out_dir / 'synthetic_edge_scores.csv'}")


if __name__ == "__main__":
    main()
