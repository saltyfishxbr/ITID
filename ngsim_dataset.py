# ngsim_dataset.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cte_lookup import (
    CTELookup,
    CTELookupConfig,
    lane_relation_python,
    LANE_SAME_FRONT,
    LANE_SAME_BACK,
    LANE_ADJACENT,
    LEVEL_DEFAULT,
    LEVEL_GLOBAL,
    LEVEL_LANE_RHO,
    LEVEL_NO_DV,
    LEVEL_FINE,
)


@dataclass
class NGSIMDatasetConfig:
    window_path: str = "./ProcessedDataset/NGSIM/0750am-0805am/windows_ngsim_0750_0805.pkl"
    cte_table_path: str = "./ProcessedDataset/NGSIM/0750am-0805am/ngsim_0750_0805_cte_table.pkl"

    split: str = "train"

    top_k_neighbors: int = 8
    max_ttc: float = 20.0

    # Default: invalid lane edges are not included.
    allow_invalid_lane_edges: bool = False

    # NGSIM raw v_Class is usually 1,2,3.
    # We map to 0,1,2. Padding uses 3.
    num_vehicle_types: int = 3
    padding_type_index: int = 3

    verbose: bool = False


IDX_X = 0
IDX_Y = 1
IDX_VX = 2
IDX_VY = 3
IDX_SPEED = 4
IDX_ACC = 5


def load_pickle(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find file: {p}")

    with open(p, "rb") as f:
        return pickle.load(f)


def extract_sequence(
    track_bank: Dict[int, Dict],
    vehicle_id: int,
    start_frame: int,
    end_frame: int,
) -> Optional[np.ndarray]:
    if vehicle_id not in track_bank:
        return None

    track = track_bank[vehicle_id]
    frames = track["frames"]
    data = track["data"]

    query_frames = np.arange(start_frame, end_frame + 1, dtype=np.int64)
    idx = np.searchsorted(frames, query_frames)

    if np.any(idx >= len(frames)):
        return None

    if not np.all(frames[idx] == query_frames):
        return None

    return data[idx].astype(np.float32)


def get_lane_at_frame(
    track_bank: Dict[int, Dict],
    vehicle_id: int,
    frame: int,
) -> int:
    if vehicle_id not in track_bank:
        return -1

    track = track_bank[vehicle_id]
    frames = track["frames"]
    lanes = track["lane"]

    idx = np.searchsorted(frames, frame)

    if idx >= len(frames) or int(frames[idx]) != int(frame):
        return -1

    return int(lanes[idx])


def get_type_index(track_bank: Dict[int, Dict], vehicle_id: int, padding_type_index: int) -> int:
    if vehicle_id not in track_bank:
        return int(padding_type_index)

    raw_type = int(track_bank[vehicle_id].get("v_class", 2))
    mapped = max(0, raw_type - 1)
    return int(mapped)


def get_raw_type(track_bank: Dict[int, Dict], vehicle_id: int) -> int:
    if vehicle_id not in track_bank:
        return -1
    return int(track_bank[vehicle_id].get("v_class", -1))


def center_history_future(
    history: np.ndarray,
    future: np.ndarray,
    origin_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    history = history.copy()
    future = future.copy()

    history[:, :, IDX_X] -= origin_xy[0]
    history[:, :, IDX_Y] -= origin_xy[1]

    future[:, :, IDX_X] -= origin_xy[0]
    future[:, :, IDX_Y] -= origin_xy[1]

    return history, future


def compute_ttc(
    delta_y: float,
    source_speed: float,
    receiver_speed: float,
    max_ttc: float,
) -> float:
    """
    TTC for source -> receiver.

    delta_y = y_source - y_receiver.

    If source is ahead and receiver is faster, receiver may catch source.
    """
    closing_speed = receiver_speed - source_speed

    if delta_y > 0.0 and closing_speed > 1e-3:
        ttc = delta_y / closing_speed
        return max(0.0, min(float(ttc), float(max_ttc)))

    return float(max_ttc)


def lane_priority_for_candidate(lane_rel: int) -> int:
    if lane_rel == LANE_SAME_FRONT:
        return 0
    if lane_rel == LANE_SAME_BACK:
        return 1
    if lane_rel == LANE_ADJACENT:
        return 2
    return 3


class NGSIMCTEDataset(Dataset):
    """
    Dataset after offline preparation.

    Responsibilities:
        1. Load compact raw perception windows.
        2. Use train CTE table to select TopK candidate neighbors.
        3. Extract target and selected neighbors' history/future.
        4. Build node tensors, edge masks, relative features, and CTE masks.

    Not responsible for:
        - dynamic graph update
        - Hard Concrete
        - GNN
        - loss computation
    """

    def __init__(self, config: NGSIMDatasetConfig):
        super().__init__()

        self.config = config

        self.payload = load_pickle(config.window_path)
        self.track_bank = self.payload["track_bank"]
        self.windows = self.payload["windows"]

        if config.split not in self.payload["splits"]:
            raise ValueError(f"Unknown split: {config.split}")

        self.indices = list(self.payload["splits"][config.split])

        self.history_len = int(self.payload["history_len"])
        self.future_len = int(self.payload["future_len"])
        self.feature_columns = self.payload["feature_columns"]

        self.num_nodes = int(config.top_k_neighbors + 1)
        self.feature_dim = len(self.feature_columns)

        self.cte_lookup = CTELookup(
            CTELookupConfig(cte_table_path=config.cte_table_path)
        )

        if self.config.padding_type_index < self.config.num_vehicle_types:
            raise ValueError(
                "padding_type_index should be >= num_vehicle_types. "
                f"Got padding_type_index={self.config.padding_type_index}, "
                f"num_vehicle_types={self.config.num_vehicle_types}."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def _select_topk_neighbors(self, window: Dict) -> Tuple[List[int], List[Dict[str, Any]]]:
        neighbor_ids = window["neighbor_ids"]
        delta_y = window["neighbor_delta_y"]
        delta_speed = window["neighbor_delta_speed"]
        lane_rel = window["neighbor_lane_relation"]
        rho = float(window["density0"])

        candidates = []

        for idx, vid in enumerate(neighbor_ids):
            dy = float(delta_y[idx])
            dv = float(delta_speed[idx])
            lr = int(lane_rel[idx])

            detail = self.cte_lookup.lookup_python_detail(
                delta_y=dy,
                delta_speed=dv,
                lane_rel=lr,
                rho=rho,
            )

            candidates.append(
                {
                    "neighbor_id": int(vid),
                    "cte": float(detail["cte"]),
                    "cte_hit": bool(detail["hit"]),
                    "cte_valid": bool(detail["valid"]),
                    "cte_level_code": int(detail["level_code"]),
                    "cte_is_significant": bool(detail["is_significant"]),
                    "abs_delta_y": abs(dy),
                    "lane_priority": lane_priority_for_candidate(lr),
                    "original_index": idx,
                }
            )

        candidates = sorted(
            candidates,
            key=lambda c: (
                -c["cte"],
                c["lane_priority"],
                c["abs_delta_y"],
                c["original_index"],
            ),
        )

        selected = candidates[: self.config.top_k_neighbors]
        selected_ids = [c["neighbor_id"] for c in selected]

        return selected_ids, selected

    def _extract_node_sequences(
        self,
        window: Dict,
        selected_neighbor_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_max = self.num_nodes
        th = self.history_len
        tf = self.future_len
        d = self.feature_dim

        history = np.zeros((n_max, th, d), dtype=np.float32)
        future = np.zeros((n_max, tf, d), dtype=np.float32)

        node_type_idx = np.full(
            (n_max,),
            fill_value=int(self.config.padding_type_index),
            dtype=np.int64,
        )
        node_type_raw = -np.ones((n_max,), dtype=np.int64)
        node_lane = -np.ones((n_max,), dtype=np.int64)
        node_mask = np.zeros((n_max,), dtype=np.bool_)
        vehicle_ids = -np.ones((n_max,), dtype=np.int64)

        hist_start = int(window["hist_start_frame"])
        hist_end = int(window["hist_end_frame"])
        fut_start = int(window["future_start_frame"])
        fut_end = int(window["future_end_frame"])
        center_frame = int(window["center_frame"])

        all_ids = [int(window["target_id"])] + [int(v) for v in selected_neighbor_ids]

        for node_idx, vid in enumerate(all_ids[:n_max]):
            hist = extract_sequence(self.track_bank, vid, hist_start, hist_end)
            fut = extract_sequence(self.track_bank, vid, fut_start, fut_end)

            if hist is None or fut is None:
                if self.config.verbose:
                    print(f"[dataset] missing sequence vid={vid}, frame={center_frame}")
                continue

            history[node_idx] = hist
            future[node_idx] = fut

            node_type_idx[node_idx] = get_type_index(
                self.track_bank,
                vid,
                padding_type_index=self.config.padding_type_index,
            )
            node_type_raw[node_idx] = get_raw_type(self.track_bank, vid)
            node_lane[node_idx] = get_lane_at_frame(self.track_bank, vid, center_frame)
            node_mask[node_idx] = True
            vehicle_ids[node_idx] = vid

        if not node_mask[0]:
            raise RuntimeError(
                f"Target sequence missing for sample target={window['target_id']} "
                f"frame={center_frame}"
            )

        target_origin_xy = history[0, -1, [IDX_X, IDX_Y]].copy()
        history, future = center_history_future(
            history=history,
            future=future,
            origin_xy=target_origin_xy,
        )

        return history, future, node_type_idx, node_type_raw, node_lane, node_mask, vehicle_ids

    def _build_edge_tensors(
        self,
        history: np.ndarray,
        node_lane: np.ndarray,
        node_mask: np.ndarray,
        density0: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.num_nodes
        f_rel = 7

        edge_mask = np.zeros((n, n), dtype=np.bool_)
        edge_lane_relation = -np.ones((n, n), dtype=np.int64)
        rel0 = np.zeros((n, n, f_rel), dtype=np.float32)

        cte0_matrix = np.zeros((n, n), dtype=np.float32)
        cte0_hit_matrix = np.zeros((n, n), dtype=np.bool_)
        cte0_valid_matrix = np.zeros((n, n), dtype=np.bool_)
        cte0_level_matrix = np.full((n, n), fill_value=LEVEL_DEFAULT, dtype=np.int64)
        cte0_significant_matrix = np.zeros((n, n), dtype=np.bool_)

        state0 = history[:, -1, :]

        for u in range(n):
            if not node_mask[u]:
                continue

            for v in range(n):
                if not node_mask[v]:
                    continue

                if u == v:
                    continue

                source_lane = int(node_lane[u])
                receiver_lane = int(node_lane[v])

                dx = float(state0[u, IDX_X] - state0[v, IDX_X])
                dy = float(state0[u, IDX_Y] - state0[v, IDX_Y])
                dspeed = float(state0[u, IDX_SPEED] - state0[v, IDX_SPEED])

                lane_rel = lane_relation_python(
                    source_lane=source_lane,
                    receiver_lane=receiver_lane,
                    delta_y=dy,
                )

                if lane_rel < 0 and not self.config.allow_invalid_lane_edges:
                    continue

                edge_mask[u, v] = True
                edge_lane_relation[u, v] = int(lane_rel)

                source_speed = float(state0[u, IDX_SPEED])
                receiver_speed = float(state0[v, IDX_SPEED])

                ttc = compute_ttc(
                    delta_y=dy,
                    source_speed=source_speed,
                    receiver_speed=receiver_speed,
                    max_ttc=self.config.max_ttc,
                )
                ttc_norm = ttc / float(self.config.max_ttc)

                rel0[u, v, 0] = dx
                rel0[u, v, 1] = dy
                rel0[u, v, 2] = dspeed
                rel0[u, v, 3] = ttc_norm

                if lane_rel == LANE_SAME_FRONT:
                    rel0[u, v, 4] = 1.0
                elif lane_rel == LANE_SAME_BACK:
                    rel0[u, v, 5] = 1.0
                elif lane_rel == LANE_ADJACENT:
                    rel0[u, v, 6] = 1.0

                # Invalid lane is uniformly CTE=0, consistent with torch lookup.
                if lane_rel < 0:
                    continue

                detail = self.cte_lookup.lookup_python_detail(
                    delta_y=dy,
                    delta_speed=dspeed,
                    lane_rel=lane_rel,
                    rho=float(density0),
                )

                cte0_matrix[u, v] = float(detail["cte"])
                cte0_hit_matrix[u, v] = bool(detail["hit"])
                cte0_valid_matrix[u, v] = bool(detail["valid"])
                cte0_level_matrix[u, v] = int(detail["level_code"])
                cte0_significant_matrix[u, v] = bool(detail["is_significant"])

        return (
            edge_mask,
            edge_lane_relation,
            rel0,
            cte0_matrix,
            cte0_hit_matrix,
            cte0_valid_matrix,
            cte0_level_matrix,
            cte0_significant_matrix,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        window = self.windows[real_idx]

        selected_neighbor_ids, selected_details = self._select_topk_neighbors(window)

        (
            history,
            future,
            node_type_idx,
            node_type_raw,
            node_lane,
            node_mask,
            vehicle_ids,
        ) = self._extract_node_sequences(
            window=window,
            selected_neighbor_ids=selected_neighbor_ids,
        )

        density0 = float(window["density0"])

        (
            edge_mask,
            edge_lane_relation,
            rel0,
            cte0_matrix,
            cte0_hit_matrix,
            cte0_valid_matrix,
            cte0_level_matrix,
            cte0_significant_matrix,
        ) = self._build_edge_tensors(
            history=history,
            node_lane=node_lane,
            node_mask=node_mask,
            density0=density0,
        )

        candidate_cte0 = np.zeros((self.num_nodes,), dtype=np.float32)
        candidate_cte_hit = np.zeros((self.num_nodes,), dtype=np.bool_)
        candidate_cte_valid = np.zeros((self.num_nodes,), dtype=np.bool_)
        candidate_cte_level = np.full((self.num_nodes,), fill_value=LEVEL_DEFAULT, dtype=np.int64)
        candidate_cte_significant = np.zeros((self.num_nodes,), dtype=np.bool_)

        for k, detail in enumerate(selected_details[: self.config.top_k_neighbors], start=1):
            candidate_cte0[k] = float(detail["cte"])
            candidate_cte_hit[k] = bool(detail["cte_hit"])
            candidate_cte_valid[k] = bool(detail["cte_valid"])
            candidate_cte_level[k] = int(detail["cte_level_code"])
            candidate_cte_significant[k] = bool(detail["cte_is_significant"])

        sample = {
            "history": torch.from_numpy(history).float(),
            "future": torch.from_numpy(future).float(),
            "state0": torch.from_numpy(history[:, -1, :]).float(),

            "node_type": torch.from_numpy(node_type_idx).long(),
            "node_type_raw": torch.from_numpy(node_type_raw).long(),
            "node_lane": torch.from_numpy(node_lane).long(),
            "node_mask": torch.from_numpy(node_mask).bool(),
            "vehicle_ids": torch.from_numpy(vehicle_ids).long(),

            "edge_mask": torch.from_numpy(edge_mask).bool(),
            "edge_lane_relation": torch.from_numpy(edge_lane_relation).long(),
            "rel0": torch.from_numpy(rel0).float(),

            "cte0_matrix": torch.from_numpy(cte0_matrix).float(),
            "cte0_hit_matrix": torch.from_numpy(cte0_hit_matrix).bool(),
            "cte0_valid_matrix": torch.from_numpy(cte0_valid_matrix).bool(),
            "cte0_level_matrix": torch.from_numpy(cte0_level_matrix).long(),
            "cte0_significant_matrix": torch.from_numpy(cte0_significant_matrix).bool(),

            "candidate_cte0": torch.from_numpy(candidate_cte0).float(),
            "candidate_cte_hit": torch.from_numpy(candidate_cte_hit).bool(),
            "candidate_cte_valid": torch.from_numpy(candidate_cte_valid).bool(),
            "candidate_cte_level": torch.from_numpy(candidate_cte_level).long(),
            "candidate_cte_significant": torch.from_numpy(candidate_cte_significant).bool(),

            "density0": torch.tensor(density0, dtype=torch.float32),
            "target_index": torch.tensor(0, dtype=torch.long),
            "sample_index": torch.tensor(int(real_idx), dtype=torch.long),
            "center_frame": torch.tensor(int(window["center_frame"]), dtype=torch.long),
            "target_id": torch.tensor(int(window["target_id"]), dtype=torch.long),
        }

        return sample

    def compute_diagnostics(self, max_samples: int = 5000) -> Dict[str, float]:
        if len(self) == 0:
            return {}

        n = min(max_samples, len(self))
        chosen = np.linspace(0, len(self) - 1, num=n, dtype=np.int64)

        raw_counts = []
        valid_nodes = []
        selected_nonzero = []
        selected_valid = []

        level_counts = {
            LEVEL_DEFAULT: 0,
            LEVEL_GLOBAL: 0,
            LEVEL_LANE_RHO: 0,
            LEVEL_NO_DV: 0,
            LEVEL_FINE: 0,
        }

        for local_idx in chosen:
            window = self.windows[self.indices[int(local_idx)]]
            raw_counts.append(len(window["neighbor_ids"]))

            selected_ids, selected_details = self._select_topk_neighbors(window)
            valid_nodes.append(1 + len(selected_ids))

            if len(selected_details) > 0:
                selected_nonzero.append(
                    np.mean([float(d["cte"]) > 0.0 for d in selected_details])
                )
                selected_valid.append(
                    np.mean([bool(d["cte_valid"]) for d in selected_details])
                )

                for d in selected_details:
                    level_counts[int(d["cte_level_code"])] += 1
            else:
                selected_nonzero.append(0.0)
                selected_valid.append(0.0)

        raw_counts = np.asarray(raw_counts, dtype=np.float64)
        valid_nodes = np.asarray(valid_nodes, dtype=np.float64)
        selected_nonzero = np.asarray(selected_nonzero, dtype=np.float64)
        selected_valid = np.asarray(selected_valid, dtype=np.float64)

        diagnostics = {
            "num_checked": float(n),
            "avg_raw_neighbors": float(raw_counts.mean()),
            "max_raw_neighbors": float(raw_counts.max()),
            "avg_valid_nodes_after_topk": float(valid_nodes.mean()),
            "padding_ratio": float(1.0 - valid_nodes.mean() / self.num_nodes),
            "avg_selected_cte_nonzero_ratio": float(selected_nonzero.mean()),
            "avg_selected_cte_valid_ratio": float(selected_valid.mean()),
            "level_default_count": float(level_counts[LEVEL_DEFAULT]),
            "level_global_count": float(level_counts[LEVEL_GLOBAL]),
            "level_lane_rho_count": float(level_counts[LEVEL_LANE_RHO]),
            "level_no_dv_count": float(level_counts[LEVEL_NO_DV]),
            "level_fine_count": float(level_counts[LEVEL_FINE]),
        }

        return diagnostics

    def summary(self, diagnostics_samples: int = 5000) -> None:
        print("[NGSIMCTEDataset] split:", self.config.split)
        print("[NGSIMCTEDataset] samples:", len(self))
        print("[NGSIMCTEDataset] top_k_neighbors:", self.config.top_k_neighbors)
        print("[NGSIMCTEDataset] num_nodes:", self.num_nodes)
        print("[NGSIMCTEDataset] history_len:", self.history_len)
        print("[NGSIMCTEDataset] future_len:", self.future_len)
        print("[NGSIMCTEDataset] feature_columns:", self.feature_columns)
        print("[NGSIMCTEDataset] padding_type_index:", self.config.padding_type_index)

        diagnostics = self.compute_diagnostics(max_samples=diagnostics_samples)
        if diagnostics:
            print("[NGSIMCTEDataset] diagnostics:")
            for k, v in diagnostics.items():
                print(f"  {k}: {v}")


def simple_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}

    keys = batch[0].keys()

    for key in keys:
        values = [b[key] for b in batch]

        if torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        else:
            out[key] = values

    return out


def main() -> None:
    cfg = NGSIMDatasetConfig(
        split="train",
        top_k_neighbors=8,
    )

    dataset = NGSIMCTEDataset(cfg)
    dataset.summary(diagnostics_samples=3000)

    sample = dataset[0]

    print("\n[sample shapes]")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        else:
            print(k, type(v))

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate_fn,
    )

    batch = next(iter(loader))

    print("\n[batch shapes]")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        else:
            print(k, type(v))


if __name__ == "__main__":
    main()