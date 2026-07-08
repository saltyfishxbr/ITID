# synthetic_sim_fixed.py
#
# Fully controllable synthetic traffic generator with known directed
# ground-truth interaction graph.
#
# Design choices in this fixed version:
#   1) Straight road, no ring-road wrap. Local_Y is continuous and compatible
#      with build_windows_ngsim.py's front/back and velocity-difference logic.
#   2) Output columns use NGSIM raw units: feet, ft/s, ft/s^2. Your existing
#      build_windows_ngsim.py will convert them back to meters using ft_to_m.
#   3) Global_Time is included.
#   4) Confounded false directed pairs are explicitly logged in the GT json.
#
# PyCharm usage:
#   Edit RUN_CFG / OUT_CSV / OUT_GT near the bottom, then run this file.
#
# CLI usage is also supported:
#   python synthetic_sim_fixed.py --scenario mixed --num_vehicles 60 --total_steps 6000

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import argparse
import json

import numpy as np
import pandas as pd


M_TO_FT = 1.0 / 0.3048


# NGSIM-like v_Class semantics:
#   1 -> motorcycle / reactive
#   2 -> small vehicle / baseline human
#   3 -> large or automated-like / smoother, larger gap
IDM_PARAMS = {
    1: dict(v0=30.0, T=1.0, a=2.5, b=2.5, s0=1.5, length=2.5),
    2: dict(v0=28.0, T=1.5, a=1.5, b=2.0, s0=2.0, length=4.5),
    3: dict(v0=26.0, T=2.2, a=1.0, b=1.5, s0=3.0, length=8.0),
}


@dataclass
class SimConfig:
    scenario: str = "mixed"  # follow | asymmetric | confounded | heterogeneous | mixed

    num_lanes: int = 3
    road_length: float = 900.0
    num_vehicles: int = 60

    dt: float = 0.1
    total_steps: int = 6000
    warmup_steps: int = 600

    accel_noise_std: float = 0.15
    pos_obs_noise_std: float = 0.05

    # Keep at 0 for clean GT interaction recovery. Lane-changing changes the
    # candidate relation while fixed GT dependencies remain fixed.
    lane_change_prob: float = 0.0
    lane_change_gap: float = 12.0

    confounded_triplets: int = 8
    seed: int = 0
    record_gt_strength: bool = True

    # Initial spacing and speed range. Lower speed variance improves stability
    # and keeps fixed leaders physically meaningful on the straight road.
    init_speed_min: float = 12.0
    init_speed_max: float = 20.0
    min_initial_spacing: float = 12.0


def idm_accel(v_ego: float, gap: Optional[float], v_lead: Optional[float], p: Dict[str, float]) -> float:
    """Standard IDM acceleration in SI units."""
    if gap is None:
        acc = p["a"] * (1.0 - (v_ego / max(p["v0"], 1e-3)) ** 4)
        return float(np.clip(acc, -8.0, 4.0))

    dv = v_ego - float(v_lead)
    s_star = p["s0"] + max(
        0.0,
        v_ego * p["T"] + (v_ego * dv) / (2.0 * np.sqrt(max(p["a"] * p["b"], 1e-9))),
    )
    gap_eff = max(float(gap), 0.1)
    acc = p["a"] * (
        1.0
        - (v_ego / max(p["v0"], 1e-3)) ** 4
        - (s_star / gap_eff) ** 2
    )
    return float(np.clip(acc, -8.0, 4.0))


class SyntheticTraffic:
    """
    Straight multi-lane road.

    fixed_dependency[receiver_idx] = list[source_idx]

    Each receiver's acceleration depends only on its listed source vehicles.
    The corresponding directed source -> receiver edges are the ground-truth
    interaction graph.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n = int(cfg.num_vehicles)

        self.confounded_false_pairs_idx: Set[Tuple[int, int]] = set()

        self.vclass = self._init_types()
        self.lane, self.pos, self.vel, self.lat = self._init_state()
        self.fixed_dependency = self._build_fixed_dependency()
        self._last_accel = np.zeros(self.n, dtype=np.float64)

    def _init_types(self) -> np.ndarray:
        n = self.n
        r = self.rng.random(n)
        # 15% moto, 70% car, 15% large/AV-like
        return np.where(r < 0.15, 1, np.where(r < 0.85, 2, 3)).astype(int)

    def _init_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.cfg
        n = self.n

        lane = self.rng.integers(0, cfg.num_lanes, size=n)
        pos = np.zeros(n, dtype=np.float64)

        # Spread vehicles along each lane on a straight road. This avoids the
        # Local_Y discontinuity of a ring road while keeping enough local pairs.
        for ln in range(cfg.num_lanes):
            idx = np.where(lane == ln)[0]
            if len(idx) == 0:
                continue

            idx = idx[self.rng.permutation(len(idx))]
            base_spacing = max(cfg.min_initial_spacing, cfg.road_length / max(len(idx), 1))
            start = self.rng.uniform(0.0, base_spacing)
            noise = self.rng.normal(0.0, 0.8, size=len(idx))
            lane_pos = start + np.arange(len(idx), dtype=np.float64) * base_spacing + noise
            lane_pos = np.maximum.accumulate(lane_pos)
            pos[idx] = lane_pos

        vel = self.rng.uniform(cfg.init_speed_min, cfg.init_speed_max, size=n)
        lat = lane.astype(np.float64) * 3.7
        return lane, pos, vel, lat

    def _same_lane_leader(self, i: int) -> Optional[int]:
        """Nearest vehicle ahead on the same lane on a straight road."""
        same = np.where(self.lane == self.lane[i])[0]
        same = same[same != i]
        if len(same) == 0:
            return None

        ahead = same[self.pos[same] > self.pos[i]]
        if len(ahead) == 0:
            return None

        gaps = self.pos[ahead] - self.pos[i]
        return int(ahead[np.argmin(gaps)])

    def _build_fixed_dependency(self) -> Dict[int, List[int]]:
        cfg = self.cfg
        dep: Dict[int, List[int]] = {i: [] for i in range(self.n)}

        if cfg.scenario in ("follow", "asymmetric", "heterogeneous", "mixed"):
            for i in range(self.n):
                leader = self._same_lane_leader(i)
                if leader is not None:
                    dep[i].append(leader)

        if cfg.scenario in ("confounded", "mixed"):
            used = set()
            triplets = 0
            for ln in range(cfg.num_lanes):
                members = np.where(self.lane == ln)[0]
                members = members[np.argsort(self.pos[members])]
                k = 0
                while k + 2 < len(members) and triplets < cfg.confounded_triplets:
                    # Sorted ascending by position: j back, i middle, L front.
                    j = int(members[k])
                    i = int(members[k + 1])
                    L = int(members[k + 2])

                    if L in used or i in used or j in used:
                        k += 1
                        continue

                    dep[i] = [L]
                    dep[j] = [L]

                    # These false edges are intentionally confounded: i and j
                    # share leader L but do not directly influence each other.
                    self.confounded_false_pairs_idx.add((i, j))
                    self.confounded_false_pairs_idx.add((j, i))

                    used.update([L, i, j])
                    triplets += 1
                    k += 3

        return dep

    def _gap_and_leadspeed(self, receiver_i: int, source_j: int) -> Tuple[float, float]:
        """Bumper gap from receiver i to source/leader j on a straight road."""
        p_i = IDM_PARAMS[int(self.vclass[receiver_i])]
        gap = self.pos[source_j] - self.pos[receiver_i] - p_i["length"]
        return float(gap), float(self.vel[source_j])

    def _accel_from_deps(self, i: int) -> Tuple[float, Dict[int, float]]:
        p = IDM_PARAMS[int(self.vclass[i])]
        deps = self.fixed_dependency[i]

        if len(deps) == 0:
            return idm_accel(self.vel[i], None, None, p), {}

        per_dep_acc = {}
        for j in deps:
            gap, v_lead = self._gap_and_leadspeed(i, j)
            per_dep_acc[j] = idm_accel(self.vel[i], gap, v_lead, p)

        # Dominant constraining dependency.
        a_full = min(per_dep_acc.values())

        strengths = {}
        if self.cfg.record_gt_strength:
            for j in deps:
                rest = [k for k in deps if k != j]
                if len(rest) == 0:
                    a_without = idm_accel(self.vel[i], None, None, p)
                else:
                    a_without = min(
                        idm_accel(
                            self.vel[i],
                            self._gap_and_leadspeed(i, k)[0],
                            self._gap_and_leadspeed(i, k)[1],
                            p,
                        )
                        for k in rest
                    )
                strengths[j] = abs(a_full - a_without)

        return float(a_full), strengths

    def step(self) -> Dict[int, Dict[int, float]]:
        cfg = self.cfg
        accels = np.zeros(self.n, dtype=np.float64)
        step_strength: Dict[int, Dict[int, float]] = {}

        for i in range(self.n):
            a, strengths = self._accel_from_deps(i)
            accels[i] = a
            if strengths:
                step_strength[i] = strengths

        accels = accels + self.rng.normal(0.0, cfg.accel_noise_std, size=self.n)
        accels = np.clip(accels, -8.0, 4.0)

        self.vel = np.clip(self.vel + accels * cfg.dt, 0.0, 40.0)
        self.pos = self.pos + self.vel * cfg.dt

        # Lateral coordinate remains lane-centered. Lane-changing is disabled by
        # default for clean GT. If enabled, it should be treated as a separate
        # robustness experiment.
        self.lat = self.lane.astype(np.float64) * 3.7

        self._last_accel = accels
        return step_strength

    def run(self) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]]:
        cfg = self.cfg
        rows = []
        vids = np.arange(1, self.n + 1, dtype=np.int64)

        strength_sum: Dict[Tuple[int, int], float] = {}
        strength_cnt: Dict[Tuple[int, int], int] = {}

        for step_idx in range(cfg.total_steps):
            step_strength = self.step()

            if step_idx < cfg.warmup_steps:
                continue

            frame_id = int(step_idx - cfg.warmup_steps)
            global_time = int(frame_id * round(cfg.dt * 1000.0))

            for rcv_i, src_map in step_strength.items():
                for src_j, val in src_map.items():
                    key = (int(vids[src_j]), int(vids[rcv_i]))
                    strength_sum[key] = strength_sum.get(key, 0.0) + float(val)
                    strength_cnt[key] = strength_cnt.get(key, 0) + 1

            for i in range(self.n):
                p = IDM_PARAMS[int(self.vclass[i])]
                x_obs_m = self.lat[i] + self.rng.normal(0.0, cfg.pos_obs_noise_std)
                y_obs_m = self.pos[i] + self.rng.normal(0.0, cfg.pos_obs_noise_std)

                rows.append(
                    {
                        "Vehicle_ID": int(vids[i]),
                        "Frame_ID": frame_id,
                        "Global_Time": global_time,
                        # NGSIM raw units are feet and ft/s. build_windows_ngsim
                        # will multiply by 0.3048 to recover meters.
                        "Local_X": float(x_obs_m * M_TO_FT),
                        "Local_Y": float(y_obs_m * M_TO_FT),
                        "v_Vel": float(self.vel[i] * M_TO_FT),
                        "v_Acc": float(self._last_accel[i] * M_TO_FT),
                        "Lane_ID": int(self.lane[i] + 1),
                        "v_Class": int(self.vclass[i]),
                        "v_Length": float(p["length"] * M_TO_FT),
                    }
                )

        df = pd.DataFrame(rows)

        gt_edges: Dict[Tuple[int, int], int] = {}
        for rcv_i, srcs in self.fixed_dependency.items():
            for src_j in srcs:
                gt_edges[(int(vids[src_j]), int(vids[rcv_i]))] = 1

        gt_strength = {}
        for key, val_sum in strength_sum.items():
            cnt = max(int(strength_cnt.get(key, 1)), 1)
            gt_strength[key] = float(val_sum / cnt)

        return df, gt_edges, gt_strength


def generate(cfg: SimConfig, out_csv: str, out_gt: str):
    sim = SyntheticTraffic(cfg)
    df, gt_edges, gt_strength = sim.run()

    out_csv_path = Path(out_csv)
    out_gt_path = Path(out_gt)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_gt_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv_path, index=False)

    vids = np.arange(1, sim.n + 1, dtype=np.int64)
    confounded_false_edges = {
        f"{int(vids[i])}->{int(vids[j])}": 1
        for (i, j) in sorted(sim.confounded_false_pairs_idx)
    }

    gt = {
        "scenario": cfg.scenario,
        "num_vehicles": cfg.num_vehicles,
        "dt": cfg.dt,
        "units_in_csv": "NGSIM_raw_feet_ftps_ftps2",
        "gt_edges": {f"{s}->{r}": 1 for (s, r) in sorted(gt_edges.keys())},
        "gt_strength": {f"{s}->{r}": float(v) for (s, r), v in sorted(gt_strength.items())},
        "confounded_false_edges": confounded_false_edges,
        "vclass": {int(i + 1): int(sim.vclass[i]) for i in range(sim.n)},
    }

    with open(out_gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"[synthetic] scenario={cfg.scenario}")
    print(f"[synthetic] rows={len(df)} vehicles={cfg.num_vehicles} frames={df['Frame_ID'].nunique()}")
    print(f"[synthetic] GT directed edges={len(gt_edges)}")
    print(f"[synthetic] confounded false directed pairs={len(confounded_false_edges)}")
    print(f"[synthetic] Local_Y range raw feet=({df['Local_Y'].min():.2f}, {df['Local_Y'].max():.2f})")
    print(f"[synthetic] saved csv -> {out_csv_path}")
    print(f"[synthetic] saved gt  -> {out_gt_path}")
    return df, gt_edges, gt_strength


# =========================
# PyCharm default config
# =========================
RUN_CFG = SimConfig(
    scenario="mixed",
    num_vehicles=60,
    total_steps=6000,
    warmup_steps=600,
    seed=0,
)
OUT_CSV = "./ProcessedDataset/SYNTH/synth_traj.csv"
OUT_GT = "./ProcessedDataset/SYNTH/synth_gt.json"


def main_with_default_config():
    generate(RUN_CFG, OUT_CSV, OUT_GT)


def main_with_argparse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=RUN_CFG.scenario)
    ap.add_argument("--num_vehicles", type=int, default=RUN_CFG.num_vehicles)
    ap.add_argument("--total_steps", type=int, default=RUN_CFG.total_steps)
    ap.add_argument("--warmup_steps", type=int, default=RUN_CFG.warmup_steps)
    ap.add_argument("--seed", type=int, default=RUN_CFG.seed)
    ap.add_argument("--out_csv", default=OUT_CSV)
    ap.add_argument("--out_gt", default=OUT_GT)
    args = ap.parse_args()

    cfg = SimConfig(
        scenario=args.scenario,
        num_vehicles=args.num_vehicles,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    generate(cfg, args.out_csv, args.out_gt)


if __name__ == "__main__":
    # For PyCharm, leave this as False and edit RUN_CFG above.
    USE_ARGPARSE = False
    if USE_ARGPARSE:
        main_with_argparse()
    else:
        main_with_default_config()
