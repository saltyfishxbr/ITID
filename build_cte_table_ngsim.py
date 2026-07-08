# build_cte_table_ngsim.py
import os
import multiprocessing as mp

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pickle

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import rankdata, norm


@dataclass
class CTEConfig:
    # window_path: str = "./ProcessedDataset/NGSIM/0750am-0805am/windows_ngsim_0750_0805.pkl"
    # output_path: str = "./ProcessedDataset/NGSIM/0750am-0805am/ngsim_0750_0805_cte_table.pkl"
    window_path = "./ProcessedDataset/SYNTH/synth_windows_compact.pkl"
    output_path = "./ProcessedDataset/SYNTH/synth_cte_table.pkl"

    # Only train split is used to estimate the CTE table.
    split_name: str = "train"

    # NGSIM is 10Hz. response_steps=5 means 0.5s response horizon.
    response_steps: int = 5

    # Context-level estimation needs enough samples.
    min_samples_per_bucket: int = 200#for synthetic, 200; for ngsim 500

    # Reservoir sampling cap per bucket.
    # KSG-style estimators are expensive, so this controls runtime.
    max_samples_per_bucket: int = 1500#for synthetic, 1500; for ngsim 2000

    num_workers: int=8

    # Frenzel-Pompe / KSG-style CMI estimator.
    k_neighbors: int = 5

    # Local conditional permutation in Z-space.
    # Larger k_perm gives more exchange candidates and faster mixing,
    # but too large weakens locality in Z.
    k_perm_neighbors: int = 20

    # Number of local conditional permutation surrogates.
    # 30 gives minimum possible p-value 1 / 31 ≈ 0.032.
    num_surrogates: int = 30
    significance_alpha: float = 0.05

    # If not significant under local conditional permutation, final cte is set to 0.
    zero_if_not_significant: bool = True

    # Preprocessing for kNN.
    # rank_gaussian is robust to non-Gaussian marginals.
    preprocess_mode: str = "rank_gaussian"  # "rank_gaussian" or "zscore"

    # Tiny noise to break exact ties after preprocessing.
    noise_std: float = 1e-10

    random_seed: int = 42
    print_every: int = 500


# Feature indices. Must match build_windows_ngsim.py
IDX_X = 0
IDX_Y = 1
IDX_VX = 2
IDX_VY = 3
IDX_SPEED = 4
IDX_ACC = 5


LANE_SAME_FRONT = 0
LANE_SAME_BACK = 1
LANE_ADJACENT = 2


# Context bins
# delta_y = source longitudinal position - target longitudinal position, unit: m
D_BINS = np.array([-np.inf, -40.0, -15.0, 15.0, 40.0, np.inf], dtype=np.float32)

# delta_speed = source speed - target speed, unit: m/s
DV_BINS = np.array([-np.inf, -5.0, -2.0, 2.0, 5.0, np.inf], dtype=np.float32)

# target-centered local density, unit: veh / m / lane
RHO_BINS = np.array([-np.inf, 0.025, 0.060, np.inf], dtype=np.float32)


def load_pickle(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find file: {p}")

    with open(p, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[save] saved CTE table to {p}")


def bin_index(value: float, bins: np.ndarray) -> int:
    idx = int(np.searchsorted(bins, value, side="right") - 1)
    idx = max(0, min(idx, len(bins) - 2))
    return idx


def make_context_keys(
    delta_y: float,
    delta_speed: float,
    lane_rel: int,
    rho: float,
) -> List[Tuple]:
    """
    Context key hierarchy.

    Fine:
        bin(delta_y), bin(delta_speed), lane_relation, bin(rho)

    Fallbacks:
        no_dv: remove relative speed
        lane_rho: lane relation and density only
        global: all train samples
    """
    d_bin = bin_index(delta_y, D_BINS)
    dv_bin = bin_index(delta_speed, DV_BINS)
    rho_bin = bin_index(rho, RHO_BINS)

    return [
        ("fine", d_bin, dv_bin, int(lane_rel), rho_bin),
        ("no_dv", d_bin, int(lane_rel), rho_bin),
        ("lane_rho", int(lane_rel), rho_bin),
        ("global",),
    ]


def extract_sequence(
    track_bank: Dict[int, Dict],
    vehicle_id: int,
    start_frame: int,
    end_frame: int,
) -> Optional[np.ndarray]:
    """
    Extract consecutive sequence [T, D] from compact track bank.

    start_frame and end_frame are inclusive.
    """
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


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

def source_history_feature(source_hist: np.ndarray, target_hist: np.ndarray) -> np.ndarray:
    """
    Reduced source history feature X_j^-.

    Important design choice:
        Do NOT include instantaneous rel_y[-1] or rel_speed[-1],
        because these are already used for context binning.

    This avoids injecting near-constant bucket variables as noisy kNN dimensions.

    Features:
        1. source current speed
        2. source current acceleration
        3. source current longitudinal velocity
        4. source current lateral velocity
        5. source speed change over history
        6. source acceleration change over history
        7. source longitudinal displacement over history
        8. source lateral displacement over history
        9. relative longitudinal distance change over history
        10. relative speed change over history
    """
    source_speed = source_hist[:, IDX_SPEED]
    source_acc = source_hist[:, IDX_ACC]

    rel_y = source_hist[:, IDX_Y] - target_hist[:, IDX_Y]
    rel_speed = source_hist[:, IDX_SPEED] - target_hist[:, IDX_SPEED]

    vec = np.array([
        source_hist[-1, IDX_SPEED],
        source_hist[-1, IDX_ACC],
        source_hist[-1, IDX_VY],
        source_hist[-1, IDX_VX],

        source_speed[-1] - source_speed[0],
        source_acc[-1] - source_acc[0],

        source_hist[-1, IDX_Y] - source_hist[0, IDX_Y],
        source_hist[-1, IDX_X] - source_hist[0, IDX_X],

        rel_y[-1] - rel_y[0],
        rel_speed[-1] - rel_speed[0],
    ], dtype=np.float32)

    return vec


def target_history_feature(target_hist: np.ndarray) -> np.ndarray:
    """
    Reduced target history condition feature X_i^-.

    Features:
        1. target current speed
        2. target current acceleration
        3. target current longitudinal velocity
        4. target current lateral velocity
        5. target speed change over history
        6. target acceleration change over history
        7. target longitudinal displacement over history
        8. target lateral displacement over history
    """
    speed = target_hist[:, IDX_SPEED]
    acc = target_hist[:, IDX_ACC]

    vec = np.array([
        target_hist[-1, IDX_SPEED],
        target_hist[-1, IDX_ACC],
        target_hist[-1, IDX_VY],
        target_hist[-1, IDX_VX],

        speed[-1] - speed[0],
        acc[-1] - acc[0],

        target_hist[-1, IDX_Y] - target_hist[0, IDX_Y],
        target_hist[-1, IDX_X] - target_hist[0, IDX_X],
    ], dtype=np.float32)

    return vec


def target_future_response(
    target_hist: np.ndarray,
    target_future: np.ndarray,
    response_steps: int,
) -> np.ndarray:
    """
    Y_i^+.

    Use a 2D summary of future velocity response instead of a 5D highly
    autocorrelated sequence.

    Response:
        1. terminal_delta_v = v(t+K) - v(t)
        2. mean_delta_v = mean_{k=1..K} [v(t+k) - v(t)]

    This reduces kNN dimensionality while preserving short-horizon response.
    """
    k = min(response_steps, target_future.shape[0])

    current_speed = float(target_hist[-1, IDX_SPEED])
    future_speed = target_future[:k, IDX_SPEED]
    delta_v_seq = future_speed - current_speed

    terminal_delta_v = float(delta_v_seq[-1])
    mean_delta_v = float(np.mean(delta_v_seq))

    return np.array([terminal_delta_v, mean_delta_v], dtype=np.float32)


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def zscore_transform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)

    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def rank_gaussian_transform(x: np.ndarray) -> np.ndarray:
    """
    Marginal rank-Gaussian transform.

    This stabilizes kNN estimation under non-Gaussian marginals and is
    approximately invariant to monotonic marginal transformations.
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape

    out = np.zeros_like(x, dtype=np.float64)

    for col in range(d):
        ranks = rankdata(x[:, col], method="average")
        u = (ranks - 0.5) / float(n)
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        out[:, col] = norm.ppf(u)

    return zscore_transform(out)


def preprocess_one_matrix(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if mode == "rank_gaussian":
        return rank_gaussian_transform(x)

    if mode == "zscore":
        return zscore_transform(x)

    raise ValueError(f"Unknown preprocess_mode: {mode}")


def preprocess_xyz_once(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    noise_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess Y, X, Z once per bucket.

    Local conditional permutation later uses the same preprocessed Z-space.
    This is important: surrogate generation and estimator geometry must match.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    valid = (
        np.isfinite(y).all(axis=1)
        & np.isfinite(x).all(axis=1)
        & np.isfinite(z).all(axis=1)
    )

    y = y[valid]
    x = x[valid]
    z = z[valid]

    y_pre = preprocess_one_matrix(y, mode=mode)
    x_pre = preprocess_one_matrix(x, mode=mode)
    z_pre = preprocess_one_matrix(z, mode=mode)

    if noise_std > 0:
        y_pre = y_pre + rng.normal(0.0, noise_std, size=y_pre.shape)
        x_pre = x_pre + rng.normal(0.0, noise_std, size=x_pre.shape)
        z_pre = z_pre + rng.normal(0.0, noise_std, size=z_pre.shape)

    return y_pre, x_pre, z_pre


# ---------------------------------------------------------------------
# kNN CMI estimator
# ---------------------------------------------------------------------

def query_tree(
    tree: cKDTree,
    points: np.ndarray,
    k: int,
    p: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compatibility wrapper for scipy versions with / without workers.
    """
    try:
        return tree.query(points, k=k, p=p, workers=1)
    except TypeError:
        return tree.query(points, k=k, p=p)


def count_neighbors_strict(
    tree: cKDTree,
    points: np.ndarray,
    radii: np.ndarray,
    p: float,
) -> np.ndarray:
    """
    Count neighbors within a radius strictly smaller than the kth-neighbor radius.

    The point itself is included by query_ball_point, so subtract 1.

    Uses vectorized return_length when available. Falls back to loop otherwise.
    """
    radii = np.asarray(radii, dtype=np.float64)
    r_strict = np.nextafter(radii, 0.0)
    r_strict = np.maximum(r_strict, 0.0)

    try:
        counts = tree.query_ball_point(
            points,
            r=r_strict,
            p=p,
            return_length=True,
        )
        counts = np.asarray(counts, dtype=np.int64)
        counts = np.maximum(0, counts - 1)
        return counts
    except TypeError:
        pass
    except Exception:
        pass

    counts = np.zeros((points.shape[0],), dtype=np.int64)

    for i in range(points.shape[0]):
        ids = tree.query_ball_point(points[i], r=float(r_strict[i]), p=p)
        counts[i] = max(0, len(ids) - 1)

    return counts


def frenzel_pompe_cmi_preprocessed(
    y_pre: np.ndarray,
    x_pre: np.ndarray,
    z_pre: np.ndarray,
    k: int,
) -> float:
    """
    Frenzel-Pompe / KSG-style conditional mutual information estimator.

    Estimate:
        I(Y ; X | Z)

    Inputs must already be preprocessed.

    Uses Chebyshev metric and strict projected-space neighbor counts.
    The returned value is NOT clipped to zero; finite-sample estimates can be
    slightly negative. Clipping is done after bias correction.
    """
    y_pre = np.asarray(y_pre, dtype=np.float64)
    x_pre = np.asarray(x_pre, dtype=np.float64)
    z_pre = np.asarray(z_pre, dtype=np.float64)

    n = y_pre.shape[0]
    if n <= k + 2:
        return 0.0

    xyz = np.concatenate([x_pre, y_pre, z_pre], axis=1)
    xz = np.concatenate([x_pre, z_pre], axis=1)
    yz = np.concatenate([y_pre, z_pre], axis=1)

    p = np.inf

    tree_xyz = cKDTree(xyz)
    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z_pre)

    # k+1 because the nearest neighbor is the point itself.
    distances, _ = query_tree(
        tree=tree_xyz,
        points=xyz,
        k=k + 1,
        p=p,
    )

    eps = distances[:, k]

    n_xz = count_neighbors_strict(tree_xz, xz, eps, p=p)
    n_yz = count_neighbors_strict(tree_yz, yz, eps, p=p)
    n_z = count_neighbors_strict(tree_z, z_pre, eps, p=p)

    estimate = digamma(k) + np.mean(
        digamma(n_z + 1)
        - digamma(n_xz + 1)
        - digamma(n_yz + 1)
    )

    if not np.isfinite(estimate):
        return 0.0

    return float(estimate)


# ---------------------------------------------------------------------
# Local conditional permutation surrogate
# ---------------------------------------------------------------------

# def conditional_permutation_in_z_space(
#     z_pre: np.ndarray,
#     rng: np.random.Generator,
#     k_perm: int,
# ) -> np.ndarray:
#     """
#     Local conditional permutation in preprocessed Z-space.
#
#     Goal:
#         Generate a surrogate X_perm that approximately preserves X-Z coupling
#         while breaking X-Y dependence conditional on Z.
#
#     For each sample i, choose a replacement index j from the nearest neighbors
#     of i in Z-space. A greedy matching is used to avoid overusing the same
#     replacement index.
#
#     If no unused local neighbor is available, we fall back to an unused index,
#     preferring self only when necessary. This makes the test conservative in
#     very small or dense-tie buckets.
#     """
#     z_pre = np.asarray(z_pre, dtype=np.float64)
#     n = z_pre.shape[0]
#
#     if n <= 1:
#         return np.arange(n, dtype=np.int64)
#
#     k_eff = min(int(k_perm), n - 1)
#     if k_eff <= 0:
#         return np.arange(n, dtype=np.int64)
#
#     tree = cKDTree(z_pre)
#
#     distances, nbrs = query_tree(
#         tree=tree,
#         points=z_pre,
#         k=k_eff + 1,
#         p=np.inf,
#     )
#
#     if nbrs.ndim == 1:
#         nbrs = nbrs.reshape(n, 1)
#
#     # Remove self neighbor.
#     nbrs = nbrs[:, 1:]
#
#     perm = np.arange(n, dtype=np.int64)
#     used = np.zeros(n, dtype=bool)
#
#     order = rng.permutation(n)
#
#     for i in order:
#         local_candidates = nbrs[i]
#         local_candidates = local_candidates[~used[local_candidates]]
#
#         if len(local_candidates) > 0:
#             j = int(rng.choice(local_candidates))
#         else:
#             unused = np.where(~used)[0]
#
#             if len(unused) == 0:
#                 j = int(i)
#             elif not used[i]:
#                 # Prefer self only if no local unused neighbor exists.
#                 # This is conservative.
#                 j = int(i)
#             else:
#                 j = int(rng.choice(unused))
#
#         perm[i] = j
#         used[j] = True
#
#     return perm

def precompute_z_neighbors(z_pre: np.ndarray, k_perm: int) -> Optional[np.ndarray]:
    """桶级预计算 Z 空间近邻。跨所有 surrogate 不变，只算一次。"""
    n = z_pre.shape[0]
    if n <= 1:
        return None
    k_eff = min(int(k_perm), n - 1)
    if k_eff <= 0:
        return None

    tree = cKDTree(z_pre)
    _, nbrs = query_tree(tree=tree, points=z_pre, k=k_eff + 1, p=np.inf)
    if nbrs.ndim == 1:
        nbrs = nbrs.reshape(n, 1)
    return nbrs[:, 1:]   # 去掉自身


def conditional_permutation_from_nbrs(
    nbrs: Optional[np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """用预算好的 Z 近邻做局部条件排列（贪心匹配）。"""
    if nbrs is None:
        return np.arange(n, dtype=np.int64)

    perm = np.arange(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)

    for i in rng.permutation(n):
        local = nbrs[i]
        local = local[~used[local]]

        if len(local) > 0:
            j = int(rng.choice(local))
        else:
            unused = np.where(~used)[0]
            if len(unused) == 0:
                j = int(i)
            elif not used[i]:
                j = int(i)            # 保守：保留自身
            else:
                j = int(rng.choice(unused))

        perm[i] = j
        used[j] = True

    return perm


def estimate_cmi_with_local_surrogates(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cfg: CTEConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Estimate raw CMI and local conditional-permutation surrogate null.

    Raw:
        I(Y ; X | Z)

    Null:
        locally permute X within preprocessed Z-neighborhoods.
        This targets X ⟂ Y | Z, not X ⟂ (Y, Z).

    Bias correction:
        cte_debiased = max(0, raw - mean(null))

    Significance:
        p = (1 + #{null >= raw}) / (B + 1)
    """
    y_pre, x_pre, z_pre = preprocess_xyz_once(
        y=y,
        x=x,
        z=z,
        mode=cfg.preprocess_mode,
        rng=rng,
        noise_std=cfg.noise_std,
    )

    n = y_pre.shape[0]

    if n < cfg.min_samples_per_bucket:
        return {
            "cte": 0.0,
            "cmi_raw_estimate": 0.0,
            "cte_raw": 0.0,
            "cte_null_mean": 0.0,
            "cte_null_std": 0.0,
            "cte_debiased": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "num_surrogates": int(cfg.num_surrogates),
            "num_valid_after_preprocess": int(n),
        }

    raw = frenzel_pompe_cmi_preprocessed(
        y_pre=y_pre,
        x_pre=x_pre,
        z_pre=z_pre,
        k=cfg.k_neighbors,
    )

    # Z 近邻在所有 surrogate 间不变，桶级算一次。
    z_nbrs = precompute_z_neighbors(z_pre, cfg.k_perm_neighbors)

    null_values = []
    for _ in range(cfg.num_surrogates):
        perm = conditional_permutation_from_nbrs(z_nbrs, n, rng)
        x_perm = x_pre[perm]

        null_cmi = frenzel_pompe_cmi_preprocessed(
            y_pre=y_pre,
            x_pre=x_perm,
            z_pre=z_pre,
            k=cfg.k_neighbors,
        )
        null_values.append(float(null_cmi))

    # null_values = []
    #
    # for _ in range(cfg.num_surrogates):
    #     perm = conditional_permutation_in_z_space(
    #         z_pre=z_pre,
    #         rng=rng,
    #         k_perm=cfg.k_perm_neighbors,
    #     )
    #
    #     x_perm = x_pre[perm]
    #
    #     null_cmi = frenzel_pompe_cmi_preprocessed(
    #         y_pre=y_pre,
    #         x_pre=x_perm,
    #         z_pre=z_pre,
    #         k=cfg.k_neighbors,
    #     )
    #
    #     null_values.append(float(null_cmi))

    null_values = np.asarray(null_values, dtype=np.float64)

    null_mean = float(np.mean(null_values)) if len(null_values) > 0 else 0.0
    null_std = float(np.std(null_values)) if len(null_values) > 0 else 0.0

    # Keep raw estimate for p-value. Clip only for CTE value.
    cte_raw = max(0.0, float(raw))
    cte_debiased = max(0.0, float(raw - null_mean))

    if len(null_values) > 0:
        p_value = float(
            (1.0 + np.sum(null_values >= raw))
            / (float(len(null_values)) + 1.0)
        )
    else:
        p_value = 1.0

    is_significant = bool(p_value <= cfg.significance_alpha)

    if cfg.zero_if_not_significant and not is_significant:
        cte_final = 0.0
    else:
        cte_final = cte_debiased

    return {
        "cte": float(cte_final),
        "cmi_raw_estimate": float(raw),
        "cte_raw": float(cte_raw),
        "cte_null_mean": float(null_mean),
        "cte_null_std": float(null_std),
        "cte_debiased": float(cte_debiased),
        "p_value": float(p_value),
        "is_significant": bool(is_significant),
        "num_surrogates": int(cfg.num_surrogates),
        "num_valid_after_preprocess": int(n),
    }


# ---------------------------------------------------------------------
# Bucket collection
# ---------------------------------------------------------------------

def reservoir_add(
    bucket: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> None:
    """
    Reservoir sampling.

    Each bucket stores at most max_samples samples,
    while count records the true number of samples seen.
    """
    bucket["count"] += 1
    count = int(bucket["count"])

    if len(bucket["X"]) < max_samples:
        bucket["X"].append(x)
        bucket["Y"].append(y)
        bucket["Z"].append(z)
        return

    replace_idx = int(rng.integers(low=0, high=count))
    if replace_idx < max_samples:
        bucket["X"][replace_idx] = x
        bucket["Y"][replace_idx] = y
        bucket["Z"][replace_idx] = z


def collect_train_buckets(payload: Dict, cfg: CTEConfig) -> Dict[Tuple, Dict[str, Any]]:
    """
    Use train split only to collect context-level CTE samples.

    For each directed raw perception pair j -> i:
        CTE^c = I(Y_i^+ ; X_j^- | X_i^-, C=c)
    """
    rng = np.random.default_rng(cfg.random_seed)

    track_bank = payload["track_bank"]
    windows = payload["windows"]
    split_indices = payload["splits"][cfg.split_name]

    buckets: Dict[Tuple, Dict[str, Any]] = {}

    for count_idx, window_idx in enumerate(split_indices):
        w = windows[window_idx]

        target_id = int(w["target_id"])
        hist_start = int(w["hist_start_frame"])
        hist_end = int(w["hist_end_frame"])
        fut_start = int(w["future_start_frame"])
        fut_end = int(w["future_end_frame"])

        target_hist = extract_sequence(track_bank, target_id, hist_start, hist_end)
        target_future = extract_sequence(track_bank, target_id, fut_start, fut_end)

        if target_hist is None or target_future is None:
            continue

        z_target = target_history_feature(target_hist)
        y_response = target_future_response(
            target_hist=target_hist,
            target_future=target_future,
            response_steps=cfg.response_steps,
        )

        neighbor_ids = w["neighbor_ids"]
        neighbor_delta_y = w["neighbor_delta_y"]
        neighbor_delta_speed = w["neighbor_delta_speed"]
        neighbor_lane_relation = w["neighbor_lane_relation"]
        rho = float(w["density0"])

        for n_idx, source_id in enumerate(neighbor_ids):
            source_id = int(source_id)

            source_hist = extract_sequence(track_bank, source_id, hist_start, hist_end)
            if source_hist is None:
                continue

            delta_y = float(neighbor_delta_y[n_idx])
            delta_speed = float(neighbor_delta_speed[n_idx])
            lane_rel = int(neighbor_lane_relation[n_idx])

            keys = make_context_keys(
                delta_y=delta_y,
                delta_speed=delta_speed,
                lane_rel=lane_rel,
                rho=rho,
            )

            x_source = source_history_feature(source_hist, target_hist)

            for key in keys:
                if key not in buckets:
                    buckets[key] = {
                        "X": [],
                        "Y": [],
                        "Z": [],
                        "count": 0,
                    }

                reservoir_add(
                    bucket=buckets[key],
                    x=x_source,
                    y=y_response,
                    z=z_target,
                    max_samples=cfg.max_samples_per_bucket,
                    rng=rng,
                )

        if cfg.print_every > 0 and (count_idx + 1) % cfg.print_every == 0:
            print(
                f"[collect] processed {count_idx + 1}/{len(split_indices)} "
                f"{cfg.split_name} windows, buckets={len(buckets)}"
            )

    return buckets


# ---------------------------------------------------------------------
# CTE table estimation
# ---------------------------------------------------------------------

# def estimate_cte_table(
#     buckets: Dict[Tuple, Dict[str, Any]],
#     cfg: CTEConfig,
# ) -> Dict[Tuple, Dict[str, Any]]:
#     rng = np.random.default_rng(cfg.random_seed + 999)
#
#     table: Dict[Tuple, Dict[str, Any]] = {}
#
#     keys = sorted(buckets.keys(), key=lambda k: str(k))
#
#     for idx, key in enumerate(keys):
#         bucket = buckets[key]
#         true_count = int(bucket["count"])
#         used_count = int(len(bucket["X"]))
#
#         if true_count < cfg.min_samples_per_bucket:
#             continue
#
#         if used_count < cfg.min_samples_per_bucket:
#             continue
#
#         x = np.stack(bucket["X"], axis=0)
#         y = np.stack(bucket["Y"], axis=0)
#         z = np.stack(bucket["Z"], axis=0)
#
#         estimate = estimate_cmi_with_local_surrogates(
#             y=y,
#             x=x,
#             z=z,
#             cfg=cfg,
#             rng=rng,
#         )
#
#         table[key] = {
#             **estimate,
#             "num_samples_total": true_count,
#             "num_samples_used": used_count,
#             "x_dim": int(x.shape[1]),
#             "y_dim": int(y.shape[1]),
#             "z_dim": int(z.shape[1]),
#         }
#
#         if cfg.print_every > 0 and (idx + 1) % cfg.print_every == 0:
#             print(
#                 f"[estimate] keys={idx + 1}/{len(keys)}, "
#                 f"valid_table={len(table)}"
#             )
#
#     return table
def _estimate_one_bucket(item):
    """多进程 worker。必须是模块级、可 pickle。"""
    key, x, y, z, cfg, child_seed, true_count, used_count = item
    rng = np.random.default_rng(child_seed)
    estimate = estimate_cmi_with_local_surrogates(y=y, x=x, z=z, cfg=cfg, rng=rng)
    return (key, estimate, true_count, used_count,
            int(x.shape[1]), int(y.shape[1]), int(z.shape[1]))


def estimate_cte_table(buckets, cfg):
    keys = sorted(buckets.keys(), key=lambda k: str(k))

    # 每个桶一条独立、可复现的随机流。
    child_seeds = np.random.SeedSequence(cfg.random_seed + 999).spawn(len(keys))

    work = []
    for key, child in zip(keys, child_seeds):
        bucket = buckets[key]
        true_count = int(bucket["count"])
        used_count = int(len(bucket["X"]))
        if true_count < cfg.min_samples_per_bucket:
            continue
        if used_count < cfg.min_samples_per_bucket:
            continue
        x = np.stack(bucket["X"], axis=0)
        y = np.stack(bucket["Y"], axis=0)
        z = np.stack(bucket["Z"], axis=0)
        work.append((key, x, y, z, cfg, child, true_count, used_count))

    print(f"[estimate] buckets to estimate: {len(work)}  workers={cfg.num_workers}")

    table = {}

    def _store(result):
        key, estimate, tc, uc, xd, yd, zd = result
        table[key] = {**estimate, "num_samples_total": tc, "num_samples_used": uc,
                      "x_dim": xd, "y_dim": yd, "z_dim": zd}

    if cfg.num_workers and cfg.num_workers > 1:
        with mp.Pool(processes=cfg.num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_estimate_one_bucket, work, chunksize=1)):
                _store(result)
                if (i + 1) % 20 == 0:
                    print(f"[estimate] done {i + 1}/{len(work)}")
    else:
        for i, item in enumerate(work):
            _store(_estimate_one_bucket(item))
            if (i + 1) % 20 == 0:
                print(f"[estimate] done {i + 1}/{len(work)}")

    return table


def lookup_cte(
    cte_package: Dict,
    delta_y: float,
    delta_speed: float,
    lane_rel: int,
    rho: float,
) -> float:
    """
    Later train / val / test can use this function to query train CTE table.

    Important:
        val/test do not estimate CTE table.
        They only use the train-estimated table.
    """
    table = cte_package["table"]

    keys = make_context_keys(
        delta_y=delta_y,
        delta_speed=delta_speed,
        lane_rel=lane_rel,
        rho=rho,
    )

    for key in keys:
        if key in table:
            return float(table[key]["cte"])

    return 0.0


def summarize_table(table: Dict[Tuple, Dict[str, Any]]) -> None:
    if len(table) == 0:
        print("[summary] empty CTE table")
        return

    values = np.asarray([v["cte"] for v in table.values()], dtype=np.float64)
    raw_values = np.asarray([v["cte_raw"] for v in table.values()], dtype=np.float64)
    raw_estimates = np.asarray([v["cmi_raw_estimate"] for v in table.values()], dtype=np.float64)
    debiased_values = np.asarray([v["cte_debiased"] for v in table.values()], dtype=np.float64)
    null_means = np.asarray([v["cte_null_mean"] for v in table.values()], dtype=np.float64)
    p_values = np.asarray([v["p_value"] for v in table.values()], dtype=np.float64)

    total_counts = np.asarray([v["num_samples_total"] for v in table.values()], dtype=np.int64)
    used_counts = np.asarray([v["num_samples_used"] for v in table.values()], dtype=np.int64)
    significant = np.asarray([v["is_significant"] for v in table.values()], dtype=np.bool_)

    print("[summary] valid keys:", len(table))

    print("[summary] final CTE mean:", float(values.mean()))
    print("[summary] final CTE std:", float(values.std()))
    print("[summary] final CTE min:", float(values.min()))
    print("[summary] final CTE max:", float(values.max()))

    print("[summary] raw clipped CTE mean:", float(raw_values.mean()))
    print("[summary] raw estimator mean:", float(raw_estimates.mean()))
    print("[summary] local null mean:", float(null_means.mean()))
    print("[summary] debiased CTE mean:", float(debiased_values.mean()))

    print("[summary] p-value mean:", float(p_values.mean()))
    print("[summary] p-value min:", float(p_values.min()))
    print("[summary] p-value max:", float(p_values.max()))

    print("[summary] significant keys:", int(significant.sum()))
    print("[summary] nonzero final CTE keys:", int(np.sum(values > 0.0)))

    print("[summary] total samples per key mean:", float(total_counts.mean()))
    print("[summary] total samples per key min:", int(total_counts.min()))
    print("[summary] total samples per key max:", int(total_counts.max()))

    print("[summary] used samples per key mean:", float(used_counts.mean()))
    print("[summary] used samples per key min:", int(used_counts.min()))
    print("[summary] used samples per key max:", int(used_counts.max()))


def main() -> None:
    cfg = CTEConfig()

    payload = load_pickle(cfg.window_path)

    print("[load] windows:", len(payload["windows"]))
    print("[load] train:", len(payload["splits"]["train"]))
    print("[load] val:", len(payload["splits"]["val"]))
    print("[load] test:", len(payload["splits"]["test"]))
    print("[load] feature_columns:", payload["feature_columns"])

    buckets = collect_train_buckets(payload, cfg)
    print("[collect] buckets:", len(buckets))

    table = estimate_cte_table(buckets, cfg)
    summarize_table(table)

    cte_package = {
        "config": asdict(cfg),
        "definition": "CTE^c = I(Y_i^+ ; X_j^- | X_i^-, C=c)",
        "estimator": {
            "name": "frenzel_pompe_ksg_conditional_mutual_information",
            "k_neighbors": cfg.k_neighbors,
            "metric": "chebyshev",
            "preprocess_mode": cfg.preprocess_mode,
            "surrogate": "local_conditional_permutation_in_preprocessed_Z_space",
            "k_perm_neighbors": cfg.k_perm_neighbors,
            "num_surrogates": cfg.num_surrogates,
            "bias_correction": "cte_debiased = max(0, cmi_raw_estimate - mean(local_conditional_null))",
            "significance_test": "p = (1 + #{null >= raw}) / (B + 1)",
            "alpha": cfg.significance_alpha,
            "zero_if_not_significant": cfg.zero_if_not_significant,
        },
        "context": {
            "fine": "bin(delta_y) x bin(delta_speed) x lane_relation x bin(rho)",
            "fallback": [
                "no_dv: bin(delta_y) x lane_relation x bin(rho)",
                "lane_rho: lane_relation x bin(rho)",
                "global: all train samples",
            ],
            "D_BINS": D_BINS.tolist(),
            "DV_BINS": DV_BINS.tolist(),
            "RHO_BINS": RHO_BINS.tolist(),
            "lane_relation_codes": {
                "same_front": LANE_SAME_FRONT,
                "same_back": LANE_SAME_BACK,
                "adjacent": LANE_ADJACENT,
            },
        },
        "features": {
            "source_feature_dim": 10,
            "target_condition_dim": 8,
            "response_dim": 2,
            "source_feature_description": [
                "source current speed",
                "source current acceleration",
                "source current longitudinal velocity",
                "source current lateral velocity",
                "source speed change over history",
                "source acceleration change over history",
                "source longitudinal displacement over history",
                "source lateral displacement over history",
                "relative longitudinal distance change over history",
                "relative speed change over history",
            ],
            "target_condition_description": [
                "target current speed",
                "target current acceleration",
                "target current longitudinal velocity",
                "target current lateral velocity",
                "target speed change over history",
                "target acceleration change over history",
                "target longitudinal displacement over history",
                "target lateral displacement over history",
            ],
            "response_description": [
                "terminal_delta_v = v_i(t+K) - v_i(t)",
                "mean_delta_v = mean_k[v_i(t+k) - v_i(t)]",
            ],
        },
        "table": table,
    }

    save_pickle(cte_package, cfg.output_path)


if __name__ == "__main__":
    main()