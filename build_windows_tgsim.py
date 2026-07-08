# build_windows_tgsim.py

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class WindowConfig:
    # Edit these for L1 / L2.
    subset_name: str = "L2"  # "L1" or "L2"
    raw_csv_path: str = "./data/RawDataset/TGSIM/I-294/I-294 L2 Trajectory/run_1.csv"
    output_path: str = "./ProcessedDataset/TGSIM/L2/windows_tgsim_L2.pkl"

    fps: int = 10
    history_seconds: float = 3.0
    future_seconds: float = 5.0
    center_frame_stride: int = 1

    perception_front_m: float = 120.0
    perception_back_m: float = 80.0
    perception_lane_span: int = 1

    density_front_m: float = 120.0
    density_back_m: float = 80.0
    density_lane_span: int = 1

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_buffer_frames: Optional[int] = None

    # TGSIM columns are already in metric units.
    max_windows: Optional[int] = None
    print_every: int = 20000

    # If one raw file contains repeated vehicle ids across runs, set this True.
    # The user said run_index is not used; by default we keep original id.
    use_run_index_in_vehicle_id: bool = False
    vehicle_id_offset: int = 1_000_000


FEATURE_COLUMNS = [
    "x_m",
    "y_m",
    "vx_mps",
    "vy_mps",
    "speed_mps",
    "acc_mps2",
]

IDX_X = 0
IDX_Y = 1
IDX_VX = 2
IDX_VY = 3
IDX_SPEED = 4
IDX_ACC = 5

LANE_SAME_FRONT = 0
LANE_SAME_BACK = 1
LANE_ADJACENT = 2


def normalize_acc_value(x) -> str:
    s = str(x).strip().lower()
    if s in {"yes", "y", "true", "1", "acc", "cav", "av"}:
        return "Yes"
    return "No"


def automation_label_from_acc(acc_value, subset_name: str) -> str:
    return str(subset_name).upper() if normalize_acc_value(acc_value) == "Yes" else "HV"


def automation_class_from_acc(acc_value) -> int:
    return 1 if normalize_acc_value(acc_value) == "Yes" else 0


def size_type_index(size_value) -> int:
    s = str(size_value).strip().lower()
    if "small" in s:
        return 0
    if "large" in s or "truck" in s:
        return 1
    return 2


def read_and_preprocess_tgsim(cfg: WindowConfig) -> pd.DataFrame:
    csv_path = Path(cfg.raw_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} does not exist")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "id",
        "time",
        "xloc_kf",
        "yloc_kf",
        "lane_kf",
        "speed_kf",
        "acceleration_kf",
        "type_most_common",
        "acc",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}")

    if cfg.use_run_index_in_vehicle_id and "run_index" in df.columns:
        df["Vehicle_ID"] = (
            df["run_index"].astype(int) * int(cfg.vehicle_id_offset)
            + df["id"].astype(int)
        )
    else:
        df["Vehicle_ID"] = df["id"].astype(int)

    # Use integer frame id at 10 Hz.
    df["Frame_ID"] = np.rint(df["time"].astype(float) * float(cfg.fps)).astype(np.int64)
    df["Global_Time"] = df["Frame_ID"].astype(np.int64)

    df["x_m"] = df["xloc_kf"].astype(float)
    df["y_m"] = df["yloc_kf"].astype(float)
    df["speed_mps"] = df["speed_kf"].astype(float)
    df["acc_mps2"] = df["acceleration_kf"].astype(float)
    df["Lane_ID"] = df["lane_kf"].astype(int)

    df["acc_norm"] = df["acc"].apply(normalize_acc_value)
    df["automation_label"] = df["acc_norm"].apply(
        lambda x: automation_label_from_acc(x, cfg.subset_name)
    )
    df["automation_class"] = df["acc_norm"].apply(automation_class_from_acc).astype(int)

    df["size_type"] = df["type_most_common"].astype(str)
    df["size_type_index"] = df["type_most_common"].apply(size_type_index).astype(int)

    # Sort before finite differences.
    df = df.sort_values(["Vehicle_ID", "Frame_ID"]).reset_index(drop=True)

    parts = []
    for _, g in df.groupby("Vehicle_ID", sort=False):
        g = g.sort_values("Frame_ID").copy()

        dt = g["Frame_ID"].diff().astype(float) / float(cfg.fps)
        vx = g["x_m"].diff() / dt
        vy = g["y_m"].diff() / dt

        vx = vx.replace([np.inf, -np.inf], np.nan)
        vy = vy.replace([np.inf, -np.inf], np.nan)

        g["vx_mps"] = vx.bfill().ffill().fillna(0.0)
        g["vy_mps"] = vy.bfill().ffill().fillna(g["speed_mps"])

        parts.append(g)

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["Frame_ID", "Vehicle_ID"]).reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
    df["Frame_ID"] = df["Frame_ID"].astype(int)
    df["Global_Time"] = df["Global_Time"].astype(np.int64)
    df["Lane_ID"] = df["Lane_ID"].astype(int)

    return df


def build_track_bank(df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Compact storage.
    For each vehicle:
        frames: [T]
        data: [T,D] with FEATURE_COLUMNS
        lane: [T]
        global_time: [T]
        automation_class: 0 HV, 1 ACC vehicle in this subset
        automation_label: HV/L1 or HV/L2
        size_type: small_vehicle / large_vehicle
    """
    track_bank: Dict[int, Dict] = {}

    for vid, g in df.groupby("Vehicle_ID", sort=False):
        g = g.sort_values("Frame_ID").copy()

        frames = g["Frame_ID"].to_numpy(dtype=np.int64)
        data = g[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        lane = g["Lane_ID"].to_numpy(dtype=np.int64)
        global_time = g["Global_Time"].to_numpy(dtype=np.int64)

        automation_class = int(g["automation_class"].mode().iloc[0])
        automation_label = str(g["automation_label"].mode().iloc[0])
        acc_raw = str(g["acc_norm"].mode().iloc[0])
        size_type = str(g["size_type"].mode().iloc[0])
        size_idx = int(g["size_type_index"].mode().iloc[0])

        track_bank[int(vid)] = {
            "frames": frames,
            "data": data,
            "lane": lane,
            "global_time": global_time,
            "automation_class": automation_class,
            "automation_label": automation_label,
            "acc": acc_raw,
            "size_type": size_type,
            "size_type_index": size_idx,

            # Compatibility aliases.
            "v_class": automation_class + 1,
        }

    return track_bank


def build_frame_index(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    frame_index: Dict[int, pd.DataFrame] = {}
    for frame_id, g in df.groupby("Frame_ID", sort=False):
        frame_index[int(frame_id)] = g.copy()
    return frame_index


def get_row_from_track(track: Dict, frame: int) -> Optional[Tuple[np.ndarray, int, int]]:
    frames = track["frames"]
    idx = np.searchsorted(frames, frame)

    if idx >= len(frames) or int(frames[idx]) != int(frame):
        return None

    data = track["data"][idx]
    lane = int(track["lane"][idx])
    global_time = int(track["global_time"][idx])
    return data, lane, global_time


def has_full_window(track: Dict, hist_frames: np.ndarray, fut_frames: np.ndarray) -> bool:
    frames = track["frames"]

    needed = np.concatenate([hist_frames, fut_frames])
    idx = np.searchsorted(frames, needed)

    if np.any(idx >= len(frames)):
        return False

    return bool(np.all(frames[idx] == needed))


def lane_relation(source_lane: int, target_lane: int, delta_y: float) -> int:
    lane_diff = int(source_lane) - int(target_lane)

    if lane_diff == 0:
        if delta_y >= 0:
            return LANE_SAME_FRONT
        return LANE_SAME_BACK

    if abs(lane_diff) == 1:
        return LANE_ADJACENT

    return -1


def select_raw_perception_neighbors(
    frame_df: pd.DataFrame,
    target_row: pd.Series,
    cfg: WindowConfig,
) -> pd.DataFrame:
    target_vid = int(target_row["Vehicle_ID"])
    y0 = float(target_row["y_m"])
    lane0 = int(target_row["Lane_ID"])

    candidates = frame_df[frame_df["Vehicle_ID"].astype(int) != target_vid].copy()
    if len(candidates) == 0:
        return candidates

    candidates["delta_y"] = candidates["y_m"].astype(float) - y0
    candidates["delta_speed"] = candidates["speed_mps"].astype(float) - float(target_row["speed_mps"])
    candidates["lane_diff"] = candidates["Lane_ID"].astype(int) - lane0

    raw_neighbors = candidates[
        (candidates["delta_y"] >= -cfg.perception_back_m)
        & (candidates["delta_y"] <= cfg.perception_front_m)
        & (candidates["lane_diff"].abs() <= cfg.perception_lane_span)
    ].copy()

    return raw_neighbors


def compute_target_centered_density(
    frame_df: pd.DataFrame,
    target_row: pd.Series,
    cfg: WindowConfig,
) -> float:
    """
    Unit: vehicles/meter/lane
    """
    y0 = float(target_row["y_m"])
    lane0 = int(target_row["Lane_ID"])

    lane_diff = frame_df["Lane_ID"].astype(int) - lane0
    delta_y = frame_df["y_m"].astype(float) - y0

    local = frame_df[
        (delta_y >= -cfg.density_back_m)
        & (delta_y <= cfg.density_front_m)
        & (lane_diff.abs() <= cfg.density_lane_span)
    ]

    road_length = cfg.density_front_m + cfg.density_back_m
    lane_count = 2 * cfg.density_lane_span + 1

    rho = float(len(local)) / float(road_length * lane_count)
    return rho


def make_window(
    target_id: int,
    center_frame: int,
    target_track: Dict,
    target_row: pd.Series,
    raw_neighbors: pd.DataFrame,
    track_bank: Dict[int, Dict],
    hist_frames: np.ndarray,
    fut_frames: np.ndarray,
    density0: float,
) -> Optional[Dict]:
    target_current_data, target_lane, target_global_time = get_row_from_track(
        target_track,
        center_frame,
    )

    if target_current_data is None:
        return None

    target_y = float(target_current_data[IDX_Y])
    target_x = float(target_current_data[IDX_X])
    target_speed = float(target_current_data[IDX_SPEED])

    neighbor_ids: List[int] = []
    neighbor_types: List[int] = []
    neighbor_size_types: List[int] = []
    neighbor_lanes: List[int] = []
    neighbor_delta_x: List[float] = []
    neighbor_delta_y: List[float] = []
    neighbor_delta_speed: List[float] = []
    neighbor_lane_diff: List[int] = []
    neighbor_lane_relation: List[int] = []

    for _, nb_row in raw_neighbors.iterrows():
        nb_id = int(nb_row["Vehicle_ID"])
        if nb_id not in track_bank:
            continue

        nb_track = track_bank[nb_id]

        if not has_full_window(nb_track, hist_frames, fut_frames):
            continue

        nb_current = get_row_from_track(nb_track, center_frame)
        if nb_current is None:
            continue

        nb_data, nb_lane, _ = nb_current

        dx = float(nb_data[IDX_X]) - target_x
        dy = float(nb_data[IDX_Y]) - target_y
        dspeed = float(nb_data[IDX_SPEED]) - target_speed
        lane_diff = int(nb_lane) - int(target_lane)
        lane_rel = lane_relation(nb_lane, target_lane, dy)

        if lane_rel < 0:
            continue

        neighbor_ids.append(nb_id)
        neighbor_types.append(int(nb_track["automation_class"]))
        neighbor_size_types.append(int(nb_track["size_type_index"]))
        neighbor_lanes.append(int(nb_lane))
        neighbor_delta_x.append(dx)
        neighbor_delta_y.append(dy)
        neighbor_delta_speed.append(dspeed)
        neighbor_lane_diff.append(lane_diff)
        neighbor_lane_relation.append(lane_rel)

    window = {
        "target_id": int(target_id),
        "center_frame": int(center_frame),
        "global_time": int(target_global_time),

        "hist_start_frame": int(hist_frames[0]),
        "hist_end_frame": int(hist_frames[-1]),
        "future_start_frame": int(fut_frames[0]),
        "future_end_frame": int(fut_frames[-1]),

        "target_type": int(target_track["automation_class"]),
        "target_size_type": int(target_track["size_type_index"]),
        "target_lane": int(target_lane),

        "density0": np.float32(density0),

        "neighbor_ids": np.asarray(neighbor_ids, dtype=np.int64),
        "neighbor_types": np.asarray(neighbor_types, dtype=np.int64),
        "neighbor_size_types": np.asarray(neighbor_size_types, dtype=np.int64),
        "neighbor_lanes": np.asarray(neighbor_lanes, dtype=np.int64),

        "neighbor_delta_x": np.asarray(neighbor_delta_x, dtype=np.float32),
        "neighbor_delta_y": np.asarray(neighbor_delta_y, dtype=np.float32),
        "neighbor_delta_speed": np.asarray(neighbor_delta_speed, dtype=np.float32),
        "neighbor_lane_diff": np.asarray(neighbor_lane_diff, dtype=np.int64),
        "neighbor_lane_relation": np.asarray(neighbor_lane_relation, dtype=np.int64),
    }

    return window


def build_windows(cfg: WindowConfig) -> Dict:
    df = read_and_preprocess_tgsim(cfg)
    track_bank = build_track_bank(df)
    frame_index = build_frame_index(df)

    history_len = int(round(cfg.history_seconds * cfg.fps))
    future_len = int(round(cfg.future_seconds * cfg.fps))

    windows: List[Dict] = []

    for target_id, target_track in track_bank.items():
        frames = target_track["frames"]

        if len(frames) < history_len + future_len:
            continue

        min_center = int(frames.min()) + history_len - 1
        max_center = int(frames.max()) - future_len

        for center_frame in range(min_center, max_center + 1, cfg.center_frame_stride):
            if center_frame not in frame_index:
                continue

            hist_frames = np.arange(
                center_frame - history_len + 1,
                center_frame + 1,
                dtype=np.int64,
            )
            fut_frames = np.arange(
                center_frame + 1,
                center_frame + future_len + 1,
                dtype=np.int64,
            )

            if not has_full_window(target_track, hist_frames, fut_frames):
                continue

            current = get_row_from_track(target_track, center_frame)
            if current is None:
                continue

            frame_df = frame_index[center_frame]

            target_rows = frame_df[frame_df["Vehicle_ID"].astype(int) == int(target_id)]
            if len(target_rows) == 0:
                continue
            target_row = target_rows.iloc[0]

            raw_neighbors = select_raw_perception_neighbors(
                frame_df=frame_df,
                target_row=target_row,
                cfg=cfg,
            )

            density0 = compute_target_centered_density(
                frame_df=frame_df,
                target_row=target_row,
                cfg=cfg,
            )

            window = make_window(
                target_id=target_id,
                center_frame=center_frame,
                target_track=target_track,
                target_row=target_row,
                raw_neighbors=raw_neighbors,
                track_bank=track_bank,
                hist_frames=hist_frames,
                fut_frames=fut_frames,
                density0=density0,
            )

            if window is None:
                continue

            windows.append(window)

            if cfg.print_every > 0 and len(windows) % cfg.print_every == 0:
                print(f"[build_windows] windows={len(windows)}")

            if cfg.max_windows is not None and len(windows) >= cfg.max_windows:
                break

        if cfg.max_windows is not None and len(windows) >= cfg.max_windows:
            break

    splits = split_windows_by_time_block(
        windows=windows,
        cfg=cfg,
        history_len=history_len,
        future_len=future_len,
    )

    payload = {
        "config": asdict(cfg),
        "dataset_name": f"TGSIM_{cfg.subset_name}",
        "type_definition": {
            "node_type": {
                "0": "HV",
                "1": str(cfg.subset_name).upper(),
                "2": "padding",
            },
            "size_type": {
                "0": "small_vehicle",
                "1": "large_vehicle",
                "2": "unknown_or_padding",
            },
        },
        "feature_columns": FEATURE_COLUMNS,
        "feature_index": {
            "x": IDX_X,
            "y": IDX_Y,
            "vx": IDX_VX,
            "vy": IDX_VY,
            "speed": IDX_SPEED,
            "acc": IDX_ACC,
        },
        "lane_relation_codes": {
            "same_front": LANE_SAME_FRONT,
            "same_back": LANE_SAME_BACK,
            "adjacent": LANE_ADJACENT,
        },
        "history_len": history_len,
        "future_len": future_len,

        "track_bank": track_bank,
        "windows": windows,
        "splits": splits,
    }

    return payload


def split_windows_by_time_block(
    windows: List[Dict],
    cfg: WindowConfig,
    history_len: int,
    future_len: int,
) -> Dict[str, List[int]]:
    if len(windows) == 0:
        return {"train": [], "val": [], "test": [], "dropped_buffer": []}

    frames = np.asarray([w["center_frame"] for w in windows], dtype=np.int64)
    min_frame = int(frames.min())
    max_frame = int(frames.max())
    span = max_frame - min_frame + 1

    train_end = min_frame + int(span * cfg.train_ratio)
    val_end = min_frame + int(span * (cfg.train_ratio + cfg.val_ratio))

    if cfg.split_buffer_frames is None:
        buffer_frames = history_len + future_len
    else:
        buffer_frames = int(cfg.split_buffer_frames)

    val_start = train_end + buffer_frames
    test_start = val_end + buffer_frames

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    dropped_buffer: List[int] = []

    for idx, w in enumerate(windows):
        f = int(w["center_frame"])

        if f <= train_end:
            train_idx.append(idx)
        elif val_start <= f <= val_end:
            val_idx.append(idx)
        elif f >= test_start:
            test_idx.append(idx)
        else:
            dropped_buffer.append(idx)

    print("[split] min_frame:", min_frame)
    print("[split] max_frame:", max_frame)
    print("[split] train_end:", train_end)
    print("[split] val_start:", val_start)
    print("[split] val_end:", val_end)
    print("[split] test_start:", test_start)
    print("[split] buffer_frames:", buffer_frames)
    print("[split] train:", len(train_idx))
    print("[split] val:", len(val_idx))
    print("[split] test:", len(test_idx))
    print("[split] dropped_buffer:", len(dropped_buffer))

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "dropped_buffer": dropped_buffer,
        "meta": {
            "min_frame": min_frame,
            "max_frame": max_frame,
            "train_end": train_end,
            "val_start": val_start,
            "val_end": val_end,
            "test_start": test_start,
            "buffer_frames": buffer_frames,
        },
    }


def save_pickle(obj: Dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[save] saved to {path}")
    print(f"[save] windows: {len(obj['windows'])}")
    print(f"[save] train: {len(obj['splits']['train'])}")
    print(f"[save] val: {len(obj['splits']['val'])}")
    print(f"[save] test: {len(obj['splits']['test'])}")


def main() -> None:
    cfg = WindowConfig()

    # Example for L2:
    # cfg.subset_name = "L2"
    # cfg.raw_csv_path = "./data/RawDataset/TGSIM/L2/tgsim_L2.csv"
    # cfg.output_path = "./ProcessedDataset/TGSIM/L2/windows_tgsim_L2.pkl"

    payload = build_windows(cfg)
    save_pickle(payload, cfg.output_path)


if __name__ == "__main__":
    main()
