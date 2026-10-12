# s2r_preprocess.py
#
# S2R-DUAL (Look-Ahead) RTK logs  ->  NGSIM-format CSV + braking events + ITID windows.
#
# What it does
#   1. Reads rtk_data_*.txt (20 Hz, three devices: leading / front / participant).
#      Keeps RTK-fixed rows with quality "Great"; uses the GNSS clock (column 3) as
#      the common time base, as in the ITSC pipeline.
#   2. Resamples to 10 Hz by averaging the records inside each 0.1 s bin.
#   3. Projects lat/lon to local metres, and keeps only frames where ALL THREE
#      vehicles are on the same straight leg of the U-turn course. Start, end,
#      the U-turn and the merge onto the road are removed by a heading test;
#      short low-speed stops inside a leg (e.g. during a braking event) are kept.
#   4. For every straight segment, builds a road-aligned frame:
#      Local_Y = distance along the direction of travel, Local_X = lateral offset,
#      and writes an NGSIM-format CSV (feet, ft/s, ft/s^2) that
#      build_windows_ngsim.py can read unchanged.
#   5. Detects the scripted braking events of the leading vehicle and the brake
#      onsets of the front vehicle and the participant, and reports the measured
#      response delays (threshold onset and cross-correlation estimates).
#   6. Picks "pseudo events" (steady cruising, no braking nearby) for controls.
#   7. Optionally builds the ITID window pickle with the repository's
#      build_windows_ngsim.py and puts every window into the "all"/"val"/"test"
#      splits (zero-shot evaluation, nothing is used for training).
#
# Usage (single file):
#   python s2r_preprocess.py --rtk path/to/rtk_data_xxx.txt --out_dir ./S2R_LA_mode1 \
#       --repo path/to/ITID --build_windows
#
# Usage (all subjects, LA = group A, mode 1, layout DATA_ROOT/<subject>/A/1/rtk_data_*.txt):
#   python s2r_preprocess.py --data_root Data --group A --mode 1 --out_dir ./S2R_LA_mode1 \
#       --repo path/to/ITID --build_windows
#
# Outputs in out_dir:
#   trial_index.csv        trial id -> file, subject, group, mode
#   vehicle_index.csv      Vehicle_ID -> trial, segment, role
#   segments.csv           kept straight segments
#   tracks_10hz.csv        all 10 Hz frames with leg labels (for checking)
#   s2r_ngsim_format.csv   input for build_windows_ngsim.py
#   events.csv             braking events with measured onsets / response delays
#   pseudo_events.csv      control time points
#   plots/trial_XXX.png    speed profiles with kept segments and detected onsets
#   s2r_windows.pkl        (with --build_windows) ITID window file

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FT_PER_M = 1.0 / 0.3048
FRAME_OFFSET_PER_TRIAL = 10_000_000          # keeps Frame_ID of different trials apart
ROLE_CODES = {"leading_vehicle": 1, "front_vehicle": 2, "participant_vehicle": 3}
ROLE_SHORT = {"leading_vehicle": "L", "front_vehicle": "F", "participant_vehicle": "P"}

RTK_COLUMNS = [
    "ts_host", "device", "gnss_time", "lat", "lon", "alt_msl", "ell_h", "geoid_sep",
    "rtk_quality", "num_sats", "hdop", "quality", "precision", "speed_kmh", "acc_dev",
    "speed_trend", "acc_raw", "async_ms",
]


@dataclass
class PrepConfig:
    # Data quality
    require_great: bool = True
    require_rtk_fixed: bool = True          # RTK quality code 4
    max_interp_gap_frames: int = 3          # fill gaps up to 0.3 s

    # Straight-leg detection
    heading_half_window: int = 5            # heading from displacement over +-0.5 s
    heading_min_disp_m: float = 1.5         # below this (about <1.5 m/s) heading is undefined
    heading_tol_deg: float = 10.0           # |heading - leg axis| tolerance
    max_fill_s: float = 20.0                # low-speed stretches inside one leg are kept up to this length
    trim_s: float = 1.0                     # drop this much at both ends of every segment
    min_segment_s: float = 12.0             # keep segments at least this long (window = 8 s)

    # Kinematics
    smooth_frames: int = 5                  # centred moving average (ITSC pipeline uses 5)

    # Event detection (accelerations in m/s^2, times in s)
    lead_trigger: float = -1.5              # leader episode must cross this
    lead_peak_min: float = -2.0             # ... and reach at least this deceleration
    onset_level: float = -0.5               # onset = start of the run below this level
    follow_trigger: float = -0.8            # follower response must cross this
    front_max_response_s: float = 8.0       # front vehicle vs leader onset (60 km/h braking ramps up slowly)
    part_max_response_s: float = 4.0        # participant vs front-vehicle onset
    pre_stimulus_s: float = 0.5             # follower onset may not precede the stimulus by more
    harsh_threshold: float = -4.0           # peak decel below this -> "harsh"
    speed_level_split_mps: float = 11.1     # 40 km/h, separates the 30 / 60 km/h runs
    min_event_gap_s: float = 8.0
    xcorr_window: Tuple[float, float] = (-1.0, 6.0)
    xcorr_max_lag_s: float = 3.0            # participant vs front vehicle; front vs leader uses 6 s

    # Pseudo events (controls): no braking of any vehicle nearby
    pseudo_quiet_before_s: float = 3.0
    pseudo_quiet_after_s: float = 5.0
    pseudo_quiet_level: float = 0.8         # |acc| of L, F and P stays below this
    pseudo_min_speed: float = 3.0
    pseudo_spacing_s: float = 8.0
    pseudo_per_trial: int = 30

    # Vehicle description for the NGSIM CSV
    v_class: int = 2                        # NGSIM "auto"
    length_m: Tuple[float, float, float] = (4.7, 4.9, 4.7)   # leading, front (SUV), participant
    lane_id: int = 1


# ---------------------------------------------------------------------------
# Reading and resampling
# ---------------------------------------------------------------------------

def gnss_seconds(x) -> float:
    s = f"{float(x):09.2f}"                  # hhmmss.ss
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + float(s[4:])


def read_rtk(path: Path, cfg: PrepConfig) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", header=None, names=RTK_COLUMNS,
                     skipinitialspace=True, engine="python")
    df = df.dropna(subset=["device", "gnss_time", "lat", "lon"])
    df["device"] = df["device"].astype(str).str.strip()
    df = df[df["device"].isin(ROLE_CODES.keys())].copy()
    df["quality"] = df["quality"].astype(str).str.strip()
    n0 = len(df)
    if cfg.require_great:
        df = df[df["quality"] == "Great"]
    if cfg.require_rtk_fixed:
        df = df[pd.to_numeric(df["rtk_quality"], errors="coerce") == 4]
    if len(df) < n0:
        print(f"  [quality] dropped {n0 - len(df)} of {n0} rows")
    df["t"] = df["gnss_time"].apply(gnss_seconds)
    for c in ["lat", "lon", "speed_kmh", "acc_dev"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def resample_10hz(df: pd.DataFrame, cfg: PrepConfig) -> Tuple[np.ndarray, Dict[str, pd.DataFrame]]:
    """Average the 20 Hz records inside each 0.1 s bin, then put all devices on one grid."""
    df = df.copy()
    df["k"] = np.round(df["t"].to_numpy() * 20.0).astype(np.int64) // 2
    agg = df.groupby(["device", "k"]).agg(
        lat=("lat", "mean"), lon=("lon", "mean"),
        speed_kmh=("speed_kmh", "mean"), acc_dev=("acc_dev", "mean"),
    ).reset_index()

    k_start = max(int(agg[agg.device == d].k.min()) for d in ROLE_CODES)
    k_end = min(int(agg[agg.device == d].k.max()) for d in ROLE_CODES)
    grid = np.arange(k_start, k_end + 1, dtype=np.int64)

    per_dev = {}
    for dev in ROLE_CODES:
        g = agg[agg.device == dev].drop(columns=["device"]).set_index("k").reindex(grid)
        g = g.astype(float).interpolate(limit=cfg.max_interp_gap_frames, limit_area="inside")
        g["speed"] = g["speed_kmh"] / 3.6
        per_dev[dev] = g[["lat", "lon", "speed", "acc_dev"]]
    return grid, per_dev


def to_local_xy(per_dev: Dict[str, pd.DataFrame]) -> None:
    lat0 = np.nanmean(np.concatenate([d["lat"].to_numpy() for d in per_dev.values()]))
    lon0 = np.nanmean(np.concatenate([d["lon"].to_numpy() for d in per_dev.values()]))
    r_earth = 6378137.0
    for d in per_dev.values():
        d["E"] = np.radians(d["lon"] - lon0) * r_earth * np.cos(np.radians(lat0))
        d["N"] = np.radians(d["lat"] - lat0) * r_earth


def centred_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(w, center=True, min_periods=max(1, w // 2 + 1)).mean().to_numpy()


def smoothed_acc(speed: np.ndarray, w: int, dt: float = 0.1) -> np.ndarray:
    v = centred_ma(speed, w)
    a = np.gradient(v, dt)
    return centred_ma(a, w)


# ---------------------------------------------------------------------------
# Straight-leg segmentation
# ---------------------------------------------------------------------------

def leg_labels(d: pd.DataFrame, axis_deg: float, cfg: PrepConfig) -> np.ndarray:
    """+1 / -1: on a straight leg travelling along / against the course axis; 0: not straight."""
    E, N = d["E"].to_numpy(), d["N"].to_numpy()
    h = cfg.heading_half_window
    dE = np.full_like(E, np.nan)
    dN = np.full_like(N, np.nan)
    dE[h:-h] = E[2 * h:] - E[:-2 * h]
    dN[h:-h] = N[2 * h:] - N[:-2 * h]
    disp = np.hypot(dE, dN)
    hdg = np.degrees(np.arctan2(dN, dE))
    dh = (hdg - axis_deg + 180.0) % 360.0 - 180.0

    lab = np.full(len(E), np.nan)
    ok = np.isfinite(disp) & (disp >= cfg.heading_min_disp_m)
    lab[ok & (np.abs(dh) <= cfg.heading_tol_deg)] = 1
    lab[ok & (np.abs(dh) >= 180.0 - cfg.heading_tol_deg)] = -1
    lab[ok & ~np.isin(lab, [1, -1])] = 0

    # Fill undefined (slow) stretches only when both sides belong to the same leg.
    max_fill = int(round(cfg.max_fill_s * 10))
    i, n = 0, len(lab)
    while i < n:
        if np.isnan(lab[i]):
            j = i
            while j < n and np.isnan(lab[j]):
                j += 1
            left = lab[i - 1] if i > 0 else np.nan
            right = lab[j] if j < n else np.nan
            fill = left if (left == right and left in (1, -1) and (j - i) <= max_fill) else 0
            lab[i:j] = fill
            i = j
        else:
            i += 1
    return lab.astype(int)


def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))          # [i, j)
            i = j
        else:
            i += 1
    return runs


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def onset_before(a: np.ndarray, i_cross: int, level: float, lo: int) -> int:
    """Walk back from a trigger crossing to the start of the run below `level`."""
    i = i_cross
    while i - 1 >= lo and a[i - 1] < level:
        i -= 1
    return i


def find_lead_events(aL: np.ndarray, vL: np.ndarray, cfg: PrepConfig) -> List[Dict]:
    events, i, n = [], 0, len(aL)
    gap = int(round(cfg.min_event_gap_s * 10))
    last_onset = -10 ** 9
    while i < n:
        if np.isfinite(aL[i]) and aL[i] < cfg.lead_trigger:
            j = i
            while j < n and np.isfinite(aL[j]) and aL[j] < cfg.onset_level:
                j += 1
            peak = float(np.nanmin(aL[i:j])) if j > i else float(aL[i])
            onset = onset_before(aL, i, cfg.onset_level, 0)
            if peak <= cfg.lead_peak_min and onset - last_onset >= gap:
                events.append({"onset": onset, "end": j, "peak": peak,
                               "speed_before": float(vL[onset])})
                last_onset = onset
            i = j + 1
        else:
            i += 1
    return events


def find_follower_onset(a: np.ndarray, stim: int, cfg: PrepConfig,
                        max_response_s: float) -> Tuple[Optional[int], bool]:
    lo = max(0, stim - int(round(cfg.pre_stimulus_s * 10)))
    hi = min(len(a), stim + int(round(max_response_s * 10)))
    for i in range(lo, hi):
        if np.isfinite(a[i]) and a[i] < cfg.follow_trigger:
            on = onset_before(a, i, cfg.onset_level, lo)
            pre_decel = bool(np.isfinite(a[stim]) and a[stim] < cfg.onset_level)
            return on, pre_decel
    return None, False


def xcorr_delay(a_stim: np.ndarray, a_resp: np.ndarray, stim: int, cfg: PrepConfig,
                max_lag_s: Optional[float] = None) -> float:
    w0 = stim + int(round(cfg.xcorr_window[0] * 10))
    w1 = stim + int(round(cfg.xcorr_window[1] * 10))
    max_lag = int(round((cfg.xcorr_max_lag_s if max_lag_s is None else max_lag_s) * 10))
    if w0 < 0 or w1 + max_lag > len(a_resp):
        return float("nan")
    x = a_stim[w0:w1]
    best, best_lag = -np.inf, None
    for lag in range(0, max_lag + 1):
        y = a_resp[w0 + lag:w1 + lag]
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.std(x) < 1e-6 or np.std(y) < 1e-6:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best:
            best, best_lag = r, lag
    return float(best_lag) / 10.0 if best_lag is not None else float("nan")


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------

def process_trial(path: Path, trial_idx: int, meta: Dict, cfg: PrepConfig, out_dir: Path):
    print(f"[trial {trial_idx}] {path}")
    raw = read_rtk(path, cfg)
    grid, per_dev = resample_10hz(raw, cfg)
    to_local_xy(per_dev)

    # Course axis from moving points of all vehicles (principal direction).
    pts = []
    for d in per_dev.values():
        m = (d["speed"] > 3.0) & d["E"].notna()
        pts.append(np.c_[d.loc[m, "E"], d.loc[m, "N"]])
    pts = np.concatenate(pts)
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    axis = vt[0]
    axis_deg = float(np.degrees(np.arctan2(axis[1], axis[0])))

    for dev, d in per_dev.items():
        d["leg"] = leg_labels(d, axis_deg, cfg)
        d["acc"] = smoothed_acc(d["speed"].to_numpy(), cfg.smooth_frames)

    L, F, P = (per_dev["leading_vehicle"], per_dev["front_vehicle"], per_dev["participant_vehicle"])
    finite = np.ones(len(grid), dtype=bool)
    for d in per_dev.values():
        finite &= d[["E", "N", "speed", "acc"]].notna().all(axis=1).to_numpy()
    same_leg = (L["leg"].to_numpy() == F["leg"].to_numpy()) & (F["leg"].to_numpy() == P["leg"].to_numpy())
    keep = finite & same_leg & np.isin(P["leg"].to_numpy(), [1, -1])

    trim = int(round(cfg.trim_s * 10))
    min_len = int(round(cfg.min_segment_s * 10))
    segments = []
    for (i0, i1) in contiguous_runs(keep):
        i0, i1 = i0 + trim, i1 - trim
        if i1 - i0 >= min_len:
            segments.append((i0, i1))

    k0 = int(grid[0])
    frame_base = trial_idx * FRAME_OFFSET_PER_TRIAL
    csv_rows, veh_rows, seg_rows, ev_rows, pseudo_rows = [], [], [], [], []

    for seg_idx, (i0, i1) in enumerate(segments):
        sl = slice(i0, i1)
        # Road-aligned frame for this straight segment.
        seg_pts = np.concatenate([np.c_[d["E"].to_numpy()[sl], d["N"].to_numpy()[sl]]
                                  for d in per_dev.values()])
        _, _, vt = np.linalg.svd(seg_pts - seg_pts.mean(0), full_matrices=False)
        u = vt[0]
        pE, pN = P["E"].to_numpy()[sl], P["N"].to_numpy()[sl]
        if (pE[-1] - pE[0]) * u[0] + (pN[-1] - pN[0]) * u[1] < 0:
            u = -u
        nvec = np.array([u[1], -u[0]])          # to the right of travel
        origin = np.array([pE[0], pN[0]])

        ys = {}
        for dev, d in per_dev.items():
            vid = trial_idx * 1000 + seg_idx * 10 + ROLE_CODES[dev]
            E, N = d["E"].to_numpy()[sl], d["N"].to_numpy()[sl]
            rel = np.c_[E - origin[0], N - origin[1]]
            y = rel @ u
            x = rel @ nvec
            ys[dev] = y
            v = d["speed"].to_numpy()[sl]
            a = d["acc"].to_numpy()[sl]
            frames = frame_base + (grid[sl] - k0)
            length = cfg.length_m[ROLE_CODES[dev] - 1]
            csv_rows.append(pd.DataFrame({
                "Vehicle_ID": vid,
                "Frame_ID": frames,
                "Global_Time": frames * 100,
                "Local_X": x * FT_PER_M,
                "Local_Y": y * FT_PER_M,
                "v_Vel": v * FT_PER_M,
                "v_Acc": a * FT_PER_M,
                "Lane_ID": cfg.lane_id,
                "v_Class": cfg.v_class,
                "v_Length": length * FT_PER_M,
            }))
            veh_rows.append({"Vehicle_ID": vid, "trial_idx": trial_idx, "seg_idx": seg_idx,
                             "role": ROLE_SHORT[dev], "device": dev, **meta})

        order_ok = bool(np.nanmedian(ys["leading_vehicle"] - ys["front_vehicle"]) > 0
                        and np.nanmedian(ys["front_vehicle"] - ys["participant_vehicle"]) > 0)
        seg_rows.append({
            "trial_idx": trial_idx, "seg_idx": seg_idx,
            "start_frame": int(frame_base + grid[i0] - k0), "end_frame": int(frame_base + grid[i1 - 1] - k0),
            "duration_s": (i1 - i0) / 10.0, "direction": int(P["leg"].to_numpy()[i0]),
            "gap_LF_median_m": float(np.nanmedian(ys["leading_vehicle"] - ys["front_vehicle"])),
            "gap_FP_median_m": float(np.nanmedian(ys["front_vehicle"] - ys["participant_vehicle"])),
            "order_L_F_P_ok": order_ok, **meta,
        })
        if not order_ok:
            print(f"  [warning] segment {seg_idx}: vehicles are not ordered L > F > P along the road")

        # Events inside this segment.
        aL, aF, aP = (L["acc"].to_numpy()[sl], F["acc"].to_numpy()[sl], P["acc"].to_numpy()[sl])
        vL = L["speed"].to_numpy()[sl]
        for ev in find_lead_events(aL, vL, cfg):
            onL = ev["onset"]
            onF, preF = find_follower_onset(aF, onL, cfg, cfg.front_max_response_s)
            onP, preP = (find_follower_onset(aP, onF, cfg, cfg.part_max_response_s)
                         if onF is not None else (None, False))
            onP_fromL, _ = find_follower_onset(aP, onL, cfg, cfg.front_max_response_s + cfg.part_max_response_s)
            to_frame = lambda i: (int(frame_base + grid[i0 + i] - k0) if i is not None else np.nan)
            ev_rows.append({
                "trial_idx": trial_idx, "seg_idx": seg_idx, "event_in_seg": len(ev_rows),
                "lead_onset_frame": to_frame(onL),
                "front_onset_frame": to_frame(onF),
                "part_onset_frame": to_frame(onP),
                "lead_peak_decel": ev["peak"],
                "lead_speed_before_mps": ev["speed_before"],
                "severity": "harsh" if ev["peak"] <= cfg.harsh_threshold else "gradual",
                "speed_level_kmh": 30 if ev["speed_before"] < cfg.speed_level_split_mps else 60,
                "rt_front_s": (onF - onL) / 10.0 if onF is not None else np.nan,
                "rt_part_s": (onP - onF) / 10.0 if (onP is not None and onF is not None) else np.nan,
                "rt_part_from_lead_s": (onP_fromL - onL) / 10.0 if onP_fromL is not None else np.nan,
                "rt_front_xcorr_s": xcorr_delay(aL, aF, onL, cfg, max_lag_s=6.0),
                "rt_part_xcorr_s": xcorr_delay(aF, aP, onF, cfg) if onF is not None else np.nan,
                "front_pre_decel": preF, "part_pre_decel": preP,
                "s_before_onset": onL / 10.0, "s_after_onset": (len(aL) - onL) / 10.0,
                **meta,
            })

        # Pseudo events: steady stretches, far from any braking.
        qb = int(round(cfg.pseudo_quiet_before_s * 10))
        qa = int(round(cfg.pseudo_quiet_after_s * 10))
        spacing = int(round(cfg.pseudo_spacing_s * 10))
        last = -10 ** 9
        vF = F["speed"].to_numpy()[sl]
        for c in range(qb + 30, len(aL) - qa - 10):
            if c - last < spacing:
                continue
            w = slice(c - qb, c + qa)
            if (np.nanmax(np.abs(aL[w])) < cfg.pseudo_quiet_level
                    and np.nanmax(np.abs(aF[w])) < cfg.pseudo_quiet_level
                    and np.nanmax(np.abs(aP[w])) < cfg.pseudo_quiet_level
                    and np.nanmin(vF[w]) > cfg.pseudo_min_speed):
                pseudo_rows.append({"trial_idx": trial_idx, "seg_idx": seg_idx,
                                    "stim_frame": int(frame_base + grid[i0 + c] - k0), **meta})
                last = c

    if len(pseudo_rows) > cfg.pseudo_per_trial:
        rng = np.random.default_rng(trial_idx)
        keep_idx = sorted(rng.choice(len(pseudo_rows), cfg.pseudo_per_trial, replace=False))
        pseudo_rows = [pseudo_rows[i] for i in keep_idx]

    # 10 Hz tracks with labels, for inspection.
    track_rows = []
    for dev, d in per_dev.items():
        t = d.reset_index(drop=True)
        t["frame"] = frame_base + (grid - k0)
        t["role"] = ROLE_SHORT[dev]
        t["kept"] = False
        for (i0, i1) in segments:
            t.loc[i0:i1 - 1, "kept"] = True
        t["trial_idx"] = trial_idx
        track_rows.append(t[["trial_idx", "frame", "role", "E", "N", "speed", "acc", "acc_dev", "leg", "kept"]])

    plot_trial(trial_idx, grid, per_dev, segments, ev_rows, out_dir, meta)

    n_ev = len(ev_rows)
    print(f"  axis={axis_deg:.1f} deg | kept {len(segments)} segments, "
          f"{sum(i1 - i0 for i0, i1 in segments) / 10:.0f} s | events={n_ev} | pseudo={len(pseudo_rows)}")
    for e in ev_rows:
        print(f"    event seg{e['seg_idx']}: {e['severity']:7s} {e['speed_level_kmh']} km/h "
              f"peak={e['lead_peak_decel']:.2f} | RT front={e['rt_front_s']:.1f}s "
              f"(xcorr {e['rt_front_xcorr_s']:.1f}) | RT participant={e['rt_part_s']:.1f}s "
              f"(xcorr {e['rt_part_xcorr_s']:.1f})")
    return csv_rows, veh_rows, seg_rows, ev_rows, pseudo_rows, track_rows


def plot_trial(trial_idx, grid, per_dev, segments, ev_rows, out_dir: Path, meta: Dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    t = (grid - grid[0]) / 10.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    colors = {"leading_vehicle": "tab:red", "front_vehicle": "tab:orange", "participant_vehicle": "tab:blue"}
    for dev, d in per_dev.items():
        axes[0].plot(t, d["speed"], color=colors[dev], lw=1, label=ROLE_SHORT[dev])
        axes[1].plot(t, d["acc"], color=colors[dev], lw=1)
    for (i0, i1) in segments:
        for ax in axes:
            ax.axvspan(t[i0], t[i1 - 1], color="0.9", zorder=0)
    frame_base = trial_idx * FRAME_OFFSET_PER_TRIAL
    for e in ev_rows:
        for key, c in [("lead_onset_frame", "tab:red"), ("front_onset_frame", "tab:orange"),
                       ("part_onset_frame", "tab:blue")]:
            f = e[key]
            if np.isfinite(f):
                tt = (f - frame_base) / 10.0
                for ax in axes:
                    ax.axvline(tt, color=c, ls="--", lw=0.8)
    axes[0].set_ylabel("speed (m/s)")
    axes[1].set_ylabel("acc (m/s$^2$)")
    axes[1].set_xlabel("time (s); grey = kept straight segments")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"trial {trial_idx} {meta}")
    fig.tight_layout()
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "plots" / f"trial_{trial_idx:03d}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Windows (uses the repository's build_windows_ngsim.py)
# ---------------------------------------------------------------------------

def build_itid_windows(repo: Path, csv_path: Path, out_pkl: Path) -> None:
    sys.path.insert(0, str(repo.resolve()))
    import build_windows_ngsim as bw

    cfg = bw.WindowConfig()
    cfg.raw_csv_path = str(csv_path)
    cfg.output_path = str(out_pkl)
    payload = bw.build_windows(cfg)

    all_idx = list(range(len(payload["windows"])))
    payload["splits"] = {
        "train": [], "val": all_idx, "test": all_idx, "all": all_idx,
        "dropped_buffer": [], "meta": {"note": "S2R-DUAL zero-shot: every window is evaluation data"},
    }
    payload["s2r_source_csv"] = str(csv_path)
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_nb = np.array([len(w["neighbor_ids"]) for w in payload["windows"]])
    print(f"[windows] saved {len(all_idx)} windows -> {out_pkl}")
    print(f"[windows] neighbours per window: mean={n_nb.mean():.2f}, min={n_nb.min()}, max={n_nb.max()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_files(args) -> List[Tuple[Path, Dict]]:
    items = []
    if args.rtk:
        for p in args.rtk:
            p = Path(p)
            items.append((p, {"subject": p.parent.parent.parent.name if len(p.parts) > 3 else "",
                              "group": args.group, "mode": args.mode, "file": p.name}))
    if args.data_root:
        root = Path(args.data_root)
        pats = [root / "*" / args.group / str(args.mode) / "rtk_data_*.txt",
                root / args.group / str(args.mode) / "rtk_data_*.txt"]
        for pat in pats:
            for f in sorted(glob.glob(str(pat))):
                f = Path(f)
                subj = f.parents[2].name if f.parents[2] != root.parent else root.name
                items.append((f, {"subject": subj, "group": args.group, "mode": args.mode, "file": f.name}))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtk", nargs="*", default=None, help="one or more rtk_data_*.txt files")
    ap.add_argument("--data_root", default=None, help="root with <subject>/<group>/<mode>/rtk_data_*.txt")
    ap.add_argument("--group", default="A", help="A = look-ahead in the original folder layout")
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--repo", default=None, help="path to the ITID repository (for --build_windows)")
    ap.add_argument("--build_windows", action="store_true")
    ap.add_argument("--config_json", default=None, help="optional JSON overriding PrepConfig fields")
    args = ap.parse_args()

    cfg = PrepConfig()
    if args.config_json:
        for k, v in json.load(open(args.config_json)).items():
            setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(args)
    if not files:
        raise SystemExit("No RTK files found. Use --rtk or --data_root.")

    all_csv, all_veh, all_seg, all_ev, all_pseudo, all_tracks, trial_rows = [], [], [], [], [], [], []
    for trial_idx, (path, meta) in enumerate(files, start=1):
        try:
            c, v, s, e, p, t = process_trial(path, trial_idx, meta, cfg, out_dir)
        except Exception as ex:                       # keep going on a bad file
            print(f"  [error] {path}: {ex}")
            continue
        all_csv += c; all_veh += v; all_seg += s; all_ev += e; all_pseudo += p; all_tracks += t
        trial_rows.append({"trial_idx": trial_idx, "path": str(path), **meta})

    if not all_csv:
        raise SystemExit("Nothing kept. Check the quality filter and the leg settings.")

    pd.DataFrame(trial_rows).to_csv(out_dir / "trial_index.csv", index=False)
    pd.DataFrame(all_veh).to_csv(out_dir / "vehicle_index.csv", index=False)
    pd.DataFrame(all_seg).to_csv(out_dir / "segments.csv", index=False)
    ev = pd.DataFrame(all_ev)
    if len(ev):
        ev.insert(0, "event_id", np.arange(len(ev)))
    ev.to_csv(out_dir / "events.csv", index=False)
    ps = pd.DataFrame(all_pseudo)
    if len(ps):
        ps.insert(0, "pseudo_id", np.arange(len(ps)))
    ps.to_csv(out_dir / "pseudo_events.csv", index=False)
    pd.concat(all_tracks, ignore_index=True).to_csv(out_dir / "tracks_10hz.csv", index=False)
    csv_path = out_dir / "s2r_ngsim_format.csv"
    pd.concat(all_csv, ignore_index=True).sort_values(["Frame_ID", "Vehicle_ID"]).to_csv(csv_path, index=False)
    with open(out_dir / "prep_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    if len(ev):
        print("\n[events] measured response delays (s)")
        print(ev[["rt_front_s", "rt_part_s", "rt_part_from_lead_s", "rt_front_xcorr_s", "rt_part_xcorr_s"]]
              .describe().round(2).to_string())
    print(f"\n[saved] {out_dir}")

    if args.build_windows:
        if not args.repo:
            raise SystemExit("--build_windows needs --repo")
        build_itid_windows(Path(args.repo), csv_path, out_dir / "s2r_windows.pkl")


if __name__ == "__main__":
    main()
