# s2r_run_model.py
#
# Zero-shot test of the NGSIM-trained ITID model on S2R-DUAL look-ahead events.
#
# Question: does the gate on the front-vehicle -> participant edge open at a fixed
# delay after the front vehicle starts braking (stimulus-locked), or at a fixed
# delay after the forecast origin (origin-locked)? And does that delay match the
# participant's measured response delay?
#
# Design
#   For every braking event, the forecast origin (last observed frame) is placed at
#   offsets o = -1.0 ... +1.5 s around the front vehicle's brake onset. For each
#   origin we read the gate open probability p(k) of the F->P edge at rollout
#   steps k = 1..50, i.e. at time tau = o + 0.1 k after the stimulus.
#     gate onset (stimulus time)  tau_on(o) = o + 0.1 * first k with p >= threshold
#     origin-locked   ->  tau_on(o) = o + const   (slope 1)
#     stimulus-locked ->  tau_on(o) = const       (slope 0)
#   Fitted over origins after the stimulus and before the participant responds.
#   Controls: pseudo events (no braking nearby), the reverse edge P->F, and the
#   L->P edge (the leader is hidden behind the SUV in mode 1).
#   Sanity: zero-shot ADE of the model vs constant velocity (and, optionally,
#   the same model with all edges removed), and the horizon profile of the gate
#   (the Fig. 2 analogue) on S2R windows.
#
# Usage
#   python s2r_run_model.py --repo path/to/ITID \
#       --checkpoint path/to/NGSIM/checkpoint_best.pt \
#       --prep_dir ./S2R_LA_mode1 [--cte_table path/to/ngsim_cte_table.pkl] \
#       [--device cuda] [--no_interaction_check]
#   Re-run only the analysis (e.g. another threshold):
#   python s2r_run_model.py --prep_dir ./S2R_LA_mode1 --analyze_only --gate_threshold 0.3
#
# Outputs in <prep_dir>/zero_shot (or --out_dir):
#   traces.npz, samples.csv        raw gate traces per window
#   sanity.json                    ADE checks and gate horizon profile
#   per_event.csv                  stimulus-locked latency, slope, measured response delay
#   onsets_long.csv                gate onset for every event x origin offset
#   curves_events.csv / curves_pseudo.csv   averaged gate curves
#   summary.json                   main numbers
#   plots/*.png

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROLE_CODE = {"L": 1, "F": 2, "P": 3}
PAIRS = [("F", "P"), ("L", "P"), ("P", "F"), ("L", "F"), ("F", "L"), ("P", "L")]
PAIR_NAMES = [f"{s}->{r}" for s, r in PAIRS]
DT = 0.1


# ===========================================================================
# Stage 1: inference
# ===========================================================================

def offsets_from_arg(s: str) -> np.ndarray:
    lo, hi, step = [float(x) for x in s.split(":")]
    n = int(round((hi - lo) / step)) + 1
    return np.round(lo + step * np.arange(n), 3)


def vid_of(trial_idx: int, seg_idx: int, role: str) -> int:
    return int(trial_idx) * 1000 + int(seg_idx) * 10 + ROLE_CODE[role]


def safe_make_config(cls, cfg_dict):
    if cfg_dict is None:
        return cls()
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in cfg_dict.items() if k in valid})


def build_requests(prep_dir: Path, payload: Dict, offsets: np.ndarray, sanity_stride: int) -> pd.DataFrame:
    """List of (window index, tag) to run."""
    lookup = {(int(w["target_id"]), int(w["center_frame"])): i for i, w in enumerate(payload["windows"])}
    events = pd.read_csv(prep_dir / "events.csv")
    pseudo = pd.read_csv(prep_dir / "pseudo_events.csv") if (prep_dir / "pseudo_events.csv").exists() else pd.DataFrame()
    reqs, missing = [], 0

    def add(kind, ref_id, recv, stim_role, stim_frame, trial, seg):
        nonlocal missing
        tvid = vid_of(trial, seg, recv)
        for o in offsets:
            cf = int(stim_frame) + int(round(o * 10))
            wi = lookup.get((tvid, cf))
            if wi is None:
                missing += 1
                continue
            reqs.append({"window_idx": wi, "kind": kind, "ref_id": int(ref_id), "receiver": recv,
                         "stim_role": stim_role, "offset_s": float(o), "center_frame": cf,
                         "stim_frame": int(stim_frame), "trial_idx": int(trial), "seg_idx": int(seg)})

    for _, e in events.iterrows():
        if np.isfinite(e["front_onset_frame"]):
            add("event", e["event_id"], "P", "F", e["front_onset_frame"], e["trial_idx"], e["seg_idx"])
        if np.isfinite(e["lead_onset_frame"]):
            add("event_lead", e["event_id"], "F", "L", e["lead_onset_frame"], e["trial_idx"], e["seg_idx"])
    for _, p in pseudo.iterrows():
        add("pseudo", p["pseudo_id"], "P", "F", p["stim_frame"], p["trial_idx"], p["seg_idx"])

    # Sanity windows: participant as target, every `sanity_stride` frames.
    for i, w in enumerate(payload["windows"]):
        if int(w["target_id"]) % 10 == ROLE_CODE["P"] and int(w["center_frame"]) % sanity_stride == 0:
            reqs.append({"window_idx": i, "kind": "sanity", "ref_id": -1, "receiver": "P", "stim_role": "",
                         "offset_s": np.nan, "center_frame": int(w["center_frame"]), "stim_frame": -1,
                         "trial_idx": int(w["target_id"]) // 1000, "seg_idx": (int(w["target_id"]) % 1000) // 10})

    print(f"[requests] {len(reqs)} window requests; {missing} requested origins had no complete window "
          f"(near a segment edge)")
    return pd.DataFrame(reqs)


def run_inference(args, prep_dir: Path, out_dir: Path) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from ngsim_dataset import NGSIMCTEDataset, NGSIMDatasetConfig, simple_collate_fn
    from dynamic_cte_gnn import DynamicCTEGNN, DynamicCTEGNNConfig

    win_path = prep_dir / "s2r_windows.pkl"
    with open(win_path, "rb") as f:
        payload = pickle.load(f)
    if "all" not in payload["splits"]:
        raise SystemExit("s2r_windows.pkl has no 'all' split. Build it with s2r_preprocess.py --build_windows.")

    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:                                   # older torch without weights_only
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = safe_make_config(DynamicCTEGNNConfig, ckpt.get("model_cfg"))
    ds_cfg_ckpt = safe_make_config(NGSIMDatasetConfig, ckpt.get("dataset_cfg"))
    cte_table = args.cte_table or ds_cfg_ckpt.cte_table_path
    if not Path(cte_table).exists():
        raise SystemExit(f"CTE table not found: {cte_table}. Pass the NGSIM table with --cte_table.")
    model_cfg.cte_table_path = cte_table          # zero-shot: keep the NGSIM table

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = DynamicCTEGNN(model_cfg).to(device)
    state = ckpt.get("model") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError("checkpoint has neither 'model' nor 'model_state_dict'")
    model.load_state_dict(state)
    model.eval()
    pred_steps = int(args.pred_steps or model_cfg.pred_steps)

    ds_cfg = NGSIMDatasetConfig(
        window_path=str(win_path), cte_table_path=cte_table, split="all",
        top_k_neighbors=ds_cfg_ckpt.top_k_neighbors, max_ttc=ds_cfg_ckpt.max_ttc,
        allow_invalid_lane_edges=ds_cfg_ckpt.allow_invalid_lane_edges,
        num_vehicle_types=ds_cfg_ckpt.num_vehicle_types, padding_type_index=ds_cfg_ckpt.padding_type_index,
        verbose=False,
    )
    dataset = NGSIMCTEDataset(ds_cfg)
    pos_of_window = {w: p for p, w in enumerate(dataset.indices)}

    offsets = offsets_from_arg(args.offsets)
    reqs = build_requests(prep_dir, payload, offsets, args.sanity_stride)
    uniq = sorted(set(reqs["window_idx"].tolist()))
    subset = Subset(dataset, [pos_of_window[w] for w in uniq])
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=simple_collate_fn, pin_memory=(device.type == "cuda"))

    vindex = pd.read_csv(prep_dir / "vehicle_index.csv")
    role_of = dict(zip(vindex["Vehicle_ID"].astype(int), vindex["role"]))

    P = len(PAIRS)
    n = len(uniq)
    op_all = np.full((n, pred_steps, P), np.nan, dtype=np.float32)
    beta_all = np.full((n, pred_steps, P), np.nan, dtype=np.float32)
    active_all = np.zeros((n, pred_steps), dtype=np.float32)      # #edges with p >= 0.5
    edges_all = np.zeros((n,), dtype=np.float32)                  # #valid edges
    ade = np.full(n, np.nan); fde = np.full(n, np.nan)
    ade_cv = np.full(n, np.nan); ade_noint = np.full(n, np.nan)

    row = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch, pred_steps=pred_steps, deterministic_gate=True)

            opb = out["open_probs"].float().cpu().numpy()            # [B,T,N,N] source u -> receiver v
            beb = out["betas"].float().cpu().numpy()
            emb = out["edge_mask"].bool().cpu().numpy()               # [B,N,N]
            vids = batch["vehicle_ids"].cpu().numpy()
            nmask = batch["node_mask"].bool().cpu().numpy()
            pred = out["pred_states"][..., :2].float().cpu().numpy()  # [B,T,N,2]
            fut = batch["future"][..., :2].float().cpu().numpy()      # [B,N,T,2]
            hist = batch["history"][..., :2].float().cpu().numpy()    # [B,N,Th,2]

            noint_pred = None
            if args.no_interaction_check:
                b_, n_, _ = emb.shape
                drop = out["edge_mask"].bool().unsqueeze(1).expand(b_, pred_steps, n_, n_).clone()
                out0 = model(batch, pred_steps=pred_steps, deterministic_gate=True, edge_drop_mask=drop)
                noint_pred = out0["pred_states"][..., :2].float().cpu().numpy()

            B = opb.shape[0]
            T = min(pred_steps, fut.shape[2])
            for b in range(B):
                node_role = {}
                for k in range(vids.shape[1]):
                    if nmask[b, k] and int(vids[b, k]) in role_of:
                        node_role[role_of[int(vids[b, k])]] = k
                for pi, (s, r) in enumerate(PAIRS):
                    if s in node_role and r in node_role and emb[b, node_role[s], node_role[r]]:
                        u, v = node_role[s], node_role[r]
                        op_all[row, :, pi] = opb[b, :, u, v]
                        beta_all[row, :, pi] = beb[b, :, u, v]
                em_t = emb[b][None, :, :]
                active_all[row] = ((opb[b] >= 0.5) & em_t).reshape(opb.shape[1], -1).sum(1)
                edges_all[row] = emb[b].sum()

                err = np.linalg.norm(pred[b, :T, 0] - fut[b, 0, :T], axis=-1)
                ade[row], fde[row] = err.mean(), err[-1]
                vel = (hist[b, 0, -1] - hist[b, 0, -6]) / 0.5
                cv = hist[b, 0, -1][None, :] + vel[None, :] * (np.arange(1, T + 1)[:, None] * DT)
                ade_cv[row] = np.linalg.norm(cv - fut[b, 0, :T], axis=-1).mean()
                if noint_pred is not None:
                    ade_noint[row] = np.linalg.norm(noint_pred[b, :T, 0] - fut[b, 0, :T], axis=-1).mean()
                row += 1
            if (bi + 1) % 20 == 0:
                print(f"[inference] {row}/{n} windows")

    uniq_arr = np.asarray(uniq)
    np.savez_compressed(out_dir / "traces.npz", window_idx=uniq_arr, op=op_all, beta=beta_all,
                        active=active_all, n_edges=edges_all, ade=ade, fde=fde, ade_cv=ade_cv,
                        ade_noint=ade_noint, pair_names=np.array(PAIR_NAMES))
    reqs.to_csv(out_dir / "samples.csv", index=False)
    meta = {"checkpoint": str(args.checkpoint), "cte_table": cte_table, "pred_steps": pred_steps,
            "offsets": offsets.tolist(), "n_windows_run": int(n)}
    json.dump(meta, open(out_dir / "inference_meta.json", "w"), indent=2)
    print(f"[inference] saved traces for {n} windows -> {out_dir}")


# ===========================================================================
# Stage 2: analysis (numpy / pandas only)
# ===========================================================================

def first_crossing(trace: np.ndarray, thr: float) -> Optional[int]:
    idx = np.where(np.isfinite(trace) & (trace >= thr))[0]
    return int(idx[0]) + 1 if len(idx) else None          # rollout step k (1-based)


def gate_onset(trace: np.ndarray, mode: str, thr: float, rel_frac: float, min_peak: float) -> Optional[int]:
    if not np.any(np.isfinite(trace)):
        return None
    if mode == "abs":
        return first_crossing(trace, thr)
    peak = np.nanmax(trace)
    if peak < min_peak:
        return None
    return first_crossing(trace, rel_frac * peak)


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) < 1e-9:
        return np.nan
    return float(np.polyfit(x[m], y[m], 1)[0])


def bootstrap_ci(values: np.ndarray, stat=np.nanmedian, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = [stat(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def pooled_within_slope(df: pd.DataFrame, group: str, x: str, y: str) -> float:
    """Slope of y on x after removing each group's mean (event fixed effects)."""
    d = df[[group, x, y]].dropna()
    if len(d) < 3:
        return np.nan
    xs = d[x] - d.groupby(group)[x].transform("mean")
    ys = d[y] - d.groupby(group)[y].transform("mean")
    den = float((xs ** 2).sum())
    return float((xs * ys).sum() / den) if den > 1e-12 else np.nan


def onset_table(samples: pd.DataFrame, op: np.ndarray, pair: str, kinds: List[str], a) -> pd.DataFrame:
    pi = PAIR_NAMES.index(pair)
    rows = []
    for ri, s in samples[samples.kind.isin(kinds)].iterrows():
        tr = op[s["_row"], :, pi]
        k = gate_onset(tr, a.onset_mode, a.gate_threshold, a.rel_frac, a.min_peak)
        kp = int(np.nanargmax(tr)) + 1 if np.any(np.isfinite(tr)) else None
        rows.append({
            "kind": s["kind"], "ref_id": s["ref_id"], "offset_s": s["offset_s"],
            "onset_step": k, "onset_from_origin_s": (k * DT if k else np.nan),
            "onset_from_stim_s": (s["offset_s"] + k * DT if k else np.nan),
            "peak_from_stim_s": (s["offset_s"] + kp * DT if kp else np.nan),
            "peak_value": float(np.nanmax(tr)) if np.any(np.isfinite(tr)) else np.nan,
            "mean_value": float(np.nanmean(tr)) if np.any(np.isfinite(tr)) else np.nan,
        })
    return pd.DataFrame(rows)


def averaged_curves(samples: pd.DataFrame, op: np.ndarray, pair: str, kind: str) -> pd.DataFrame:
    pi = PAIR_NAMES.index(pair)
    rows = []
    sub = samples[samples.kind == kind]
    for o, g in sub.groupby("offset_s"):
        tr = op[g["_row"].to_numpy(), :, pi]                      # [n, T]
        T = tr.shape[1]
        mean = np.nanmean(tr, axis=0) if len(g) else np.full(T, np.nan)
        for k in range(T):
            rows.append({"offset_s": o, "step": k + 1, "t_from_origin_s": (k + 1) * DT,
                         "t_from_stim_s": o + (k + 1) * DT, "mean_open_prob": float(mean[k]),
                         "n": int(np.isfinite(tr[:, k]).sum())})
    return pd.DataFrame(rows)


def analyze(prep_dir: Path, out_dir: Path, a) -> Dict:
    tr = np.load(out_dir / "traces.npz", allow_pickle=True)
    samples = pd.read_csv(out_dir / "samples.csv")
    row_of = {int(w): i for i, w in enumerate(tr["window_idx"])}
    samples["_row"] = samples["window_idx"].map(row_of)
    samples = samples.dropna(subset=["_row"]).copy()
    samples["_row"] = samples["_row"].astype(int)
    op = tr["op"]
    events = pd.read_csv(prep_dir / "events.csv")
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    summary: Dict = {"settings": {"onset_mode": a.onset_mode, "gate_threshold": a.gate_threshold,
                                  "rel_frac": a.rel_frac, "min_peak": a.min_peak,
                                  "fit_range_s": [a.fit_lo, a.fit_hi],
                                  "stop_before_response": a.stop_before_response}}

    # ---------------- sanity ----------------
    san = samples[samples.kind == "sanity"]
    rows = san["_row"].to_numpy()
    sanity = {
        "n_windows": int(len(rows)),
        "ADE_model_m": float(np.nanmean(tr["ade"][rows])) if len(rows) else np.nan,
        "ADE_const_velocity_m": float(np.nanmean(tr["ade_cv"][rows])) if len(rows) else np.nan,
        "ADE_no_interaction_m": (float(np.nanmean(tr["ade_noint"][rows]))
                                 if len(rows) and np.isfinite(tr["ade_noint"][rows]).any() else None),
        "FDE_model_m": float(np.nanmean(tr["fde"][rows])) if len(rows) else np.nan,
    }
    if len(rows):
        act = tr["active"][rows].sum(0) / np.maximum(tr["n_edges"][rows].sum(), 1)
        sanity["active_ratio_by_step"] = [round(float(x), 4) for x in act]
        sanity["active_ratio_peak_time_s"] = float((int(np.argmax(act)) + 1) * DT)
        for pi, name in enumerate(PAIR_NAMES):
            sanity[f"mean_open_prob_{name}"] = float(np.nanmean(op[rows, :, pi]))
    json.dump(sanity, open(out_dir / "sanity.json", "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    summary["sanity"] = {k: v for k, v in sanity.items() if k != "active_ratio_by_step"}

    # ---------------- event-locked F->P ----------------
    ons = onset_table(samples, op, "F->P", ["event", "pseudo"], a)
    ons.to_csv(out_dir / "onsets_long.csv", index=False)

    rt = events.set_index("event_id")
    per_event = []
    ev_ons = ons[ons.kind == "event"].copy()
    for eid, g in ev_ons.groupby("ref_id"):
        rt_p = float(rt.loc[eid, "rt_part_s"]) if eid in rt.index else np.nan
        hi = a.fit_hi
        if a.stop_before_response and np.isfinite(rt_p):
            hi = min(hi, rt_p - 0.1)
        fit = g[(g.offset_s >= a.fit_lo - 1e-9) & (g.offset_s <= hi + 1e-9)]
        per_event.append({
            "event_id": eid,
            "n_origins_fit": int(fit["onset_from_stim_s"].notna().sum()),
            "slope_onset_vs_origin": fit_slope(fit["offset_s"].to_numpy(), fit["onset_from_stim_s"].to_numpy()),
            "stim_locked_latency_s": float(np.nanmedian(fit["onset_from_stim_s"])) if fit["onset_from_stim_s"].notna().any() else np.nan,
            "peak_latency_s": float(np.nanmedian(fit["peak_from_stim_s"])) if fit["peak_from_stim_s"].notna().any() else np.nan,
            "opened_fraction": float(g["onset_step"].notna().mean()),
            "rt_part_s": rt_p,
            "rt_part_xcorr_s": float(rt.loc[eid, "rt_part_xcorr_s"]) if eid in rt.index else np.nan,
            "severity": rt.loc[eid, "severity"] if eid in rt.index else "",
            "speed_level_kmh": rt.loc[eid, "speed_level_kmh"] if eid in rt.index else np.nan,
            "subject": rt.loc[eid, "subject"] if eid in rt.index else "",
        })
    per_event = pd.DataFrame(per_event)
    per_event.to_csv(out_dir / "per_event.csv", index=False)

    def block(df_on: pd.DataFrame, df_ev: pd.DataFrame, kind: str) -> Dict:
        d = df_on[(df_on.kind == kind) & (df_on.offset_s >= a.fit_lo - 1e-9) & (df_on.offset_s <= a.fit_hi + 1e-9)]
        res = {"n_refs": int(d["ref_id"].nunique()),
               "opened_fraction": float(d["onset_step"].notna().mean()) if len(d) else np.nan,
               "pooled_within_slope": pooled_within_slope(d, "ref_id", "offset_s", "onset_from_stim_s"),
               "median_onset_from_origin_s": float(np.nanmedian(d["onset_from_origin_s"])) if len(d) else np.nan}
        boots = []
        refs = d["ref_id"].unique()
        rng = np.random.default_rng(1)
        for _ in range(1000 if len(refs) >= 3 else 0):
            pick = rng.choice(refs, len(refs), replace=True)
            dd = pd.concat([d[d.ref_id == r].assign(ref_id=f"{r}_{j}") for j, r in enumerate(pick)])
            boots.append(pooled_within_slope(dd, "ref_id", "offset_s", "onset_from_stim_s"))
        res["pooled_within_slope_CI95"] = ([float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
                                           if boots else [np.nan, np.nan])
        return res

    ev_block = block(ons, per_event, "event")
    if len(per_event):
        lat = per_event["stim_locked_latency_s"].to_numpy()
        rtp = per_event["rt_part_s"].to_numpy()
        ev_block.update({
            "median_stim_locked_latency_s": float(np.nanmedian(lat)),
            "median_stim_locked_latency_CI95": bootstrap_ci(lat),
            "median_measured_rt_part_s": float(np.nanmedian(rtp)),
            "median_latency_minus_rt_s": float(np.nanmedian(lat - rtp)),
            "median_latency_minus_rt_CI95": bootstrap_ci(lat - rtp),
            "spearman_latency_vs_rt": spearman(lat, rtp),
            "spearman_latency_vs_rt_xcorr": spearman(lat, per_event["rt_part_xcorr_s"].to_numpy()),
            "median_per_event_slope": float(np.nanmedian(per_event["slope_onset_vs_origin"])),
            "by_condition": per_event.groupby(["severity", "speed_level_kmh"])[
                ["stim_locked_latency_s", "rt_part_s"]].median().round(3).reset_index().to_dict("records"),
        })
    summary["events_F_to_P"] = ev_block
    summary["pseudo_F_to_P"] = block(ons, None, "pseudo")

    # ---------------- edge controls on event windows ----------------
    ctrl = {}
    ev_rows = samples[(samples.kind == "event") & (samples.offset_s >= a.fit_lo - 1e-9)
                      & (samples.offset_s <= a.fit_hi + 1e-9)]["_row"].to_numpy()
    for name in ["F->P", "P->F", "L->P", "L->F"]:
        pi = PAIR_NAMES.index(name)
        x = op[ev_rows, :, pi]
        ctrl[name] = {"mean_open_prob": float(np.nanmean(x)) if x.size else np.nan,
                      "fraction_steps_open": float(np.nanmean(x >= a.gate_threshold)) if x.size else np.nan}
    summary["edge_controls_event_windows"] = ctrl

    # ---------------- secondary: L->F with the leader onset ----------------
    ons_lead = onset_table(samples, op, "L->F", ["event_lead"], a)
    if len(ons_lead):
        summary["events_L_to_F"] = block(ons_lead.assign(kind="event_lead"), None, "event_lead")

    # ---------------- curves + plots ----------------
    cur_e = averaged_curves(samples, op, "F->P", "event")
    cur_p = averaged_curves(samples, op, "F->P", "pseudo")
    cur_e.to_csv(out_dir / "curves_events.csv", index=False)
    cur_p.to_csv(out_dir / "curves_pseudo.csv", index=False)
    make_plots(out_dir, cur_e, cur_p, per_event, sanity, a)

    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print_summary(summary)
    return summary


def make_plots(out_dir: Path, cur_e, cur_p, per_event, sanity, a) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    for cur, name in [(cur_e, "events"), (cur_p, "pseudo")]:
        if not len(cur):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        offs = sorted(cur.offset_s.unique())
        cmap = plt.get_cmap("viridis")
        for i, o in enumerate(offs):
            c = cur[cur.offset_s == o]
            col = cmap(i / max(len(offs) - 1, 1))
            axes[0].plot(c.t_from_stim_s, c.mean_open_prob, color=col, lw=1)
            axes[1].plot(c.t_from_origin_s, c.mean_open_prob, color=col, lw=1, label=f"{o:+.1f}")
        axes[0].set_xlabel("time since front-vehicle brake onset (s)")
        axes[1].set_xlabel("time since forecast origin (s)")
        axes[0].set_ylabel("mean open prob, F->P")
        axes[0].set_title("aligned to stimulus: curves overlap if stimulus-locked")
        axes[1].set_title("aligned to origin: curves overlap if origin-locked")
        axes[1].legend(title="origin offset (s)", fontsize=6, ncol=2)
        for ax in axes:
            ax.axhline(a.gate_threshold, color="0.6", ls=":")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"fp_gate_{name}.png", dpi=130)
        plt.close(fig)
    if len(per_event) and per_event["stim_locked_latency_s"].notna().any():
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(per_event.rt_part_s, per_event.stim_locked_latency_s, s=18)
        lim = [0, max(3.0, np.nanmax(per_event[["rt_part_s", "stim_locked_latency_s"]].to_numpy()) + 0.2)]
        ax.plot(lim, lim, color="0.6", ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("measured participant response delay (s)")
        ax.set_ylabel("gate latency after stimulus (s)")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "latency_vs_rt.png", dpi=130)
        plt.close(fig)
    if "active_ratio_by_step" in sanity:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        act = sanity["active_ratio_by_step"]
        ax.plot(np.arange(1, len(act) + 1) * DT, act)
        ax.set_xlabel("prediction horizon (s)")
        ax.set_ylabel("active edge ratio")
        ax.set_title("S2R windows (Fig. 2 analogue)")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "horizon_profile.png", dpi=130)
        plt.close(fig)


def print_summary(s: Dict) -> None:
    print("\n================ S2R zero-shot summary ================")
    sa = s.get("sanity", {})
    fmt = lambda v: "n/a" if v is None or not np.isfinite(v) else f"{v:.3f} m"
    print(f"ADE model {fmt(sa.get('ADE_model_m'))} | const. velocity {fmt(sa.get('ADE_const_velocity_m'))}"
          f" | no interaction {fmt(sa.get('ADE_no_interaction_m'))}")
    print(f"gate horizon profile peaks at {sa.get('active_ratio_peak_time_s', np.nan)} s")
    for key in ["events_F_to_P", "pseudo_F_to_P", "events_L_to_F"]:
        b = s.get(key)
        if not b:
            continue
        print(f"\n[{key}] refs={b.get('n_refs')} opened={b.get('opened_fraction', np.nan):.2f} "
              f"pooled slope={b.get('pooled_within_slope', np.nan):.2f} CI95={b.get('pooled_within_slope_CI95')}")
        if "median_stim_locked_latency_s" in b:
            print(f"  gate latency after stimulus: median {b['median_stim_locked_latency_s']:.2f} s "
                  f"CI95 {b['median_stim_locked_latency_CI95']}")
            print(f"  measured participant delay:  median {b['median_measured_rt_part_s']:.2f} s")
            print(f"  latency - delay: median {b['median_latency_minus_rt_s']:.2f} s CI95 {b['median_latency_minus_rt_CI95']}")
            print(f"  Spearman(latency, delay) = {b['spearman_latency_vs_rt']:.2f}")
    print("\nedge controls (event windows):")
    for k, v in s.get("edge_controls_event_windows", {}).items():
        print(f"  {k}: mean p={v['mean_open_prob']:.3f}, fraction open={v['fraction_steps_open']:.3f}")
    print("Reading: slope ~0 = stimulus-locked, slope ~1 = origin-locked.")


# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cte_table", default=None, help="NGSIM CTE table (default: from the checkpoint)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pred_steps", type=int, default=None)
    ap.add_argument("--offsets", default="-1.0:1.5:0.1", help="lo:hi:step, seconds around the stimulus")
    ap.add_argument("--sanity_stride", type=int, default=5)
    ap.add_argument("--no_interaction_check", action="store_true")
    ap.add_argument("--analyze_only", action="store_true")
    # analysis settings
    ap.add_argument("--onset_mode", choices=["abs", "rel"], default="abs")
    ap.add_argument("--gate_threshold", type=float, default=0.5)
    ap.add_argument("--rel_frac", type=float, default=0.5)
    ap.add_argument("--min_peak", type=float, default=0.05)
    ap.add_argument("--fit_lo", type=float, default=0.0)
    ap.add_argument("--fit_hi", type=float, default=1.0)
    ap.add_argument("--stop_before_response", type=int, default=1)
    a = ap.parse_args()

    prep_dir = Path(a.prep_dir)
    out_dir = Path(a.out_dir) if a.out_dir else prep_dir / "zero_shot"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not a.analyze_only:
        if not (a.repo and a.checkpoint):
            raise SystemExit("--repo and --checkpoint are required unless --analyze_only")
        run_inference(a, prep_dir, out_dir)
    analyze(prep_dir, out_dir, a)


if __name__ == "__main__":
    main()
