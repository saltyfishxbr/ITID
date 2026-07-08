# cte_lookup.py

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pickle

import numpy as np
import torch


LANE_SAME_FRONT = 0
LANE_SAME_BACK = 1
LANE_ADJACENT = 2

LEVEL_DEFAULT = -1
LEVEL_GLOBAL = 0
LEVEL_LANE_RHO = 1
LEVEL_NO_DV = 2
LEVEL_FINE = 3


@dataclass
class CTELookupConfig:
    cte_table_path: str = "./ProcessedDataset/NGSIM/0750am-0805am/ngsim_0750_0805_cte_table.pkl"
    default_value: float = 0.0


def load_pickle(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cannot find file: {p}")

    with open(p, "rb") as f:
        return pickle.load(f)


def bin_index_python(value: float, bins: np.ndarray) -> int:
    """
    Match numpy searchsorted side='right' convention.

    For bins [-inf, b1, b2, ..., inf], boundary values go to the right bin.
    """
    idx = int(np.searchsorted(bins, value, side="right") - 1)
    idx = max(0, min(idx, len(bins) - 2))
    return idx


def torch_bin_index(values: torch.Tensor, bins: np.ndarray) -> torch.Tensor:
    """
    Torch version of bin_index_python.

    For full bins:
        [-inf, b1, b2, ..., inf]

    We bucketize over internal boundaries [b1, b2, ...] with right=True,
    so values exactly equal to a boundary go to the right bin.
    """
    device = values.device
    internal = bins[1:-1]
    boundaries = torch.as_tensor(internal, dtype=values.dtype, device=device)

    idx = torch.bucketize(values, boundaries, right=True)
    return idx.long()


class CTELookup:
    """
    Query train-estimated context-level CTE table.

    Context:
        c = bin(delta_y) x bin(delta_speed) x lane_relation x bin(rho)

    Fallback order:
        fine -> no_dv -> lane_rho -> global -> default

    Level codes:
        fine:     3
        no_dv:    2
        lane_rho: 1
        global:   0
        default: -1
    """

    def __init__(self, config: CTELookupConfig):
        self.config = config
        self.package = load_pickle(config.cte_table_path)

        self.table: Dict[Tuple, Dict[str, Any]] = self.package["table"]
        context = self.package["context"]

        self.d_bins = np.asarray(context["D_BINS"], dtype=np.float32)
        self.dv_bins = np.asarray(context["DV_BINS"], dtype=np.float32)
        self.rho_bins = np.asarray(context["RHO_BINS"], dtype=np.float32)

        self.num_d_bins = len(self.d_bins) - 1
        self.num_dv_bins = len(self.dv_bins) - 1
        self.num_rho_bins = len(self.rho_bins) - 1
        self.num_lane_rel = 3

        self.default_value = float(config.default_value)

        self._dense_cache: Dict[str, Dict[str, torch.Tensor]] = {}

        self._build_dense_tables()

    def _build_dense_tables(self) -> None:
        self.fine_np = np.full(
            (self.num_d_bins, self.num_dv_bins, self.num_lane_rel, self.num_rho_bins),
            np.nan,
            dtype=np.float32,
        )

        self.no_dv_np = np.full(
            (self.num_d_bins, self.num_lane_rel, self.num_rho_bins),
            np.nan,
            dtype=np.float32,
        )

        self.lane_rho_np = np.full(
            (self.num_lane_rel, self.num_rho_bins),
            np.nan,
            dtype=np.float32,
        )

        self.fine_sig_np = np.zeros_like(self.fine_np, dtype=np.bool_)
        self.no_dv_sig_np = np.zeros_like(self.no_dv_np, dtype=np.bool_)
        self.lane_rho_sig_np = np.zeros_like(self.lane_rho_np, dtype=np.bool_)

        self.global_value_np = np.nan
        self.global_significant_np = False

        for key, value in self.table.items():
            cte_value = float(value.get("cte", self.default_value))
            is_significant = bool(value.get("is_significant", False))

            if len(key) == 0:
                continue

            level = key[0]

            if level == "fine":
                _, d_bin, dv_bin, lane_rel, rho_bin = key
                if self._valid_fine_indices(d_bin, dv_bin, lane_rel, rho_bin):
                    self.fine_np[d_bin, dv_bin, lane_rel, rho_bin] = cte_value
                    self.fine_sig_np[d_bin, dv_bin, lane_rel, rho_bin] = is_significant

            elif level == "no_dv":
                _, d_bin, lane_rel, rho_bin = key
                if self._valid_no_dv_indices(d_bin, lane_rel, rho_bin):
                    self.no_dv_np[d_bin, lane_rel, rho_bin] = cte_value
                    self.no_dv_sig_np[d_bin, lane_rel, rho_bin] = is_significant

            elif level == "lane_rho":
                _, lane_rel, rho_bin = key
                if self._valid_lane_rho_indices(lane_rel, rho_bin):
                    self.lane_rho_np[lane_rel, rho_bin] = cte_value
                    self.lane_rho_sig_np[lane_rel, rho_bin] = is_significant

            elif level == "global":
                self.global_value_np = cte_value
                self.global_significant_np = is_significant

        if not np.isfinite(self.global_value_np):
            self.global_value_np = self.default_value
            self.global_significant_np = False

    def _valid_fine_indices(self, d_bin: int, dv_bin: int, lane_rel: int, rho_bin: int) -> bool:
        return (
            0 <= d_bin < self.num_d_bins
            and 0 <= dv_bin < self.num_dv_bins
            and 0 <= lane_rel < self.num_lane_rel
            and 0 <= rho_bin < self.num_rho_bins
        )

    def _valid_no_dv_indices(self, d_bin: int, lane_rel: int, rho_bin: int) -> bool:
        return (
            0 <= d_bin < self.num_d_bins
            and 0 <= lane_rel < self.num_lane_rel
            and 0 <= rho_bin < self.num_rho_bins
        )

    def _valid_lane_rho_indices(self, lane_rel: int, rho_bin: int) -> bool:
        return (
            0 <= lane_rel < self.num_lane_rel
            and 0 <= rho_bin < self.num_rho_bins
        )

    def make_context_keys_python(
        self,
        delta_y: float,
        delta_speed: float,
        lane_rel: int,
        rho: float,
    ) -> List[Tuple]:
        d_bin = bin_index_python(delta_y, self.d_bins)
        dv_bin = bin_index_python(delta_speed, self.dv_bins)
        rho_bin = bin_index_python(rho, self.rho_bins)

        return [
            ("fine", d_bin, dv_bin, int(lane_rel), rho_bin),
            ("no_dv", d_bin, int(lane_rel), rho_bin),
            ("lane_rho", int(lane_rel), rho_bin),
            ("global",),
        ]

    @staticmethod
    def level_name_to_code(level_name: str) -> int:
        if level_name == "fine":
            return LEVEL_FINE
        if level_name == "no_dv":
            return LEVEL_NO_DV
        if level_name == "lane_rho":
            return LEVEL_LANE_RHO
        if level_name == "global":
            return LEVEL_GLOBAL
        return LEVEL_DEFAULT

    def lookup_python_detail(
        self,
        delta_y: float,
        delta_speed: float,
        lane_rel: int,
        rho: float,
    ) -> Dict[str, Any]:
        """
        Detailed scalar lookup.

        Returns:
            cte: float
            hit: whether any table entry is found, including global
            valid: whether non-global context-level table entry is found
            level: level name
            level_code: int
            is_significant: significance flag stored in table
        """
        lane_rel = int(lane_rel)

        if lane_rel < 0 or lane_rel >= self.num_lane_rel:
            return {
                "cte": self.default_value,
                "hit": False,
                "valid": False,
                "level": "default",
                "level_code": LEVEL_DEFAULT,
                "is_significant": False,
            }

        keys = self.make_context_keys_python(
            delta_y=delta_y,
            delta_speed=delta_speed,
            lane_rel=lane_rel,
            rho=rho,
        )

        for key in keys:
            if key in self.table:
                entry = self.table[key]
                level_name = key[0]
                level_code = self.level_name_to_code(level_name)

                return {
                    "cte": float(entry.get("cte", self.default_value)),
                    "hit": True,
                    "valid": level_code >= LEVEL_LANE_RHO,
                    "level": level_name,
                    "level_code": level_code,
                    "is_significant": bool(entry.get("is_significant", False)),
                }

        return {
            "cte": self.default_value,
            "hit": False,
            "valid": False,
            "level": "default",
            "level_code": LEVEL_DEFAULT,
            "is_significant": False,
        }

    def lookup_python(
        self,
        delta_y: float,
        delta_speed: float,
        lane_rel: int,
        rho: float,
    ) -> float:
        return float(
            self.lookup_python_detail(
                delta_y=delta_y,
                delta_speed=delta_speed,
                lane_rel=lane_rel,
                rho=rho,
            )["cte"]
        )

    def _get_dense_torch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Lazy cache dense tensors by device.

        This avoids repeated numpy -> torch -> GPU transfers in training loops.
        """
        key = str(device)

        if key not in self._dense_cache:
            self._dense_cache[key] = {
                "fine": torch.as_tensor(self.fine_np, dtype=torch.float32, device=device),
                "no_dv": torch.as_tensor(self.no_dv_np, dtype=torch.float32, device=device),
                "lane_rho": torch.as_tensor(self.lane_rho_np, dtype=torch.float32, device=device),
                "global": torch.tensor(float(self.global_value_np), dtype=torch.float32, device=device),

                "fine_sig": torch.as_tensor(self.fine_sig_np, dtype=torch.bool, device=device),
                "no_dv_sig": torch.as_tensor(self.no_dv_sig_np, dtype=torch.bool, device=device),
                "lane_rho_sig": torch.as_tensor(self.lane_rho_sig_np, dtype=torch.bool, device=device),
                "global_sig": torch.tensor(bool(self.global_significant_np), dtype=torch.bool, device=device),
            }

        return self._dense_cache[key]

    def clear_cache(self) -> None:
        self._dense_cache.clear()

    def lookup_torch_detail(
        self,
        delta_y: torch.Tensor,
        delta_speed: torch.Tensor,
        lane_rel: torch.Tensor,
        rho: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Batched torch lookup with details.

        Returns:
            cte: same broadcasted shape
            hit: True for fine/no_dv/lane_rho/global
            valid: True only for fine/no_dv/lane_rho
            level_code: fine=3, no_dv=2, lane_rho=1, global=0, default=-1
            is_significant: table significance flag for selected fallback level
        """
        device = delta_y.device

        delta_y, delta_speed, lane_rel, rho = torch.broadcast_tensors(
            delta_y,
            delta_speed,
            lane_rel.to(device),
            rho.to(device),
        )

        d_bin = torch_bin_index(delta_y, self.d_bins)
        dv_bin = torch_bin_index(delta_speed, self.dv_bins)
        rho_bin = torch_bin_index(rho, self.rho_bins)

        lane_rel = lane_rel.long()
        valid_lane = (lane_rel >= 0) & (lane_rel < self.num_lane_rel)
        lane_safe = torch.clamp(lane_rel, min=0, max=self.num_lane_rel - 1)

        dense = self._get_dense_torch(device)

        fine = dense["fine"][d_bin, dv_bin, lane_safe, rho_bin]
        no_dv = dense["no_dv"][d_bin, lane_safe, rho_bin]
        lane_rho = dense["lane_rho"][lane_safe, rho_bin]
        global_value = dense["global"]

        fine_sig = dense["fine_sig"][d_bin, dv_bin, lane_safe, rho_bin]
        no_dv_sig = dense["no_dv_sig"][d_bin, lane_safe, rho_bin]
        lane_rho_sig = dense["lane_rho_sig"][lane_safe, rho_bin]
        global_sig = dense["global_sig"].expand_as(fine_sig)

        has_fine = ~torch.isnan(fine)
        has_no_dv = ~torch.isnan(no_dv)
        has_lane_rho = ~torch.isnan(lane_rho)

        cte = fine
        level_code = torch.full_like(lane_safe, fill_value=LEVEL_FINE, dtype=torch.long)
        is_significant = fine_sig

        use_no_dv = ~has_fine & has_no_dv
        cte = torch.where(use_no_dv, no_dv, cte)
        level_code = torch.where(use_no_dv, torch.full_like(level_code, LEVEL_NO_DV), level_code)
        is_significant = torch.where(use_no_dv, no_dv_sig, is_significant)

        use_lane_rho = ~has_fine & ~has_no_dv & has_lane_rho
        cte = torch.where(use_lane_rho, lane_rho, cte)
        level_code = torch.where(use_lane_rho, torch.full_like(level_code, LEVEL_LANE_RHO), level_code)
        is_significant = torch.where(use_lane_rho, lane_rho_sig, is_significant)

        use_global = ~has_fine & ~has_no_dv & ~has_lane_rho
        cte = torch.where(use_global, global_value.expand_as(cte), cte)
        level_code = torch.where(use_global, torch.full_like(level_code, LEVEL_GLOBAL), level_code)
        is_significant = torch.where(use_global, global_sig, is_significant)

        cte = torch.where(torch.isnan(cte), torch.full_like(cte, self.default_value), cte)

        # Invalid lane is always default.
        cte = torch.where(valid_lane, cte, torch.full_like(cte, self.default_value))
        level_code = torch.where(valid_lane, level_code, torch.full_like(level_code, LEVEL_DEFAULT))
        is_significant = torch.where(valid_lane, is_significant, torch.zeros_like(is_significant))

        hit = level_code >= LEVEL_GLOBAL
        valid = level_code >= LEVEL_LANE_RHO

        return {
            "cte": cte,
            "hit": hit,
            "valid": valid,
            "level_code": level_code,
            "is_significant": is_significant,
        }

    def lookup_torch(
        self,
        delta_y: torch.Tensor,
        delta_speed: torch.Tensor,
        lane_rel: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        return self.lookup_torch_detail(
            delta_y=delta_y,
            delta_speed=delta_speed,
            lane_rel=lane_rel,
            rho=rho,
        )["cte"]

    def summary(self) -> None:
        fine_count = int(np.isfinite(self.fine_np).sum())
        no_dv_count = int(np.isfinite(self.no_dv_np).sum())
        lane_rho_count = int(np.isfinite(self.lane_rho_np).sum())

        print("[CTELookup] table path:", self.config.cte_table_path)
        print("[CTELookup] fine entries:", fine_count)
        print("[CTELookup] no_dv entries:", no_dv_count)
        print("[CTELookup] lane_rho entries:", lane_rho_count)
        print("[CTELookup] global value:", float(self.global_value_np))
        print("[CTELookup] D bins:", self.d_bins)
        print("[CTELookup] DV bins:", self.dv_bins)
        print("[CTELookup] RHO bins:", self.rho_bins)


def lane_relation_python(source_lane: int, receiver_lane: int, delta_y: float) -> int:
    """
    Directed lane relation for source -> receiver.

    same lane:
        source ahead of receiver: LANE_SAME_FRONT
        source behind receiver: LANE_SAME_BACK
    adjacent lane:
        LANE_ADJACENT
    invalid:
        -1
    """
    lane_diff = int(source_lane) - int(receiver_lane)

    if lane_diff == 0:
        if delta_y >= 0:
            return LANE_SAME_FRONT
        return LANE_SAME_BACK

    if abs(lane_diff) == 1:
        return LANE_ADJACENT

    return -1


def lane_relation_torch(
    source_lane: torch.Tensor,
    receiver_lane: torch.Tensor,
    delta_y: torch.Tensor,
) -> torch.Tensor:
    source_lane, receiver_lane, delta_y = torch.broadcast_tensors(
        source_lane,
        receiver_lane,
        delta_y,
    )

    lane_diff = source_lane.long() - receiver_lane.long()

    out = torch.full_like(lane_diff, fill_value=-1, dtype=torch.long)

    same = lane_diff == 0
    adjacent = lane_diff.abs() == 1

    out = torch.where(same & (delta_y >= 0), torch.zeros_like(out), out)
    out = torch.where(same & (delta_y < 0), torch.ones_like(out), out)
    out = torch.where(adjacent, torch.full_like(out, LANE_ADJACENT), out)

    return out


def test_bin_consistency(lookup: CTELookup) -> None:
    test_sets = [
        ("D_BINS", lookup.d_bins, [-100.0, -40.0, -15.0, 15.0, 40.0, 100.0]),
        ("DV_BINS", lookup.dv_bins, [-10.0, -5.0, -2.0, 2.0, 5.0, 10.0]),
        ("RHO_BINS", lookup.rho_bins, [0.0, 0.025, 0.060, 0.1]),
    ]

    for name, bins, values in test_sets:
        values_np = np.asarray(values, dtype=np.float32)
        values_torch = torch.tensor(values_np, dtype=torch.float32)

        py_bins = [bin_index_python(float(v), bins) for v in values_np]
        torch_bins = torch_bin_index(values_torch, bins).cpu().numpy().tolist()

        print(f"[test_bin_consistency] {name}")
        print("  values:", values)
        print("  python:", py_bins)
        print("  torch: ", torch_bins)

        assert py_bins == torch_bins, (
            f"Bin mismatch for {name}: python={py_bins}, torch={torch_bins}"
        )

    print("[test_bin_consistency] passed.")


def main() -> None:
    cfg = CTELookupConfig()
    lookup = CTELookup(cfg)
    lookup.summary()

    test_bin_consistency(lookup)

    value_detail = lookup.lookup_python_detail(
        delta_y=20.0,
        delta_speed=-1.0,
        lane_rel=LANE_SAME_FRONT,
        rho=0.04,
    )
    print("[test] python detail:", value_detail)

    dy = torch.tensor([[20.0, -10.0]])
    dv = torch.tensor([[-1.0, 2.0]])
    lane = torch.tensor([[0, 1]])
    rho = torch.tensor([[0.04, 0.04]])

    detail = lookup.lookup_torch_detail(dy, dv, lane, rho)
    print("[test] torch cte:", detail["cte"])
    print("[test] torch valid:", detail["valid"])
    print("[test] torch level_code:", detail["level_code"])


if __name__ == "__main__":
    main()