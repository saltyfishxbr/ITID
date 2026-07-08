# dynamic_cte_gnn.py

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import temperature

from hard_concrete import HardConcreteGate, HardConcreteConfig
from cte_lookup import (
    CTELookup,
    CTELookupConfig,
    LEVEL_DEFAULT,
)

# State feature order must match dataset.
IDX_X = 0
IDX_Y = 1
IDX_VX = 2
IDX_VY = 3
IDX_SPEED = 4
IDX_ACC = 5

LANE_SAME_FRONT = 0
LANE_SAME_BACK = 1
LANE_ADJACENT = 2

@dataclass
class DynamicCTEGNNConfig:
    # Data dimensions
    state_dim: int = 6
    history_dim: int = 6

    # Vehicle type
    # NGSIM valid type indices: 0,1,2; padding index: 3.
    num_type_embeddings: int = 4
    padding_type_index: int = 3
    type_emb_dim: int = 8

    # Network dimensions
    hidden_dim: int = 128
    edge_hidden_dim: int = 128
    rel_dim: int = 8
    message_dim: int = 128

    # GRU temporal encoder
    temporal_gru_layers: int = 1
    temporal_dropout: float = 0.0

    # Rollout
    pred_steps: int = 50
    dt: float = 0.1

    # Decoder output constraints
    max_delta_pos: float = 3.0
    max_delta_vel: float = 3.0

    # Message normalization
    lambda_self: float = 1.0
    norm_eps: float = 1e-6

    # Edge masking
    invalid_score_value: float = -20.0
    invalid_gate_logit_value: float = -20.0

    # CTE lookup
    cte_table_path: str = "./ProcessedDataset/NGSIM/0750am-0805am/ngsim_0750_0805_cte_table.pkl"

    # We follow your decision:
    # CTE does not enter edge feature; it only supervises score.
    use_cte_as_edge_feature: bool = False

    # We follow your decision:
    # use dynamic lookup based on predicted states.
    # For ablation/debug, "initial" can be added later if needed.
    cte_lookup_mode: str = "dynamic"

    # Hard Concrete
    hard_concrete_temperature: float = 2.0 / 3.0
    hard_concrete_gamma: float = -0.1
    hard_concrete_zeta: float = 1.1
    hard_concrete_eval_threshold: float = 0.5

    # Warm-up option. Later trainer can set this false/true.
    use_hard_concrete_gate: bool = True

    interaction_mode: str="normal"

    gate_init_open_prob: float=0.2


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    dropout: float = 0.0,
    activation: nn.Module = nn.ReLU(),
    final_activation: Optional[nn.Module] = None,
) -> nn.Sequential:
    layers = []

    if num_layers <= 1:
        layers.append(nn.Linear(input_dim, output_dim))
        if final_activation is not None:
            layers.append(final_activation)
        return nn.Sequential(*layers)

    dim_in = input_dim
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(dim_in, hidden_dim))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim_in = hidden_dim

    layers.append(nn.Linear(dim_in, output_dim))
    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)

def hard_concrete_logit_for_open_prob(
        open_prob: float,
        temperature: float,
        gamma: float,
        zeta: float,
        eps: float=1e-5,
)->float:
    p=float(open_prob)
    p=max(eps,min(1-eps,p))

    beta=float(temperature)
    gamma=float(gamma)
    zeta=float(zeta)

    if gamma>=0:
        raise ValueError(f"Hard Concrete gamma should be negative, got {gamma}")

    if zeta<=0:
        raise ValueError(f"Hard Concrete zeta should be positive, got {zeta}")

    logit_p=math.log(p/(1.0-p))
    offset=beta*math.log(-gamma/zeta)

    return float(logit_p+offset)

class TemporalEncoder(nn.Module):
    """
    GRU temporal encoder.

    Input:
        history: [B, N, Th, D]
        node_type: [B, N]

    Output:
        h0: [B, N, H]
    """

    def __init__(self, cfg: DynamicCTEGNNConfig):
        super().__init__()
        self.cfg = cfg

        self.gru = nn.GRU(
            input_size=cfg.history_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.temporal_gru_layers,
            batch_first=True,
            dropout=cfg.temporal_dropout if cfg.temporal_gru_layers > 1 else 0.0,
        )

        self.type_embedding = nn.Embedding(
            num_embeddings=cfg.num_type_embeddings,
            embedding_dim=cfg.type_emb_dim,
            padding_idx=cfg.padding_type_index,
        )

        self.type_proj = nn.Linear(cfg.type_emb_dim, cfg.hidden_dim)

        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        history: torch.Tensor,
        node_type: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, th, d = history.shape

        history_flat = history.reshape(b * n, th, d)

        _, h_last = self.gru(history_flat)
        h = h_last[-1]
        h = h.reshape(b, n, self.cfg.hidden_dim)

        type_emb = self.type_embedding(node_type)
        h = h + self.type_proj(type_emb)

        h = self.out_norm(h)

        h = h * node_mask.unsqueeze(-1).to(h.dtype)
        return h


class DynamicEdgeEncoder(nn.Module):
    """
    Build edge hidden representation from:
        h_source, h_receiver, dynamic relative features, source type emb, receiver type emb.

    CTE value is intentionally NOT used as edge input.
    """

    def __init__(self, cfg: DynamicCTEGNNConfig):
        super().__init__()
        self.cfg = cfg

        self.type_embedding = nn.Embedding(
            num_embeddings=cfg.num_type_embeddings,
            embedding_dim=cfg.type_emb_dim,
            padding_idx=cfg.padding_type_index,
        )

        input_dim = (
            cfg.hidden_dim * 2
            + cfg.rel_dim
            + cfg.type_emb_dim * 2
        )

        self.edge_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=cfg.edge_hidden_dim,
            output_dim=cfg.edge_hidden_dim,
            num_layers=3,
            dropout=0.0,
            activation=nn.ReLU(),
        )

        self.edge_norm = nn.LayerNorm(cfg.edge_hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        rel_feat: torch.Tensor,
        node_type: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, N, H]
            rel_feat: [B, N, N, F_rel]
            node_type: [B, N]
            edge_mask: [B, N, N]

        Direction:
            edge u -> v is indexed [u, v].
        """
        b, n, hdim = h.shape

        h_source = h.unsqueeze(2).expand(b, n, n, hdim)
        h_receiver = h.unsqueeze(1).expand(b, n, n, hdim)

        type_emb = self.type_embedding(node_type)
        type_source = type_emb.unsqueeze(2).expand(b, n, n, self.cfg.type_emb_dim)
        type_receiver = type_emb.unsqueeze(1).expand(b, n, n, self.cfg.type_emb_dim)

        edge_input = torch.cat(
            [
                h_source,
                h_receiver,
                rel_feat,
                type_source,
                type_receiver,
            ],
            dim=-1,
        )

        edge_hidden = self.edge_mlp(edge_input)
        edge_hidden = self.edge_norm(edge_hidden)

        edge_hidden = edge_hidden * edge_mask.unsqueeze(-1).to(edge_hidden.dtype)
        return edge_hidden


class MessagePassingLayer(nn.Module):
    """
    One directed message passing step.

    beta[u, v] controls source u -> receiver v.
    """

    def __init__(self, cfg: DynamicCTEGNNConfig):
        super().__init__()
        self.cfg = cfg

        self.message_mlp = build_mlp(
            input_dim=cfg.edge_hidden_dim,
            hidden_dim=cfg.message_dim,
            output_dim=cfg.message_dim,
            num_layers=2,
            dropout=0.0,
            activation=nn.ReLU(),
        )

        self.message_to_hidden = nn.Linear(cfg.message_dim, cfg.hidden_dim)

        self.updater = nn.GRUCell(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
        )

        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_hidden: torch.Tensor,
        beta: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, N, H]
            edge_hidden: [B, N, N, He]
            beta: [B, N, N]
            node_mask: [B, N]
            edge_mask: [B, N, N]

        Returns:
            h_next: [B, N, H]
        """
        b, n, hdim = h.shape

        beta = beta * edge_mask.to(beta.dtype)

        incoming_sum = beta.sum(dim=1, keepdim=True)
        pi = beta / (float(self.cfg.lambda_self) + incoming_sum + float(self.cfg.norm_eps))
        pi = pi * edge_mask.to(pi.dtype)

        edge_message = self.message_mlp(edge_hidden)
        weighted_message = pi.unsqueeze(-1) * edge_message

        # Sum over source dimension, aggregate into receiver.
        node_message = weighted_message.sum(dim=1)
        node_message = self.message_to_hidden(node_message)

        h_flat = h.reshape(b * n, hdim)
        msg_flat = node_message.reshape(b * n, hdim)

        h_next = self.updater(msg_flat, h_flat)
        h_next = h_next.reshape(b, n, hdim)

        h_next = self.out_norm(h_next)
        h_next = h_next * node_mask.unsqueeze(-1).to(h_next.dtype)

        return h_next


class TrajectoryDecoder(nn.Module):
    """
    Decode one-step state increment.

    Decoder raw output:
        [delta_x_raw, delta_y_raw, delta_vx_raw, delta_vy_raw]

    We apply tanh scaling for rollout stability.
    """

    def __init__(self, cfg: DynamicCTEGNNConfig):
        super().__init__()
        self.cfg = cfg

        self.mlp = build_mlp(
            input_dim=cfg.hidden_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=4,
            num_layers=3,
            dropout=0.0,
            activation=nn.ReLU(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(h)

        delta_pos = torch.tanh(raw[..., 0:2]) * float(self.cfg.max_delta_pos)
        delta_vel = torch.tanh(raw[..., 2:4]) * float(self.cfg.max_delta_vel)

        delta = torch.cat([delta_pos, delta_vel], dim=-1)
        return delta


class DynamicCTEGNN(nn.Module):
    """
    Dynamic CTE-supervised GNN.

    Core design:
        - Dataset provides complete directed graph over target + TopK neighbors.
        - Hard Concrete learns edge existence.
        - Score head learns CTE-calibrated interaction intensity.
        - CTE does not enter edge feature.
        - Dynamic CTE teacher is obtained by lookup based on predicted states.
        - Node hidden h is updated by GNN recurrently, not by re-encoding predicted states.
    """

    def __init__(self, cfg: DynamicCTEGNNConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = TemporalEncoder(cfg)
        self.edge_encoder = DynamicEdgeEncoder(cfg)

        self.gate_head = nn.Linear(cfg.edge_hidden_dim, 1)
        self.score_head = nn.Linear(cfg.edge_hidden_dim, 1)

        self.initialize_gate_head()

        hc_cfg = HardConcreteConfig(
            temperature=cfg.hard_concrete_temperature,
            gamma=cfg.hard_concrete_gamma,
            zeta=cfg.hard_concrete_zeta,
            eval_threshold=cfg.hard_concrete_eval_threshold,
        )
        self.hard_concrete = HardConcreteGate(hc_cfg)

        self.message_layer = MessagePassingLayer(cfg)
        self.decoder = TrajectoryDecoder(cfg)

        self.cte_lookup = CTELookup(
            CTELookupConfig(cte_table_path=cfg.cte_table_path)
        )

    def initialize_gate_head(self)->None:
        bias_value=hard_concrete_logit_for_open_prob(
            open_prob=float(self.cfg.gate_init_open_prob),
            temperature=float(self.cfg.hard_concrete_temperature),
            gamma=float(self.cfg.hard_concrete_gamma),
            zeta=float(self.cfg.hard_concrete_zeta),
        )
        with torch.no_grad():
            nn.init.normal_(self.gate_head.weight,mean=0.0,std=1e-3)
            self.gate_head.bias.fill_(bias_value)

    def build_dynamic_relative_features(
        self,
        state: torch.Tensor,
        edge_lane_relation: torch.Tensor,
        density0: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Build dynamic edge relative features from predicted state.

        Args:
            state: [B, N, 6]
            edge_lane_relation: [B, N, N]
            density0: [B]
            edge_mask: [B, N, N]

        Returns:
            rel_feat: [B, N, N, rel_dim]
            delta_y: [B, N, N]
            delta_speed: [B, N, N]
            ttc_norm: [B, N, N]
        """
        b, n, _ = state.shape
        device = state.device
        dtype = state.dtype

        x = state[..., IDX_X]
        y = state[..., IDX_Y]
        speed = state[..., IDX_SPEED]

        x_source = x.unsqueeze(2)
        x_receiver = x.unsqueeze(1)
        y_source = y.unsqueeze(2)
        y_receiver = y.unsqueeze(1)
        speed_source = speed.unsqueeze(2)
        speed_receiver = speed.unsqueeze(1)

        delta_x = x_source - x_receiver
        delta_y = y_source - y_receiver
        delta_speed = speed_source - speed_receiver

        closing_speed = speed_receiver - speed_source

        max_ttc = 20.0
        ttc = torch.full_like(delta_y, fill_value=max_ttc)

        valid_ttc = (delta_y > 0.0) & (closing_speed > 1e-3)
        ttc_value = delta_y / torch.clamp(closing_speed, min=1e-3)
        ttc_value = torch.clamp(ttc_value, min=0.0, max=max_ttc)
        ttc = torch.where(valid_ttc, ttc_value, ttc)

        ttc_norm = ttc / max_ttc

        lane_rel = edge_lane_relation.long()

        same_front = (lane_rel == LANE_SAME_FRONT).to(dtype)
        same_back = (lane_rel == LANE_SAME_BACK).to(dtype)
        adjacent = (lane_rel == LANE_ADJACENT).to(dtype)

        rho = density0.view(b, 1, 1).expand(b, n, n).to(dtype)

        rel_feat = torch.stack(
            [
                delta_x,
                delta_y,
                delta_speed,
                ttc_norm,
                same_front,
                same_back,
                adjacent,
                rho,
            ],
            dim=-1,
        )

        rel_feat = rel_feat * edge_mask.unsqueeze(-1).to(dtype)

        return {
            "rel_feat": rel_feat,
            "delta_y": delta_y,
            "delta_speed": delta_speed,
            "ttc_norm": ttc_norm,
        }

    def lookup_dynamic_cte_teacher(
        self,
        delta_y: torch.Tensor,
        delta_speed: torch.Tensor,
        edge_lane_relation: torch.Tensor,
        density0: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Dynamic CTE lookup based on current predicted state.

        CTE teacher is not used as edge input.
        It is returned for loss computation.
        """
        b, n, _ = delta_y.shape
        rho = density0.view(b, 1, 1).expand(b, n, n)

        detail = self.cte_lookup.lookup_torch_detail(
            delta_y=delta_y,
            delta_speed=delta_speed,
            lane_rel=edge_lane_relation,
            rho=rho,
        )

        cte = detail["cte"] * edge_mask.to(delta_y.dtype)
        valid = detail["valid"] & edge_mask
        hit = detail["hit"] & edge_mask
        significant = detail["is_significant"] & edge_mask
        level_code = detail["level_code"].masked_fill(~edge_mask, LEVEL_DEFAULT)

        return {
            "cte": cte,
            "valid": valid,
            "hit": hit,
            "significant": significant,
            "level_code": level_code,
        }

    def update_state(
        self,
        state: torch.Tensor,
        delta: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update predicted physical state.

        State layout:
            [x, y, vx, vy, speed, acc]
        """
        dt = float(self.cfg.dt)

        x = state[..., IDX_X]
        y = state[..., IDX_Y]
        vx = state[..., IDX_VX]
        vy = state[..., IDX_VY]
        speed = state[..., IDX_SPEED]

        dx = delta[..., 0]
        dy = delta[..., 1]
        dvx = delta[..., 2]
        dvy = delta[..., 3]

        x_next = x + dx
        y_next = y + dy
        vx_next = vx + dvx
        vy_next = vy + dvy

        speed_next = torch.sqrt(vx_next ** 2 + vy_next ** 2 + 1e-8)
        acc_next = (speed_next - speed) / dt

        next_state = torch.stack(
            [
                x_next,
                y_next,
                vx_next,
                vy_next,
                speed_next,
                acc_next,
            ],
            dim=-1,
        )

        next_state = next_state * node_mask.unsqueeze(-1).to(next_state.dtype)
        return next_state

    def applY_interaction_mode(
            self,
            gates: torch.Tensor,
            open_probs: torch.Tensor,
            alphas: torch.Tensor,
            edge_mask: torch.Tensor,
            dtype: torch.dtype,
    )->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode=str(self.cfg.interaction_mode)

        if mode=="normal":
            betas=gates*alphas*edge_mask.to(dtype)
            return gates,open_probs,betas

        if mode=="no_interaction":
            betas=torch.zeros_like(alphas)
            return gates,open_probs,betas

        if mode=="all_open":
            gates=edge_mask.to(dtype)
            open_probs=edge_mask.to(dtype)
            betas=alphas*edge_mask.to(dtype)
            return gates,open_probs,betas

        raise ValueError(
            f"Unknown interaction_mode: {mode}"
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        pred_steps: Optional[int] = None,
        deterministic_gate: Optional[bool] = None,
        edge_drop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch:
                From NGSIMCTEDataset.

            pred_steps:
                Number of rollout steps. If None, cfg.pred_steps is used.

            deterministic_gate:
                If None, HardConcrete uses module training/eval mode.
                If True, deterministic hard gate.
                If False, relaxed stochastic gate.

        Returns:
            Dictionary with predictions and edge diagnostics.
        """
        history = batch["history"]
        node_type = batch["node_type"]
        node_mask = batch["node_mask"]
        edge_mask = batch["edge_mask"]
        edge_lane_relation = batch["edge_lane_relation"]
        density0 = batch["density0"]

        device = history.device
        dtype = history.dtype

        b, n, th, d = history.shape
        t_steps = int(pred_steps if pred_steps is not None else self.cfg.pred_steps)

        node_mask = node_mask.bool()
        edge_mask = edge_mask.bool()

        # Enforce node-pair mask in case dataset edge_mask is incomplete.
        pair_mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        non_self = ~torch.eye(n, dtype=torch.bool, device=device).unsqueeze(0)
        edge_mask = edge_mask & pair_mask & non_self

        if edge_drop_mask is not None:
            edge_drop_mask = edge_drop_mask.to(device=device).bool()

            expected_shape = (b, t_steps, n, n)
            if tuple(edge_drop_mask.shape) != expected_shape:
                raise ValueError(
                    f"edge_drop_mask shape mismatch. "
                    f"Expected {expected_shape}, got {tuple(edge_drop_mask.shape)}"
                )

            edge_drop_mask = edge_drop_mask & edge_mask.unsqueeze(1)

        h = self.encoder(
            history=history,
            node_type=node_type,
            node_mask=node_mask,
        )

        state = history[:, :, -1, :].clone()
        state = state * node_mask.unsqueeze(-1).to(dtype)

        pred_states = []

        all_scores = []
        all_alphas = []
        all_gate_logits = []
        all_gates = []
        all_open_probs = []
        all_betas = []

        all_cte_teacher = []
        all_cte_valid = []
        all_cte_hit = []
        all_cte_significant = []
        all_cte_level = []

        for _ in range(t_steps):
            rel_out = self.build_dynamic_relative_features(
                state=state,
                edge_lane_relation=edge_lane_relation,
                density0=density0,
                edge_mask=edge_mask,
            )

            rel_feat = rel_out["rel_feat"]
            delta_y = rel_out["delta_y"]
            delta_speed = rel_out["delta_speed"]

            cte_out = self.lookup_dynamic_cte_teacher(
                delta_y=delta_y,
                delta_speed=delta_speed,
                edge_lane_relation=edge_lane_relation,
                density0=density0,
                edge_mask=edge_mask,
            )

            edge_hidden = self.edge_encoder(
                h=h,
                rel_feat=rel_feat,
                node_type=node_type,
                edge_mask=edge_mask,
            )

            gate_logits = self.gate_head(edge_hidden).squeeze(-1)
            scores = self.score_head(edge_hidden).squeeze(-1)

            gate_logits = gate_logits.masked_fill(
                ~edge_mask,
                float(self.cfg.invalid_gate_logit_value),
            )
            scores = scores.masked_fill(
                ~edge_mask,
                float(self.cfg.invalid_score_value),
            )

            if self.cfg.use_hard_concrete_gate:
                gate_out = self.hard_concrete(
                    log_alpha=gate_logits,
                    mask=edge_mask,
                    deterministic=deterministic_gate,
                )
                gates = gate_out["gate"]
                open_probs = gate_out["open_prob"]
            else:
                gates = edge_mask.to(dtype)
                open_probs = edge_mask.to(dtype)

            alphas = torch.sigmoid(scores) * edge_mask.to(dtype)
            gates,open_probs,betas=self.applY_interaction_mode(
                gates=gates,
                open_probs=open_probs,
                alphas=alphas,
                edge_mask=edge_mask,
                dtype=dtype,
            )

            if edge_drop_mask is not None:
                step_drop_mask = edge_drop_mask[:, len(pred_states)]  # [B, N, N]
                step_drop_mask = step_drop_mask & edge_mask

                betas = betas.masked_fill(step_drop_mask, 0.0)
                gates = gates.masked_fill(step_drop_mask, 0.0)

            h_next = self.message_layer(
                h=h,
                edge_hidden=edge_hidden,
                beta=betas,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )

            delta_state = self.decoder(h_next)
            state_next = self.update_state(
                state=state,
                delta=delta_state,
                node_mask=node_mask,
            )

            pred_states.append(state_next)

            all_scores.append(scores)
            all_alphas.append(alphas)
            all_gate_logits.append(gate_logits)
            all_gates.append(gates)
            all_open_probs.append(open_probs)
            all_betas.append(betas)

            all_cte_teacher.append(cte_out["cte"])
            all_cte_valid.append(cte_out["valid"])
            all_cte_hit.append(cte_out["hit"])
            all_cte_significant.append(cte_out["significant"])
            all_cte_level.append(cte_out["level_code"])

            h = h_next
            state = state_next

        out = {
            "pred_states": torch.stack(pred_states, dim=1),          # [B, T, N, 6]

            "scores": torch.stack(all_scores, dim=1),                # [B, T, N, N]
            "alphas": torch.stack(all_alphas, dim=1),                # [B, T, N, N]
            "gate_logits": torch.stack(all_gate_logits, dim=1),      # [B, T, N, N]
            "gates": torch.stack(all_gates, dim=1),                  # [B, T, N, N]
            "open_probs": torch.stack(all_open_probs, dim=1),        # [B, T, N, N]
            "betas": torch.stack(all_betas, dim=1),                  # [B, T, N, N]

            "cte_teacher": torch.stack(all_cte_teacher, dim=1),      # [B, T, N, N]
            "cte_valid": torch.stack(all_cte_valid, dim=1),          # [B, T, N, N]
            "cte_hit": torch.stack(all_cte_hit, dim=1),              # [B, T, N, N]
            "cte_significant": torch.stack(all_cte_significant, dim=1),
            "cte_level": torch.stack(all_cte_level, dim=1),          # [B, T, N, N]

            "edge_mask": edge_mask,
            "node_mask": node_mask,
        }

        return out


def main() -> None:
    """
    Simple PyCharm smoke test.

    This assumes you already built:
        - ngsim_0750_0805_windows_compact.pkl
        - ngsim_0750_0805_cte_table.pkl
    """

    from ngsim_dataset import NGSIMCTEDataset, NGSIMDatasetConfig, simple_collate_fn

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

    edge_mask_t=out["edge_mask"].unsqueeze(1).expand_as(out["open_probs"])
    valid_open_probs=out["open_probs"][edge_mask_t]

    print("[gate init check] target open prob:",model_cfg.gate_init_open_prob)
    print("[gate init check] valid open prob mean:", float(valid_open_probs.mean()))
    print("[gate init check] valid open prob min:", float(valid_open_probs.min()))
    print("[gate init check] valid open prob max:", float(valid_open_probs.max()))

    print("[model output shapes]")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype)
        else:
            print(k, type(v))


if __name__ == "__main__":
    main()