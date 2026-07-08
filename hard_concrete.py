from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn


@dataclass
class HardConcreteConfig:
    temperature: float = 2.0 / 3.0
    gamma: float = -0.1
    zeta: float = 1.1

    # In evaluation mode, use deterministic hard gate:
    # gate = 1[p_open > threshold]
    eval_threshold: float = 0.5

    # Clamp random uniform sample for numerical stability.
    eps: float = 1e-6

class HardConcreteGate(nn.Module):
    """
    Hard Concrete gate for differentiable L0-style edge selection.

    Input:
        log_alpha: arbitrary shape, typically [B, N, N]

    Output:
        {
            "gate":      sampled or deterministic gate, same shape
            "open_prob": P(gate > 0), same shape
        }

    Training:
        gate is a relaxed sample in [0, 1].

    Evaluation:
        gate is deterministic hard binary gate based on open_prob.
    """
    def __init__(self, config: HardConcreteConfig) -> None:
        super().__init__()
        self.config = config

    def open_probability(self,log_alpha: torch.Tensor) -> torch.Tensor:
        """
        P(g > 0) for stretched hard concrete.

        p_open = sigmoid(log_alpha - beta * log(-gamma / zeta))
        """
        beta=float(self.config.temperature)
        gamma=float(self.config.gamma)
        zeta=float(self.config.zeta)

        offset=beta*torch.log(
            torch.tensor(-gamma/zeta,dtype=log_alpha.dtype,device=log_alpha.device)
        )

        return torch.sigmoid(log_alpha-offset)

    def sample_gate(self,log_alpha: torch.Tensor) -> torch.Tensor:
        beta=float(self.config.temperature)
        gamma=float(self.config.gamma)
        zeta=float(self.config.zeta)
        eps=float(self.config.eps)

        u=torch.rand_like(log_alpha)
        u=torch.clamp(u,min=eps,max=1.0-eps)

        logistic_noise=torch.log(u)-torch.log1p(-u)

        s=torch.sigmoid((logistic_noise+log_alpha)/beta)
        s_bar=s*(zeta-gamma)+gamma

        gate=torch.clamp(s_bar,min=0.0,max=1.0)
        return gate

    def deterministic_gate(self,log_alpha: torch.Tensor) -> torch.Tensor:
        open_prob=self.open_probability(log_alpha)
        gate=(open_prob>float(self.config.eval_threshold)).to(log_alpha.dtype)
        return gate

    def forward(
            self,
            log_alpha: torch.Tensor,
            mask:Optional[torch.Tensor] = None,
            deterministic: Optional[bool] = None,
    )->Dict[str, torch.Tensor]:
        """
        Args:
            log_alpha:
                Gate logits, arbitrary shape.

            mask:
                Optional bool or float mask, same broadcastable shape as log_alpha.
                Invalid positions are forced to zero gate and zero open probability.

            deterministic:
                If None, use not self.training.
                If True, use deterministic hard gate.
                If False, sample relaxed gate.

        Returns:
            gate:
                Relaxed or hard gate.

            open_prob:
                Probability gate is nonzero.
        """
        if deterministic is None:
            deterministic = not self.training

        open_prob = self.open_probability(log_alpha)

        if deterministic:
            gate = self.deterministic_gate(log_alpha)
        else:
            gate = self.sample_gate(log_alpha)

        if mask is not None:
            mask = mask.to(dtype=gate.dtype, device=gate.device)
            gate = gate * mask
            open_prob = open_prob * mask

        return {
            "gate": gate,
            "open_prob": open_prob,
        }

def main() -> None:
    cfg = HardConcreteConfig()
    gate_module = HardConcreteGate(cfg)

    log_alpha = torch.randn(2, 3, 3)
    mask = torch.ones(2, 3, 3).bool()

    gate_module.train()
    out_train = gate_module(log_alpha, mask=mask)
    print("[train] gate shape:", out_train["gate"].shape)
    print("[train] gate min/max:", out_train["gate"].min().item(), out_train["gate"].max().item())
    print("[train] open_prob min/max:", out_train["open_prob"].min().item(), out_train["open_prob"].max().item())

    gate_module.eval()
    out_eval = gate_module(log_alpha, mask=mask)
    print("[eval] gate unique:", torch.unique(out_eval["gate"]))
    print("[eval] open_prob min/max:", out_eval["open_prob"].min().item(), out_eval["open_prob"].max().item())


if __name__ == "__main__":
    main()