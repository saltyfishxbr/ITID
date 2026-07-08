# train_tgsim.py

from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tgsim_dataset import TGSIMCTEDataset, TGSIMDatasetConfig, simple_collate_fn
from dynamic_cte_gnn import DynamicCTEGNN, DynamicCTEGNNConfig
from losses import LossConfig

from counterfactual import CounterfactualConfig
from trainer import Trainer, TrainerConfig

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def build_subset(dataset, subset_size: int, seed: int):
    if subset_size is None:
        return dataset

    subset_size = int(subset_size)

    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False)
    indices = indices.tolist()

    return Subset(dataset, indices)


def build_dataloaders(
    dataset_cfg_base: TGSIMDatasetConfig,
    batch_size: int,
    num_workers: int,
    use_train_subset: bool,
    train_subset_size: int,
    use_val_subset: bool,
    val_subset_size: int,
    seed: int,
):
    train_cfg = TGSIMDatasetConfig(
        window_path=dataset_cfg_base.window_path,
        cte_table_path=dataset_cfg_base.cte_table_path,
        split="train",
        automation_level=dataset_cfg_base.automation_level,
        top_k_neighbors=dataset_cfg_base.top_k_neighbors,
        max_ttc=dataset_cfg_base.max_ttc,
        allow_invalid_lane_edges=dataset_cfg_base.allow_invalid_lane_edges,
        num_vehicle_types=dataset_cfg_base.num_vehicle_types,
        padding_type_index=dataset_cfg_base.padding_type_index,
        verbose=dataset_cfg_base.verbose,
    )

    val_cfg = TGSIMDatasetConfig(
        window_path=dataset_cfg_base.window_path,
        cte_table_path=dataset_cfg_base.cte_table_path,
        split="val",
        automation_level=dataset_cfg_base.automation_level,
        top_k_neighbors=dataset_cfg_base.top_k_neighbors,
        max_ttc=dataset_cfg_base.max_ttc,
        allow_invalid_lane_edges=dataset_cfg_base.allow_invalid_lane_edges,
        num_vehicle_types=dataset_cfg_base.num_vehicle_types,
        padding_type_index=dataset_cfg_base.padding_type_index,
        verbose=dataset_cfg_base.verbose,
    )

    train_dataset = TGSIMCTEDataset(train_cfg)
    val_dataset = TGSIMCTEDataset(val_cfg)

    print("\n[train dataset]")
    train_dataset.summary(diagnostics_samples=3000)

    print("\n[val dataset]")
    val_dataset.summary(diagnostics_samples=1000)

    if use_train_subset:
        print(f"[Data] Using train subset: {train_subset_size}")
        train_dataset = build_subset(
            train_dataset,
            subset_size=train_subset_size,
            seed=seed,
        )

    if use_val_subset:
        print(f"[Data] Using val subset: {val_subset_size}")
        val_dataset = build_subset(
            val_dataset,
            subset_size=val_subset_size,
            seed=seed + 1,
        )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=simple_collate_fn,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=simple_collate_fn,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


def main() -> None:
    seed = 42
    seed_everything(seed)

    # =========================
    # Experiment switch
    # ======================

    subset_name = "L2"  # "L1" or "L2"
    experiment_name = f"tgsim_{subset_name}_ctegnn_nocf_debug"

    # Debug first:
    #   pred_steps=10, train subset, use_cf=False
    #
    # Full no-CF:
    #   pred_steps=50, use_cf=False
    #
    # CF ablation:
    #   pred_steps=50, use_cf=True, cf_start_epoch late

    use_cf = False

    debug_mode = False

    if debug_mode:
        pred_steps = 10
        batch_size = 4
        max_epochs = 2
        use_train_subset = True
        train_subset_size = 5000
        use_val_subset = True
        val_subset_size = 1000
        max_train_batches_per_epoch = 500
        max_val_batches = 100
        experiment_name = f"tgsim_{subset_name}_ctegnn_debug"
    else:
        pred_steps = 50
        batch_size = 16
        max_epochs = 100
        use_train_subset = False
        train_subset_size = 250000
        use_val_subset = False
        val_subset_size = 10000
        max_train_batches_per_epoch = None
        max_val_batches = 300

        if use_cf:
            experiment_name = f"tgsim_{subset_name}_ctegnn_with_cf"
        else:
            experiment_name = f"tgsim_{subset_name}_ctegnn_no_cf"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./Runs/TGSIM_CTEGNN/{experiment_name}_{timestamp}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # =========================
    # Dataset config
    # =========================

    # Choose one subset each run:
    #   subset_name = "L1"  -> node_type 0=HV, 1=L1, 2=padding
    #   subset_name = "L2"  -> node_type 0=HV, 1=L2, 2=padding

    dataset_cfg = TGSIMDatasetConfig(
        window_path=f"./ProcessedDataset/TGSIM/{subset_name}/windows_tgsim_{subset_name}.pkl",
        cte_table_path=f"./ProcessedDataset/TGSIM/{subset_name}/tgsim_{subset_name}_cte_table.pkl",

        split="train",
        automation_level=subset_name,
        top_k_neighbors=8,
        max_ttc=20.0,
        allow_invalid_lane_edges=False,

        # TGSIM automation-level types:
        # 0=HV, 1=L1/L2 depending on subset, 2=padding.
        num_vehicle_types=2,
        padding_type_index=2,

        verbose=False,
    )

    train_loader, val_loader = build_dataloaders(
        dataset_cfg_base=dataset_cfg,
        batch_size=batch_size,
        num_workers=0,
        use_train_subset=use_train_subset,
        train_subset_size=train_subset_size,
        use_val_subset=use_val_subset,
        val_subset_size=val_subset_size,
        seed=seed,
    )

    # =========================
    # Model config
    # =========================

    model_cfg = DynamicCTEGNNConfig(
        state_dim=6,
        history_dim=6,

        num_type_embeddings=3,
        padding_type_index=2,
        type_emb_dim=8,

        hidden_dim=128,
        edge_hidden_dim=128,
        rel_dim=8,
        message_dim=128,

        pred_steps=pred_steps,
        dt=0.1,

        max_delta_pos=3.0,
        max_delta_vel=3.0,

        lambda_self=1.0,

        cte_table_path=dataset_cfg.cte_table_path,
        use_cte_as_edge_feature=False,
        cte_lookup_mode="dynamic",
        interaction_mode="normal",

        use_hard_concrete_gate=True,
        hard_concrete_eval_threshold=0.5,
    )

    model = DynamicCTEGNN(model_cfg)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] trainable parameters: {num_params / 1e6:.3f} M")

    # =========================
    # Loss config
    # =========================

    loss_cfg = LossConfig(
        lambda_traj=1.0,
        lambda_cte=0.05,
        lambda_l0=1e-2,

        lambda_vel=0.2,
        neighbor_weight=0.5,
        target_weight=1.0,

        cte_temperature=0.2,
        cte_use_valid_only=True,
        cte_use_significant_only=False,
        cte_require_positive=True,
        cte_min_edges=2,
    )

    # =========================
    # CF config
    # =========================

    cf_cfg = CounterfactualConfig(
        cf_step_indices=(9, 19, 29, 39, 49),
        top_k_edges=16,

        cte_temperature=0.2,
        use_cte_valid_only=True,
        require_positive_teacher_group=True,
        min_edges_per_receiver=2,

        min_open_prob=0.2,
        min_expected_beta=1e-3,

        drop_from_step_to_end=True,
        deterministic_gate_for_cf=True,
        no_grad_cf_rollout=True,
        eval_mode_for_cf=True,

        label_temperature=1.0,
        min_degradation_for_positive=1e-6,
        supervise_segment_mean=True,
    )

    # =========================
    # Trainer config
    # =========================

    trainer_cfg = TrainerConfig(
        output_dir=output_dir,

        max_epochs=max_epochs,
        lr=1e-4,
        weight_decay=1e-5,
        grad_clip_norm=5.0,

        device="cuda" if torch.cuda.is_available() else "cpu",

        log_every=50,
        histogram_every=200,

        val_every=1,
        save_every=1,

        max_train_batches_per_epoch=max_train_batches_per_epoch,
        max_val_batches=max_val_batches,

        use_cf=use_cf,
        lambda_cf=1e-4,
        cf_start_epoch=max_epochs - 5 if max_epochs >= 10 else max_epochs + 1,
        cf_every=5,

        use_amp=False,
        resume_path=None,
    )

    print("\n[Config]")
    print("dataset_cfg:", asdict(dataset_cfg))
    print("model_cfg:", asdict(model_cfg))
    print("loss_cfg:", asdict(loss_cfg))
    print("cf_cfg:", asdict(cf_cfg))
    print("trainer_cfg:", asdict(trainer_cfg))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_cfg=loss_cfg,
        trainer_cfg=trainer_cfg,
        cf_cfg=cf_cfg,
        model_cfg=model_cfg,
        dataset_cfg=dataset_cfg,
    )

    trainer.train()


if __name__ == "__main__":
    main()