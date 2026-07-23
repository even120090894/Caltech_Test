from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_path: Path = Path("Caltech/calms21_task1_train_windowed_distance_lt_330.npy")
    feature_cache_dir: Path = Path("Caltech/cache")
    num_mice: int = 2
    num_joints: int = 7
    coord_dim: int = 2
    num_behavior_classes: int = 4
    num_role_classes: int = 2
    num_self_distances: int = 12
    model_name: str = "multi_predict"
    # model_name: str = "single_predict"
    target_branch: str = "intruder"
    context_branch: str = "resident"
    history_frames: int = 24
    include_context_current: bool = True
    include_interaction_current: bool = True
    behavior_label_mode: str = "history_plus_current"
    use_scores: bool = False

    @property
    def a_window_length(self) -> int:
        return self.history_frames

    @property
    def b_window_length(self) -> int:
        return self.history_frames + int(self.include_context_current)

    @property
    def interaction_window_length(self) -> int:
        return self.history_frames + int(self.include_interaction_current)

    @property
    def min_target_index(self) -> int:
        return self.history_frames


@dataclass
class EmbeddingConfig:
    joint_input_dim: int = 2
    joint_embedding_dim: int = 16
    pose_embedding_dim: int = 64
    velocity_embedding_dim: int = 32
    self_distance_embedding_dim: int = 32
    behavior_embedding_dim: int = 16
    role_embedding_dim: int = 8
    mouse_embedding_dim: int = 128
    interaction_input_dim: int = 7
    interaction_embedding_dim: int = 32
    frame_mlp_hidden_dim: int = 128
    dropout: float = 0.0


@dataclass
class SequenceConfig:
    model_type: str = "lstm"
    input_dim: int = 128
    interaction_input_dim: int = 32
    hidden_dim: int = 128
    interaction_hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    @property
    def output_dim(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)

    @property
    def interaction_output_dim(self) -> int:
        return self.interaction_hidden_dim * (2 if self.bidirectional else 1)


@dataclass
class PoolingConfig:
    pooling_type: str = "auto"
    attention_hidden_dim: int = 64


@dataclass
class HeadConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    output_dim: int = 14
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    num_folds: int = 2
    split_mode: str = "sequence_level"
    # Used when num_folds == 1: reserve this ratio as a single validation split.
    # 当 num_folds == 1 时，使用该比例划出一次性的验证集。
    holdout_val_ratio: float = 0.2
    seed: int = 42
    # Larger batch_size usually increases GPU utilization but also uses more VRAM.
    # 增大 batch_size 通常能提高 GPU 利用率，但会占用更多显存。
    batch_size: int = 256
    epochs: int = 120
    learning_rate: float = 1e-3 # //TODO 学习率可变？cos 退火？
    weight_decay: float = 1e-4
    # More workers can speed up data loading, but too many may overload CPU/RAM.
    # 增加 num_workers 可以加快数据加载，但过高会占用 CPU/RAM。
    num_workers: int = 0
    # Use "cuda" to require GPU, "cpu" to force CPU, or "auto" to choose automatically.
    # 使用 "cuda" 强制 GPU，"cpu" 强制 CPU，"auto" 自动选择。
    device: str = "auto"
    # pin_memory speeds up CPU-to-GPU transfer when training on CUDA.
    # CUDA 训练时 pin_memory 可以加速 CPU 到 GPU 的数据拷贝。
    pin_memory: bool = True
    # Number of batches each worker preloads; useful only when num_workers > 0.
    # 每个 worker 预取的 batch 数；仅在 num_workers > 0 时生效。
    prefetch_factor: int = 2
    # Keep DataLoader workers alive between epochs to reduce restart overhead.
    # 在 epoch 之间保留 DataLoader worker，减少反复启动的开销。
    persistent_workers: bool = False
    # AMP can reduce VRAM usage and improve GPU throughput on supported CUDA devices.
    # 混合精度可降低显存占用，并在支持的 CUDA 设备上提高吞吐。
    use_amp: bool = False
    # Print step-level loss every N training batches.
    # 每隔 N 个训练 batch 打印一次 step 级 loss。
    log_every_n_steps: int = 50
    # Limit train/validation batches for smoke tests or quick load experiments.
    # 限制训练/验证 batch 数，便于 smoke test 或快速压测。
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    output_root: Path = Path("Model/asymmetric_pose_case/runs")
    experiment_name: str = "asymmetric_pose"
    save_checkpoints: bool = True
    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True
    export_validation_predictions: bool = True
    use_feature_cache: bool = True


@dataclass
class NormalizationConfig:
    eps: float = 1e-6


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()
