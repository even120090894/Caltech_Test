import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data import (
    CalMS21AsymmetricPoseDataset,
    WindowIndex,
    frame_annotation_id,
    frame_position,
    get_branch_frames,
    load_calms21_sequences,
)
from ..runtime import (
    build_modules_from_checkpoint,
    config_from_dict,
    forward_batch,
    move_batch_to_device,
    resolve_device,
    to_jsonable,
)
from .metrics import compute_pose_errors


def _find_sequence(records, sequence_id: str) -> tuple[int, Any]:
    for sequence_index, record in enumerate(records):
        if record.sequence_id == sequence_id:
            return sequence_index, record
    available = ", ".join(record.sequence_id for record in records[:5])
    raise ValueError(
        f"Unknown sequence_id: {sequence_id}. First available sequences: {available}"
    )


def _branch_pose(record, window_index: int, branch_name: str, target_t: int) -> np.ndarray:
    trial_window = record.windows[window_index]
    frames = get_branch_frames(trial_window, branch_name)
    if target_t < 0 or target_t >= len(frames):
        raise ValueError(
            f"target_t={target_t} is outside branch '{branch_name}' length {len(frames)}."
        )
    return frame_position(frames[target_t]).astype(np.float32)


def _branch_annotation(record, window_index: int, target_branch: str, target_t: int) -> np.int64:
    frames = get_branch_frames(record.windows[window_index], target_branch)
    return frame_annotation_id(frames[target_t])


def _to_calms_pair(resident_pose_xy: np.ndarray, intruder_pose_xy: np.ndarray) -> np.ndarray:
    pair = np.empty((resident_pose_xy.shape[0], 2, 2, 7), dtype=np.float32)
    pair[:, 0] = np.transpose(resident_pose_xy, (0, 2, 1))
    pair[:, 1] = np.transpose(intruder_pose_xy, (0, 2, 1))
    return pair


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_rollout(
    run_dir: Path,
    fold: int,
    checkpoint_name: str,
    sequence_id: str,
    start_t: int,
    rollout_length: int,
    output_name: str,
    window_index: int = 0,
    device_name: str = "auto",
    batch_size: int = 512,
    history_frames: int | None = None,
) -> Path:
    """Export consecutive one-step predictions for visualization.

    This is teacher-forced with respect to history: every target frame uses the
    true windowed features from the preprocessed dataset. It is intentionally
    compatible with visualization.ipynb's existing rollout npz reader.
    """

    fold_dir = run_dir / f"fold_{fold:02d}"
    checkpoint_path = fold_dir / "checkpoints" / checkpoint_name
    output_dir = fold_dir / "rollouts"
    output_path = output_dir / output_name
    summary_path = output_dir / "rollout_summary.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if rollout_length <= 0:
        raise ValueError("rollout_length must be positive.")

    device = resolve_device(device_name)
    modules, normalizers, checkpoint = build_modules_from_checkpoint(
        checkpoint_path, device
    )
    trained_cfg = config_from_dict(checkpoint["config"])
    inference_cfg = config_from_dict(checkpoint["config"])
    trained_history_frames = int(trained_cfg.data.history_frames)
    rollout_history_frames = (
        trained_history_frames if history_frames is None else int(history_frames)
    )
    if rollout_history_frames <= 0:
        raise ValueError("history_frames must be positive.")
    inference_cfg.data.history_frames = rollout_history_frames
    history_override = rollout_history_frames != trained_history_frames
    if history_override:
        print(
            "Warning: rollout history_frames override is an out-of-distribution "
            "inference experiment. "
            f"trained_history_frames={trained_history_frames}, "
            f"rollout_history_frames={rollout_history_frames}"
        )

    records = load_calms21_sequences(trained_cfg.data.data_path)
    sequence_index, record = _find_sequence(records, sequence_id)
    if window_index < 0 or window_index >= len(record.windows):
        raise ValueError(
            f"window_index={window_index} is outside sequence window count "
            f"{len(record.windows)}."
        )

    target_frames = get_branch_frames(
        record.windows[window_index], inference_cfg.data.target_branch
    )
    end_t = start_t + rollout_length
    if start_t < inference_cfg.data.min_target_index:
        raise ValueError(
            f"start_t={start_t} must be >= "
            f"min_target_index={inference_cfg.data.min_target_index}."
        )
    if end_t > len(target_frames):
        raise ValueError(
            f"start_t + rollout_length = {end_t} exceeds target branch length "
            f"{len(target_frames)}."
        )

    target_ts = np.arange(start_t, end_t, dtype=np.int64)
    windows = [
        WindowIndex(
            sequence_index=sequence_index,
            window_index=window_index,
            target_t=int(target_t),
        )
        for target_t in target_ts
    ]
    dataset = CalMS21AsymmetricPoseDataset(
        records, windows, inference_cfg.data, normalizers
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for module in modules.values():
        module.eval()

    predicted_delta_chunks: list[np.ndarray] = []
    predicted_pose_chunks: list[np.ndarray] = []
    true_delta_chunks: list[np.ndarray] = []
    true_pose_chunks: list[np.ndarray] = []
    previous_pose_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            predicted_delta_norm = forward_batch(batch, modules, inference_cfg, device)
            predicted_delta = normalizers.target_delta.inverse(
                predicted_delta_norm.detach().cpu()
            )
            true_delta = normalizers.target_delta.inverse(
                batch["target_delta"].detach().cpu()
            )
            previous_pose = batch["previous_pose"].detach().cpu()
            predicted_pose = previous_pose + predicted_delta
            true_pose = batch["target_pose"].detach().cpu()

            predicted_delta_chunks.append(predicted_delta.numpy())
            predicted_pose_chunks.append(predicted_pose.numpy())
            true_delta_chunks.append(true_delta.numpy())
            true_pose_chunks.append(true_pose.numpy())
            previous_pose_chunks.append(previous_pose.numpy())

    target_pred_pose_xy = np.concatenate(predicted_pose_chunks, axis=0).reshape(
        rollout_length, inference_cfg.data.num_joints, inference_cfg.data.coord_dim
    )
    target_true_pose_xy = np.concatenate(true_pose_chunks, axis=0).reshape(
        rollout_length, inference_cfg.data.num_joints, inference_cfg.data.coord_dim
    )
    target_pred_delta_xy = np.concatenate(predicted_delta_chunks, axis=0).reshape(
        rollout_length, inference_cfg.data.num_joints, inference_cfg.data.coord_dim
    )
    target_true_delta_xy = np.concatenate(true_delta_chunks, axis=0).reshape(
        rollout_length, inference_cfg.data.num_joints, inference_cfg.data.coord_dim
    )
    previous_pose_xy = np.concatenate(previous_pose_chunks, axis=0).reshape(
        rollout_length, inference_cfg.data.num_joints, inference_cfg.data.coord_dim
    )

    resident_true_pose_xy = np.stack(
        [
            _branch_pose(record, window_index, "resident", int(target_t))
            for target_t in target_ts
        ],
        axis=0,
    )
    intruder_true_pose_xy = np.stack(
        [
            _branch_pose(record, window_index, "intruder", int(target_t))
            for target_t in target_ts
        ],
        axis=0,
    )
    resident_pred_pose_xy = resident_true_pose_xy.copy()
    intruder_pred_pose_xy = intruder_true_pose_xy.copy()
    if inference_cfg.data.target_branch == "resident":
        resident_pred_pose_xy = target_pred_pose_xy
        context_pose_xy = intruder_true_pose_xy
    elif inference_cfg.data.target_branch == "intruder":
        intruder_pred_pose_xy = target_pred_pose_xy
        context_pose_xy = resident_true_pose_xy
    else:
        raise ValueError(
            f"Unsupported target_branch: {inference_cfg.data.target_branch}"
        )

    keypoints_pred_pair = _to_calms_pair(resident_pred_pose_xy, intruder_pred_pose_xy)
    keypoints_true_pair = _to_calms_pair(resident_true_pose_xy, intruder_true_pose_xy)
    errors = compute_pose_errors(target_pred_pose_xy, target_true_pose_xy)
    annotations = np.asarray(
        [
            _branch_annotation(
                record, window_index, inference_cfg.data.target_branch, int(target_t)
            )
            for target_t in target_ts
        ],
        dtype=np.int64,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sequence_id=np.asarray(sequence_id),
        window_index=np.asarray(window_index, dtype=np.int64),
        rollout_length=np.asarray(rollout_length, dtype=np.int64),
        target_t=target_ts,
        annotation=annotations,
        target_branch=np.asarray(inference_cfg.data.target_branch),
        context_branch=np.asarray(inference_cfg.data.context_branch),
        trained_history_frames=np.asarray(trained_history_frames, dtype=np.int64),
        rollout_history_frames=np.asarray(rollout_history_frames, dtype=np.int64),
        a_pose_xy=context_pose_xy.astype(np.float32),
        b_pred_pose_xy=target_pred_pose_xy.astype(np.float32),
        b_true_pose_xy=target_true_pose_xy.astype(np.float32),
        b_pred_delta_xy=target_pred_delta_xy.astype(np.float32),
        b_true_delta_xy=target_true_delta_xy.astype(np.float32),
        previous_pose_xy=previous_pose_xy.astype(np.float32),
        resident_true_pose_xy=resident_true_pose_xy.astype(np.float32),
        intruder_true_pose_xy=intruder_true_pose_xy.astype(np.float32),
        resident_pred_pose_xy=resident_pred_pose_xy.astype(np.float32),
        intruder_pred_pose_xy=intruder_pred_pose_xy.astype(np.float32),
        keypoints_pred_pair=keypoints_pred_pair,
        keypoints_true_pair=keypoints_true_pair,
        per_joint_l2=errors["per_joint_l2"],
        per_frame_mse=errors["per_frame_mse"],
        per_frame_rmse=errors["per_frame_rmse"],
    )
    _write_json(
        summary_path,
        {
            "sequence_id": sequence_id,
            "window_index": window_index,
            "start_t": start_t,
            "rollout_length": rollout_length,
            "target_branch": inference_cfg.data.target_branch,
            "context_branch": inference_cfg.data.context_branch,
            "trained_history_frames": trained_history_frames,
            "rollout_history_frames": rollout_history_frames,
            "history_override": history_override,
            "checkpoint_path": checkpoint_path,
            "output_path": output_path,
            "mode": "teacher_forced_windowed_one_step_export",
            "metrics": {
                key: value
                for key, value in errors.items()
                if isinstance(value, float)
            },
        },
    )
    print(f"Saved rollout export to: {output_path}")
    print(f"Saved rollout summary to: {summary_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export consecutive windowed one-step pose predictions for visualization. "
            "导出连续帧 one-step 预测结果，供 visualization.ipynb 可视化。"
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Training run directory containing config.json and fold_XX folders. "
            "训练输出目录，例如 Model/asymmetric_pose_case/runs/20260718_081453_asymmetric_pose。"
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="One-based fold id; 1 maps to fold_01. 从 1 开始的 fold 编号，默认 1。",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best.pt",
        help=(
            "Checkpoint filename under fold_XX/checkpoints. "
            "checkpoint 文件名，通常是 best.pt 或 last.pt，默认 best.pt。"
        ),
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        required=True,
        help=(
            "CalMS21 sequence id to visualize. "
            "要导出的序列 id，例如 task1/train/mouse003_task1_annotator1。"
        ),
    )
    parser.add_argument(
        "--start-t",
        type=int,
        required=True,
        help=(
            "First target frame index to predict. Must be >= rollout history_frames. "
            "第一个预测帧编号，必须不小于 history_frames。"
        ),
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=20,
        help=(
            "Number of consecutive target frames to export. "
            "连续导出的帧数，默认 20。"
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="rollout_predictions.npz",
        help=(
            "Output .npz filename under fold_XX/rollouts. "
            "输出 npz 文件名，保存到 fold_XX/rollouts 下。"
        ),
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help=(
            "Trial-window index inside the selected sequence. "
            "同一个 sequence 内的 trial window 编号，默认 0。"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Inference device: auto, cpu, or cuda. "
            "推理设备，可选 auto/cpu/cuda，默认 auto。"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help=(
            "Batch size for export inference only; it does not affect training. "
            "导出推理时的 batch size，不影响训练结果，默认 512。"
        ),
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help=(
            "Override historical frame count for rollout inference only. "
            "Defaults to checkpoint config history_frames. A different value is "
            "an out-of-distribution comparison experiment."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_rollout(
        run_dir=args.run_dir,
        fold=args.fold,
        checkpoint_name=args.checkpoint_name,
        sequence_id=args.sequence_id,
        start_t=args.start_t,
        rollout_length=args.rollout_length,
        output_name=args.output_name,
        window_index=args.window_index,
        device_name=args.device,
        batch_size=args.batch_size,
        history_frames=args.history_frames,
    )
