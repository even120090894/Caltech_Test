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
    frame_self_distance,
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


DEFAULT_TEST_DATA_PATH = Path("Caltech/calms21_task1_test_windowed_distance_lt_330.npy")
FRAME_RATE_HZ = 30.0
BRANCH_BY_WINDOW_MOUSE_ID = {0: "intruder", 1: "resident"}
BRANCH_INDEX = {"intruder": 0, "resident": 1}
INTERNODE_INDEX_PAIRS = (
    (0, 1),
    (0, 2),
    (2, 3),
    (1, 3),
    (0, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 6),
    (5, 6),
    (1, 2),
    (4, 5),
)


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


def _target_context_from_mouse_id(target_mouse_id: int) -> tuple[str, str]:
    if target_mouse_id not in BRANCH_BY_WINDOW_MOUSE_ID:
        raise ValueError("target_mouse_id must be 0 or 1.")
    target_branch = BRANCH_BY_WINDOW_MOUSE_ID[target_mouse_id]
    context_branch = "resident" if target_branch == "intruder" else "intruder"
    return target_branch, context_branch


def _pose_stack(frames: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([frame_position(frame) for frame in frames], axis=0).astype(np.float32)


def _velocity_stack(frames: list[dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                [
                    frame["body"]["node"][node_name]["node_velocity"]
                    for node_name in (
                        "head",
                        "headLeft",
                        "headRight",
                        "neck",
                        "bodyLeft",
                        "bodyRight",
                        "tail",
                    )
                ],
                dtype=np.float32,
            )
            for frame in frames
        ],
        axis=0,
    )


def _self_distance_stack(frames: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([frame_self_distance(frame) for frame in frames], axis=0).astype(
        np.float32
    )


def _behavior_stack(frames: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([frame_annotation_id(frame) for frame in frames], dtype=np.int64)


def _role_stack(frames: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [np.int64(frame["interaction"]["intruder_or_resident_tag"]) for frame in frames],
        dtype=np.int64,
    )


def _internode_distances(pose_xy: np.ndarray) -> np.ndarray:
    distances = [
        np.linalg.norm(pose_xy[..., node_a, :] - pose_xy[..., node_b, :], axis=-1)
        for node_a, node_b in INTERNODE_INDEX_PAIRS
    ]
    return np.stack(distances, axis=-1).astype(np.float32)


def _head_angle_sin_cos(own_pose_xy: np.ndarray, other_pose_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    own_center = np.nanmean(own_pose_xy, axis=-2)
    other_center = np.nanmean(other_pose_xy, axis=-2)
    del own_center
    nose = own_pose_xy[..., 0, :]
    neck = own_pose_xy[..., 3, :]
    heading = nose - neck
    to_other = other_center - nose
    cross = heading[..., 0] * to_other[..., 1] - heading[..., 1] * to_other[..., 0]
    dot = heading[..., 0] * to_other[..., 0] + heading[..., 1] * to_other[..., 1]
    angle = np.arctan2(cross, dot)
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def _interaction_features(target_pose_xy: np.ndarray, context_pose_xy: np.ndarray) -> np.ndarray:
    target_center = np.nanmean(target_pose_xy, axis=-2)
    context_center = np.nanmean(context_pose_xy, axis=-2)
    center_delta = context_center - target_center
    midpoint_distance = np.linalg.norm(center_delta, axis=-1)
    target_sin, target_cos = _head_angle_sin_cos(target_pose_xy, context_pose_xy)
    context_sin, context_cos = _head_angle_sin_cos(context_pose_xy, target_pose_xy)
    return np.stack(
        [
            center_delta[..., 0],
            center_delta[..., 1],
            midpoint_distance,
            target_sin,
            target_cos,
            context_sin,
            context_cos,
        ],
        axis=-1,
    ).astype(np.float32)


def _normalizer_transform(normalizer: Any, value: np.ndarray) -> torch.Tensor:
    return normalizer.transform(torch.as_tensor(value, dtype=torch.float32))


def _build_target_segments(
    target_pose_xy: np.ndarray,
    start_ts: np.ndarray,
    history_frames: int,
    rollout_length: int,
) -> np.ndarray:
    segments = np.empty(
        (
            len(start_ts),
            history_frames + rollout_length + 1,
            target_pose_xy.shape[1],
            target_pose_xy.shape[2],
        ),
        dtype=np.float32,
    )
    for row, start_t in enumerate(start_ts):
        segment_start = int(start_t) - history_frames
        previous_index = max(segment_start - 1, 0)
        segments[row, 0] = target_pose_xy[previous_index]
        segments[row, 1:] = target_pose_xy[segment_start : int(start_t) + rollout_length]
    return segments


def _make_recursive_batch(
    target_segments: np.ndarray,
    start_ts: np.ndarray,
    step: int,
    target_behavior: np.ndarray,
    target_role: np.ndarray,
    true_target_pose_xy: np.ndarray,
    context_pose_xy: np.ndarray,
    context_velocity: np.ndarray,
    context_self_distance: np.ndarray,
    context_behavior: np.ndarray,
    context_role: np.ndarray,
    normalizers: Any,
    cfg: Any,
    sequence_id: str,
) -> dict[str, torch.Tensor | list[str]]:
    history_frames = cfg.data.history_frames
    batch_size = len(start_ts)
    target_t = start_ts + step

    a_offsets = 1 + step + np.arange(history_frames)
    a_xy = target_segments[:, a_offsets]
    a_velocity = (target_segments[:, a_offsets] - target_segments[:, a_offsets - 1]) * FRAME_RATE_HZ
    a_self_distance = _internode_distances(a_xy)
    a_indices = target_t[:, None] - history_frames + np.arange(history_frames)

    b_length = cfg.data.b_window_length
    b_indices = target_t[:, None] - history_frames + np.arange(b_length)

    interaction_length = cfg.data.interaction_window_length
    interaction_indices = target_t[:, None] - history_frames + np.arange(interaction_length)
    interaction_offsets = 1 + step + np.arange(interaction_length)
    if cfg.data.include_interaction_current:
        # Current target pose is not known before this step's prediction. Use the
        # last available recursive target estimate for the current interaction slot.
        interaction_offsets = interaction_offsets.copy()
        interaction_offsets[-1] = 1 + history_frames + step - 1
    target_interaction_pose = target_segments[:, interaction_offsets]
    context_interaction_pose = context_pose_xy[interaction_indices]
    interaction = _interaction_features(target_interaction_pose, context_interaction_pose)

    true_pose = true_target_pose_xy[target_t].reshape(batch_size, -1)
    previous_pose = target_segments[:, 1 + history_frames + step - 1].reshape(
        batch_size, -1
    )
    true_delta = (
        true_target_pose_xy[target_t] - true_target_pose_xy[target_t - 1]
    ).reshape(batch_size, -1)

    return {
        "a_xy": _normalizer_transform(normalizers.coord, a_xy),
        "a_velocity": _normalizer_transform(normalizers.velocity, a_velocity),
        "a_self_distance": _normalizer_transform(normalizers.self_distance, a_self_distance),
        "a_behavior": torch.as_tensor(target_behavior[a_indices], dtype=torch.long),
        "a_role": torch.as_tensor(target_role[a_indices], dtype=torch.long),
        "b_xy": _normalizer_transform(normalizers.coord, context_pose_xy[b_indices]),
        "b_velocity": _normalizer_transform(normalizers.velocity, context_velocity[b_indices]),
        "b_self_distance": _normalizer_transform(
            normalizers.self_distance, context_self_distance[b_indices]
        ),
        "b_behavior": torch.as_tensor(context_behavior[b_indices], dtype=torch.long),
        "b_role": torch.as_tensor(context_role[b_indices], dtype=torch.long),
        "interaction": _normalizer_transform(normalizers.interaction, interaction),
        "current_behavior": torch.as_tensor(target_behavior[target_t], dtype=torch.long),
        "target_delta": _normalizer_transform(normalizers.target_delta, true_delta),
        "previous_pose": torch.as_tensor(previous_pose, dtype=torch.float32),
        "target_pose": torch.as_tensor(true_pose, dtype=torch.float32),
        "target_t": torch.as_tensor(target_t, dtype=torch.long),
        "sequence_id": [sequence_id] * batch_size,
    }


def _delta_mse(pred_delta_xy: np.ndarray, true_delta_xy: np.ndarray) -> np.ndarray:
    return np.mean(np.square(pred_delta_xy - true_delta_xy), axis=(1, 2)).astype(
        np.float32
    )


def _empty_loss_accumulator() -> dict[str, float | int]:
    return {
        "delta_mse_sum": 0.0,
        "delta_mse_count": 0,
        "pose_mse_sum": 0.0,
        "pose_mse_count": 0,
        "rollout_count": 0,
    }


def _accumulate_rollout_losses(
    accumulator: dict[str, float | int],
    rollout: dict[str, Any],
) -> None:
    delta = np.asarray(rollout["per_frame_delta_mse"], dtype=np.float64)
    pose = np.asarray(rollout["per_frame_pose_mse"], dtype=np.float64)
    delta_valid = np.isfinite(delta)
    pose_valid = np.isfinite(pose)

    accumulator["delta_mse_sum"] = float(accumulator["delta_mse_sum"]) + float(
        delta[delta_valid].sum()
    )
    accumulator["delta_mse_count"] = int(accumulator["delta_mse_count"]) + int(
        delta_valid.sum()
    )
    accumulator["pose_mse_sum"] = float(accumulator["pose_mse_sum"]) + float(
        pose[pose_valid].sum()
    )
    accumulator["pose_mse_count"] = int(accumulator["pose_mse_count"]) + int(
        pose_valid.sum()
    )
    accumulator["rollout_count"] = int(accumulator["rollout_count"]) + 1


def _finalize_loss_metrics(accumulator: dict[str, float | int]) -> dict[str, float | int | None]:
    delta_count = int(accumulator["delta_mse_count"])
    pose_count = int(accumulator["pose_mse_count"])
    delta_sum = float(accumulator["delta_mse_sum"])
    pose_sum = float(accumulator["pose_mse_sum"])
    return {
        "delta_mse_sum": delta_sum,
        "delta_mse_count": delta_count,
        "delta_mse_mean": delta_sum / delta_count if delta_count else None,
        "pose_mse_sum": pose_sum,
        "pose_mse_count": pose_count,
        "pose_mse_mean": pose_sum / pose_count if pose_count else None,
        "rollout_count": int(accumulator["rollout_count"]),
    }


def _window_true_arrays(
    record: Any,
    window_index: int,
    target_branch: str,
    context_branch: str,
) -> dict[str, np.ndarray]:
    trial_window = record.windows[window_index]
    target_frames = get_branch_frames(trial_window, target_branch)
    context_frames = get_branch_frames(trial_window, context_branch)
    resident_frames = get_branch_frames(trial_window, "resident")
    intruder_frames = get_branch_frames(trial_window, "intruder")
    return {
        "target_pose": _pose_stack(target_frames),
        "target_behavior": _behavior_stack(target_frames),
        "target_role": _role_stack(target_frames),
        "context_pose": _pose_stack(context_frames),
        "context_velocity": _velocity_stack(context_frames),
        "context_self_distance": _self_distance_stack(context_frames),
        "context_behavior": _behavior_stack(context_frames),
        "context_role": _role_stack(context_frames),
        "resident_pose": _pose_stack(resident_frames),
        "intruder_pose": _pose_stack(intruder_frames),
    }


def _rollout_dict_from_arrays(
    sequence_id: str,
    window_index: int,
    start_t: int,
    target_branch: str,
    context_branch: str,
    target_ts: np.ndarray,
    annotations: np.ndarray,
    target_pred_pose_xy: np.ndarray,
    target_true_pose_xy: np.ndarray,
    target_pred_delta_xy: np.ndarray,
    target_true_delta_xy: np.ndarray,
    context_pose_xy: np.ndarray,
    resident_true_pose_xy: np.ndarray,
    intruder_true_pose_xy: np.ndarray,
) -> dict[str, Any]:
    resident_pred_pose_xy = resident_true_pose_xy.copy()
    intruder_pred_pose_xy = intruder_true_pose_xy.copy()
    if target_branch == "resident":
        resident_pred_pose_xy = target_pred_pose_xy
    elif target_branch == "intruder":
        intruder_pred_pose_xy = target_pred_pose_xy
    else:
        raise ValueError(f"Unsupported target_branch: {target_branch}")

    keypoints_pred_pair = _to_calms_pair(resident_pred_pose_xy, intruder_pred_pose_xy)
    keypoints_true_pair = _to_calms_pair(resident_true_pose_xy, intruder_true_pose_xy)
    pose_errors = compute_pose_errors(target_pred_pose_xy, target_true_pose_xy)
    per_frame_delta_mse = _delta_mse(target_pred_delta_xy, target_true_delta_xy)

    return {
        "sequence_id": sequence_id,
        "window_index": int(window_index),
        "start_t": int(start_t),
        "rollout_length": int(len(target_ts)),
        "target_t": target_ts.astype(np.int64),
        "annotation": annotations.astype(np.int64),
        "target_branch": np.asarray(target_branch),
        "context_branch": np.asarray(context_branch),
        "a_pose_xy": context_pose_xy.astype(np.float32),
        "b_pred_pose_xy": target_pred_pose_xy.astype(np.float32),
        "b_true_pose_xy": target_true_pose_xy.astype(np.float32),
        "b_pred_delta_xy": target_pred_delta_xy.astype(np.float32),
        "b_true_delta_xy": target_true_delta_xy.astype(np.float32),
        "resident_true_pose_xy": resident_true_pose_xy.astype(np.float32),
        "intruder_true_pose_xy": intruder_true_pose_xy.astype(np.float32),
        "resident_pred_pose_xy": resident_pred_pose_xy.astype(np.float32),
        "intruder_pred_pose_xy": intruder_pred_pose_xy.astype(np.float32),
        "keypoints_pred_pair": keypoints_pred_pair,
        "keypoints_true_pair": keypoints_true_pair,
        "per_frame_delta_mse": per_frame_delta_mse,
        "per_frame_pose_mse": pose_errors["per_frame_mse"],
        "per_frame_pose_rmse": pose_errors["per_frame_rmse"],
        "per_joint_pose_l2": pose_errors["per_joint_l2"],
        "per_frame_mse": pose_errors["per_frame_mse"],
        "per_frame_rmse": pose_errors["per_frame_rmse"],
        "per_joint_l2": pose_errors["per_joint_l2"],
    }


def _run_window_recursive_rollouts(
    record: Any,
    window_index: int,
    start_ts: np.ndarray,
    target_branch: str,
    context_branch: str,
    rollout_length: int,
    rollout_batch_size: int,
    modules: dict[str, torch.nn.Module],
    normalizers: Any,
    cfg: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    arrays = _window_true_arrays(record, window_index, target_branch, context_branch)
    output_rollouts: list[dict[str, Any]] = []

    for batch_start in range(0, len(start_ts), rollout_batch_size):
        batch_start_ts = start_ts[batch_start : batch_start + rollout_batch_size]
        batch_size = len(batch_start_ts)
        target_segments = _build_target_segments(
            arrays["target_pose"],
            batch_start_ts,
            cfg.data.history_frames,
            rollout_length,
        )
        predicted_delta = np.empty(
            (batch_size, rollout_length, cfg.data.num_joints, cfg.data.coord_dim),
            dtype=np.float32,
        )
        predicted_pose = np.empty_like(predicted_delta)

        with torch.no_grad():
            for step in range(rollout_length):
                batch = _make_recursive_batch(
                    target_segments=target_segments,
                    start_ts=batch_start_ts,
                    step=step,
                    target_behavior=arrays["target_behavior"],
                    target_role=arrays["target_role"],
                    true_target_pose_xy=arrays["target_pose"],
                    context_pose_xy=arrays["context_pose"],
                    context_velocity=arrays["context_velocity"],
                    context_self_distance=arrays["context_self_distance"],
                    context_behavior=arrays["context_behavior"],
                    context_role=arrays["context_role"],
                    normalizers=normalizers,
                    cfg=cfg,
                    sequence_id=record.sequence_id,
                )
                batch = move_batch_to_device(batch, device)
                predicted_delta_norm = forward_batch(batch, modules, cfg, device)
                predicted_delta_flat = normalizers.target_delta.inverse(
                    predicted_delta_norm.detach().cpu()
                ).numpy()
                predicted_delta_xy = predicted_delta_flat.reshape(
                    batch_size, cfg.data.num_joints, cfg.data.coord_dim
                )
                previous_pose_xy = target_segments[:, 1 + cfg.data.history_frames + step - 1]
                predicted_pose_xy = previous_pose_xy + predicted_delta_xy
                target_segments[:, 1 + cfg.data.history_frames + step] = predicted_pose_xy
                predicted_delta[:, step] = predicted_delta_xy
                predicted_pose[:, step] = predicted_pose_xy

        target_t_matrix = batch_start_ts[:, None] + np.arange(rollout_length)
        true_pose = arrays["target_pose"][target_t_matrix]
        true_delta = true_pose - arrays["target_pose"][target_t_matrix - 1]
        context_pose = arrays["context_pose"][target_t_matrix]
        resident_true_pose = arrays["resident_pose"][target_t_matrix]
        intruder_true_pose = arrays["intruder_pose"][target_t_matrix]
        annotations = arrays["target_behavior"][target_t_matrix]

        for row, start_t in enumerate(batch_start_ts):
            output_rollouts.append(
                _rollout_dict_from_arrays(
                    sequence_id=record.sequence_id,
                    window_index=window_index,
                    start_t=int(start_t),
                    target_branch=target_branch,
                    context_branch=context_branch,
                    target_ts=target_t_matrix[row],
                    annotations=annotations[row],
                    target_pred_pose_xy=predicted_pose[row],
                    target_true_pose_xy=true_pose[row],
                    target_pred_delta_xy=predicted_delta[row],
                    target_true_delta_xy=true_delta[row],
                    context_pose_xy=context_pose[row],
                    resident_true_pose_xy=resident_true_pose[row],
                    intruder_true_pose_xy=intruder_true_pose[row],
                )
            )

    return output_rollouts


def run_test_recursive_rollouts(
    run_dir: Path,
    fold: int = 1,
    checkpoint_name: str = "best.pt",
    test_data_path: Path = DEFAULT_TEST_DATA_PATH,
    target_mouse_id: int = 0,
    rollout_length: int = 50,
    stride: int = 1,
    output_name: str | None = None,
    device_name: str = "auto",
    rollout_batch_size: int = 512,
    history_frames: int | None = None,
    max_trials: int | None = None,
    max_windows_per_trial: int | None = None,
    max_rollouts_per_window: int | None = None,
) -> Path:
    if rollout_length <= 0:
        raise ValueError("rollout_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be positive.")

    fold_dir = run_dir / f"fold_{fold:02d}"
    checkpoint_path = fold_dir / "checkpoints" / checkpoint_name
    output_dir = fold_dir / "rollouts"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_branch, context_branch = _target_context_from_mouse_id(target_mouse_id)
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
    inference_cfg.data.data_path = test_data_path
    inference_cfg.data.history_frames = rollout_history_frames
    inference_cfg.data.target_branch = target_branch
    inference_cfg.data.context_branch = context_branch

    for module in modules.values():
        module.eval()

    records = load_calms21_sequences(test_data_path)
    if max_trials is not None:
        records = records[:max_trials]

    if output_name is None:
        output_name = (
            f"test_recursive_rollouts_target_mouse_{target_mouse_id}_"
            f"hist_{rollout_history_frames}_stride_{stride}.npz"
        )
    output_path = output_dir / output_name
    summary_path = output_path.with_suffix(".summary.json")

    min_window_length = rollout_history_frames + rollout_length
    trials: list[dict[str, Any]] = []
    total_windows = 0
    exported_windows = 0
    skipped_windows = 0
    total_rollouts = 0
    loss_accumulator = _empty_loss_accumulator()

    for trial_index, record in enumerate(records):
        trial_windows: list[dict[str, Any]] = []
        windows_iter = list(enumerate(record.windows))
        if max_windows_per_trial is not None:
            windows_iter = windows_iter[:max_windows_per_trial]

        trial_skipped = 0
        for window_index, trial_window in windows_iter:
            total_windows += 1
            target_len = len(get_branch_frames(trial_window, target_branch))
            context_len = len(get_branch_frames(trial_window, context_branch))
            window_length = min(target_len, context_len)
            if window_length < min_window_length:
                skipped_windows += 1
                trial_skipped += 1
                continue

            start_ts = np.arange(
                rollout_history_frames,
                window_length - rollout_length + 1,
                stride,
                dtype=np.int64,
            )
            if max_rollouts_per_window is not None:
                start_ts = start_ts[:max_rollouts_per_window]
            if len(start_ts) == 0:
                skipped_windows += 1
                trial_skipped += 1
                continue

            rollouts = _run_window_recursive_rollouts(
                record=record,
                window_index=window_index,
                start_ts=start_ts,
                target_branch=target_branch,
                context_branch=context_branch,
                rollout_length=rollout_length,
                rollout_batch_size=rollout_batch_size,
                modules=modules,
                normalizers=normalizers,
                cfg=inference_cfg,
                device=device,
            )
            exported_windows += 1
            total_rollouts += len(rollouts)
            for rollout in rollouts:
                _accumulate_rollout_losses(loss_accumulator, rollout)
            trial_windows.append(
                {
                    "window_index": int(window_index),
                    "window_length": int(window_length),
                    "target_mouse_id": int(target_mouse_id),
                    "target_branch": target_branch,
                    "context_branch": context_branch,
                    "rollouts": rollouts,
                }
            )
            print(
                f"{record.sequence_id} window={window_index}: "
                f"rollouts={len(rollouts)} length={window_length}"
            )

        trials.append(
            {
                "sequence_id": record.sequence_id,
                "trial_index": int(trial_index),
                "window_count_total": int(len(windows_iter)),
                "window_count_exported": int(len(trial_windows)),
                "window_count_skipped": int(trial_skipped),
                "windows": trial_windows,
            }
        )

    branch_warning = None
    if trained_cfg.data.target_branch != target_branch:
        branch_warning = (
            "Requested target branch differs from checkpoint training target_branch. "
            "The model weights and normalizers are reused as requested."
        )
        print(f"Warning: {branch_warning}")

    summary = {
        "mode": "recursive_full_test_rollout",
        "source_data_path": test_data_path,
        "checkpoint_training_data_path": trained_cfg.data.data_path,
        "checkpoint_path": checkpoint_path,
        "output_path": output_path,
        "target_mouse_id": target_mouse_id,
        "target_branch": target_branch,
        "context_branch": context_branch,
        "trained_target_branch": trained_cfg.data.target_branch,
        "trained_context_branch": trained_cfg.data.context_branch,
        "branch_warning": branch_warning,
        "trained_history_frames": trained_history_frames,
        "rollout_history_frames": rollout_history_frames,
        "rollout_length": rollout_length,
        "stride": stride,
        "rollout_batch_size": rollout_batch_size,
        "min_window_length": min_window_length,
        "trial_count": len(trials),
        "total_windows": total_windows,
        "exported_windows": exported_windows,
        "skipped_windows": skipped_windows,
        "total_rollouts": total_rollouts,
        "loss_metrics": _finalize_loss_metrics(loss_accumulator),
        "interaction_current_policy": (
            "When include_interaction_current=True, recursive rollout fills the "
            "current target interaction slot with the last available predicted "
            "target pose to avoid using true future target pose."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        trials=np.asarray(trials, dtype=object),
        summary=np.asarray(summary, dtype=object),
    )
    _write_json(summary_path, summary)
    print(f"Saved recursive rollout export to: {output_path}")
    print(f"Saved recursive rollout summary to: {summary_path}")
    return output_path


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
    data_path: Path = DEFAULT_TEST_DATA_PATH,
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

    inference_cfg.data.data_path = data_path
    records = load_calms21_sequences(data_path)
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


def parse_args_v2() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export recursive full-test rollouts for visualization. If both "
            "--sequence-id and --start-t are provided, exports the legacy "
            "single windowed one-step rollout instead."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing config.json and fold_XX folders.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="One-based fold id; 1 maps to fold_01.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best.pt",
        help="Checkpoint filename under fold_XX/checkpoints.",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=DEFAULT_TEST_DATA_PATH,
        help="Windowed CalMS21 test npy used for recursive full-test rollout.",
    )
    parser.add_argument(
        "--target-mouse-id",
        type=int,
        choices=(0, 1),
        default=0,
        help="Window-order mouse id to predict: 0=intruder, 1=resident.",
    )
    parser.add_argument(
        "--sequence-id",
        type=str,
        default=None,
        help="Legacy single-rollout sequence id. If omitted, full-test rollout is used.",
    )
    parser.add_argument(
        "--start-t",
        type=int,
        default=None,
        help="Legacy single-rollout first target frame. Must be paired with --sequence-id.",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=50,
        help="Recursive rollout length.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Start-frame stride for full-test recursive rollouts.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output npz filename under fold_XX/rollouts.",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Legacy single-rollout window index inside the selected sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=512,
        help="Number of rollout start points processed together.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Backward-compatible alias for --rollout-batch-size.",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help="Override historical frame count for rollout inference only.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional smoke-test limit for the number of trials.",
    )
    parser.add_argument(
        "--max-windows-per-trial",
        type=int,
        default=None,
        help="Optional smoke-test limit for windows per trial.",
    )
    parser.add_argument(
        "--max-rollouts-per-window",
        type=int,
        default=None,
        help="Optional smoke-test limit for start points per window.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args_v2()
    rollout_batch_size = (
        args.rollout_batch_size if args.batch_size is None else args.batch_size
    )
    if args.sequence_id is not None or args.start_t is not None:
        if args.sequence_id is None or args.start_t is None:
            raise ValueError("--sequence-id and --start-t must be provided together.")
        run_rollout(
            run_dir=args.run_dir,
            fold=args.fold,
            checkpoint_name=args.checkpoint_name,
            sequence_id=args.sequence_id,
            start_t=args.start_t,
            rollout_length=args.rollout_length,
            output_name=args.output_name or "rollout_predictions.npz",
            window_index=args.window_index,
            device_name=args.device,
            batch_size=rollout_batch_size,
            history_frames=args.history_frames,
            data_path=args.test_data_path,
        )
    else:
        run_test_recursive_rollouts(
            run_dir=args.run_dir,
            fold=args.fold,
            checkpoint_name=args.checkpoint_name,
            test_data_path=args.test_data_path,
            target_mouse_id=args.target_mouse_id,
            rollout_length=args.rollout_length,
            stride=args.stride,
            output_name=args.output_name,
            device_name=args.device,
            rollout_batch_size=rollout_batch_size,
            history_frames=args.history_frames,
            max_trials=args.max_trials,
            max_windows_per_trial=args.max_windows_per_trial,
            max_rollouts_per_window=args.max_rollouts_per_window,
        )
