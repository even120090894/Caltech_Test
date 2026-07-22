# Validation Rollout And Visualization Notes

This document describes the current `Model/asymmetric_pose_case/validation`
export path and how to use its output in `visualization.ipynb`.

## 1. Current Scope

`validation/rollout.py` currently implements a **windowed one-step prediction
export**, not a closed-loop recursive rollout.

Meaning:

```text
for each target_t in [start_t, start_t + rollout_length):
  use the true preprocessed window features for that target_t
  predict normalized target_delta
  inverse-normalize target_delta
  reconstruct predicted target pose
  export arrays for visualization
```

This is teacher-forced with respect to history: every frame uses the true
windowed features from `Caltech/calms21_task1_train_windowed_distance_lt_330.npy`.

By default, rollout uses the `history_frames` value saved in the checkpoint.
For comparison experiments, `--history-frames` can override the number of
historical frames used only during rollout inference. This does not change the
checkpoint config, model weights, or normalizers.

This export is intended to produce an intermediate visualization state for
`visualization.ipynb`. It does not feed frame `t` prediction back into frame
`t + 1`.

## 2. Model Output Semantics

The model predicts a normalized pose displacement:

```text
target_delta = target_pose[t] - target_pose[t - 1]
```

During export:

```text
pred_delta_xy = inverse_normalize(model_output)
pred_pose_xy  = previous_pose_xy + pred_delta_xy
```

The `.npz` output keeps both delta and absolute pose forms:

```text
b_pred_delta_xy
b_pred_pose_xy
```

For visualization, use absolute pose:

```python
keypoints_pred_pair = rollout["keypoints_pred_pair"]
```

## 3. Command

Example using a trained run:

```bash
python -m Model.asymmetric_pose_case.validation.rollout \
  --run-dir Model/asymmetric_pose_case/runs/20260718_081453_asymmetric_pose \
  --fold 1 \
  --checkpoint-name best.pt \
  --sequence-id task1/train/mouse003_task1_annotator1 \
  --start-t 100 \
  --rollout-length 50 \
  --output-name mouse003_start100_len50.npz \
  --device cuda
```

Example: train a 49-frame model:

```bash
python -m Model.asymmetric_pose_case.train \
  --history-frames 49 \
  --device cuda \
  --batch-size 4096 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --use-amp
```

Example: use a 49-frame checkpoint but feed only the latest 9 frames at rollout
time:

```bash
python -m Model.asymmetric_pose_case.validation.rollout \
  --run-dir Model/asymmetric_pose_case/runs/<49_frame_run> \
  --fold 1 \
  --checkpoint-name best.pt \
  --sequence-id task1/train/mouse003_task1_annotator1 \
  --start-t 100 \
  --rollout-length 300 \
  --history-frames 9 \
  --output-name rollout_predictions_hist9.npz \
  --device cuda
```

Output path:

```text
Model/asymmetric_pose_case/runs/20260718_081453_asymmetric_pose/fold_01/rollouts/mouse003_start100_len50.npz
```

Summary path:

```text
Model/asymmetric_pose_case/runs/20260718_081453_asymmetric_pose/fold_01/rollouts/rollout_summary.json
```

## 4. CLI Parameters

```text
--run-dir
  Training run directory containing config.json and fold_XX folders.
  Example: Model/asymmetric_pose_case/runs/20260718_081453_asymmetric_pose

--fold
  One-based fold id. fold=1 maps to fold_01.
  Default: 1

--checkpoint-name
  Checkpoint filename under fold_XX/checkpoints.
  Common values: best.pt, last.pt
  Default: best.pt

--sequence-id
  CalMS21 sequence id to export and visualize.
  Example: task1/train/mouse003_task1_annotator1

--start-t
  First target frame index to predict.
  Must be >= the rollout history frame count.

--rollout-length
  Number of consecutive target frames to export.
  Default: 20

--output-name
  Output .npz filename under fold_XX/rollouts.
  Default: rollout_predictions.npz

--window-index
  Trial-window index inside the selected sequence.
  Default: 0

--device
  Inference device: auto, cpu, or cuda.
  Default: auto

--batch-size
  Batch size for export inference only. It does not affect training.
  Default: 512

--history-frames
  Optional inference-time history length override.
  Default: checkpoint config data.history_frames
  When different from the trained value, this is an out-of-distribution
  comparison experiment.
```

You can also inspect these directly:

```bash
python -m Model.asymmetric_pose_case.validation.rollout --help
```

## 5. Output Arrays

The export writes:

```text
rollout_predictions.npz
rollout_summary.json
```

Important `.npz` arrays:

```text
sequence_id
  shape: scalar string
  meaning: exported CalMS21 sequence id

window_index
  shape: scalar int64
  meaning: selected trial-window index within the sequence

rollout_length
  shape: scalar int64
  meaning: number of exported frames

target_t
  shape: [T]
  meaning: original target frame ids

annotation
  shape: [T]
  meaning: behavior label ids from the target branch frames

target_branch
  shape: scalar string
  meaning: model target branch, usually "intruder"

context_branch
  shape: scalar string
  meaning: model context branch, usually "resident"

trained_history_frames
  shape: scalar int64
  meaning: history_frames used when the checkpoint was trained

rollout_history_frames
  shape: scalar int64
  meaning: history_frames actually used to build rollout inputs

a_pose_xy
  shape: [T, 7, 2]
  meaning: true context branch pose. Kept for visualization notebook compatibility.

b_pred_pose_xy
  shape: [T, 7, 2]
  meaning: predicted target branch absolute pose

b_true_pose_xy
  shape: [T, 7, 2]
  meaning: true target branch absolute pose

b_pred_delta_xy
  shape: [T, 7, 2]
  meaning: predicted target branch pose displacement

b_true_delta_xy
  shape: [T, 7, 2]
  meaning: true target branch pose displacement

previous_pose_xy
  shape: [T, 7, 2]
  meaning: target branch pose at target_t - 1

resident_true_pose_xy
  shape: [T, 7, 2]
  meaning: true resident pose

intruder_true_pose_xy
  shape: [T, 7, 2]
  meaning: true intruder pose

resident_pred_pose_xy
  shape: [T, 7, 2]
  meaning: resident pose used in predicted pair

intruder_pred_pose_xy
  shape: [T, 7, 2]
  meaning: intruder pose used in predicted pair

keypoints_pred_pair
  shape: [T, 2, 2, 7]
  meaning: CalMS21-style pair for visualization; mouse 0 resident, mouse 1 intruder

keypoints_true_pair
  shape: [T, 2, 2, 7]
  meaning: CalMS21-style true resident/intruder pair

per_joint_l2
  shape: [T, 7]
  meaning: per-frame, per-joint L2 error for the target branch

per_frame_mse
  shape: [T]
  meaning: per-frame MSE for target branch pose

per_frame_rmse
  shape: [T]
  meaning: per-frame RMSE for target branch pose
```

`keypoints_pred_pair` follows CalMS21 visualization layout:

```text
keypoints_pred_pair[frame, mouse, coord, joint]

mouse = 0: resident
mouse = 1: intruder
coord = 0: x
coord = 1: y
joint = 0..6
```

## 6. Visualization Notebook Usage

In `visualization.ipynb`, update:

```python
rollout_path = Path(
    "Model/asymmetric_pose_case/runs/20260718_081453_asymmetric_pose/"
    "fold_01/rollouts/mouse003_start100_len50.npz"
)
```

The notebook already reads:

```python
rollout = np.load(rollout_path, allow_pickle=True)
keypoints_pred_pair = rollout["keypoints_pred_pair"]
a_pose_xy = rollout["a_pose_xy"]
b_true_pose_xy = rollout["b_true_pose_xy"]
per_frame_rmse = rollout["per_frame_rmse"]
per_frame_mse = rollout["per_frame_mse"]
target_t = rollout["target_t"]
sequence_id = str(rollout["sequence_id"])
rollout_length = int(rollout["rollout_length"])
```

Then it builds:

```python
keypoints_true_pair = build_pair_keypoints(a_pose_xy, b_true_pose_xy)
```

and visualizes:

```python
animate_pose_sequence(..., keypoints_pred_pair, ...)
animate_pose_sequence(..., keypoints_true_pair, ...)
```

## 7. Important Caveats

- The current export is not recursive closed-loop rollout.
- Each frame uses the true preprocessed window for that exact `target_t`.
- `start_t` must be at least `rollout_history_frames`.
- `start_t + rollout_length` must not exceed the selected branch length.
- If `rollout_history_frames != trained_history_frames`, the export is a useful
  comparison experiment, but it is not equivalent to training a model with that
  shorter or longer history length.
- `window_index` defaults to 0; choose another value only when the selected
  `sequence_id` contains multiple trial windows and you intentionally want one
  of them.
- `batch-size` is only for export inference throughput; it does not change model
  outputs except for tiny floating-point ordering differences.
- Historical notes or examples that describe feeding predicted pose back into the
  next frame refer to a future closed-loop rollout implementation, not the
  current script.

## 8. Future Closed-Loop Rollout

To implement a true recursive rollout later, the script must maintain a mutable
pose buffer:

```text
1. start from true target branch history before start_t
2. predict target_delta at target_t
3. reconstruct predicted pose
4. write predicted pose into the buffer
5. build the next target_t + 1 input using that predicted history
```

That future implementation must also decide how to handle future context branch
features, future interaction-current features, and future behavior labels.
