# AMP, EMA, and Gradient Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in AMP, EMA-backed validation/checkpointing, and CLIP Transformer activation checkpointing while preserving the legacy launcher.

**Architecture:** Add focused precision/EMA helpers in `utils/training.py`, thread them through the existing trainer and checkpointer, and put activation checkpointing at the existing Transformer block boundary. Keep every new option disabled by default and provide a separate launcher that enables the complete feature set.

**Tech Stack:** Python 3.11, PyTorch 2.9.1 CUDA 12.8, `unittest`, Bash, uv

## Global Constraints

- Existing `2024-CVPR-RDE/run_rde.sh` behavior must remain unchanged.
- AMP dtype choices are exactly `fp16` and `bf16`; the default is `fp16`.
- EMA decay defaults to `0.999` and must satisfy `0 < decay < 1`.
- Gradient checkpointing applies only to gradient-enabled training in CLIP Transformer blocks and uses `use_reentrant=False`.
- New checkpoints prefer EMA weights for evaluation while remaining compatible with checkpoints that contain only `model`.
- Python commands use `uv run python`.

---

### Task 1: Training Feature Options and Utilities

**Files:**
- Create: `2024-CVPR-RDE/utils/training.py`
- Modify: `2024-CVPR-RDE/utils/options.py`
- Create: `tests/test_training_features.py`

**Interfaces:**
- Produces: `validate_training_options(args) -> None`
- Produces: `get_autocast_dtype(name: str) -> torch.dtype`
- Produces: `autocast_context(enabled: bool, dtype_name: str)`
- Produces: `build_grad_scaler(enabled: bool, dtype_name: str) -> torch.amp.GradScaler`
- Produces: `optimizer_step(loss, optimizer, scaler) -> bool`
- Produces: `unwrap_model(model) -> torch.nn.Module`
- Produces: `build_ema_model(model, decay: float) -> AveragedModel`

- [ ] **Step 1: Write failing CLI and utility tests**

```python
def test_training_feature_options_are_disabled_by_default():
    args = parse_args(["train.py"])
    assert not args.amp
    assert args.amp_dtype == "fp16"
    assert not args.ema
    assert args.ema_decay == 0.999
    assert not args.gradient_checkpointing

def test_optimizer_step_reports_scaler_overflow():
    scaler = FakeScaler(old_scale=8.0, new_scale=4.0)
    assert optimizer_step(loss, optimizer, scaler) is False
```

- [ ] **Step 2: Run the focused tests and verify missing options/helpers fail**

Run: `uv run python -m unittest tests.test_training_features -v`

Expected: FAIL because the CLI fields and `utils.training` do not exist.

- [ ] **Step 3: Implement options, validation, and helpers**

```python
def optimizer_step(loss, optimizer, scaler):
    scaler.scale(loss).backward()
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= old_scale
```

Use `torch.autocast(device_type="cuda", dtype=...)`,
`torch.amp.GradScaler("cuda", enabled=...)`, and
`AveragedModel(unwrap_model(model), multi_avg_fn=get_ema_multi_avg_fn(decay),
use_buffers=True)`.

- [ ] **Step 4: Run focused tests**

Run: `uv run python -m unittest tests.test_training_features -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 2024-CVPR-RDE/utils/options.py 2024-CVPR-RDE/utils/training.py tests/test_training_features.py
git commit -m "feat: add AMP and EMA training utilities" \
  -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 2: CLIP Transformer Activation Checkpointing

**Files:**
- Modify: `2024-CVPR-RDE/model/clip_model.py`
- Modify: `2024-CVPR-RDE/model/build.py`
- Modify: `2024-CVPR-RDE/model/CrossEmbeddingLayer_tse.py`
- Modify: `tests/test_training_features.py`

**Interfaces:**
- Consumes: `args.gradient_checkpointing: bool`
- Produces: `Transformer.set_gradient_checkpointing(enabled: bool) -> None`
- Produces: `CLIP.set_gradient_checkpointing(enabled: bool) -> None`

- [ ] **Step 1: Write failing Transformer behavior tests**

```python
def test_transformer_checkpoints_each_block_only_during_training():
    transformer = Transformer(width=8, layers=2, heads=1)
    transformer.set_gradient_checkpointing(True)
    transformer.train()
    with patch("model.clip_model.checkpoint", wraps=checkpoint) as wrapped:
        transformer([torch.randn(3, 2, 8, requires_grad=True)])
    assert wrapped.call_count == 2
    assert all(call.kwargs["use_reentrant"] is False for call in wrapped.call_args_list)
```

Also assert no checkpoint calls in eval mode or under `torch.no_grad()`.

- [ ] **Step 2: Run the focused test and verify the missing setter fails**

Run: `uv run python -m unittest tests.test_training_features.TrainingFeatureTest.test_transformer_checkpoints_only_during_training -v`

Expected: FAIL because checkpointing is not implemented.

- [ ] **Step 3: Implement block-level checkpointing and dtype compatibility**

Replace `nn.Sequential` execution with an explicit residual-block loop. Propagate
the CLI option through `CLIP` and build the RDE model in FP32 when AMP is
enabled. Replace TSE `.half()` input casts with casts based on the destination
layer's parameter dtype.

- [ ] **Step 4: Run focused tests**

Run: `uv run python -m unittest tests.test_training_features -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 2024-CVPR-RDE/model/clip_model.py 2024-CVPR-RDE/model/build.py \
  2024-CVPR-RDE/model/CrossEmbeddingLayer_tse.py tests/test_training_features.py
git commit -m "feat: checkpoint CLIP transformer blocks" \
  -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 3: EMA Training, Validation, and Checkpoint State

**Files:**
- Modify: `2024-CVPR-RDE/train.py`
- Modify: `2024-CVPR-RDE/processor/processor.py`
- Modify: `2024-CVPR-RDE/utils/checkpoint.py`
- Modify: `2024-CVPR-RDE/test.py`
- Modify: `tests/test_training_tracking.py`
- Create: `tests/test_checkpoint_training_state.py`

**Interfaces:**
- Consumes: `do_train(..., scaler=None, ema_model=None)`
- Produces: `Checkpointer(..., scaler=None, ema_model=None)`
- Produces: `Checkpointer.load(f=None, prefer_ema: bool = False)`
- Produces: `Checkpointer.resume(f=None)` restoring raw model, optimizer,
  scheduler, scaler, and EMA state when available

- [ ] **Step 1: Write failing checkpoint and trainer tests**

```python
def test_checkpoint_prefers_ema_weights_when_present():
    checkpointer = Checkpointer(target)
    checkpointer.load(path, prefer_ema=True)
    assert target.weight.item() == ema_weight

def test_one_epoch_updates_and_validates_ema_model():
    do_train(..., scaler=FakeScaler(), ema_model=ema_model)
    ema_model.update_parameters.assert_called_once_with(raw_model)
    evaluator.eval.assert_called_with(ema_model.module.eval(), ...)
```

Add a legacy checkpoint assertion showing `prefer_ema=True` falls back to
`model`, and a scaler/EMA resume round trip.

- [ ] **Step 2: Run the focused tests and verify expected failures**

Run: `uv run python -m unittest tests.test_checkpoint_training_state tests.test_training_tracking -v`

Expected: FAIL because extra checkpoint state and EMA trainer arguments are not
implemented.

- [ ] **Step 3: Implement the AMP/EMA trainer path and checkpoint state**

Wrap noisy-loss and training forwards in `autocast_context`. Replace direct
backward/step with `optimizer_step`, update EMA only when it returns true, and
validate the EMA module on rank 0. Extend checkpoint save/resume and make
standalone/post-training evaluation pass `prefer_ema=True`.

- [ ] **Step 4: Run focused tests**

Run: `uv run python -m unittest tests.test_checkpoint_training_state tests.test_training_tracking -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 2024-CVPR-RDE/train.py 2024-CVPR-RDE/test.py \
  2024-CVPR-RDE/processor/processor.py 2024-CVPR-RDE/utils/checkpoint.py \
  tests/test_training_tracking.py tests/test_checkpoint_training_state.py
git commit -m "feat: train and evaluate with EMA weights" \
  -m "Co-authored-by: Codex <codex@openai.com>"
```

### Task 4: Enhanced Launcher and Regression Verification

**Files:**
- Create: `2024-CVPR-RDE/run_rde_amp_ema_gc.sh`
- Create: `tests/test_enhanced_training_script.py`

**Interfaces:**
- Produces: executable `run_rde_amp_ema_gc.sh`

- [ ] **Step 1: Write a failing launcher test**

```python
def test_launcher_invokes_training_with_all_features():
    with fake_uv_executable() as invocation_file:
        subprocess.run(
            ["bash", "2024-CVPR-RDE/run_rde_amp_ema_gc.sh"],
            check=True,
            env=fake_uv_environment(invocation_file),
        )
    invocation = invocation_file.read_text()
    assert invocation.startswith("run python train.py ")
    assert "--amp" in invocation.split()
    assert ["--amp_dtype", "fp16"] == adjacent_args(invocation, "--amp_dtype")
    assert "--ema" in invocation.split()
    assert ["--ema_decay", "0.999"] == adjacent_args(invocation, "--ema_decay")
    assert "--gradient_checkpointing" in invocation.split()
```

- [ ] **Step 2: Run the launcher test and verify the missing file fails**

Run: `uv run python -m unittest tests.test_enhanced_training_script -v`

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Add the launcher without modifying `run_rde.sh`**

Copy the existing experiment variables and arguments, change the Python command
to `uv run python`, and append the five feature arguments from the global
constraints.

- [ ] **Step 4: Run focused and full verification**

Run: `bash -n 2024-CVPR-RDE/run_rde_amp_ema_gc.sh`

Run: `uv run python -m unittest discover -s tests -v`

Expected: both commands PASS.

- [ ] **Step 5: Run CUDA smoke verification when CUDA is available**

Run:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

If CUDA is available, instantiate a small `Transformer`, run FP16 autocast plus
scaled backward, and confirm gradients are finite. Do not start dataset
training.

- [ ] **Step 6: Commit**

```bash
git add 2024-CVPR-RDE/run_rde_amp_ema_gc.sh tests/test_enhanced_training_script.py
git commit -m "feat: add enhanced RDE training launcher" \
  -m "Co-authored-by: Codex <codex@openai.com>"
```
