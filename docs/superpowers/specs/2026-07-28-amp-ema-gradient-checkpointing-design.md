# AMP, EMA, and Gradient Checkpointing Design

## Goal

Add optional automatic mixed precision (AMP), exponential moving average
(EMA), and activation gradient checkpointing to RDE training while preserving
the behavior of the existing `run_rde.sh`.

## Command-Line Interface

The training CLI gains these options:

- `--amp` enables CUDA automatic mixed precision.
- `--amp_dtype {fp16,bf16}` selects the autocast dtype and defaults to `fp16`.
- `--ema` enables an exponential moving average of model parameters and
  buffers.
- `--ema_decay` controls EMA decay, defaults to `0.999`, and must be strictly
  between zero and one.
- `--gradient_checkpointing` enables activation checkpointing in CLIP
  Transformer blocks.

All three features are disabled by default. Consequently, existing commands
and `run_rde.sh` retain their current training behavior.

## Training Flow

With AMP enabled, model parameters remain FP32 and both the noisy-sample loss
pass and training forward pass run under CUDA autocast. FP16 uses
`torch.amp.GradScaler`; BF16 does not scale gradients. An unsupported BF16
configuration fails before training begins with a clear error.

After a successful optimizer step, rank 0 updates an EMA model using
`torch.optim.swa_utils.AveragedModel`,
`torch.optim.swa_utils.get_ema_multi_avg_fn`, and `use_buffers=True`. If FP16
overflow causes `GradScaler` to skip the optimizer step, the corresponding EMA
update is also skipped.

Validation uses the EMA model when enabled, and its retrieval result determines
the best checkpoint. Training and noisy correspondence selection continue to
use the current raw model.

## Gradient Checkpointing

Both image and text CLIP Transformers use activation checkpointing at the
`ResidualAttentionBlock` boundary. A block is checkpointed only when the option
is enabled, the Transformer is in training mode, and autograd is enabled.
Checkpointing uses the recommended non-reentrant implementation:
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.

Validation and inference do not recompute blocks. The TSE MLPs are not
checkpointed because their activation footprint is comparatively small.
Hard-coded FP16 input conversions in those layers are replaced with
parameter-dtype-aware conversions so FP16, BF16 autocast, and the legacy
half-precision path remain compatible.

## Distributed Training

The EMA model wraps the unwrapped RDE module rather than its DDP wrapper. It is
created, updated, validated, and saved only on rank 0, avoiding one EMA replica
per worker. DDP continues to synchronize the raw training model normally.

## Checkpoint Compatibility

New checkpoints contain:

- `model`
- `ema_model` when enabled, including its averaging counter
- `optimizer`
- `scheduler`
- `scaler` when supplied
- existing epoch and iteration metadata

Resume restores every available state. Missing EMA or scaler state is accepted
so older checkpoints remain usable. Standalone evaluation and the post-training
evaluation prefer EMA weights when present and otherwise load `model`.

## Launcher

`run_rde.sh` is unchanged. A new `run_rde_amp_ema_gc.sh` copies the current
experiment configuration, invokes training through `uv run python`, and enables
AMP FP16, EMA with decay `0.999`, and gradient checkpointing.

## Validation

Automated tests cover:

- CLI defaults, explicit feature options, and EMA decay validation
- autocast/scaler optimizer stepping and overflow detection
- EMA update, validation, and best-checkpoint selection
- checkpoint save/resume and EMA-preferred backward-compatible loading
- checkpoint activation only during gradient-enabled training
- launcher syntax and required flags
- the complete existing test suite

A small CUDA smoke check is run when CUDA is available. Full dataset training is
outside the implementation verification scope.
