# RDE W&B Tracking Design

## Goal

Add opt-in Weights & Biases tracking to RDE with the same run lifecycle,
primary metric names, environment-variable precedence, and efficiency
measurements as the local IRRA project.

## Run configuration

- `--wandb` enables tracking; without it, all tracking calls are no-ops.
- `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` are read from the
  process environment first and then from the repository's `env/.env`.
- `--wandb_env_file` overrides the env-file path.
- `--wandb_project`, `--wandb_entity`, `--wandb_run_name`,
  `--wandb_group`, `--wandb_tags`, and `--wandb_notes` override optional run
  metadata.
- The project fallback is `rde`; the normal configured value is
  `WANDB_PROJECT=rde`.
- The run name defaults to the timestamped output-directory basename and the
  group defaults to the dataset name.
- Only rank 0 owns a W&B run.
- Run metadata is written to `wandb_meta.json` and `wandb_run_id` inside the
  timestamped output directory.

## Metrics

All metrics use `epoch` as their W&B step metric.

Training logs the existing RDE epoch meters and IRRA efficiency keys:

- `train/loss`
- `train/bge_loss`
- `train/tse_loss`
- `train/id_loss`
- `train/img_acc`
- `train/txt_acc`
- `train/lr`
- `train/temperature`
- `train/epoch_seconds`
- `train/examples_per_second`
- `train/cumulative_gpu_hours`
- `train/peak_vram_allocated_mb`
- `train/peak_vram_reserved_mb`

The final BGE+TSE retrieval result is the primary IRRA-compatible validation
series:

- `val/t2i_R{1,5,10}`, `val/t2i_mAP`, `val/t2i_mINP`
- `val/i2t_R{1,5,10}`, `val/i2t_mAP`, `val/i2t_mINP`
- `val/t2i_error@{1,5,10}` and `val/i2t_error@{1,5,10}`

BGE and TSE component results are retained under
`val/bge_{t2i,i2t}_*` and `val/tse_{t2i,i2t}_*`. Validation also logs
`val/epoch_seconds`, `val/peak_vram_allocated_mb`, and
`val/peak_vram_reserved_mb`.

The final summary records `val/best_t2i_R1`, `val/best_t2i_error@1`,
`val/best_epoch`, the best-checkpoint path, and the output directory.

## Integration

IRRA's focused `utils/wandb_tracking.py` and `utils/efficiency.py` patterns are
ported into `2024-CVPR-RDE/utils/`, keeping RDE independent from the IRRA
checkout.

`Evaluator.eval()` keeps returning final BGE+TSE T2I R@1 by default. With
`return_metrics=True`, it returns the complete flat dictionary for all three
similarity branches and both retrieval directions. The epoch training
measurement includes RDE's confidence-estimation/GMM pass and optimizer pass,
but excludes validation and checkpoint I/O. Validation has its own timer and
peak-VRAM reset. Existing checkpoint selection remains based on final BGE+TSE
T2I R@1.

The training entry point creates and finalizes the rank-0 W&B session. The
existing shell launcher enables `--wandb`.

## Failure behavior

- Missing `wandb` is an error only when `--wandb` is supplied.
- A missing API key emits a warning and allows an existing W&B login.
- Tracking is finalized in `finally`, including when training raises.
- Existing training, TensorBoard logging, checkpointing, and inference remain
  unchanged when W&B is disabled.

## Testing

Standard-library `unittest` tests cover disabled sessions, environment
precedence, project fallback, primary/component payload names, error metrics,
metric-return compatibility, timing/VRAM helpers, and the one-epoch
integration boundary. Tests use fake W&B runs and synthetic similarity
matrices; they do not contact W&B or require a GPU.
