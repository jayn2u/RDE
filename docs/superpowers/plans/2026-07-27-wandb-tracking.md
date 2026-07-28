# RDE W&B Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add IRRA-compatible, opt-in W&B retrieval and efficiency tracking to RDE.

**Architecture:** Independent W&B and efficiency utilities are ported into RDE. The evaluator exposes a backward-compatible dictionary path whose primary keys represent BGE+TSE, while the processor measures and logs the full RDE training phase and independent validation phase.

**Tech Stack:** Python 3.12, PyTorch, unittest, Weights & Biases, uv.

## Global Constraints

- Process environment values override the repository root `.env`.
- `WANDB_PROJECT` is normally `rde`, with `rde` as the code fallback.
- Tracking is enabled only by `--wandb`; `run_rde.sh` supplies the flag.
- Only rank 0 initializes and writes to W&B.
- Existing TensorBoard, checkpoint, evaluator, and inference behavior remain compatible.
- W&B tests must not contact the network or require a GPU.

---

### Task 1: W&B lifecycle and efficiency utilities

**Files:**
- Create: `2024-CVPR-RDE/utils/wandb_tracking.py`
- Create: `2024-CVPR-RDE/utils/efficiency.py`
- Create: `tests/__init__.py`
- Create: `tests/test_wandb_tracking.py`
- Modify: `2024-CVPR-RDE/utils/options.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Create locally: `.env`

**Interfaces:**
- Produces: `WandbSession`, `start_train_run(args)`, `log_train_epoch_metrics(...)`, `log_val_metrics(...)`, `finish_train_run(...)`
- Produces: `start_measurement(device)`, `finish_cuda_timer(device, started_at)`, `get_global_processed_examples(...)`, `get_peak_vram_metrics(device)`, `build_epoch_efficiency_metrics(...)`

- [ ] **Step 1: Write failing configuration and payload tests**

```python
def test_project_falls_back_to_rde(self):
    run = start_train_run(self.args, wandb_module=self.fake_wandb)
    self.assertEqual(self.fake_wandb.kwargs["project"], "rde")

def test_primary_validation_errors_use_combined_metrics(self):
    session = RecordingSession()
    log_val_metrics(session, 3, {"t2i_R1": 75.94, "bge_t2i_R1": 73.49})
    self.assertEqual(session.payloads[0]["val/t2i_error@1"], 24.06)
    self.assertEqual(session.payloads[0]["val/bge_t2i_R1"], 73.49)
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `python -m unittest tests.test_wandb_tracking -v`

Expected: FAIL because the tracking utilities do not exist.

- [ ] **Step 3: Port the IRRA utilities with RDE defaults**

Use:

```python
DEFAULT_WANDB_PROJECT = "rde"
output_dir = args.output_dir
group = args.wandb_group or args.dataset_name
```

Resolve the default env file against the RDE repository root so it works from
both the root and `2024-CVPR-RDE/`. Add the IRRA W&B CLI options and
`wandb>=0.28.1`.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_wandb_tracking -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore pyproject.toml uv.lock 2024-CVPR-RDE/utils/options.py 2024-CVPR-RDE/utils/efficiency.py 2024-CVPR-RDE/utils/wandb_tracking.py tests
git commit -m "feat: add rde wandb tracking utilities"
```

### Task 2: Return primary and component retrieval metrics

**Files:**
- Modify: `2024-CVPR-RDE/utils/metrics.py`
- Create: `tests/test_retrieval_metrics.py`

**Interfaces:**
- Extends: `Evaluator.eval(model, i2t_metric=False, return_metrics=False)`
- Produces primary `t2i_*` and `i2t_*` keys from BGE+TSE
- Produces component `bge_{t2i,i2t}_*` and `tse_{t2i,i2t}_*` keys

- [ ] **Step 1: Write failing evaluator-contract tests**

```python
def test_combined_branch_is_primary(self):
    metrics = evaluator.eval(model, i2t_metric=True, return_metrics=True)
    self.assertEqual(metrics["t2i_R1"], 100.0)
    self.assertIn("bge_t2i_R1", metrics)
    self.assertIn("tse_i2t_mAP", metrics)
```

Also assert that `return_metrics=False` still returns the BGE+TSE T2I R@1
float used for checkpoint selection.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_retrieval_metrics -v`

Expected: FAIL because the evaluator has no dictionary return path.

- [ ] **Step 3: Build a flat metrics dictionary during the existing branch loop**

Map `BGE+TSE` to the primary keys (`t2i_R1`, `t2i_mAP`, and peers).
Compute I2T metrics for each branch when requested. Return the dictionary only
when `return_metrics=True`.

- [ ] **Step 4: Run retrieval tests**

Run: `python -m unittest tests.test_retrieval_metrics -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add 2024-CVPR-RDE/utils/metrics.py tests/test_retrieval_metrics.py
git commit -m "feat: expose complete rde retrieval metrics"
```

### Task 3: Instrument RDE training and validation

**Files:**
- Modify: `2024-CVPR-RDE/processor/processor.py`
- Modify: `2024-CVPR-RDE/train.py`
- Modify: `2024-CVPR-RDE/run_rde.sh`
- Create: `tests/test_training_tracking.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Extends: `do_train(..., wandb_session=None) -> tuple[float, int]`
- Adds: `_evaluate_with_efficiency(evaluator, model, device)`

- [ ] **Step 1: Write a failing one-epoch processor test**

Patch the GMM pre-pass, one optimizer batch, CUDA measurement helpers, and
evaluator. Assert the training timer surrounds both the confidence-estimation
and optimizer passes, validation has an independent scope, and only rank 0
logs.

```python
self.assertEqual(train_efficiency["examples_per_second"], 0.4)
self.assertEqual(val_payload["val/t2i_R1"], 75.94)
checkpointer.save.assert_called_once_with("best", num_epoch=1, iteration=0, epoch=1)
```

- [ ] **Step 2: Run the integration test and verify failure**

Run: `python -m unittest tests.test_training_tracking -v`

Expected: FAIL because the processor does not accept a session or return the
best epoch.

- [ ] **Step 3: Add the measurement and logging boundaries**

Start training measurement before `get_loss`, finish after the optimizer loop,
count optimized examples from `meters["loss"].count`, multiply cumulative
seconds by world size, and run validation through
`Evaluator.eval(..., i2t_metric=True, return_metrics=True)`.

- [ ] **Step 4: Own the run in the entry point**

Create the rank-0 session after the timestamped output directory and config
exist. Wrap `do_train` with `try/finally`, write best summaries, and retain the
post-training best/last inference sequence.

- [ ] **Step 5: Enable and document the launcher**

Add `--wandb` to `run_rde.sh` and document:

```bash
WANDB_PROJECT=rde RDE_DATA_ROOT=/mnt/data/lab_datasets uv run bash 2024-CVPR-RDE/run_rde.sh
```

- [ ] **Step 6: Run all tests**

Run: `python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add 2024-CVPR-RDE/processor/processor.py 2024-CVPR-RDE/train.py 2024-CVPR-RDE/run_rde.sh README.md AGENTS.md tests/test_training_tracking.py
git commit -m "feat: log rde training metrics to wandb"
```
