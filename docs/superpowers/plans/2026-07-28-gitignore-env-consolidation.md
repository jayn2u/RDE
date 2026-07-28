# Gitignore and Environment Consolidation Implementation Plan

> Superseded for environment-file placement by
> `2026-07-28-env-directory-layout.md`. This document records the earlier
> implementation sequence.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate local configuration into one ignored repository-root `.env` and prevent generated Python/ML artifacts from being tracked.

**Architecture:** The existing W&B configuration loader remains responsible for reading a dotenv-style file, but its default moves to root `.env`. The argument parser reuses the same precedence for `RDE_DATA_ROOT`, and the shell launcher leaves that default intact. Git ignore rules define the repository boundary for secrets and generated artifacts, while current documentation and tests enforce the new location.

**Tech Stack:** Git, `.gitignore`, Bash, Python 3.12, `unittest`, `uv`

## Global Constraints

- Preserve all current uncommitted source changes.
- Work on `codex/gitignore-env-consolidation`, not `main`.
- Use one ignored root `.env` containing `RDE_DATA_ROOT` and the three W&B settings.
- Retain the current deletions of tracked `__pycache__/*.pyc` and `.DS_Store`.
- Keep source, tests, `uv.lock`, documentation, and `2024-CVPR-RDE/noiseindex/*.npy` trackable.
- Run Python through `uv run python` from `2024-CVPR-RDE/`.

---

### Task 1: Make root `.env` the observable W&B default

**Files:**
- Modify: `tests/test_wandb_tracking.py`
- Modify: `2024-CVPR-RDE/utils/wandb_tracking.py`
- Modify: `2024-CVPR-RDE/utils/options.py`
- Modify: `2024-CVPR-RDE/run_rde.sh`

**Interfaces:**
- Consumes: `start_train_run(args)` and its existing environment/CLI precedence.
- Produces: default repository-root `.env` resolution when `wandb_env_file` is empty or omitted.
- Produces: `RDE_DATA_ROOT` resolution from process environment, root `.env`, then the built-in fallback.

- [x] **Step 1: Write the failing behavior test**

Add a test that patches `utils.wandb_tracking.PROJECT_ROOT` to a temporary
directory, writes `.env` there with literal W&B settings, invokes
`start_train_run()` with no explicit environment-file override, and asserts
that the real configuration flow passes those settings to `wandb.init`.

- [x] **Step 2: Run the focused test and verify the expected failure**

Run from `2024-CVPR-RDE/`:

```bash
uv run python -m unittest discover -s ../tests -p test_wandb_tracking.py -v
```

Expected: the new test fails because the current loader checks
`<PROJECT_ROOT>/env/.env`, so it falls back to project `rde` instead of the
literal project stored in `<PROJECT_ROOT>/.env`.

- [x] **Step 3: Write and verify the dataset-root failing test**

Patch the options module's project root to a temporary directory containing a
root `.env`, parse otherwise-default arguments, and assert `args.root_dir`
uses the file's `RDE_DATA_ROOT`. Verify it fails with the built-in fallback
before implementation.

- [x] **Step 4: Implement the minimal path changes**

Change `_env_file()` to return `os.path.join(PROJECT_ROOT, ".env")` when no
explicit path is configured. Change `--wandb_env_file` to default to `.env`
and describe it as repository-root-relative. Resolve `--root_dir` through the
existing dotenv reader, and remove the launcher's explicit fallback
`--root_dir` argument so it does not mask the parser default.

- [x] **Step 5: Run the focused tests and verify they pass**

```bash
uv run python -m unittest discover -s ../tests -p test_wandb_tracking.py -v
```

Expected: all environment and W&B tracking tests pass.

### Task 2: Consolidate environment files and define ignore policy

**Files:**
- Create locally, ignored: `.env`
- Modify: `.gitignore`
- Delete: `env/.env`
- Delete: `env/.env.example`
- Delete: `2024-CVPR-RDE/env/.env.example`

**Interfaces:**
- Consumes: root `.env` path from Task 1.
- Produces: one local dotenv file and deterministic Git visibility rules.

- [x] **Step 1: Populate the single root environment file**

Use these keys:

```dotenv
RDE_DATA_ROOT=/mnt/data/lab_datasets
WANDB_API_KEY=your-wandb-api-key
WANDB_ENTITY=your-wandb-entity
WANDB_PROJECT=rde
```

- [x] **Step 2: Replace the empty `.gitignore`**

Add focused groups for root `.env`, Python bytecode and caches, virtual
environments, OS/editor files, coverage/build outputs, training outputs,
W&B state, model checkpoints, downloaded artifacts, and local agent
worktrees. Do not ignore `uv.lock`, tests, docs, or noise indexes.

- [x] **Step 3: Remove legacy environment files**

Delete all three legacy files after their settings have been represented in
root `.env`. The old `DATASET_ROOT` key is corrected to the runtime-supported
`RDE_DATA_ROOT`.

- [x] **Step 4: Verify Git behavior**

Run:

```bash
git check-ignore -v .env 2024-CVPR-RDE/run_logs/example \
  2024-CVPR-RDE/results/example 2024-CVPR-RDE/__pycache__/x.pyc \
  2024-CVPR-RDE/model.pth downloads/example .claude/worktrees/example
git check-ignore -q README.md && exit 1 || true
git check-ignore -q uv.lock && exit 1 || true
git check-ignore -q tests/test_wandb_tracking.py && exit 1 || true
git check-ignore -q 2024-CVPR-RDE/noiseindex/CUHK-PEDES_0.2.npy && exit 1 || true
```

Expected: generated/local examples report matching ignore rules; trackable
examples do not match.

### Task 3: Update current instructions and remove stale paths

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-07-27-wandb-tracking-design.md`
- Modify: `docs/superpowers/plans/2026-07-27-wandb-tracking.md`

**Interfaces:**
- Consumes: root `.env` layout from Tasks 1–2.
- Produces: one consistent setup path across current repository guidance.

- [x] **Step 1: Update setup instructions**

Replace copy-from-example directions with an explicit root `.env` snippet
containing `RDE_DATA_ROOT` and the W&B keys. State that process variables
override root `.env`.

- [x] **Step 2: Update repository guidance and historical references**

Change supported environment-file paths to `<repository>/.env`, remove
instructions to keep `.env.example`, and update the earlier W&B design/plan
where they prescribe `env/.env`.

- [x] **Step 3: Search for stale environment layout references**

Run:

```bash
rg -n 'env/\.env|\.env\.example|(^|[^A-Z_])DATASET_ROOT' \
  README.md AGENTS.md 2024-CVPR-RDE tests \
  docs/superpowers/specs/2026-07-27-wandb-tracking-design.md \
  docs/superpowers/plans/2026-07-27-wandb-tracking.md
```

Expected: no matches.

### Task 4: Verify and commit the complete change

**Files:**
- Review all files changed by Tasks 1–3 plus the inherited generated-file deletions.

**Interfaces:**
- Consumes: all preceding task outputs.
- Produces: a verified Git commit on the dedicated branch.

- [x] **Step 1: Run the full test suite**

From `2024-CVPR-RDE/`:

```bash
uv run python -m unittest discover -s ../tests -v
```

Expected: 10 tests pass after adding the two environment-path tests.

- [x] **Step 2: Review repository state and whitespace**

```bash
git diff --check
git status --short --branch
git diff --stat
```

Confirm root `.env` is ignored, legacy environment files are absent, intended
source and documentation changes remain present, and generated files no
longer appear as untracked clutter.

- [x] **Step 3: Commit without exposing `.env`**

Stage the intended tracked changes explicitly, verify the staged file list,
and commit with the human user's configured author plus:

```text
Co-authored-by: Codex <codex@openai.com>
```
