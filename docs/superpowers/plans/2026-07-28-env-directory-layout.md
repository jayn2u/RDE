# Environment Directory Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move local configuration to ignored `env/.env` and add a tracked, copyable `env/.env.example`.

**Architecture:** The existing dotenv parser remains unchanged; only its repository-relative default moves to `env/.env`. Git tracks the safe example while excluding the local file, and all setup guidance uses the same copy command.

**Tech Stack:** Git, `.gitignore`, Python 3.12, `unittest`, `uv`, Markdown

## Global Constraints

- Preserve the values currently stored in the local root `.env`.
- `env/.env` contains machine-specific paths and credentials and is ignored.
- `env/.env.example` contains safe placeholder/default values and is tracked.
- Both files contain `RDE_DATA_ROOT`, `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT`.
- Process environment variables override file values; explicit CLI values override defaults.
- Work on `codex/gitignore-env-consolidation` and update draft PR #2.

---

### Task 1: Require the environment-directory path

**Files:**
- Modify: `tests/test_wandb_tracking.py`
- Modify: `2024-CVPR-RDE/utils/wandb_tracking.py`
- Modify: `2024-CVPR-RDE/utils/options.py`

**Interfaces:**
- Consumes: `start_train_run(args)` and `get_args()`.
- Produces: default configuration lookup at `<PROJECT_ROOT>/env/.env`.

- [x] **Step 1: Change the behavior fixtures before production code**

Update both default-path tests to create a temporary `env/` directory and
write their fixture to `env/.env`:

```python
env_dir = os.path.join(project_root, "env")
os.makedirs(env_dir)
env_path = os.path.join(env_dir, ".env")
```

Rename the tests to
`test_default_env_file_is_repository_env_dotenv` and
`test_dataset_root_defaults_from_repository_env_dotenv`.

- [x] **Step 2: Run focused tests and verify the expected failures**

From `2024-CVPR-RDE/`, run:

```bash
uv run python -m unittest discover -s ../tests -p test_wandb_tracking.py -v
```

Expected: the two renamed tests fail because production still reads
`<PROJECT_ROOT>/.env`.

- [x] **Step 3: Change the two production defaults**

In `utils/wandb_tracking.py`, use:

```python
return op.join(PROJECT_ROOT, "env", ".env")
```

In `utils/options.py`, build the dataset env path the same way and set
`--wandb_env_file` to `env/.env`.

- [x] **Step 4: Rerun focused tests**

Run the same focused command. Expected: all seven tests pass.

### Task 2: Create the local file and tracked template

**Files:**
- Move locally: `.env` to `env/.env`
- Create: `env/.env.example`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: the environment path from Task 1.
- Produces: an ignored local file and a trackable template with identical keys.

- [x] **Step 1: Move the local values**

Create `env/.env` with the exact current root `.env` contents, then remove
the root file.

- [x] **Step 2: Add the tracked example**

Create `env/.env.example`:

```dotenv
RDE_DATA_ROOT=/mnt/data/lab_datasets
WANDB_API_KEY=your-wandb-api-key
WANDB_ENTITY=your-wandb-entity
WANDB_PROJECT=rde
```

- [x] **Step 3: Adjust ignore rules**

Remove `env/` from the virtual-environment patterns and add:

```gitignore
!env/.env.example
```

The existing `.env` and `.env.*` patterns continue to ignore local dotenv
files.

- [x] **Step 4: Verify Git visibility and inventory**

Run:

```bash
git check-ignore -q env/.env
! git check-ignore -q env/.env.example
test -f env/.env
test -f env/.env.example
test ! -e .env
```

Expected: every command exits successfully.

### Task 3: Align documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: existing W&B and environment design/plan documents under `docs/superpowers/`

**Interfaces:**
- Consumes: the file layout from Task 2.
- Produces: one consistent setup command and environment path across guidance.

- [x] **Step 1: Update user setup**

Document:

```bash
cp env/.env.example env/.env
```

Explain that users then fill in W&B credentials and dataset path in
`env/.env`.

- [x] **Step 2: Update repository guidance and prior documents**

Replace active root-`.env` descriptions with `env/.env`, describe the tracked
example, and retain the existing precedence rules.

- [x] **Step 3: Search active references**

Run:

```bash
rg -n 'root `\.env`|root \.env|PROJECT_ROOT, "\.env"|default="\.env"|<repository>/\.env' \
  README.md AGENTS.md 2024-CVPR-RDE tests \
  docs/superpowers/specs/2026-07-27-wandb-tracking-design.md \
  docs/superpowers/specs/2026-07-28-gitignore-env-consolidation-design.md
```

Expected: no stale root-layout matches.

### Task 4: Verify, commit, and update the PR

**Files:**
- Review all files changed by Tasks 1–3.

**Interfaces:**
- Consumes: the complete layout update.
- Produces: a verified follow-up commit pushed to PR #2.

- [x] **Step 1: Run full verification**

From `2024-CVPR-RDE/`:

```bash
uv run python -m unittest discover -s ../tests -v
bash -n run_rde.sh
```

Expected: 10 tests pass and shell syntax is valid.

- [x] **Step 2: Audit staged scope**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm `env/.env` is absent from the staged file list and
`env/.env.example` is present.

- [x] **Step 3: Commit and push**

Commit with:

```text
refactor: move configuration into env directory

Co-authored-by: Codex <codex@openai.com>
```

Push `codex/gitignore-env-consolidation` to refresh draft PR #2.
