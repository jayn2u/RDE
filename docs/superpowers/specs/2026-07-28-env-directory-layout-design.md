# Environment Directory Layout Design

## Goal

Store local dataset and W&B configuration in an ignored `env/.env` while
providing a tracked `env/.env.example` that users can copy.

## File layout

The repository uses exactly two environment files:

- `env/.env` contains machine-specific paths and credentials and is ignored.
- `env/.env.example` contains safe placeholder/default values and is tracked.

Both files use the same keys:

```dotenv
RDE_DATA_ROOT=/mnt/data/lab_datasets
WANDB_API_KEY=your-wandb-api-key
WANDB_ENTITY=your-wandb-entity
WANDB_PROJECT=rde
```

The existing root `.env` is moved to `env/.env` without changing its local
values.

## Loading and precedence

The W&B loader and dataset argument default resolve `env/.env` relative to
the repository root. Process environment variables continue to override file
values, explicit CLI arguments continue to override defaults, and missing
values retain the existing built-in fallbacks.

## Ignore policy and documentation

The `env/` directory is no longer ignored as a generic virtual environment.
Git ignores `env/.env` and allows `env/.env.example`. Other conventional
virtual-environment directories such as `.venv/`, `venv/`, and `ENV/` remain
ignored.

Setup instructions use:

```bash
cp env/.env.example env/.env
```

Current guidance and historical design documents are updated to describe the
same layout.

## Verification

1. A behavior test proves the default W&B and dataset configuration path is
   `<repository>/env/.env`.
2. The full Python test suite passes.
3. `git check-ignore` confirms `env/.env` is ignored and
   `env/.env.example` is trackable.
4. Environment-file inventory confirms the root `.env` is absent and only
   the two files under `env/` exist.
5. The branch is pushed to update draft PR #2.
