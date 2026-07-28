# Gitignore and Environment Consolidation Design

## Goal

Keep generated, local, and sensitive files out of version control while
consolidating repository configuration under `env/`.

## Environment layout

The repository uses an ignored `env/.env` and a tracked
`env/.env.example`. Both contain:

```dotenv
RDE_DATA_ROOT=/mnt/data/lab_datasets
WANDB_API_KEY=your-wandb-api-key
WANDB_ENTITY=your-wandb-entity
WANDB_PROJECT=rde
```

The W&B loader and CLI default resolve `env/.env` relative to the repository
root. The dataset argument default reads `RDE_DATA_ROOT` from the same file,
and the shell launcher does not mask that default with its own fallback.
Process environment variables and explicit CLI settings continue to take
precedence.

## Ignore policy

The root `.gitignore` uses focused repository-specific groups:

- secrets and local environment configuration;
- Python bytecode, virtual environments, and tool caches;
- operating-system and editor metadata;
- build and test outputs;
- training logs, results, W&B state, checkpoints, and downloaded artifacts;
- local agent worktrees.

Source code, tests, documentation, dependency locks, and the supplied noise
indexes remain trackable. Existing deletions of tracked Python bytecode and
`.DS_Store` are retained so Git stops tracking those generated files.

## Documentation and compatibility

Current setup instructions copy `env/.env.example` to `env/.env` and use
`RDE_DATA_ROOT`, matching the existing Python and shell configuration.
Historical W&B design and plan documents describe the same environment-file
location.

## Verification

Verification covers:

1. `git check-ignore` confirms secrets and generated artifacts are ignored
   while source, lock, and noise-index files remain trackable.
2. A search of active code and guidance finds no stale root environment path
   or unsupported dataset-root key.
3. Tests confirm the default W&B environment path and dataset root use the
   repository `env/.env` while preserving process-environment precedence.
4. The full existing test suite runs with `uv run python`.
