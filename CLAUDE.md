# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`pan_lab` is the research library for **Phase Accumulator Networks (PAN)** — networks whose primitive operation is sinusoidal phase addition (integer-modular) rather than float multiply-accumulate. It is the successor to a monolithic `pan.py` and is organized as a library + YAML-driven experiment runner used to produce figures for the companion paper.

## Commands

Dependencies are managed with `uv`. Shell out via `uv run ...`.

```
uv run pytest                                   # full suite (~25s on CPU)
uv run pytest tests/test_trainer.py             # one file
uv run pytest tests/test_models.py::test_name   # one test
make test                                       # same as pytest
make smoke                                      # ~10s end-to-end sanity run at P=11, K=3

uv run python -m pan_lab --list                 # registered experiment names
uv run python -m pan_lab experiments/FOO.yaml --dry-run   # print plan, no training
uv run python -m pan_lab experiments/FOO.yaml             # actually run
uv run python -m pan_lab --replot results/tier3           # regen plots from CSVs
uv run python -m pan_lab --ad-hoc grid_sweep --p 113 --k 9 --steps 50000

make plan                                       # dry-run every YAML, verify plans
make paper                                      # runs the paper-submission experiment set
```

Single-experiment make targets exist for every YAML (`make tier3`, `make k8_sweep`, etc.) — see the `Makefile`.

## Architecture

The codebase separates **what you run** (`experiments/*.yaml`) from **how it runs** (`pan_lab.trainer`) from **the primitives** (`pan_lab.models`, `pan_lab.data`). Everything that shapes a run lives on one flat `RunConfig` dataclass; nothing about a run is implicit.

### Data flow of a single run

1. `cli.py` loads a YAML → `run_from_yaml` in `experiments.py`.
2. For the common case (`experiment: grid_sweep`), `pan_lab.grid_sweep.run_grid_sweep` expands the `grid:` into a list of `RunConfig`s and calls `_run_cfgs`. The three bespoke experiments (`sifp16_inference`, `decoder_swap`, `decoder_analysis`) have their own functions in `experiments.py`.
3. For each cfg: `make_model(cfg)` (seeds RNG first), `make_modular_dataset(...)`, `train(model, cfg, tx, ty, vx, vy, hooks=...)`.
4. `ExperimentReporter` accumulates per-run rows, writes `runs.csv`, `curves.csv`, optional `slots.csv`/`ablations.csv`/`checkpoints.csv`, and a `manifest.json` with provenance (git SHA, torch version, device, hostname, argv).
5. Plots in `plots.py` consume DataFrames only — so `--replot` regenerates figures from the CSVs without retraining.

### Module map

- `config.py` — `RunConfig` (flat dataclass, YAML-serializable, `short_id()` hashes it), device selection, `capture_provenance()`, constants `TWO_PI`, `PHASE_SCALE=65536`, `SIFP16_QUANT_ERR = 2π/65536`.
- `data.py` — `make_modular_dataset` for `mod_add`, `mod_mul`, `mod_two_step`. `mod_two_step` uses 3 inputs; the others use 2.
- `models/pan.py` — `PhaseEncoder` (learnable freqs, `fourier` or `random` init), `PhaseMixingLayer` (linear + `% 2π`), `PhaseGate` (`(1+cos(φ-φ_ref))/2`, ref wrapped in forward), `PhaseAccumulatorNetwork` (N encoders → concat → mix → gate → linear decoder).
- `models/transformer.py` — Nanda-style 1-layer baseline. `weight_decay=1.0` is the tuned default (vs `0.01` for PAN).
- `models/quantize.py` — SIFP-16 straight-through quantizer.
- `trainer.py` — one generic loop for all models/tasks. Handles AdamW, eval cadence, grok detection, early stop, and diversity regularization.
- `hooks.py` — duck-typed callbacks with optional `on_run_start` / `on_step_start` / `on_eval` / `on_end`. `CheckpointLogger` records frequencies + decoder Fourier concentration per eval; `CSVStreamLogger` streams each eval to disk so crashes don't lose curves.
- `analysis.py` — post-hoc mechanistic metrics (ablations, `fourier_concentration`, `slot_activation_census`, `detect_mode_collapse`). Analyzers never mutate model state and always return plain dicts/DataFrames.
- `reporting.py` — `ExperimentReporter` turns `TrainResult`s into DataFrames and CSVs. Every varying knob ends up as a column.
- `experiments.py` — `EXPERIMENT_REGISTRY` populated via `@register("name")`. Registers `grid_sweep` (the generic sweep) plus three bespoke post-training-analysis experiments: `sifp16_inference`, `decoder_swap`, `decoder_analysis`. Also defines `load_experiment_yaml` and `run_from_yaml`.
- `grid_sweep.py` — the single generic experiment function that every sweep YAML dispatches to. Holds `HOOK_REGISTRY` (string → hook class) and `PLOT_REGISTRY` (string → plot_fn + required DataFrames + default filename). `_expand_grid` accepts either a `dict[field, list]` (Cartesian product) or a `list[dict]` (explicit per-sub-run overrides, for coupled axes like `tf_sweep`'s `d_model`/`n_heads`/`d_mlp`).
- `plots.py` — matplotlib figures; all take DataFrames only. Used both at run end and from `--replot`.

### Invariants to preserve when editing

- **Diversity regularization must go through `model.mix_features()` under autograd.** The old pan.py computed `phi_a`/`phi_b` inside `torch.no_grad()` before the Gram penalty, so encoder frequencies never received the diversity gradient. `tests/test_gradient_flow.py` locks this in — do not break it.
- **`make_model` seeds the RNG before constructing the model.** Two runs with identical `RunConfig`s must produce bit-identical initial weights on the same device. Seeding inside `train()` (as the old code did) is not enough.
- **`use_compile=False` is the default** because `torch.compile` changes MPS float accumulation order and hurts reproducibility. Don't flip this casually.
- **`PhaseGate.ref_phase` is stored unconstrained; the wrap happens in `forward` via `torch.remainder`.** Don't "fix" this by wrapping the parameter in place — you'll cause gradient spikes at cosine inflection points.
- **Unknown fields in YAML `base:` raise a warning, not an error** (so YAML can carry future-reader comments-as-fields). Preserve that behavior in `RunConfig.from_dict`.
- **`n_inputs` is derived from `task_kind`** (3 for `mod_two_step`, else 2) in `make_model`. Legacy `encoder_a`/`encoder_b` properties on `PhaseAccumulatorNetwork` are there for back-compat with analysis call sites.

### Adding a new experiment

Most new experiments are pure YAML — write `experiments/<name>.yaml` with:

```yaml
experiment: grid_sweep
out_dir:    results/<name>
base:                               # RunConfig fields shared by every sub-run
  p: 113
  k_freqs: 9
  n_steps: 100000
grid:                               # dict → Cartesian product over fields
  seed: [42, 123, 456]
# OR list-of-dicts for coupled axes (e.g. transformer d_model/n_heads/d_mlp):
# grid:
#   - {d_model: 16, n_heads: 1, d_mlp: 64, seed: 42, label: TF-d16-s42}
options:                            # optional
  ablations: false                  # default true; whether to run ablations
  slots:     true                   # default false; write slots.csv
  hooks:     [checkpoint_logger]    # see HOOK_REGISTRY in grid_sweep.py
plots:                              # optional, declarative list
  - {type: training_curves, title: "..."}
  - {type: sweep_reliability, group_by: seed}
```

Add a convenience target in `Makefile` if it's one you'll run often. Only write Python (a `@register` function in `experiments.py`) for experiments that do bespoke post-training analysis — the grid-sweep function does not fit them. The three that exist today (`sifp16_inference`, `decoder_swap`, `decoder_analysis`) are the template.

### Test suite conventions

Every test uses the `tiny_cfg` fixture in `tests/conftest.py` (P=11, K=3, 200 steps) so the full suite stays under a minute on CPU. Tests never depend on grokking actually happening — only on the mechanics of the code path. Follow that convention when adding tests.
