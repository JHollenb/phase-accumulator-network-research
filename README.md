# pan_lab — Phase Accumulator Network research library

Experiment framework for the PAN architecture: a neural network whose
primitive operation is sinusoidal phase addition rather than
floating-point multiply-accumulate. Replaces the monolithic `pan.py`
with a library + YAML-driven experiment runner.

Everything a run produces — metrics, curves, checkpoints, ablations —
lands in CSV alongside a `manifest.json` with full provenance (config,
git SHA, torch version, device, timestamp).

## Install

```
cd pan_lab
pip install -e .
pytest                    # 55 tests, ~25s on CPU
```

`torch`, `numpy`, `pandas`, `matplotlib`, `tabulate`, and `pyyaml` are the only
runtime dependencies.

## Quick start

```
# List available experiments
uv run python -m pan_lab --list

# Dry-run any experiment: print every sub-run's config, no training
uv run python -m pan_lab experiments/k8_sweep.yaml --dry-run

# Actually run
uv run python -m pan_lab experiments/tier3.yaml

# Regenerate plots from existing CSVs (no retraining)
uv run python -m pan_lab --replot results/tier3

# Ad-hoc run without a YAML file
uv run python -m pan_lab --ad-hoc compare --p 113 --k 9 --steps 50000
```

## What's in the box

**Registered experiments** (see `experiments/*.yaml`):

| Experiment | Addresses | Scale |
|---|---|---|
| `compare` | PAN vs transformer head-to-head | 2 runs |
| `k_sweep` | Minimum K for reliable grokking | 45 runs |
| `k8_sweep` | K=8 anomaly investigation (§4) | 10 runs |
| `dw_sweep` | Diversity-weight reliability | 30 runs |
| `wd_sweep` | Weight-decay reliability | 18 runs |
| `primes` | Cross-prime generalization | 5 runs |
| `held_out_primes` | Primes unseen in development | 3 runs |
| `tier3` | Mechanistic equivalence (§5.1) | 1 long run |
| `slot_census` | **Exp A**: slot activation census | 20 runs |
| `freq_init_ablation` | **Exp H**: Fourier vs random init | 10 runs |
| `sifp16_inference` | **Exp E**: 16-bit quant at inference | 3 runs |
| `decoder_swap` | **Exp I**: swap to Fourier decoder | 3 runs |
| `mod_mul` | **Exp B**: modular multiplication | 3 runs |
| `mod_two_step` | **Exp C**: (a+b)·c mod P | 3 runs |
| `tf_sweep` | Minimum transformer d_model | 15 runs |

"Exp X" labels reference the experiment-ideas table from the
critical review — the ones the paper needs before submission.

## CSV schema

Every experiment writes (at minimum) these to its `out_dir`:

**runs.csv** — one row per training run.
```
run_id, experiment, label, p, task_kind, model_kind, k_freqs, d_model,
seed, weight_decay, diversity_weight, freq_init,
n_steps_planned, n_steps_actual, grok_step, grokked,
final_val_acc, peak_val_acc, final_train_loss, final_val_loss,
elapsed_s, param_count, mode_collapsed
```

**curves.csv** — one row per eval step per run.
```
run_id, step, train_loss, val_loss, val_acc
```

**ablations.csv** (when `ablations=True`) — one row per
(run, intervention).
```
run_id, intervention, val_acc
```
where `intervention` is one of `baseline`, `zero_phase_mixing`,
`randomize_frequencies`, `zero_ref_phases`.

**checkpoints.csv** (when `record_checkpoints=true`) — one row per
(run, step, encoder, k). The Tier 3 mechanistic data.
```
run_id, step, encoder, k, theoretical, learned, error
```

**slots.csv** (when `slots=True`) — frequency-slot activation census.
```
model_idx, encoder, k, theoretical, learned, learned_raw,
error, converged, run_id, seed
```

**manifest.json** — experiment name, number of runs, provenance
dict, list of written files.

## Directory layout

```
pan_lab/
├── pan_lab/
│   ├── config.py          RunConfig, constants, provenance capture
│   ├── data.py            mod_add, mod_mul, mod_two_step datasets
│   ├── models/
│   │   ├── pan.py         PhaseEncoder, PhaseMixingLayer, PhaseGate, PAN
│   │   ├── transformer.py Nanda 1-layer baseline
│   │   └── quantize.py    SIFP-16 straight-through quantizer
│   ├── trainer.py         generic loop, hooks, seed discipline
│   ├── hooks.py           CheckpointLogger, CSVStreamLogger
│   ├── analysis.py        ablations, freq errors, slot census
│   ├── reporting.py       ExperimentReporter, pandas aggregation, CSV
│   ├── plots.py           matplotlib figures (consume DataFrames only)
│   ├── experiments/
│   │   ├── __init__.py    EXPERIMENT_REGISTRY + YAML loader + dispatch
│   │   ├── base.py        BaseExperiment + shared run helpers
│   │   ├── compare.py     one experiment per file (example)
│   │   └── ...            other experiment modules
│   └── cli.py             python -m pan_lab entry
├── experiments/           15 YAML specs
├── tests/                 55 pytest tests, ~25s on CPU
└── pyproject.toml
```

## YAML schema

```
experiment: k8_sweep         # name from EXPERIMENT_REGISTRY
out_dir:    results/k8_sweep
dry_run:    false            # can be forced via --dry-run

base:                        # RunConfig fields — any subset
  p:            113
  k_freqs:      8
  n_steps:      200000
  weight_decay: 0.01
  ...

experiment_args:             # experiment-specific overrides (optional)
  seeds: [42, 123, 456, ...]
```

Unknown fields in `base` raise a warning but do not fail — you can put
comments-as-fields in YAML for future readers.

## Adding a new experiment

Create one module per experiment under `pan_lab/experiments/`, define a
`BaseExperiment` subclass, import it in `pan_lab/experiments/__init__.py`,
and add it to `_register_default_experiments()`.

```python
# pan_lab/experiments/my_experiment.py
from pan_lab.config import RunConfig
from pan_lab.experiments.base import BaseExperiment

class MyExperiment(BaseExperiment):
    name = "my_experiment"

    def build_configs(self, base: RunConfig, seeds=None, **_):
        seeds = seeds or [42, 123, 456]
        return [
            base.with_overrides(seed=s, label=f"my-s{s}")
            for s in seeds
        ]

    def handle_result(self, reporter, result, vx, vy, cfg, state):
        reporter.add_run(result, val_x=vx, val_y=vy, ablations=True)

    def finalize(self, reporter, state, out_dir):
        # optional extra CSVs/plots
        pass
```

```python
# pan_lab/experiments/__init__.py
from .my_experiment import MyExperiment

def _register_default_experiments() -> None:
    for exp in [
        ...,
        MyExperiment(),
    ]:
        EXPERIMENT_REGISTRY[exp.name] = exp
```

```
# experiments/my_experiment.yaml
experiment: my_experiment
out_dir:    results/my_experiment
base:
  p:       113
  k_freqs: 9
  n_steps: 50000
experiment_args:
  seeds: [42, 123, 456, 789]
```

## What's different from the original `pan.py`

1. **Diversity regularizer bug is fixed.** The old code computed
   `phi_a/phi_b` under `torch.no_grad()` before feeding them to the
   Gram penalty, so the encoder frequencies were detached from the
   diversity gradient. Every reported DW sweep result was produced
   with a partially-disabled regularizer. `pan_lab.trainer` routes
   DW through `model.mix_features()`, which keeps the encoder
   gradient edges live.
   `tests/test_gradient_flow.py` contains both the old-bug
   demonstration and the fix confirmation.

2. **Deterministic model init.** `make_model()` seeds the RNG before
   constructing the model. The old code only seeded inside `train()`,
   after the model was built, so two runs with identical configs
   produced different initializations depending on ambient RNG state.

3. **Every run is serializable.** `RunConfig.short_id()` produces a
   stable hash of the config; `TrainResult.provenance` captures env
   metadata at run time; CSVs contain every config knob that varied.

4. **Hooks replace hard-coded branches.** The original
   `if record_checkpoints:` branch is now `CheckpointLogger`, a class.
   Adding a new per-step probe is ~10 lines.

5. **Generalized to N inputs.** `mod_add`/`mod_mul` use 2 inputs; the
   two-step task uses 3. A single `n_inputs` arg replaces the
   hardcoded `encoder_a`/`encoder_b` split.

## Test suite

```
uv run pytest
```

Covers models (shapes, param counts, init), data (correctness, split
determinism), gradient flow (the critical diversity-reg regression
test), trainer (dry-run, determinism, early stop), reporting (CSV
schemas, manifest, summaries), and experiments (YAML parse, registry
dispatch, every shipped YAML validates).

## License & provenance

Research code from Jacob Hollenbeck's SPF/PAN project (Apr 2026).
Companion to `pan_paper.md` and the SPF whitepaper.
