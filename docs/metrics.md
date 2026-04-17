# Metrics reference

Mechanistic instrumentation of PAN training: what each per-eval scalar
measures, what plot panel it ends up in, and which YAML knob controls
its cadence. Source of truth is `pan_lab/metrics.py`,
`pan_lab/training_dynamics.py`, and `pan_lab/plots.py` — this doc is
a map over them.

## Overview

Each eval step calls `MetricsLogger.on_eval` (`metrics.py:391`),
which appends one row to `metrics.csv`. Transformer runs are a no-op;
only PAN runs populate rows.

Two cost classes:

- **Cheap** — pure functions of model weights. Run every eval
  (M1–M5, M8). Columns are dense.
- **Expensive** — need a forward pass (M6, M7) or extra FFTs (M9).
  Run every `metrics_expensive_every` steps (default 5000). Columns
  are NaN at evals where they weren't sampled.

CSVs written to `out_dir/`:

| File | Row per | Populated by |
|---|---|---|
| `metrics.csv` | (run, eval step) | `MetricsLogger` |
| `metrics_spectra.csv` | (run, metric, freq bin) | `_write_spectra` in `grid_sweep.py:259` |
| `metrics_peaks.csv` | (run, metric) | same |

Access from code via `ExperimentReporter.metrics_df()`,
`.spectra_df()`, `.peaks_df()`.

## Cheap per-eval metrics

Columns populated every eval. All run through `cheap_metrics`
(`metrics.py:336`).

### M1 · `fourier_snap` — `metrics.py:65`

For each learned encoder frequency `ω_k`, circular distance to the
nearest integer Fourier mode `n · 2π/p` for `n ∈ [1, p/2]`.

- **Captures:** how close each slot is to a rational-multiple-of-`p`
  frequency. At init: ~uniform random over [0, π]. Grokked: → 0.
- **Columns:** `enc{0,1}_snap_mean`, `enc{0,1}_snap_max`,
  `enc{0,1}_snap_min`, `enc{0,1}_active_n` (string: comma-separated
  snapped mode indices).

### M2 · `clock_pair_compliance` — `metrics.py:79`

Fraction of mix-matrix rows whose top-2 `|weights|` come from
different encoders and have matched magnitudes (within `mag_tol=0.20`).

- **Captures:** how much of the mixing layer has locked onto
  Clock-pair structure (one slot from encoder 0, matched-magnitude
  slot from encoder 1).
- **Column:** `clock_compliance`.

### M3 · `clock_freq_alignment` — `metrics.py:102`

For rows that pass M2, mean circular distance between the two paired
frequencies. Guards M2 against false positives (a pair can have
matched magnitudes but unrelated frequencies).

- **Captures:** whether the paired slots encode the *same* Fourier
  mode — a real Clock — or just two arbitrary cross-encoder slots.
- **Columns:** `clock_freq_align_mean` (NaN when M2 finds nothing),
  `clock_freq_align_n` (pair count).
- **Status:** kept but provisional; hasn't cleanly discriminated a
  single K13 failure. Revisit at 15–20 failed runs.

### M4 · `mix_row_entropy` — `metrics.py:134`

Shannon entropy of each `|W_mix|` row (row treated as a probability
distribution over its `2K` columns); `eff_n = exp(entropy)` is the
"effective number of contributing slots."

- **Captures:** row-level sparsification. Uniform row → `eff_n = 2K`.
  Clean Clock pair → `eff_n ≈ 2`.
- **Columns:** `mix_row_entropy_mean`, `mix_row_eff_n_mean`,
  `mix_row_eff_n_min`.

### M5 · `active_frequencies` — `metrics.py:155`

Snap each slot's frequency to an integer mode, then count distinct
modes used by at least one mix-row weight above `0.1`.

- **Captures:** how many Fourier modes the circuit actually uses.
  K=9 grokked runs typically converge to `count ≈ 3–5`.
- **Columns:** `active_freq_count` (int), `active_freq_set` (string:
  comma-separated sorted mode indices).

### M8 · `decoder_fourier_projection` — `metrics.py:239`

For each of the `K` decoder columns (viewed as a length-`p` signal
over class index), fraction of spectral energy in its single largest
DFT bin.

- **Captures:** whether each decoder column is a pure sinusoid over
  classes (Clock decoder). Pure cosine → peak ≈ 1; random → ≈ 1/(p/2).
- **Columns:** `decoder_fourier_peak_mean`,
  `decoder_fourier_peak_max`.
- **Discrimination:** clean on real runs — 0.356 failed vs 0.625
  grokked on the single K13 case that prompted disabling M9.

## Expensive per-eval metrics

Populated every `metrics_expensive_every` steps (default 5000). Rows
between these steps are NaN for these columns.

### M6 · `gate_linear_decodability` — `metrics.py:183`

Fits a multinomial logistic regression (sklearn, `max_iter=200`) on
`(gate_output → label)` over a subsample of the val set
(`metrics_gate_decode_max_rows`, default 4000).

- **Captures:** ceiling accuracy a *linear* decoder could get from
  the current gate representation — i.e. "has the gate layer become
  computationally sufficient?"
- **Column:** `gate_linear_acc`.

### M7 · `sifp16_robustness` — `metrics.py:215`

Val acc at fp32 vs val acc with every PAN phase rounded to the
SIFP-16 lattice (`sifp16_context` monkey-patch).

- **Captures:** whether the frequency set is clean enough to survive
  hardware quantization. `quant_delta` ≈ 0 on a grokked, Fourier-
  locked circuit.
- **Columns:** `fp32_acc`, `sifp16_acc`, `quant_delta`
  (`fp32_acc − sifp16_acc`).

### Derived · `gate_decoder_gap`

Pure CSV arithmetic: `gate_linear_acc − fp32_acc`. Computed inside
`expensive_metrics` (`metrics.py:373`) so it shares the M6/M7 cadence
and NaN pattern automatically.

Three-regime reading:

| `gate_decoder_gap` | `fp32_acc` | Interpretation |
|---|---|---|
| large positive | low | **decoder-limited** — gate is sufficient, trained decoder hasn't caught up |
| near zero | high | **circuit complete** — decoder has fully exploited the gate |
| near zero | low | **representation-limited** — gate itself is the bottleneck (e.g. K13-s1) |

**Column:** `gate_decoder_gap`.

## Disabled by default · M9 logit 2D spectrum

`logit_2d_spectrum` (`metrics.py:266`) computes the 2D DFT of
`logit(c | a, b)` over the P×P input grid, averaged across sampled
classes. Returns three aggregates: `logit_spec_diag_frac_mean`,
`logit_spec_peak_sparsity_mean`, `logit_spec_active_count_mean`.

**Why disabled:** on real sweeps these saturate to 0.99+ by step
~5K even on runs stuck at 12% val accuracy. They don't discriminate
grokked vs failed. And they cost a P² forward pass + per-class 2D
FFT — the single most expensive call in `expensive_metrics`.

**Why not deleted:** the function and its unit tests stay, in case
a later experiment cares. M8 already covers decoder-Fourier
structure with cleaner discrimination.

**To re-enable on one experiment:**

```yaml
options:
  metrics_logit_spectrum: true
  # optional — which classes to sample; default is 8 evenly-spaced
  metrics_logit_spectrum_classes: [0, 7, 14, 21, 28]
```

Flipping the flag puts the three `logit_spec_*` columns back into
`metrics.csv`.

## Post-hoc training dynamics

After training, `_write_spectra` (`grid_sweep.py:259`) runs a DFT of
each numeric column's time series per run. Output lands in
`metrics_spectra.csv` (long form) and `metrics_peaks.csv` (one row
per run × metric).

Pipeline — `training_dynamics_spectrum` (`training_dynamics.py:32`):

1. Drop NaN samples (lets sparse expensive metrics participate).
2. Quadratic detrend — removes DC, linear, and quadratic drift, so
   a monotone formation curve like `snap_mean → 0` doesn't dominate
   the spectrum.
3. Hann window — reduces leakage at bin edges.
4. `np.fft.rfft`, power = `|FFT|²`.
5. `eval_interval_steps = median(diff(step))` per run, so a run with
   `log_every=500` and one with `log_every=50` produce spectra in
   the same (cycles per training step) units.

Series shorter than 4 finite samples are skipped.

Columns excluded from DFT (non-scalar): `run_id`, `step`,
`enc0_active_n`, `enc1_active_n`, `active_freq_set`. See
`_NON_SPECTRAL_COLS` in `training_dynamics.py:24`.

`summarize_metrics_spectra` (`training_dynamics.py:127`) picks the
single dominant non-DC peak per (run, metric) and writes
`metrics_peaks.csv` with columns `peak_freq`, `peak_power`,
`peak_timescale_steps`.

## Plots

Three figures; all dispatched through `PLOT_REGISTRY` in
`grid_sweep.py:56`. All are pure DataFrame-in, PNG-out and can be
regenerated without retraining via `python -m pan_lab --replot DIR`.

Default panel selection comes from `DEFAULT_FORMATION_METRICS`
(`plots.py:26`), which is 8 entries post-M9-disable:

```
enc0_snap_mean              # M1
clock_compliance            # M2
mix_row_eff_n_mean          # M4
active_freq_count           # M5
decoder_fourier_peak_mean   # M8
gate_linear_acc             # M6
gate_decoder_gap            # derived
sifp16_acc                  # M7
```

Any YAML can override with `metrics: [...]` inside a plot spec.

### `metric_formation_curves.png` — `plots.py:417`

Grid of small panels — one per metric. Within a panel: step on x,
metric value on y, one line per run (up to `max_lines=20`), colored
by `tab10`. Dotted vertical at each run's grok step (from
`runs.csv`).

- **Input:** `metrics_df`, `runs_df`.
- **Shape:** 3-column grid, `ceil(len(metrics) / 3)` rows. With the
  default 8 metrics, 3 rows × 3 cols with 1 blank panel.
- **How to read:** M1 `snap_mean` should decay monotonically before
  grokking; M2 `clock_compliance` rises late; M4 `eff_n_mean` drops
  from ~6 toward 2 on K=3 as the circuit sparsifies. Sparse dots on
  M6/M7/`gate_decoder_gap` panels are the expensive-cadence rows.

### `metric_spectra.png` — `plots.py:488`

Log-log overlay of DFT power vs timescale. x-axis is inverted so
slow timescales (≥ 10⁴ steps) sit on the left, fast on the right.
Excludes the DC bin.

- **Input:** `spectra_df`.
- **Modes:**
  - `aggregate="median"` *(default)* — one line per metric, median
    power across runs, shaded min/max envelope.
  - `aggregate="per_run"` — one line per (run, metric); denser and
    noisier, useful for spotting per-seed outliers.
- **How to read:** distinct peaks at different timescales across
  metrics supports the "grokking is multi-timescale" story (encoders
  lock fast, mix sparsifies slowly, decoder Clock projection in
  between). A flat spectrum means that metric is essentially
  monotone — the detrend removed everything.

### `metric_peak_timescales.png` — `plots.py:561`

Horizontal bar chart: y = metric name, x = mean peak timescale
across runs (log scale). Whiskers at min/max, annotated with
`n={run count}`. Sorted fastest-at-top.

- **Input:** `peaks_df`.
- **How to read:** direct comparison of "how slow does each metric
  evolve, in training steps." M1 snap usually shortest timescale;
  M2 compliance and M4 `eff_n_mean` usually longest. Order
  inversions at K=8 anomaly or K=13 failures are the signal.

## YAML knobs

All under `options:` in the experiment YAML.

| Key | Default | Effect |
|---|---|---|
| `metrics` | `true` | Master switch. `false` means MetricsLogger isn't attached — no metrics.csv. |
| `metrics_expensive_every` | `5000` | Steps between M6/M7 (+ `gate_decoder_gap`) samples. `0` disables. |
| `metrics_gate_decode_max_rows` | `4000` | Val-set subsample for the sklearn probe in M6. |
| `metrics_logit_spectrum` | `false` | Opt in to M9 (three `logit_spec_*` columns). |
| `metrics_logit_spectrum_classes` | `None` | Int count or explicit list of class indices. No-op when M9 is off. |
| `metrics_spectra` | `true` | Write `metrics_spectra.csv` + `metrics_peaks.csv` post-training. |

## Cross-reference

| File | Role |
|---|---|
| `pan_lab/metrics.py` | M1–M9 functions, `cheap_metrics`, `expensive_metrics`, `MetricsLogger` hook |
| `pan_lab/training_dynamics.py` | Post-hoc DFT + peak summarization |
| `pan_lab/plots.py` | Three plot functions + `DEFAULT_FORMATION_METRICS` |
| `pan_lab/grid_sweep.py` | `PLOT_REGISTRY`, `_write_spectra`, `run_grid_sweep` (threads YAML options) |
| `pan_lab/experiments.py` | `_run_cfgs` builds `MetricsLogger` per run |
| `pan_lab/reporting.py` | `ExperimentReporter.metrics_df()`, `.spectra_df()`, `.peaks_df()` accessors |
