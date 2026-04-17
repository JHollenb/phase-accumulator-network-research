# Experiment 1 — Decoder vs Representation Over Time in PAN

## Purpose

Test whether PAN learns a **usable spectral scaffold early** and only groks later because the decoder and/or late-stage representation refinement lag behind.

This experiment is designed around the updated picture from the K sweep:

- `mix_entropy_min` and `mix_entropy_mean` collapse very early
- `fourier_conc` spikes very early, then gradually declines
- `fourier_norm`, `decoder_norm`, and `circuit_ratio` keep growing much later
- long plateaus appear to be a late-stage phenomenon, not an absence of early structure

So the right question is **not** just “does the decoder lag?”  
It is:

> At what point does the gate representation become linearly usable, and what part of late training is decoder alignment versus representation redistribution/strengthening?

---

## Core Questions

1. Does the gate representation become linearly decodable **well before** the model groks?
2. If yes, is the trained decoder simply lagging behind?
3. Or does late training still materially improve the representation itself by:
   - increasing margin
   - redistributing spectral mass
   - strengthening the usable circuit
   - reducing brittleness of an early concentrated scaffold

---

## Main Hypotheses

### H1 — Early scaffold, late usable code
PAN rapidly forms a Fourier/Clock-like scaffold, but the early code is too concentrated or brittle to support full generalization.

### H2 — Late training is not just decoder cleanup
Even after early scaffold formation, the representation continues changing in meaningful ways:
- `fourier_norm` keeps rising
- `circuit_ratio` keeps rising
- `fourier_conc` declines from its early spike
- successful runs differ from failed runs mainly in these late-phase refinements

### H3 — Some runs are decoder-limited, others are representation-limited
There may be multiple regimes:
- some checkpoints where a fresh linear probe succeeds much better than the trained decoder
- some checkpoints where both trained decoder and fresh probe fail similarly
- some checkpoints where both already succeed

---

## Experimental Design

## Setup

Use a fixed task and architecture:
- task: mod-113 addition
- model: PAN
- primary regime: `k_freqs = 8, 9, 10`
- seeds: at least 5 per K if feasible
- training mode: **no early stop**
- save dense checkpoints

Recommended checkpoint schedule:
- every 5 steps from 0–100
- every 10 steps from 100–300
- every 25 steps from 300 onward

If storage is an issue, save full weights less frequently and cache gate activations more frequently.

---

## Per-Checkpoint Quantities to Save

### Model state
- encoder frequencies
- mixing weights
- reference phases
- decoder weights and bias

### Trajectory metrics
- validation accuracy
- train loss / val loss
- `fourier_norm`
- `fourier_conc`
- `mix_entropy_min`
- `mix_entropy_mean`
- `decoder_norm`
- `circuit_ratio`
- `plateau_steps`

### New gate-level quantities
For all residues or all validation pairs:
- gate activations
- logits from trained decoder
- labels

---

## Decoder / Probe Analyses at Each Checkpoint

For the same frozen gate activations, evaluate:

### A. Trained decoder accuracy
The actual model accuracy using the checkpoint’s decoder.

### B. Fresh linear probe accuracy
Train a fresh multiclass linear probe on the gate activations.

Use:
- multinomial logistic regression or equivalent linear classifier
- train on the same train split the model used
- evaluate on held-out validation

This answers:

> How much information is already present in the gate representation, independent of the current trained decoder?

### C. Decoder-reset recovery
Replace the decoder with a fresh randomly initialized decoder and retrain **decoder only** on frozen gates for a small number of steps.

This answers:

> If the representation is already good, can the decoder rapidly recover performance?

### D. Frozen-representation continuation
At selected checkpoints, freeze encoder + mixing + ref phases and continue training the decoder only.

### E. Frozen-decoder continuation
At the same checkpoints, freeze the decoder and continue training the representation only.

These two interventions separate:
- decoder-limited runs
- representation-limited runs

---

## Key Derived Metrics

### 1. Probe gap
`probe_gap = probe_val_acc - trained_decoder_val_acc`

Interpretation:
- large positive gap → representation ahead of decoder
- near zero gap → decoder already aligned to current representation

### 2. Decoder recoverability
Accuracy reached by decoder-only retraining from a checkpoint.

Interpretation:
- rapid recovery → representation already usable
- poor recovery → representation still immature

### 3. Representation maturation
Late-phase change after scaffold formation:
- late slope of `fourier_norm`
- late slope of `circuit_ratio`
- late drop in `fourier_conc`
- late drop or stabilization in plateau burden

### 4. Margin diagnostics
For both trained decoder and fresh probe:
- mean correct-class logit margin
- median margin
- fraction of near-ties

A code can be linearly separable before it is robustly margin-separated.

---

## Primary Plots

### Plot 1 — Checkpoint panel for representative runs
Same x-axis (step), with:
- validation accuracy
- trained decoder accuracy
- fresh probe accuracy
- `fourier_conc`
- `fourier_norm`
- `mix_entropy_min`
- `decoder_norm`
- `circuit_ratio`

Use one panel each for:
- low-K failure
- plateauing borderline run
- successful K=9 or K=10 run

### Plot 2 — Probe gap over time
Across runs, show when `probe_gap` becomes positive and whether it closes before grok.

### Plot 3 — Representation maturation vs success
Scatter:
- late `fourier_norm` slope vs peak accuracy
- late `circuit_ratio` slope vs peak accuracy
- late `fourier_conc` drop vs peak accuracy

### Plot 4 — Decoder recoverability
For several checkpoints, compare:
- original trained decoder accuracy
- accuracy after decoder-only retraining on frozen gates

---

## Decision Logic

### Outcome A — Probe succeeds early, trained decoder lags
Interpretation:
- strong evidence that a usable representation forms early
- late grok is at least partly decoder alignment / margin growth

### Outcome B — Probe and trained decoder both fail early, both improve later
Interpretation:
- late training is genuinely improving the representation
- early scaffold exists, but it is not yet enough

### Outcome C — Probe succeeds modestly early, then improves further later
Interpretation:
- mixed story
- early scaffold is real
- late refinement still materially improves code quality

This is the outcome I currently expect.

---

## Most Important Claim This Experiment Can Support

If successful, this experiment could justify a statement like:

> PAN forms a concentrated Fourier/Clock-like scaffold early in training, but grokking depends on a later refinement phase in which the gate representation becomes more robustly linearly usable through continued circuit strengthening and redistribution of spectral mass.

That is stronger and safer than:
- “the decoder is the only thing lagging”
- “the full algorithm is completely learned early”

---

## Practical Notes

- Do **not** use OLS as a proxy for optimal linear decoding.
- Use multinomial logistic regression or an equivalent proper linear classifier.
- Keep train/val splits identical to the model’s own split.
- For small gate dimensionality, probe training should be cheap.
- Save a few “failed but highly structured” checkpoints; those are likely the most revealing.

---

## Minimal Version

If time is limited, run only:

1. trained decoder accuracy over checkpoints  
2. fresh probe accuracy over checkpoints  
3. `fourier_conc`, `fourier_norm`, `mix_entropy_min`, `circuit_ratio`

Even that reduced version will answer the main question:
**is there a real gap between early scaffold formation and late usable representation?**
