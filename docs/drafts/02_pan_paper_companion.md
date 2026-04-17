# PAN Research Companion

*Unofficial working document — Apr 2026 — Jacob Hollenbeck*
*Companion to `pan_paper.md` (v1 Mar 2026; v2 draft in `pan_paper_v2.md`)*
*Context: Phase 0 of the SPF/PAN Project Roadmap — paper submission prep.*

This document is for the next person (human or LLM) picking up the PAN work.
It captures what we found, what broke, what's still open, and how the code
is laid out. It is *not* the paper. Read this before re-running experiments
or revising the paper draft.

---

## Table of Contents

1. [What PAN is and why it exists](#1-what-pan-is-and-why-it-exists)
2. [Timeline of findings from this work session](#2-timeline-of-findings-from-this-work-session)
3. [Bugs caught and fixed](#3-bugs-caught-and-fixed)
4. [Current state of the headline claims](#4-current-state-of-the-headline-claims)
5. [Open questions and live hypotheses](#5-open-questions-and-live-hypotheses)
6. [What the paper needs before submission](#6-what-the-paper-needs-before-submission)
7. [Codebase map and design](#7-codebase-map-and-design)
8. [How to extend the codebase](#8-how-to-extend-the-codebase)
9. [Experiment catalog with current status](#9-experiment-catalog-with-current-status)
10. [Datafile formats](#10-datafile-formats)
11. [Known traps and gotchas](#11-known-traps-and-gotchas)

---

## 1. What PAN is and why it exists

**Phase Accumulator Networks (PAN)** is a neural architecture whose primitive
operation is sinusoidal phase addition on the unit circle rather than
floating-point multiply-accumulate. The motivating observation:

- Nanda et al. (2023), Kantamneni & Tegmark (2025), and Zhou et al. (2024)
  show that transformers trained on modular arithmetic converge on an
  explicit Fourier / "Clock" algorithm — discovered through gradient descent
  on a general-purpose multiply-accumulate architecture.
- PAN makes phase arithmetic the architectural primitive rather than something
  the network has to discover. Inputs become phases; the forward pass mixes
  phases, gates them, and decodes.

The hypothesis: if the computation a transformer is trying to learn is
sinusoidal phase rotation, an architecture where phase rotation is native
should solve the same tasks with dramatically fewer parameters.

PAN also serves a second purpose in the broader project: it is the ML-side
evidence for **Spectral IEEE 754 (SPF)**, a proposed log-polar number format
where phase addition is a 16-bit integer add. If PAN works, SPF has a
concrete ML use case. If SIFP-16 phase quantization preserves PAN accuracy,
SPF has a concrete hardware-relevance claim.

---

## 2. Timeline of findings from this work session

The session started with the v1 paper draft and the original monolithic
`pan.py` (~1,400 lines in one file). Over the session we:

### Phase A: Critical review of the existing code and paper

- Identified that **the diversity regularizer was silently broken** in
  `pan.py`: `phi_a`/`phi_b` were computed under `torch.no_grad()` before
  being fed into the Gram-matrix diversity penalty. That detached the
  encoder frequencies from the diversity gradient, so every reported DW
  sweep was produced with a half-disabled regularizer. (Details in §3.)
- Identified that **model initialization was non-deterministic** —
  `make_model` didn't seed the RNG before constructing the model, so two
  runs with the same seed produced different `decoder.weight` inits
  depending on ambient RNG state. This quietly broke the paper's
  reproducibility claim across process invocations.
- Identified that the paper's "Tier 3 mechanistic equivalence" section
  was based on raw-frequency plotting, which misreads the data. Learned
  frequencies drift modulo 2π; plotting raw values hides convergence
  to the Fourier basis. The right plot is **angular error** on a log
  y-axis.

### Phase B: Rewrote pan.py as pan_lab library

Split the monolithic file into a proper package (§7) with YAML-driven
experiments, pandas-based reporting, and pytest coverage. Added the
`test_diversity_reg_grads_reach_encoder_freqs` regression test that
would have caught the bug-zero and locks in the fix. Built 15 experiment
YAMLs (the headline runs for the paper plus sweeps for characterization).

### Phase C: Tier 3 mechanistic run on the new library

Your tier3 run at P=113, K=9, seed=42 produced the data that resolved the
paper's Section 5.1 open question. Findings:

1. **PAN grokks at step 46,000.** Val accuracy hits 99% and holds.
2. **5 of 18 encoder slots converge within SIFP-16 precision** (< 9.6e-5
   rad) to theoretical Fourier basis vectors of ℤ_113. The distinct
   frequencies are {k=1, k=3, k=4, k=9}.
3. **All 9 mixing-layer output channels show Clock structure.** Every row
   has its top two weights drawn from different encoders, with matched
   magnitude near 1.0. This is the Nanda "Clock algorithm" — confirmed
   mechanically by direct weight inspection, not by interpretability
   probing.
4. **Circuit formation is post-grokking.** Generalization happens at
   46K, but encoder frequencies only reach their final
   SIFP-16-precision values around step 80K. This matches Varma et al.'s
   "circuit efficiency" prediction: generalization and cleanup are
   distinct events.

### Phase D: Slot census at 20 seeds

To confirm the Clock finding isn't seed-specific, we ran 20 seeds at K=9.
Results:

- **16/20 grokked (80%).** This is the first measured K=9 reliability
  number — the v1 paper implied K=9 was "minimum reliable" without
  measuring.
- **Mean Clock-compliance score: 0.82.** Five runs hit perfect 1.00
  (every mixing-layer output is a clean Clock pair).
- **Mean distinct Fourier frequencies per circuit: 4.62** (range 3–8).
- **Frequency preference is not Nanda's.** PAN seeds most often converge
  on k=7 (69%) and mid-range k=5,6,8,9 (44% each). Nanda's transformer
  preferred k ∈ {14, 34, 41, 42, 52}. Some PAN seeds find high-k modes
  (k=23, 25, 33, 47, 48) — legitimate basis vectors of ℤ_113, just not
  the "canonical low-k" ones.

**Refined claim:** PAN and the transformer converge on the same
*algorithm* (Clock), but not the same *frequency subset*. This is a more
precise and more defensible claim than "PAN implements Nanda's
algorithm."

### Phase E: Frequency-init ablation surprise

5-seed pilot at K=9 comparing Fourier vs random initialization:

| | Fourier init | Random init |
|---|---|---|
| Grok rate | 4/5 | 5/5 |
| Mean grok step | 70,375 | 24,100 |
| Mean final acc | 97.8% | 99.5% |

This suggested random init is **strictly better** — 2.9× faster, more
reliable, higher accuracy. Would have been the most surprising result
in the paper. Promoted to headline finding in the v2 draft.

**But then 20-seed replication of random init showed 15/20 = 75%.**
Fourier at 20 seeds is 16/20 = 80%. The two are statistically
comparable. **The v2 "random strictly better" claim does not
survive at larger sample size.** What survives: **the architecture is
insensitive to initialization choice** — a weaker but still publishable
claim.

### Phase F: SIFP-16 inference quantization

Clean result. 3 seeds:

| Seed | fp32 val_acc | SIFP-16 val_acc | Δ |
|---|---|---|---|
| 42  | 99.23% | 99.23% | 0.000 |
| 123 | 99.14% | 99.14% | 0.000 |
| 456 | 94.36% | 94.06% | −0.003 |

**16-bit phase quantization is effectively free.** This is a single-line
claim in the paper that backs up the entire SPF hardware argument. One
of the cleanest results in the project.

### Phase G: K=8 anomaly dissolves

v1 paper called K=8 "the most surprising result" — 0/3 seeds grokked
while K=7 grokked 1/3 and K=9 grokked 2/3. With 10 seeds, K=8 grokked
6/10. The anomaly was sampling noise; the reliability curve is smooth
(K=8 at 60%, K=9 at 80%). **Section 4 of v1 paper is substantially
weakened and needs rewriting or cutting.**

### Phase H: Decoder-swap confusion

The v1-designed `decoder_swap` experiment asked: "at grokking, swap the
learned decoder for a theoretical Fourier decoder and measure accuracy."
Results: catastrophic −97% accuracy drop. Initially flagged as a bug in
the experiment formula — the formula used `cos(encoders[0].freq[j] * c)`
ignoring the mixing matrix and gate reference phases.

Wrote a corrected formula that uses the **effective channel frequency**
(extracted from the mixing matrix's top-2 weights per row) and the gate
reference phase. Smoke-tested on small non-grokked runs: corrected
formula was 20× better than buggy one but still didn't preserve
accuracy.

**Decided to replace decoder_swap entirely with decoder_analysis** — a
new experiment that doesn't ask "does pure Clock work as a decoder?"
but "what does the learned decoder contain that Clock doesn't?" Key
outputs: Clock-explained energy fraction, residual FFT spectrum,
basis-expansion recovery curve. Unit-tested the decomposition math on
synthetic pure-Clock and Clock-plus-extra-frequency decoders; math
verifies correctly. **Not yet run on real trained models.**

### Phase I: Held-out primes

Tested K=9 on primes unseen during development: P ∈ {59, 71, 97}.
Results at 200K steps:

| P | Grok step | Peak acc |
|---|---|---|
| 59 | 4,500  | 99.1% |
| 71 | 22,000 | 99.2% |
| 97 | —      | 83.4% |

P=97 didn't grok. Re-ran at 500K steps (expected it to be an
optimization wall like P=89 was). P=97 still plateaued at **~85%**.
**This is a capacity wall, not an optimization wall** — other primes
grok in 20–140K steps; 500K is not the limit.

**Revised cross-prime claim:** 7/8 primes grok at K=9. P=97 needs a
larger K (hypothesis: the units-group structure of ℤ/97ℤ differs from
ℤ/113ℤ in a way that matters for sparse Fourier approximation). Not
confirmed; left for future work.

---

## 3. Bugs caught and fixed

The following were found during this work session. All fixes are in
`pan_lab`; the original `pan.py` still has them.

### 3.1 Diversity regularizer — encoder gradients dropped

**What was happening (`pan.py` lines around the diversity reg):**

```python
if cfg.diversity_weight > 0 and hasattr(model, 'phase_mix'):
    with torch.no_grad():                       # ← kill switch
        phi_a = model.encoder_a(x_batch[:, 0])
        phi_b = model.encoder_b(x_batch[:, 1])
    mix_out = model.phase_mix(torch.cat([phi_a, phi_b], dim=-1))
    # ... Gram penalty on mix_out ...
```

The `torch.no_grad()` block detaches `phi_a` and `phi_b`, so autograd
edges from the diversity penalty can only reach `phase_mix.weight`. The
encoder frequencies never get a gradient signal from the Gram penalty.

**Fix (`pan_lab/trainer.py`):** Route the diversity penalty through
`model.mix_features()`, which runs the full encode+mix stack under
autograd.

**Impact on prior results:** Every DW sweep in the v1 paper was produced
with a half-disabled regularizer. The optimal DW value may shift when
re-run with the fix.

**Regression test:** `tests/test_gradient_flow.py::test_mix_features_routes_grads_to_encoder_freqs`
— asserts `encoders[0].freq.grad` is non-zero after a diversity-only
backward pass. Also
`test_no_grad_encoding_drops_encoder_grads` — demonstrates the old bug
is real by reproducing it deliberately.

### 3.2 Model initialization was non-deterministic

**What was happening:** `train()` seeded the RNG inside its loop, but
models were constructed *before* `train()` was called. Two runs with
identical configs produced different `decoder.weight` inits depending on
whatever code ran before `make_model()`.

**Fix (`pan_lab/models/__init__.py`):** Seed `torch.manual_seed`,
`np.random.seed`, and `random.seed` at the top of `make_model(cfg)`,
before any model construction.

**Impact:** All v1 paper reproducibility claims across process
invocations were nominally false. The effect was small in practice (init
noise is bounded) but the claim "same seed → same result" was only true
within a single process.

**Regression test:** `tests/test_trainer.py::test_same_seed_same_config_same_history`.

### 3.3 save_model was a dead field

**What was happening:** `RunConfig.save_model: bool` existed, was
referenced in the README and the tier3 YAML, but no code path actually
honored it. No .pt files were produced.

**Fix (`pan_lab/experiments.py`):** Added a `save_model_weights()`
function in `reporting.py` and a call-site in `_run_cfgs` that honors
`cfg.save_model`.

**Impact:** Every post-hoc analysis that needed trained weights was
silently blocked. The tier3 mechanistic inspection had to be redone
after this fix.

**Regression test:** `tests/test_save_model.py::test_save_model_writes_pt_file`.

### 3.4 `plot_freq_trajectories` plotted the wrong axis

**What was happening:** The function plotted `learned` (raw wrapped
frequency) on the y-axis. Because learned frequencies drift mod 2π,
convergence to the Fourier basis looks like a collapse from 2π down to
0.5 — visually very confusing, and it hides which basis vector each
slot converged to.

**Fix (`pan_lab/plots.py`):** Plot `error` (angular distance to nearest
theoretical basis vector) on log y-axis, with SIFP-16 quantization line
drawn for reference. User applied this patch as a new function
`plot_freq_err_trajectories` without removing the old one.

### 3.5 paper_extract.py needed decoder_analysis section (if we run it)

The extraction script reads CSVs from `results/` and produces a summary
markdown. It originally had sections for all the v1/v2 experiments. The
new `decoder_analysis` experiment (Phase H) writes three CSVs; the
patched paper_extract.py renders them under a new §9 and bumps the
manifest section to §10.

---

## 4. Current state of the headline claims

Paper-ready claims, in order of confidence, with actual data backing them:

### Very strong (ready to publish)

- **Mechanistic equivalence to the Clock algorithm.** 16/20 K=9 PANs
  show Clock-structured mixing matrices with mean compliance 0.82.
  The tier3 single-seed run has 9/9 Clock compliance — every output
  channel is a clean Clock pair. Source: §3.3 in v2 draft, data in
  `results/slot_census/` and `results/tier3/`.

- **Parameter efficiency.** At P=113, K=9 PAN has 1,319 parameters
  vs 227,200 for the transformer baseline (172× fewer). Both solve
  the task. Unchanged from v1.

- **Ablation specificity.** Zeroing any PAN component collapses
  accuracy to chance. Source: `results/tier3/ablations.csv`.

- **SIFP-16 inference tolerance.** 3 seeds, mean accuracy delta
  zero. The hardware-relevance claim for SPF. Source:
  `results/sifp16_inference/quant_eval.csv`.

### Strong (ready to publish with care)

- **K=9 reliability at 80%.** Measured on 20 seeds. Non-trivial
  failure rate even at K=9 — 4/20 runs fail via structural defect
  (mixing collapse or near-grok plateau). This is not in the v1
  paper; needs to be added honestly.

- **Initialization insensitivity.** Fourier and random init perform
  comparably at 20 seeds (80% vs 75%). The v2 draft's "random strictly
  better" was a 5-seed artifact.

- **Post-grokking circuit crystallization.** Tier3 shows generalization
  at step 46K but frequency convergence not until ~80K. Matches Varma
  et al.'s circuit efficiency prediction.

### Moderate (need more work or more honest framing)

- **Cross-prime generalization: 7/8 primes.** Previously "5/5" on
  development primes. After held-out testing, P=97 does not grok at
  K=9 even with 500K steps. Not an optimization wall. Paper needs to
  soften the claim.

### Weakened / killed

- **~~K=8 anomaly~~.** v1 paper Section 4 called this "the most
  surprising result." With 10 seeds, K=8 grokks 6/10. Smooth
  reliability curve. Section needs to be cut or rewritten as a
  one-paragraph note.

- **~~PAN finds Nanda's 5 frequencies~~.** The v1 "5 active
  frequencies" observation was a single-run artifact. Real mean is
  4.62 (range 3–8). The frequency subset varies by seed and doesn't
  match Nanda's specific preference.

### Not yet resolved

- **What the learned decoder contains beyond Clock.** The original
  decoder_swap experiment was uninformative due to a formula bug. The
  replacement `decoder_analysis` experiment is implemented and
  math-tested but has not been run on real data. Until it runs, we
  know the Clock circuit is in the mixing/gate stack but we don't
  know whether the *decoder* is also canonically Clock or has residual
  structure.

---

## 5. Open questions and live hypotheses

### 5.1 Why does P=97 fail at K=9 when P=89 and P=113 succeed?

Plausible hypotheses:

- **Group-structural.** Units group of ℤ/97ℤ has order 96 = 2⁵·3;
  ℤ/113ℤ has order 112 = 2⁴·7. Different factorizations may affect
  which Fourier subsets suffice for Clock.
- **Capacity.** K=9 is below some minimum-K threshold specific to
  P=97. A K=12 or K=15 sweep on P=97 would test this.
- **Optimization landscape.** The specific basin around K=9 init may
  have more saddle points for P=97 than for other primes. A seed
  sweep at K=9 on P=97 (not just one seed) would partially address.

**Cheapest next experiment:** run K=12 on P=97 at 200K steps, 1 seed.
If it grokks, it's a capacity wall. If it doesn't, the hypothesis is
wrong and we need to investigate further. If it does, a K=12 run on
other primes to confirm 12 is the new headline K.

### 5.2 What's in the decoder beyond Clock?

Run `decoder_analysis` (Phase H). If the residual spectrum has distinct
peaks at specific k values, the circuit is "Clock + N extras" —
interesting. If the residual is flat noise, Clock captures the
structure and the decoder_swap gap is from something non-spectral
(scaling, bias absorption, imperfect effective-frequency estimation).

### 5.3 Does the diversity regularizer actually help now that it's fixed?

Re-run `dw_sweep` with the trainer bug fixed. The v1 paper's DW=0.01
recommendation was derived from a half-disabled regularizer, so the
optimum may be different. **This is important because DW affects the
optimization landscape that produces the Clock circuits we just
characterized.** A fully-working DW might change the frequency
preferences in §3.3.

### 5.4 What distinguishes structural-failure seeds from plateau seeds?

In the 20-seed slot census, 4 seeds failed:
- Seeds 12, 19: catastrophic (peak <50%). Mixing collapse likely.
- Seeds 11, 18: plateau (peak 92–98%). Near-grok but didn't complete.

In the 20-seed random-init run, similar:
- Seeds 10, 15: catastrophic (peak <35%).
- Seeds 0, 17, 18: plateau (peak 72–98%).

The mode_collapsed detector (top-K-slot-dominance check) missed all of
these — `mode_collapsed: False` in every case. The current detector is
too coarse. A sharper detector might use mixing-matrix rank or
effective-rank of the phase-mix output. Worth a quick investigation
since the current 80% reliability number would improve if we could
reliably detect and retry failure cases.

### 5.5 Are the high-k frequencies (k=47, k=48) in some seeds real?

In `results/slot_census/`, seeds 4, 7, 15, 16 converge to frequencies
like k=33, 47, 48, 25, 27, 29. These are valid basis vectors of ℤ_113,
but they're surprising given that Nanda's transformer consistently
preferred k ≤ 52. Are these seeds implementing a different circuit
(still Clock-structured but at different frequencies) or is the
effective-frequency extraction noisy?

The mixing-matrix analysis (already done) showed these seeds still have
high Clock compliance, so the circuit is Clock. The question is
whether the *interpretation* of the high-k as a "real" frequency is
correct or whether the effective-frequency extraction is landing on
conjugate or aliased frequencies.

Can test by checking whether `channel_freq ≈ TWO_PI * k / P` holds
with `k` as computed vs `P - k` (the conjugate).

### 5.6 Tier 5 — language modeling probe

Not attempted this session. Still marked "future work" in v1 and v2
paper drafts. See roadmap task 0.3.

---

## 6. What the paper needs before submission

In priority order:

### Must-do

1. **Apply decoder_swap bug analysis to v2 paper.** Drop the current
   decoder_swap numbers from the paper entirely. They are measurements
   of a formula bug, not mechanism. Either run decoder_analysis and
   use its numbers, or drop the experiment from the paper.

2. **Revise §3.4 with 20-seed random-init numbers.** The v2 "random
   strictly better" draft is wrong. Use "initialization insensitivity"
   framing instead (80% Fourier vs 75% random is a wash).

3. **Revise §3.5 with P=97 honesty.** v1 claimed 5/5 primes grok.
   With held-out testing it's 7/8 with P=97 as a specific open case.

4. **Rewrite or cut §4 (K=8 anomaly).** 6/10 at K=8 with 10 seeds
   dissolves the anomaly. Either one-paragraph note or removed.

5. **Tighten §3.2/§3.3 Clock compliance claim.** The paper should
   explicitly note: PAN and transformer converge on the same
   *algorithmic structure* (Clock pairs with matched frequencies from
   each input), not the same *Fourier subset*. Current v2 draft has
   this but the language could be sharper.

### Should-do

6. **Re-run dw_sweep with the fixed regularizer.** If DW=0.01 is still
   optimal, the paper's hyperparameter section is consistent. If it's
   different, the tier3 and slot_census results were produced under
   subtly different regularization than the sweep was characterizing.

7. **One K=12 run on P=97.** Tests whether P=97's failure is a
   capacity wall. If it grokks cleanly, the headline cross-prime claim
   becomes "all 8 primes grok with K appropriately chosen; K=9 is
   sufficient for 7/8, K=12 for 8/8."

8. **Run decoder_analysis (already implemented).** 5 seeds, K=9. The
   residual spectrum result either adds a finding or closes a
   hypothesis; either outcome goes in §5.

### Nice-to-have

9. **mod_mul and mod_two_step.** §5.3 open questions. Strong test of
   whether Clock generalizes beyond group addition. If they work,
   big paper upgrade. If they fail cleanly, constrains the claim —
   also a win.

10. **k_sweep with the fixed regularizer.** The v1 K-sweep table is
    paper-facing and is referenced in the min-K claim. Worth
    re-confirming after the trainer fix.

### Won't-do for this submission

- Tier 5 (language modeling) — belongs in the v2 paper or a follow-up.
- tf_sweep (minimum viable transformer) — interesting but not
  blocking; the 172× headline from the existing comparison stands.
- held_out_primes with more primes — 8 is enough for the cross-prime
  claim.

---

## 7. Codebase map and design

### 7.1 Package structure

```
pan_lab/
├── pyproject.toml          Editable install config
├── Makefile                Common operations: install, test, plan, smoke, paper
├── README.md
├── .gitignore
├── pan_lab/                The library itself
│   ├── __init__.py         Public API surface
│   ├── __main__.py         Enables `python -m pan_lab`
│   ├── config.py           RunConfig, constants, provenance
│   ├── data.py             mod_add, mod_mul, mod_two_step datasets
│   ├── models/
│   │   ├── __init__.py     make_model() factory; re-exports
│   │   ├── pan.py          PhaseEncoder, PhaseMixingLayer, PhaseGate, PAN
│   │   ├── transformer.py  Nanda 1-layer baseline
│   │   └── quantize.py     SIFP-16 straight-through fake-quant
│   ├── trainer.py          Generic loop + diversity-reg fix + hooks
│   ├── hooks.py            CheckpointLogger, CSVStreamLogger
│   ├── analysis.py         Ablations, freq errors, slot census, Clock detection
│   ├── reporting.py        ExperimentReporter + pandas aggregation + save_model
│   ├── plots.py            matplotlib — reads DataFrames, writes PNG
│   ├── experiments.py      EXPERIMENT_REGISTRY + YAML loader + CLI dispatch
│   ├── decoder_analysis.py New as of Phase H — basis-projection experiment
│   └── cli.py              python -m pan_lab entry point
├── experiments/            YAML specs — one per experiment
│   ├── compare.yaml
│   ├── tier3.yaml
│   ├── slot_census.yaml
│   ├── k_sweep.yaml
│   ├── k8_sweep.yaml
│   ├── dw_sweep.yaml
│   ├── wd_sweep.yaml
│   ├── primes.yaml
│   ├── held_out_primes.yaml
│   ├── freq_init_ablation.yaml
│   ├── sifp16_inference.yaml
│   ├── decoder_swap.yaml        (deprecated, kept for comparison)
│   ├── decoder_analysis.yaml    (new, Phase H)
│   ├── mod_mul.yaml
│   ├── mod_two_step.yaml
│   └── tf_sweep.yaml
├── tests/                  pytest — 57 tests, ~25s on CPU
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_data.py
│   ├── test_gradient_flow.py      ← the critical bug-fix test
│   ├── test_trainer.py
│   ├── test_reporting.py
│   ├── test_experiments.py
│   ├── test_analysis_and_quant.py
│   └── test_save_model.py
└── paper_extract.py        Standalone script — reads results/ and writes
                            paper_extract.md for hand-off to LLMs.
```

### 7.2 Design principles

- **Declarative over imperative for experiments.** Each experiment is
  a YAML (*what to run*) plus a function in `experiments.py`
  (*how to run it*). Functions are named in `EXPERIMENT_REGISTRY`.
  Adding a new experiment = one function + one YAML.

- **Pure functions over methods where possible.** `analysis.py`
  functions take a model or a DataFrame and return a dict or a
  DataFrame. They never mutate state. Ablations save/restore
  parameters around their interventions.

- **CSV as the canonical data interchange.** Every experiment writes
  `runs.csv` (one row per run), `curves.csv` (one row per eval step),
  and optionally `ablations.csv`, `slots.csv`, `checkpoints.csv`. Plots
  are always regenerated from CSVs via `--replot` — no in-memory-only
  figures.

- **Provenance captured at run time.** `manifest.json` per experiment
  contains git SHA, torch version, device, hostname, timestamp, and
  argv. Every result traces back to a specific commit.

- **Hooks for training-time side effects.** `CheckpointLogger`
  (mechanistic snapshots) and `CSVStreamLogger` (live CSV streaming)
  are hooks. Adding a new per-step probe is a ~10-line class.

- **Seeding discipline.** `make_model(cfg)` seeds *before* construction.
  `train(model, cfg)` seeds *again* before the loop. This is
  belt-and-suspenders because data loader randomization also uses the
  RNG.

### 7.3 Data flow for a typical experiment

```
CLI argv
   │
   ▼
cli.py::main                 parse args, find yaml, dispatch
   │
   ▼
experiments.py::run_from_yaml
   │
   ├─► config.py::RunConfig.from_dict    parse YAML base config
   │
   ▼
experiments.py::<registered_function>   e.g. exp_tier3, exp_slot_census
   │
   ├─► builds list of RunConfigs (sub-runs)
   │
   └─► experiments.py::_run_cfgs
         │
         └─► for each cfg:
               ├─► data.py::make_modular_dataset    seeded splits
               ├─► models.make_model(cfg).to(DEVICE) seeded init
               ├─► hooks.py::{CheckpointLogger, CSVStreamLogger}
               ├─► trainer.py::train(model, cfg, ...)
               │     │
               │     ├─► loss = CE + dw * diversity_penalty(mix_features(x))
               │     ├─► hook.on_eval(step, model, history, val_loss, val_acc)
               │     └─► returns TrainResult
               │
               ├─► reporter.add_run(result, ablations=, slots=)
               │     │
               │     └─► appends to internal list of dicts
               │
               └─► if cfg.save_model: save_model_weights(result, out_dir)

         ├─► rep.write_all()    runs.csv, curves.csv, ablations.csv,
         │                       slots.csv, checkpoints.csv, manifest.json
         ├─► rep.print_summary() stdout summary table
         └─► plots.py::plot_*    PNG files read from CSVs
```

### 7.4 Public API

What an external user needs to know:

```python
from pan_lab import (
    RunConfig, DEVICE,
    make_modular_dataset,
    PhaseAccumulatorNetwork, TransformerBaseline, make_model,
    train, TrainResult, TrainHistory,
    analyze_pan, ablation_test, fourier_concentration,
    compute_frequency_errors, detect_mode_collapse,
    slot_activation_census,
)

# Basic use:
cfg = RunConfig(p=113, k_freqs=9, seed=42)
tx, ty, vx, vy = make_modular_dataset(p=cfg.p, seed=cfg.seed)
model = make_model(cfg).to(DEVICE)
result = train(model, cfg, tx, ty, vx, vy)
abl = ablation_test(result.model, vx, vy)
```

For extending the framework:

```python
from pan_lab.experiments import register, EXPERIMENT_REGISTRY

@register("my_new_experiment")
def exp_my_new(base: RunConfig, out_dir: str, dry_run=False, **kwargs):
    # build cfgs, call _run_cfgs, return reporter
    ...
```

### 7.5 Testing philosophy

Tests are *light on unit tests, heavy on regression tests*. The three
most important tests:

- `test_gradient_flow.py::test_mix_features_routes_grads_to_encoder_freqs`
  — the fix for the diversity-reg bug. If this test starts failing,
  the bug is back.
- `test_trainer.py::test_same_seed_same_config_same_history`
  — the reproducibility guarantee. If this fails, seeding discipline
  has broken somewhere.
- `test_save_model.py::test_save_model_writes_pt_file`
  — the save_model-is-wired test. If this fails, post-hoc analyses
  will be blocked again.

The full suite runs in ~25 seconds on CPU. Tests deliberately use tiny
configs (P=11, K=3, a few hundred steps) so the test feedback loop is
fast. Correctness of behavior is tested separately from convergence to
grokking (grokking is tested by the experiment runs themselves, not by
unit tests).

---

## 8. How to extend the codebase

### 8.1 Adding a new dataset / task

```python
# In pan_lab/data.py:
def make_modular_dataset(p, task_kind, ...):
    if task_kind == "my_new_task":
        inputs = _all_pairs(p)
        labels = my_new_function(inputs)
        ...
```

Then register the task kind in the YAML with `task_kind: my_new_task`.
The PAN `n_inputs` must match the task's input arity.

### 8.2 Adding a new model architecture

```python
# In pan_lab/models/my_arch.py:
class MyArchitecture(nn.Module):
    def __init__(self, p, ...): ...
    def forward(self, inputs): ...     # returns (B, P) logits
    def count_parameters(self): ...

# In pan_lab/models/__init__.py, extend make_model:
def make_model(cfg):
    if cfg.model_kind == "my_arch":
        return MyArchitecture(cfg.p, ...)
```

If the model supports the PAN-specific diversity-reg path, it needs a
`mix_features(x)` method that returns a (B, K) tensor with autograd
edges live.

### 8.3 Adding a new experiment

Write the function. Register it. Write the YAML.

```python
# In pan_lab/experiments.py (or a new module that experiments.py imports):
@register("my_experiment")
def exp_my_experiment(base, out_dir, dry_run=False, seeds=None, **_):
    seeds = seeds or [42, 123, 456]
    cfgs = [base.with_overrides(seed=s, label=f"myexp-s{s}")
            for s in seeds]
    rep = _run_cfgs(cfgs, "my_experiment", out_dir, dry_run,
                    ablations=True)
    if not dry_run:
        # custom plotting, extra CSV writes, etc.
        pass
    return rep
```

```yaml
# experiments/my_experiment.yaml
experiment: my_experiment
out_dir:    results/my_experiment
base:
  p:       113
  k_freqs: 9
  n_steps: 50000
experiment_args:
  seeds: [42, 123, 456]
```

Dry-run first: `python -m pan_lab experiments/my_experiment.yaml --dry-run`.

### 8.4 Adding a new analysis / plot

If it only needs CSVs (doesn't load .pt models), add it to
`plots.py`. Each function takes DataFrames and a path, writes a PNG.
No side effects on in-memory state.

If it needs trained model weights, add it as a post-hoc script (like
`paper_extract.py` does with mixing matrix analysis). Load from the .pt
files, never from in-memory references. Keep the analysis independent
of training.

---

## 9. Experiment catalog with current status

As of this session:

| Experiment | Purpose | Current status | Paper role |
|---|---|---|---|
| compare | PAN vs TF head-to-head | ✓ run | §3.1 |
| tier3 | Mechanistic single-seed | ✓ run; .pt saved | §3.2 |
| slot_census | 20-seed Clock verification | ✓ run; 20 .pt saved | §3.3 |
| k_sweep | Minimum-K characterization | not re-run with trainer fix | §3.7 (cite v1) |
| k8_sweep | K=8 anomaly investigation | ✓ run | §3.7 footnote |
| freq_init_ablation | Fourier vs random | ✓ 5-seed + ✓ 20-seed random rerun | §3.4 |
| sifp16_inference | 16-bit quantization | ✓ run | §3.6 |
| decoder_swap | Canonical decoder test | ✓ run but formula buggy — drop from paper | — |
| decoder_analysis | Decoder decomposition | ✗ not yet run (replaces decoder_swap) | §5.2 |
| held_out_primes | Reviewer robustness | ✓ run; P=97 = 500K run also | §3.5 |
| dw_sweep | Diversity-weight tuning | ✗ not re-run with fix | not paper-facing |
| wd_sweep | Weight-decay tuning | ✗ not run | not paper-facing |
| primes | Development primes | v1 data (cite) | §3.5 |
| mod_mul | Multiplicative group | ✗ not run | §5.3 open |
| mod_two_step | Composition | ✗ not run | §5.3 open |
| tf_sweep | Minimum transformer | ✗ not run | supplementary |

Experiments marked ✗ are either open tasks or were never needed for
this paper submission.

---

## 10. Datafile formats

Every experiment output directory looks like:

```
results/<experiment_name>/
├── runs.csv                         one row per training run
├── curves.csv                       one row per eval step per run
├── curves_stream.csv                same data, written during training
├── manifest.json                    experiment metadata + provenance
├── ablations.csv                    if ablations=True: per-intervention acc
├── slots.csv                        if slots=True: freq-slot census long-format
├── checkpoints.csv                  if record_checkpoints: mechanistic snapshots
├── model_<run_id>.pt                if save_model: trained weights
├── <experiment-specific>.csv        e.g. decoder_swap.csv, quant_eval.csv
└── *.png                            plots generated by plots.py
```

### runs.csv columns (required)

```
run_id, experiment, label, p, task_kind, model_kind, k_freqs, d_model,
seed, weight_decay, diversity_weight, freq_init, n_steps_planned,
n_steps_actual, grok_step, grokked, final_val_acc, peak_val_acc,
final_train_loss, final_val_loss, elapsed_s, param_count, mode_collapsed
```

`grok_step` is `-1` when the run did not grok. `grokked` is a boolean.

### curves.csv columns

```
run_id, step, train_loss, val_loss, val_acc
```

### checkpoints.csv columns

```
run_id, step, encoder, k, theoretical, learned, error
```

`encoder` is 0 or 1 (or up to `n_inputs-1` for two-step tasks). `k` is
the 1-indexed slot number within that encoder. `theoretical = k * 2π/P`.
`learned` is the wrapped frequency value. `error` is angular distance
on S¹.

### manifest.json

```json
{
  "experiment": "tier3",
  "n_runs": 1,
  "provenance": {
    "timestamp": "2026-04-16T15:45:00",
    "hostname": "...",
    "torch": "2.11.0",
    "device": "mps",
    "git_sha": "...",
    ...
  },
  "files": {"runs.csv": "...", ...}
}
```

### Model .pt files

```python
torch.save({
    "state_dict":  model.state_dict(),
    "arch":        "PAN" | "TransformerBaseline",
    "config":      RunConfig.as_dict(),
    "grok_step":   int | None,
    "param_count": int,
}, path)
```

To load:

```python
ckpt = torch.load(path, weights_only=False)
from pan_lab.models.pan import PhaseAccumulatorNetwork
cfg  = ckpt["config"]
pan  = PhaseAccumulatorNetwork(
    p         = cfg["p"],
    k_freqs   = cfg["k_freqs"],
    n_inputs  = 3 if cfg["task_kind"] == "mod_two_step" else 2,
    freq_init = cfg["freq_init"],
)
pan.load_state_dict(ckpt["state_dict"])
```

---

## 11. Known traps and gotchas

### 11.1 torch.compile and MPS

`use_compile=True` can work on CUDA but has inconsistent MPS support
depending on PyTorch version. More importantly, **compile changes
floating-point accumulation order**, so compiled runs are not bitwise
reproducible against eager runs even with the same seed. All the
experiments in this repo default to `use_compile=False` for exactly
this reason. Don't flip this to True without explicit justification.

### 11.2 Raw frequencies vs wrapped frequencies

The encoder `freq` parameter is a vanilla `nn.Parameter` in ℝ. Adam
can push it anywhere. The *forward pass* only sees the value mod 2π.
So when you read `model.encoders[0].freq.data`, you may see values like
6.28 or -12.3 that are unremarkable modulo 2π. Always wrap before
comparing to theoretical basis values:

```python
wrapped = float(model.encoders[0].freq[k] % TWO_PI)
```

The `get_learned_frequencies()` method does this. Direct parameter
access does not.

### 11.3 The encoder-slot labeling is arbitrary

When the analysis says "encoder 0 slot 8 converged to k=9," what that
really means is: "the 8th row of encoder 0's freq parameter happens to
currently hold a value close to 9·2π/P." The slot index has no
semantic meaning — Adam could have moved any slot to any frequency.
For cross-seed comparisons, always reorganize by *nearest theoretical
k*, not by slot index.

### 11.4 FFT conjugate symmetry

When analyzing residual spectra of real-valued decoder weights,
every frequency component at bin `k` has a conjugate at bin `P - k`
of equal magnitude. Don't double-count. Iterate over the first half
of the FFT bins only, or deduplicate by taking `min(k, P-k)`.

### 11.5 Val_samples subsampling

`RunConfig.val_samples` is provided for speed (full val on P=113 is
7,661 samples, ~338ms per eval on CPU). If set, the val set is
subsampled once at run start and the same subset is used for every
eval. **For paper-grade results, set this to None** so val_acc is
measured on the full held-out set.

### 11.6 `mode_collapsed` detector is coarse

The current `detect_mode_collapse` checks whether every output
channel's top-|weight| comes from the same input slot. This catches
extreme collapse but misses subtler failures: seeds that converge to
a 2-slot rotation, seeds whose mixing matrix is low-rank without being
single-slot-dominant, etc. Multiple grokking failures in the 20-seed
census had `mode_collapsed=False` despite clearly degenerate circuits.
Don't trust the flag as the sole indicator of mechanistic health.

### 11.7 Early stop hides training dynamics

With `early_stop=True`, training halts at the first eval where
val_acc crosses `grok_threshold`. For mechanistic runs (tier3,
slot_census, decoder_analysis) you want `early_stop=False` so the
post-grokking crystallization phase can complete. The relevant YAMLs
set this explicitly.

### 11.8 Checkpoints can be large

`CheckpointLogger` records `get_learned_frequencies()` at every eval
step. For a 100K-step run at log_every=500 that's 200 snapshots × 2
encoders × K frequencies per snapshot. Not huge, but note that
`checkpoints.csv` grows linearly with `n_steps / log_every`. The
20-seed slot_census does not log checkpoints by default — only tier3
does.

### 11.9 PAN initialization now seeds three RNGs

`make_model(cfg)` calls `torch.manual_seed`, `np.random.seed`, and
`random.seed` all with `cfg.seed`. This is wasteful but defensive —
any library code that reaches for numpy or Python random will get
the same values across runs. If you see reproducibility drift, check
whether some new dependency is using a non-seeded RNG (e.g. `secrets`
or a C-backed RNG).

---

## Where to go from here

If you're continuing this work:

1. **First**: run `pytest` to confirm the library still works in
   your environment.
2. **Second**: read the v2 paper draft (`pan_paper_v2.md`) to see the
   intended claim structure.
3. **Third**: pick one item from §6 "must-do" list. The cheapest high-value
   item is running `decoder_analysis` — the code is written, the math
   is tested, the YAML is there; you just need to invoke it.
4. **If you have more time**: re-run `dw_sweep` with the trainer fix.
   This is the single cleanest way to verify that the fixed regularizer
   doesn't materially change the paper's headline results, and if it
   does, we need to know before submission.

Questions worth keeping in mind while you work:

- **Is the claim I'm about to make about PAN actually supported by 16+
  seeds of data, or is it a lucky artifact of one run?** The Clock
  structure survives this test; "random init is strictly better" did
  not.
- **If I were a NeurIPS reviewer, what would I ask for that I don't
  currently have?** The decoder_analysis residual spectrum is high on
  this list.
- **Is this experiment actually testing what I think it's testing?**
  The decoder_swap debacle is the cautionary tale. Pure Clock decoders
  need the mixing-matrix effective frequency AND the gate reference
  phase; either omission makes the experiment measure something else.

---

*End of research companion.*
