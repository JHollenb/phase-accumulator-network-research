# Experiment 01 — The Diffusion Phase of PAN Grokking

**Author:** jholl · **Date:** 2026-04-18
**Notebook:** [`notebooks/01_diffusion_investigation.ipynb`](../../notebooks/01_diffusion_investigation.ipynb)
**Companion module:** [`notebooks/diffusion_analysis.py`](../../notebooks/diffusion_analysis.py)
**Preceding plan:** [`docs/todo/experiment-decoder.md`](../todo/experiment-decoder.md)

---

## Abstract

We test the hypothesis that the Clock circuit in a Phase Accumulator Network (PAN) forms early in training, and that the remaining steps constitute a "diffusion" phase that produces grokking. Across 465 runs (K=1..20 × ≈20 seeds per K, pooled from four training runs) we extract per-run milestone timelines and identify which of eight candidate milestones is the last to fire before generalization. The Clock-first hypothesis is confirmed — median `t_clock` = 1,000 steps vs median `t_grok` = 15,000 steps, so 89 % of training occurs after Clock-pair structure has formed. However, **the diffusion phase is not decoder catch-up**: the decoder's Fourier alignment (M8) fires early (median 1,500 steps) and is the bottleneck in only 1 of 276 grokked runs. Instead, diffusion is **mix-layer pruning** — `mix_row_eff_n_mean` drops from 9.4 to 4.2 and `active_freq_count` from 9 to 5 during the diffusion window, while encoder and decoder metrics stay flat. Grokking happens when this pruning converges on a minimal Fourier subset that makes the gate representation linearly decodable; this representation milestone (M6) is the modal bottleneck (44 % of runs). No grokked run and zero failed runs in the sample are decoder-limited.

## Headline findings

| # | Claim | Number | Evidence |
|---|---|---|---|
| 1 | Clock-first is confirmed | median `t_clock` = 1,000 vs `t_grok` = 15,000 | [Fig 3–4](figs/fig03_04_clock_first_gap.png) |
| 2 | Diffusion fraction of total training | 89 % (median across 276 grokked runs) | [Fig 3–4](figs/fig03_04_clock_first_gap.png) |
| 3 | Decoder is never the bottleneck | 1 of 276 grokked runs (0.4 %) | [Fig 7–8](figs/fig07_08_bottleneck.png) |
| 4 | Gate-probe is the modal bottleneck | 44 % of grokked runs | [Fig 7–8](figs/fig07_08_bottleneck.png) |
| 5 | Mix-layer pruning dominates the diffusion window | `mix_row_eff_n`: 9.40 → 4.20; `active_freq_count`: 9 → 5 | [Fig 6](figs/fig06_diffusion_traces.png) |
| 6 | No failed run is decoder-limited | 0 / 190 non-grokked runs | [Fig 11–12](figs/fig11_12_failure_modes.png) |

## Question

The Clock algorithm for modular addition (Nanda et al. 2023) has two moving parts in a transformer: a sparse Fourier encoding of the inputs and a decoder that projects out pairwise products of the encodings. The original framing of this investigation (see [`docs/todo/experiment-decoder.md`](../todo/experiment-decoder.md)) asked:

> At what point does the gate representation become linearly usable, and what part of late training is decoder alignment versus representation redistribution/strengthening?

The implicit expectation was **Outcome A** from that document: *"Probe succeeds early, trained decoder lags."* If true, one would expect the probe accuracy (`gate_linear_acc`) to cross 0.9 well before the trained-decoder accuracy, and the trained-decoder accuracy to be the last milestone before grok.

This experiment tests that expectation against the full K-sweep data.

## Method

### Datasets

| Dataset | Description | Runs used |
|---|---|---|
| `interesting_results/k_census_n20_random` | K=1..20 × 20 seeds, random freq init | 284 |
| `interesting_results/k_census_n20_fourier` | K=1..20 × ~20 seeds, Fourier freq init | ~50 (grokked only) |
| `results/paper_k13_fourier` | 15 K=13 seeds with dense per-step checkpoints | 15 |
| `interesting_results/init_random_primary_k` | K=10 × 20 seeds, random init, mechanistic logging | 20 |
| `interesting_results/tier3` | K=9 single run with 500-step per-slot checkpoints | 1 |

### Milestone extraction

For each run we extract the first training step where each metric crosses a threshold. Thresholds are calibrated so that ≥ 90 % of grokked runs actually reach them. Definitions:

| Milestone | Column | Threshold | Direction | Meaning |
|---|---|---|---|---|
| `t_snap_{a,b}` | `enc{0,1}_snap_mean` | < 0.30 rad | ↓ | encoder Fourier-lattice lock (M1) |
| `t_freqs_stable` | `active_freq_count` | no change for 5 evals | — | active subset chosen (M5) |
| `t_sparse` | `mix_row_eff_n_mean` | < 4.0 | ↓ | mix rows sparsify (M4) |
| `t_clock` | `clock_compliance` | > 0.50 | ↑ | mix-layer Clock pairs formed (M2) |
| `t_align` | `clock_freq_align_mean` | < 0.30 rad | ↓ | paired slots share a Fourier mode (M3) |
| `t_decoder` | `decoder_fourier_peak_mean` | > 0.40 | ↑ | decoder columns are sinusoidal (M8) |
| `t_probe` | `gate_linear_acc` | > 0.80 | ↑ | gate linearly decodable (M6) |
| `t_fp32_high` | `fp32_acc` | > 0.80 | ↑ | trained decoder 80 % (M7) |
| `t_grok` | `val_acc` | ≥ 0.99 | ↑ | generalization |

Full metric definitions: [`docs/metrics.md`](../metrics.md). A sensitivity check in the notebook (§1b) perturbs every threshold by ±50 % and confirms the ordering is stable.

## Findings

### Temporal ordering

Grokked K=7..14 runs show a consistent timeline:

**`t_snap` ≈ `t_align` ≈ `t_decoder` ≪ `t_clock` ≈ `t_sparse` ≈ `t_freqs_stable` ≪ `t_probe` ≈ `t_fp32_high` ≪ `t_grok`**

![Milestone timing, K=7..14 grokked runs](figs/fig01b_milestones_K7to14.png)

Spearman correlations with `t_grok` are 0.86–0.87 for `t_probe` and `t_fp32_high` and only 0.17–0.32 for the scaffold milestones. **When a run groks is determined by when the gate becomes decodable, not by when the scaffold forms.**

### Clock-first gap

For each grokked run, `gap = t_grok − t_clock` and `frac = gap / t_grok`. Median `frac` is 0.89.

![Clock-first gap: histogram and scatter](figs/fig03_04_clock_first_gap.png)

Every grokked point lies above `y = x` in the right panel — Clock-pair structure forms before generalization without exception in our sample.

### What diffuses

Each run's metric trajectory is resampled onto normalized phase-progress `[0 = t_clock, 1 = t_grok]` and pooled across K=7..14 (n = 248).

![Diffusion-window median trajectories](figs/fig06_diffusion_traces.png)

| Metric | Value at `t_clock` | Value at `t_grok` | Δ |
|---|---|---|---|
| `mix_row_eff_n_mean` | 9.40 | 4.20 | **−5.20** |
| `active_freq_count` | 9 | 5 | **−4** |
| `clock_compliance` (M2) | 0.60 | 0.82 | +0.22 |
| `clock_freq_align_mean` | 0.377 | 0.297 | −0.080 |
| `enc0_snap_mean` | 0.306 | 0.242 | −0.064 |
| `decoder_fourier_peak_mean` | 0.580 | 0.628 | +0.048 |

Encoder lock and decoder Fourier peak barely move; the mix-row effective-n drops by more than half and the active-frequency count makes a step-function cut. **Diffusion is mix-layer pruning toward a minimal Clock-shaped basis.**

### The bottleneck

For each grokked run we find `argmin_i (t_grok − t_i)` over milestones `t_i ≤ t_grok` — the last milestone to fire before grokking.

![Bottleneck frequency across runs](figs/fig07_08_bottleneck.png)

| Bottleneck | Count | Fraction |
|---|---|---|
| `t_probe` (M6 gate decodability) | 121 | **44 %** |
| `t_sparse` (M4 mix sparsify) | 62 | 22 % |
| `t_freqs_stable` (M5 active freqs) | 57 | 21 % |
| `t_snap_{a,b}` (M1 encoder lock) | 29 | 11 % |
| `t_clock` (M2 clock-pair) | 6 | 2 % |
| **`t_decoder` (M8 decoder Fourier)** | **1** | **0.4 %** |

The "decoder lags representation" framing is refuted. The decoder is Fourier-aligned on average 13,500 steps before grok; the last thing that happens is the gate becoming linearly readable.

### Probe-gap trajectory

`gate_decoder_gap = gate_linear_acc − fp32_acc`. A positive gap would mean the representation is decodable even though the trained decoder hasn't caught up.

![Probe gap: grokked clean runs and failed low-K runs](figs/fig09_10_probe_gap.png)

- **Grokked K=9..11 (Fig 9).** Median gap peaks slightly positive (~+0.05) around step 5K and decays to ~0 by grok. Terminal cohort median = −0.030. The "representation ahead of decoder" effect is real but small and transient.
- **Failed low-K (Fig 10).** Median gap is ≤ 0 at every step. At terminal eval, cohort `probe_median = 0.152` and `fp32_median = 0.216` — **both probe and trained decoder fail together**. No sign of a ready-but-unused representation.

This is the `experiment-decoder.md` *Outcome B* ("both improve together, representation-limited"), not *Outcome A*.

### Failure-mode taxonomy

Non-grokked runs are classified into five categories based on terminal metric values:

![Failure modes per K and peak_val_acc by mode](figs/fig11_12_failure_modes.png)

- **K=1** — 20/20 **collapse** (`active_freq_count → 1`). One Fourier channel is insufficient for mod-113 addition.
- **K=2–5** — primarily `rep_weak` and `collapse`. The model selects a Fourier subset that is too narrow for the task.
- **K=6–14** — primarily `plateau` (peak_val_acc in [0.50, 0.98]) and `rep_weak`. The scaffold forms but the diffusion phase fails to converge.
- **`decoder_limited` category: zero runs in 190.** The decoder is never the reason a failed run fails.

### Per-slot dynamics (tier3)

The `tier3` K=9 run has per-slot (18 slots = 2 encoders × 9 frequencies) error checkpoints every 500 steps, where error = `|learned_freq − theoretical_freq|`. The theoretical basis is `k · 2π / 113`; SIFP-16 quantization floor is `2π / 65536 ≈ 9.6e-5` rad. `t_grok = 16,500`.

![tier3 per-slot error trajectories](figs/fig13_tier3_slot_dynamics.png)

Of 18 slots, only **3 converge to SIFP-16 precision** by the final step: `enc0-k3` (step 2,000), `enc1-k5` (step 2,500), `enc0-k8` (step 7,000) — all **before** `t_grok = 16,500`. One more (`enc1-k8`) hits SIFP-16 transiently at step 10,500 then drifts back. The remaining 14 slots stay stuck at initialization noise.

This matches the companion paper's §3.2 claim that 3–5 Clock slots suffice for mod-113 addition. It also **refines** the companion's reading of tier3: the companion framed the 80K-step SIFP-16 convergence as post-grok "cleanup after the Clock does its job," but the converged slots actually hit SIFP-16 *during* the diffusion phase. What happens post-grok is further tightening of already-converged slots, not initial precision achievement.

### K-dependence

![Per-metric peak timescale vs K, with grok-rate overlay](figs/fig14_k_dependence.png)

![Trajectory cohorts: failed K=2-3 vs clean K=9 vs edge K=13](figs/fig15_k_trajectories.png)

The failure regime is **low K**, not high K. Grok rates by K: K=1 (0 %), K=2 (0 %), K=3 (0 %), K=4 (0 %), K=5 (25 %), K=6 (45 %), rising to ~80 %+ at K ≥ 7. K=13 — which we originally expected to be a failure mode — groks 19/20 times and is one of the more reliable regimes.

Failed K=2–3 runs (Fig 15) show `mix_row_eff_n_mean` never dropping below 5 and `active_freq_count` stuck at K. With a tight K budget the network cannot prune to a usable subset because there is no redundancy to throw away. Failure is **scaffold-never-forms**, not scaffold-forms-then-stuck.

### Init sensitivity

![Init comparison at K=10: Fourier vs Random](figs/fig16_init_comparison.png)

Under Fourier init at K=10, `t_snap_{a,b}`, `t_align`, and `t_decoder` fire at step 0–1,000. Under random init, the same milestones take 5,000–20,000 steps. **But `t_probe`, `t_fp32_high`, and `t_grok` are similar across both inits** — median `t_grok` is in the 15K–30K range regardless. This is the cleanest causal evidence that diffusion is not scaffold-formation in disguise: when the scaffold is pre-installed, grokking still takes ~15K+ steps.

## Refined mechanism

Across the four datasets, PAN grokking on mod-113 addition proceeds in three phases:

**Phase 1 — Scaffold (step 0 → ~2K).**
Encoders lock to the Fourier lattice, decoder columns become sinusoidal, and mix-row pairs that will eventually participate in Clock are already roughly aligned. Under Fourier init this phase is ~instantaneous; under random init it takes 5K–20K steps. Importantly, the scaffold forms regardless of whether the run will ultimately grok — at low K the encoders still lock, they just don't have enough degrees of freedom for a usable subset to be selected later.

**Phase 2 — Diffusion (step ~2K → ~14K).**
The mix layer prunes. `mix_row_eff_n_mean` falls from 9.4 to 4.2; `active_freq_count` drops from 9 to 5; `clock_compliance` rises from 0.60 to 0.82. Encoder and decoder metrics are essentially flat. In tier3, 3 slots converge to SIFP-16 precision during this phase while the remaining 15 stay at noise. **This phase is the model selecting which slots run the Clock circuit and discarding the rest.**

**Phase 3 — Usable code (step ~10K → ~15K).**
The gate representation becomes linearly decodable (M6 > 0.8), closely followed by trained decoder hitting 80 % (M7), followed ~4K steps later by full generalization. M6 is the modal bottleneck (44 % of runs) and correlates with `t_grok` at ρ = 0.86.

**Post-grok cleanup** (paper companion §2.C) is a fourth phase — further tightening of converged slots and margin growth.

### Refined hypothesis statement

> *The Clock circuit scaffold forms early and fast (Phase 1). PAN then spends ~89 % of training in a diffusion phase (Phase 2) during which the mix layer prunes redundant Fourier slots until a minimal usable subset remains. Grokking (Phase 3) happens when this pruning completes: the gate representation becomes linearly decodable, and the already-Fourier-aligned decoder immediately capitalizes. Decoder misalignment is not the bottleneck — representation selection is.*

## Implications

- **`experiment-decoder.md` reframing.** The original question "does the decoder lag?" is answered negatively. The right question is "what triggers the gate representation becoming decodable?" and the answer from this experiment is "mix-layer pruning to a minimal Fourier subset." A follow-up experiment targeting that transition should manipulate the pruning process directly — e.g. by freezing the mix layer at various points to see how `t_probe` shifts — rather than by retraining the decoder on frozen gates.
- **Paper-companion §3.2 refinement.** The claim that SIFP-16 precision is a post-grok cleanup artifact is partially correct for the *remaining* (untight) slots but wrong for the *converged* slots — those hit SIFP-16 during Phase 2. The cleanup phase tightens slots that were already within SIFP-16 range, it does not achieve SIFP-16 initially.
- **K-sensitivity narrative.** Low-K failures are representation collapse (Phase 1 scaffold forms but Phase 2 cannot prune a minimal subset because there is no redundancy). Mid-K plateaus are incomplete Phase 2. High K is not a failure regime in this sample. The paper section on K should reflect the three-phase-breakdown view rather than a simple "too-few-channels" story.

## Reproducibility

```bash
# Regenerate every figure under docs/experiments/figs/ from the raw experiment CSVs.
uv run python notebooks/diffusion_analysis.py

# Re-execute the notebook in place (imports diffusion_analysis.py, embeds the same figs).
uv run jupyter nbconvert --to notebook --execute notebooks/01_diffusion_investigation.ipynb --inplace
```

Read-only inputs: `interesting_results/k_census_n20_{random,fourier}/`, `interesting_results/init_random_primary_k/`, `interesting_results/tier3/`, `results/paper_k13_fourier/`. The analysis module writes nothing outside `docs/experiments/figs/`.
