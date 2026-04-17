# Experiment 2 — Why Random Initialization Wins in PAN

## Purpose

Explain why `freq_init = random` appears to outperform Fourier initialization in PAN.

The updated interpretation is no longer just:

> random init is empirically better

The sharper question is:

> Does random initialization help because it improves early spectral coverage, or because it breaks harmful symmetry and allows channels to specialize into a usable scaffold faster?

This experiment is designed to separate those mechanisms.

---

## Core Questions

1. Why does random initialization grok faster and more reliably than Fourier init?
2. Is the benefit primarily:
   - **symmetry breaking**
   - **frequency-space coverage**
   - **better channel differentiation**
   - or some combination of these?
3. Does init mainly affect:
   - the **early scaffold formation phase**
   - the **late refinement phase**
   - or both?

---

## Main Hypotheses

### H1 — Symmetry-breaking hypothesis
Exact Fourier initialization puts channels in a high-symmetry state that is analytically meaningful but optimization-hostile.

Prediction:
- early organization is slower or more ambiguous
- channels differentiate more slowly
- plateau burden is higher
- small perturbations to Fourier init should help a lot

### H2 — Coverage hypothesis
Random init works because it seeds a broader or more useful spread of frequencies.

Prediction:
- random init with enforced spread should outperform clustered random init
- Fourier init with permutation or diversity-preserving noise may behave more like random

### H3 — Scaffold-timing hypothesis
Init affects **when and how** the early scaffold forms:
- random init may create a rough scaffold earlier
- or create a scaffold that is easier to redistribute later

---

## Experimental Design

## Fixed setup

Hold constant:
- task: mod-113 addition
- PAN architecture
- `k_freqs = 9` as primary condition
- same optimizer and training schedule
- no early stop
- dense logging of trajectory metrics

Recommended seeds:
- at least 5 per init family

---

## Initialization Families

## A. Exact Fourier init
Canonical baseline:
- `f_k = k * 2π / P` or current Fourier scheme

## B. Random uniform init
Current empirical winner:
- frequencies sampled uniformly on `[0, 2π)`

## C. Jittered Fourier init
Start from Fourier init, then add small noise:
- e.g. Gaussian or uniform jitter with small angular scale

Purpose:
- isolates symmetry-breaking with minimal departure from Fourier structure

## D. Permuted / reassigned Fourier init
Keep the same Fourier values but shuffle them across slots or create repeated assignments intentionally

Purpose:
- tests whether exact canonical ordering matters

## E. Spread random init
Random initialization with minimum pairwise separation constraint

Purpose:
- isolates the role of broad spectral coverage

## F. Clustered random init
Random initialization but deliberately cluster several frequencies near each other

Purpose:
- tests whether random only works when it avoids early crowding

---

## Quantities to Log

Use the same metrics as the K sweep:

- validation accuracy
- train loss / val loss
- `fourier_norm`
- `fourier_conc`
- `mix_entropy_min`
- `mix_entropy_mean`
- `decoder_norm`
- `circuit_ratio`
- `plateau_steps`

Also add:

### Frequency-geometry metrics
Computed from encoder frequencies at each checkpoint:
- minimum pairwise frequency distance
- mean pairwise distance
- entropy of frequency distribution over nearest theoretical bins
- number of effectively distinct frequencies
- slot crowding score

### Channel-differentiation metrics
If possible:
- gate correlation matrix statistics
- mixing-row similarity
- fraction of near-duplicate channels
- active-slot count per encoder

---

## Key Derived Metrics

### 1. Time to scaffold formation
Same idea as in the decoder experiment:
- time to early entropy drop
- time to early `fourier_conc` peak
- early `fourier_norm` gain

### 2. Early crowding score
A scalar summary of how bunched the initialized or early-learned frequencies are.

### 3. Refinement score
Late:
- `fourier_norm` slope
- `circuit_ratio` slope
- `fourier_conc` drop
- plateau resolution

### 4. Plateau burden
Final or cumulative plateau behavior.

---

## Crucial Ablation Pairs

## Ablation 1 — Fourier vs jittered Fourier
This is the most important one.

If tiny jitter closes much of the gap to random init, then exact Fourier symmetry is the main problem.

### Interpretation
- large improvement from tiny jitter → symmetry-breaking explanation
- little improvement → need broader explanation

## Ablation 2 — Random vs spread-random vs clustered-random
This tests whether random wins because it samples better frequency geometry.

### Interpretation
- spread-random best → coverage matters
- clustered-random worst → early crowding is harmful
- all random variants similar → symmetry breaking may dominate

## Ablation 3 — Late decoder-only rescue across init families
At matched early checkpoints, freeze the representation and retrain decoder only.

This checks whether some init families produce an early scaffold that is already usable, while others do not.

---

## Primary Analyses

### Analysis 1 — Which init family forms the scaffold fastest?
Compare by init:
- time to entropy collapse
- time to `fourier_conc` peak
- early `fourier_norm` gain

### Analysis 2 — Which init family produces the most harmful plateaus?
Compare by init:
- `plateau_steps`
- plateau onset
- plateau growth rate
- final / peak accuracy

### Analysis 3 — Is the main gap early or late?
Compare init families on:
- early scaffold metrics
- late refinement metrics

Possible outcomes:
- early metrics differ strongly → init mostly affects scaffold formation
- late metrics differ strongly but early metrics similar → init mostly affects refinement
- both differ → init affects the whole trajectory

### Analysis 4 — Does symmetry-breaking alone explain the gain?
Compare:
- exact Fourier
- jittered Fourier
- random

If jittered Fourier behaves much more like random than exact Fourier, that is strong evidence for symmetry-breaking.

### Analysis 5 — Does coverage matter independently?
Compare:
- spread-random
- ordinary random
- clustered-random

If spread-random is reliably best and clustered-random worst, then geometry of initial spectral coverage matters in its own right.

---

## Primary Plots

### Plot 1 — Overlay by init family
For each init family, aggregate trajectories of:
- `fourier_conc`
- `fourier_norm`
- `mix_entropy_min`
- `circuit_ratio`
- val accuracy

### Plot 2 — Scaffold timing by init
Boxplots of:
- time to entropy drop
- time to `fourier_conc` peak
- early `fourier_norm` gain

### Plot 3 — Plateau burden by init
Compare:
- final `plateau_steps`
- plateau onset
- peak accuracy

### Plot 4 — Init geometry vs success
Scatter:
- initial min pairwise frequency distance vs peak accuracy
- initial crowding score vs plateau burden

---

## Decision Logic

### Outcome A — Jittered Fourier ≈ random
Interpretation:
- exact symmetry is the main problem
- architecture already contains the right inductive bias
- analytically “correct” init over-constrains optimization

### Outcome B — Spread-random > random > clustered-random
Interpretation:
- spectral coverage matters substantially
- random helps partly because it avoids early crowding

### Outcome C — Random wins mainly in late refinement, not early scaffold timing
Interpretation:
- random init does not necessarily create structure sooner
- it creates a scaffold that is easier to redistribute and strengthen later

### Outcome D — Random wins in both early scaffold timing and late refinement
Interpretation:
- random init helps the entire trajectory

---

## Most Important Claim This Experiment Can Support

If successful, this experiment could justify a statement like:

> Random initialization outperforms Fourier initialization in PAN not because it is closer to the analytically correct solution, but because it breaks high-symmetry starting conditions and promotes earlier channel differentiation and more effective refinement of the learned spectral scaffold.

That is a much stronger result than simply reporting a sweep table.

---

## Recommended Minimal Version

If time is limited, do only three init families:

1. exact Fourier  
2. jittered Fourier  
3. random uniform

and measure only:

- val accuracy
- `fourier_conc`
- `fourier_norm`
- `mix_entropy_min`
- `plateau_steps`

That reduced experiment will still answer the most important question:
**is exact Fourier bad because it is too symmetric?**
