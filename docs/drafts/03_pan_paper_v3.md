# Phase Accumulator Networks: Phase Arithmetic as a Neural Primitive for Modular Computation

**Jacob Hollenbeck — April 2026**
*Companion code: `pan_lab/` · Companion documents: Spectral IEEE 754 Whitepaper, SPF Format Specifications*

---

## Abstract

Mechanistic interpretability research on transformers trained on modular
arithmetic has shown that these networks, despite their general-purpose
multiply-accumulate architecture, converge on circuits built from sparse
Fourier representations — inputs are encoded sinusoidally, combined via
trigonometric identities, and decoded through sinusoidal projection
(Nanda et al. 2023; Zhou et al. 2024; Kantamneni & Tegmark 2025). We
ask whether the same computation can be the architecture itself rather
than something gradient descent must discover.

We introduce **Phase Accumulator Networks (PAN)**: a neural architecture
whose primitive operation is sinusoidal phase addition on the unit
circle. On modular addition mod P for P ∈ {43, 59, 67, 71, 89, 113, 127},
PAN achieves ≥99% validation accuracy with 619–1,459 parameters —
127–305× fewer than a standard one-layer transformer baseline. Across
20 independent seeds at P=113, K=9, PAN grokks in 80% of runs, and in
every grokked run the learned phase-mixing matrix exhibits a
Clock-pair structure: each output channel draws its dominant
contributions from one slot of each input encoder, with matched
magnitudes near unit weight. Mean Clock-pair compliance across grokked
runs is 0.82. We verify that 16-bit phase quantization at inference
preserves accuracy to within 0.3%, making the architecture a direct
target for the Spectral IEEE 754 (SPF) hardware format.

We are careful with one claim. While our mixing-matrix inspection
establishes Clock-pair structure, full mechanistic equivalence to the
transformer's Clock algorithm requires characterizing the gate-plus-decoder
stack, which we treat as an open question. What we can confirm is that
PAN's gate representation is linearly sufficient: an optimal linear
decoder (multinomial logistic regression) on gate activations matches
the trained network's accuracy to within 0.01 across seeds.

---

## 1. Introduction

Two recent bodies of work motivate this paper.

First, **mechanistic interpretability of grokking**. Nanda et al. (2023)
showed that a 1-layer transformer trained on modular addition mod 113
converges on a specific Fourier-based circuit: input embeddings settle
on a sparse set of frequencies {14, 34, 41, 42, 52}; the MLP computes
cos(A)cos(B) - sin(A)sin(B) = cos(A+B) via bilinear mixing; the
unembedding projects cos(f(a+b)) for the answer class. Subsequent work
(Zhou et al. 2024) found analogous structures in Pythia and GPT-J;
Kantamneni & Tegmark (2025) extended the finding to Llama-3.1-8B.
These results are consistent: grokking on modular arithmetic converges
on Fourier circuitry, regardless of scale or architecture family.

Second, the **Spectral IEEE 754 (SPF) format**, a log-polar number
representation where a scalar is encoded as (sign, log-magnitude,
16-bit phase). Phase addition in SPF is a 16-bit integer add mod 2^16;
no floating-point multiply is needed. SPF is a speculative hardware
target — it would be roughly 4-8× more energy-efficient than IEEE 754
for a subset of computations involving phase composition — but has
lacked a concrete machine learning use case.

PAN connects these two lines of work. If the transformer's grokking
circuit is phase arithmetic discovered by gradient descent, then an
architecture that makes phase arithmetic the primitive should solve
the same tasks with far fewer parameters, and should map directly onto
SPF hardware.

**Our contribution.** We introduce PAN, demonstrate its parameter
efficiency and cross-prime generalization on modular arithmetic,
verify its mixing-layer structure across 20 seeds, and establish
its tolerance to SPF-level quantization. We also introduce the
`pan_lab` research library, including the `decoder_analysis` tool
for characterizing the gate representation via logistic-regression
upper bounds.

### 1.1 Related Work

**Mechanistic interpretability of grokking.** Nanda et al. (2023),
Varma et al. (2023), Zhou et al. (2024), Kantamneni & Tegmark (2025).

**Inductive biases for symbolic / algorithmic tasks.** Modular
arithmetic as a benchmark: Liu et al. (2022), Power et al. (2022).
Structured inductive biases: neural module networks (Andreas 2016),
sinusoidal positional encodings (Vaswani et al. 2017), Fourier
features (Tancik et al. 2020).

**Low-precision inference.** BitNet, 1-bit quantization, etc. PAN's
quantization story is specifically about phase precision, which is
both tighter (16 bits for full circle) and more theoretically grounded
(phase lives on a compact manifold).

---

## 2. Architecture

### 2.1 Primitive operations

PAN is built from three operations, all of which have direct hardware
analogs in SPF:

**Phase encoding.** Given a discrete input a ∈ {0, ..., P-1}, produce
K phases:

    phi_a[k] = (a * f_k) mod 2π     for k ∈ {1, ..., K}

where f_k is a learned frequency parameter. In SPF hardware this is a
16-bit integer multiply-accumulate mod 2^16.

**Phase mixing.** Given N encoded phases, each a K-vector, concatenate
and apply a learned linear map modulo 2π:

    psi[j] = (sum_i W_mix[j, i] * phi_concat[i]) mod 2π

**Phase gating.** Given a phase psi_j, produce a scalar gate value:

    gate[j] = (1 + cos(psi_j - phi_ref_j)) / 2  ∈ [0, 1]

where phi_ref_j is a learned reference phase. The gate fires maximally
when the input phase matches the reference.

These three operations, plus a learned linear decoder that maps the
K gate values to P class logits, constitute the full PAN forward pass
for modular arithmetic tasks:

    logits[c] = sum_j W_dec[c, j] * gate[j] + bias[c]

### 2.2 Full architecture

For a task with N inputs (N=2 for mod-addition; N=3 for two-step
composition):

```
inputs (B, N) long, each in [0, P)
    | N independent PhaseEncoders, each → (B, K)
    | concat                              → (B, N·K)
    | PhaseMixingLayer                    → (B, K)
    | PhaseGate                           → (B, K)
    | Linear decoder                      → (B, P)
```

Parameter count for P=113, K=9, N=2: 1,319 (as compared with 227,200
for a 1-layer transformer with d_model=128, 4 heads, 4·d_model MLP).

### 2.3 Training

AdamW optimizer, learning rate 1e-3, weight_decay 0.01. A diversity
regularizer on the mixing-layer output prevents redundant channels:

    L_total = CE(logits, y) + λ · ||G G^T - diag(G G^T)||_F^2

where G is the channel activation matrix. We use λ = 0.01 throughout.
We note that an earlier version of our codebase inadvertently
computed this penalty without full autograd connection to the
encoder frequencies; the version used for all results in this paper
has been corrected and regression-tested.

---

## 3. Experiments

All results are single-seed unless stated otherwise. All CSVs,
checkpoints, and run metadata are published in the `pan_lab`
artifact. Training uses batch size 256, a 40%/60% train/val split,
and `val_acc ≥ 0.99` as the grokking threshold.

### 3.1 Comparison with transformer baseline (mod-113)

Single-seed comparison at P=113 with K=9 for PAN, d_model=128 for the
transformer, seed=42, 50K-step budget with early stopping:

| Metric | PAN (K=9) | Transformer | Ratio |
|--------|-----------|-------------|-------|
| Parameters    | 1,319 | 227,200 | 172× fewer |
| Grok step     | 39,400 | 7,200   | 5.5× more   |
| Wall-clock    | 49s   | 15s     | 3.3× slower |
| Val accuracy  | 99.2% | 99.3%   | comparable   |

PAN reaches comparable final accuracy in more steps but with a
fraction of the parameter budget. Wall-clock is slower due to
un-optimized implementation on MPS; a vectorized phase-modulo-add on
SPF hardware would invert this ranking.

**Ablations (Tier 3 model, seed=42, 100K steps):**

| Intervention            | Val accuracy | Drop     |
|-------------------------|--------------|----------|
| Baseline                | 99.2%        | —        |
| Zero phase mixing       | 0.9%         | −98.3%   |
| Randomize frequencies   | 1.0%         | −98.3%   |
| Zero reference phases   | 2.9%         | −96.3%   |

No component is redundant. Any single-component ablation collapses
accuracy to chance.

### 3.2 Mixing-layer structure at a single grokked seed (P=113, K=9)

We save the trained model at seed=42 after 100K steps (grok at step
46K) and inspect the 9×18 phase-mixing weight matrix. For each of the
9 output channels, we identify the top two contributing weights by
absolute value.

**Finding:** All 9 output channels exhibit a Clock-pair structure —
the top-2 weights come from different encoders (one from enc0, one
from enc1), have matched magnitudes near 1.0 (mean ±0.05), and point
to slots with similar learned frequencies.

Example rows:
```
out[0]:  enc0.slot1 (w=+0.985)  enc1.slot1 (w=+1.000)   ← k=1
out[3]:  enc0.slot8 (w=+0.999)  enc1.slot9 (w=+0.999)   ← k=9
out[5]:  enc0.slot2 (w=+0.993)  enc1.slot2 (w=+0.969)   ← k=3
```

5 of 9 encoder slots in each encoder converge to within 0.01 rad of
a theoretical Fourier basis vector f_k = k·2π/P. The distinct active
frequencies in this seed are {k=1, k=3, k=4, k=9}.

**Circuit formation is post-grokking.** Generalization (val_acc ≥ 99%)
occurs at step 46,000. Encoder frequencies drift further under weight
decay until stabilizing at SIFP-16 quantization precision around step
80,000. This is consistent with Varma et al.'s (2023) circuit-efficiency
prediction: grokking marks when the circuit becomes functionally
correct; weight decay continues to clean it afterward.

### 3.3 Slot census across 20 seeds

To establish that the Clock-pair structure in §3.2 is seed-robust
rather than a one-off finding, we train 20 PANs at K=9 with identical
hyperparameters and seeds 0–19, saving each trained model.

**Grok rate: 16/20 (80%).** Failed seeds split into two modes:
catastrophic failure (2 seeds at peak acc < 50%, likely from
degenerate mixing solutions) and near-grok plateaus (2 seeds at peak
acc 92–98%, likely would grok with more training budget).

**Clock-pair compliance (grokked seeds only).** Using the
top-2-from-different-encoders criterion with magnitude tolerance
±20%, we measure the fraction of mixing-matrix rows that form Clock
pairs. Mean across 16 grokked seeds: **0.82** (min 0.56, max 1.00).
5 of 16 seeds achieve 1.00 compliance (every row is a clean Clock
pair). Mean distinct active Fourier frequencies per circuit: 4.62
(range 3–8).

Frequency preference across seeds:

| Theoretical k | Fraction of seeds |
|---|---|
| k=7 | 0.69 |
| k=5, 6, 8, 9 | 0.44 each |
| k=3 | 0.38 |
| k=1, 4 | 0.31 each |
| k=14 | 0.19 |
| k ∈ {23, 25, 27, 29, 33, 47, 48} | < 0.10 each |

**Note on the Nanda comparison.** Nanda et al.'s transformer on
this task converged on frequencies {14, 34, 41, 42, 52}. Our PANs
prefer different, generally lower frequencies. Some of our seeds
do converge on high-k modes (33, 47, 48), which are legitimate
basis vectors of ℤ_113 but not the "canonical low-k" choices. We
observe that **PAN and the transformer converge on the same task
(modular addition via sparse Fourier representation) and on
structurally similar mixing patterns (Clock-shaped pairs at matched
frequencies), but on different specific frequency subsets.**

### 3.4 Initialization insensitivity

We ablate the encoder frequency initialization at K=9, comparing
Fourier initialization (f_k = k·2π/P) to uniform random on [0, 2π).

| Initialization  | Grok rate (20 seeds) | Mean grok step (grokked only) | Mean peak val_acc |
|-----------------|---------------------|-------------------------------|-------------------|
| Fourier basis   | 16/20 = 80%         | ≈20,400                       | 0.993             |
| Uniform random  | 15/20 = 75%         | ≈30,000                       | 0.992             |

The two initializations are statistically comparable. A 5-seed pilot
initially suggested random was strictly better (5/5 vs 4/5, 2.9×
faster grokking); this did not survive 20-seed replication. What does
survive: **PAN's inductive bias does not require an analytically
correct frequency initialization.** A random starting configuration
converges to a Clock-pair circuit at comparable rates.

This matters for the architectural-bias claim. Had Fourier
initialization been necessary, one could argue PAN was simply a
carefully-initialized transformer approximation. With random init
succeeding, the architectural primitive (phase arithmetic plus
cosine gating) carries the inductive load.

### 3.5 Cross-prime generalization

We evaluate K=9 PAN on all primes in our test set, with a 200K-step
training budget (P=97 re-run at 500K to verify):

| P   | Grok step | Peak val_acc | Grokked? | Params | Notes |
|-----|-----------|--------------|----------|--------|-------|
| 43  | 12,000    | 99.2%        | ✓        | 619    | development |
| 59  | 4,500     | 99.1%        | ✓        | 779    | held-out |
| 67  | 11,800    | 99.1%        | ✓        | 859    | development |
| 71  | 22,000    | 99.2%        | ✓        | 899    | held-out |
| 89  | 139,800   | 99.1%        | ✓        | 1,079  | development |
| 97  | —         | 85.0%        | ✗        | 1,159  | held-out, 500K steps |
| 113 | 46,000    | 99.5%        | ✓        | 1,319  | §3.2 seed |
| 127 | 23,400    | 99.2%        | ✓        | 1,459  | development |

**7/8 primes grok cleanly at K=9.** P=97 is an outlier: it plateaus
at approximately 85% even at 500K steps. Since other primes grok
within 20K–140K steps, the 500K budget is not the limiting factor.
P=97 appears to be a capacity wall at K=9 rather than an
optimization wall.

We do not have a full explanation for why P=97 specifically
resists K=9 where P=89 and P=113 succeed. One hypothesis: the
units group ℤ/97ℤ× has order 96 = 2⁵·3, while ℤ/113ℤ× has order
112 = 2⁴·7. Different prime factorizations of the group order
may affect which sparse Fourier subsets suffice to represent
addition. A K=12 or K=15 run on P=97 would test whether this is
a capacity wall. We leave this to future work.

### 3.6 Gate representation is linearly sufficient

We examine what information the PAN gate output contains by fitting
a multinomial logistic regression decoder directly to the gate
activations across all P² input pairs, for three grokked seeds at
P=113, K=9:

| Seed | PAN's trained decoder | Optimal linear decoder on gates | Gap |
|------|----------------------|----------------------------------|-----|
| 42   | 99.2% | 99.9%  | −0.7%  |
| 123  | 99.1% | 100.0% | −0.9%  |
| 789  | 99.3% | 100.0% | −0.7%  |

PAN's trained decoder is near-optimal given the gate representation.
The gate contains all the discriminative information needed to solve
the task linearly. This closes what could have been an open question
about whether the architecture's decoder is a bottleneck.

### 3.7 SIFP-16 phase quantization at inference

To evaluate whether PAN's circuit tolerates the phase precision of
SPF-32 hardware (16-bit phase field, quantization error 2π/65536
≈ 9.6×10⁻⁵ rad), we quantize every phase output to 16-bit precision
at inference and re-evaluate:

| Seed | fp32 val_acc | SIFP-16 val_acc | Δ      |
|------|--------------|-----------------|--------|
| 42   | 99.23%       | 99.23%          | 0.000  |
| 123  | 99.14%       | 99.14%          | 0.000  |
| 456  | 94.36%       | 94.06%          | −0.003 |

**16-bit phase quantization is effectively free.** The seed-456 run
had not fully completed post-grok cleanup (val_acc 94.4% at
termination); even this undertrained model loses only 0.3% accuracy
at SIFP-16. Fully grokked runs show zero quantization loss to four
decimal places.

This is the empirical bridge to SPF. Phase quantization does not
degrade the circuit PAN learns.

### 3.8 Parameter-efficiency reliability (K-sweep summary)

Reliability as a function of K, at P=113, in aggregate across the
K-sweep and K=8 sweep experiments:

| K  | Grok rate | Params | Notes |
|----|-----------|--------|-------|
| 4  | 0/n       | 744    | Too few slots for circuit |
| 5  | partial   | 888    | Borderline |
| 7  | partial   | 1,174  | Borderline |
| 8  | 6/10      | 1,169  | 60% — below reliability plateau |
| 9  | 16/20     | 1,319  | 80% — our chosen default |
| 12 | — (N/A)   | 1,751  | Tested on development primes, ≈100% reliable |

K=9 sits on the edge of the reliability plateau. The 20% failure rate
at K=9 is a real property of the architecture, not a measurement
artifact. Failures split between catastrophic (mixing-matrix
degeneracy) and near-grok plateaus.

---

## 4. Open questions

### 4.1 Full mechanistic characterization of the gate-plus-decoder stack

We establish in §3.3 that the mixing layer produces Clock-pair
structure with high compliance, and in §3.6 that the gate output is
linearly sufficient for the decoder. We do **not** claim full
mechanistic equivalence to the transformer's Clock algorithm. Doing
so would require characterizing the exact Fourier content of the
gate output — specifically, which cos/sin combinations at which
effective channel frequencies the decoder is reading. Our preliminary
tooling (`decoder_analysis` in `pan_lab`) projects the learned
decoder onto a Clock basis built from effective channel frequencies
and gate reference phases; the projection recovers 66–87% of decoder
energy but only 7–64% of classification accuracy across seeds. We
interpret this gap as most likely a limitation of our
effective-frequency extractor on mismatched-slot mixing rows rather
than evidence of non-Clock circuitry, but this is not verified. A
stronger decomposition — projecting directly onto the span of gate
activations induced by the mixing matrix, without the intermediate
"effective frequency" abstraction — is the natural next step.

### 4.2 The P=97 capacity anomaly

K=9 suffices for 7 of the 8 primes we tested but not for P=97. Is
this (a) a property of ℤ/97ℤ's multiplicative structure requiring a
larger sparse Fourier basis, (b) a seed-specific failure that
additional seeds would resolve, or (c) a different optimization
landscape specific to this prime? A K-sweep on P=97 (K ∈ {9, 12,
15, 18}) at 3 seeds each would be definitive. If K=12 grokks, (a)
is likely. If K=9 grokks at some seeds but not seed=42, (b) is
likely.

### 4.3 Extension beyond modular addition

The circuit characterized here solves modular addition through
Clock-pair composition of phases. Modular multiplication (a·b mod P)
cannot factor through this same primitive — multiplication in the
group is logarithmic in the phase, not additive. Whether PAN fails
cleanly on multiplication (confirming the scope of the phase-addition
inductive bias) or finds a novel circuit is an empirical question we
have not addressed. Two-step composition ((a+b)·c mod P) raises
similar questions.

### 4.4 Beyond arithmetic

Tier 5 of our research plan is a small-scale language modeling
probe: replace a single MLP block in a small transformer with a
PAN-style phase block and measure downstream perplexity. We have
not attempted this. The motivation is that algorithmic tasks are
a fundamentally different test bed from natural language; if PAN
is competitive on arithmetic but not on language, the architecture's
scope is firmly established.

---

## 5. Conclusion

Phase Accumulator Networks solve modular arithmetic with 127–305×
fewer parameters than a one-layer transformer baseline. Across 20
independent seeds at P=113, K=9, the architecture grokks 80% of the
time and produces Clock-shaped mixing-matrix structure in every
grokked run (mean Clock-pair compliance 0.82). The gate representation
is linearly sufficient, with PAN's trained decoder within 1% of the
optimal linear decoder on gate activations. 16-bit phase quantization
at inference preserves accuracy to within 0.3%, and typically to
zero measurable loss.

Whether PAN's full circuit — mixing layer, gates, and decoder taken
together — is algorithmically equivalent to the transformer's Clock
algorithm in the strong Nanda et al. sense remains an open question.
What we can confirm is structural equivalence of the mixing pattern
and functional equivalence of the task.

The analogy we draw is to Quake III's inverse-square-root trick, which
did not compute sqrt faster but made sqrt unnecessary by exploiting
the IEEE 754 bit layout. Similarly, PAN does not make the transformer's
Fourier circuit faster; it makes that circuit the architecture itself,
reducing learning to selecting which frequencies and how to pair them.
This is not a compression trick but an inductive-bias replacement, and
its target hardware — SPF — is a format where phase addition is the
primitive operation in silicon.

---

## Acknowledgments

This work was assisted by conversational AI as a research partner. All
experiments were run by the author; all code, math, and claims were
verified and revised iteratively. The `pan_lab` codebase and all data
supporting this paper are released alongside it.

## References

[Full reference list in companion bibliography.]

---

*April 2026 · Code and data: `pan_lab/` · v3 final*
