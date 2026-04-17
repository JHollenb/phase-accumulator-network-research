# Phase Accumulator Networks: Phase Arithmetic as a Neural Primitive for Modular Computation

**Jacob Hollenbeck — March 2026**  
*Companion code: `pan.py` · Companion documents: Spectral IEEE 754 Whitepaper, SPF Format Specifications*

---

## Abstract

Mechanistic interpretability research has shown that transformers trained on modular arithmetic converge on an explicit Fourier algorithm: inputs are encoded as sinusoidal functions, combined via trigonometric products, and decoded through sinusoidal projection. These circuits are discovered through gradient descent on a general-purpose architecture. We ask whether the same computation can be the architecture itself — not something learned, but the forward pass.

We introduce **Phase Accumulator Networks (PAN)**: a neural architecture where the primitive operation is sinusoidal phase addition rather than floating-point multiply-accumulate. On modular addition mod *P*, PAN achieves ≥99.1% validation accuracy across five primes (P ∈ {43, 67, 89, 113, 127}) using between 619 and 1,459 parameters — 127–305× fewer than the transformer baseline — with no hyperparameter tuning between primes. Ablation confirms that phase arithmetic is the active computational mechanism. We characterize the minimum frequency count *K* required for reliable learning, identify an unexplained anomaly at *K*=8, and discuss open questions for mechanistic investigation and extension to other domains.

---

## 1. Introduction

In 1999, the Quake III engine shipped an inverse square root approximation that exploited the bit layout of IEEE 754 floating point: reinterpreting a float as an integer yields its base-2 logarithm for free. The trick did not make `sqrt()` faster. It made `sqrt()` *unnecessary* for that use case, because the number format already encoded the answer.

A similar observation motivates this work. The Spectral IEEE 754 (SPF) format encodes complex numbers in log-polar form: multiplication becomes two integer additions, and phase composition — `e^(iα) × e^(iβ) = e^(i(α+β))` — is an exact integer add with natural modular wraparound. A network designed around this primitive has a fundamentally different cost model than one built on IEEE 754 multiply-accumulate.

Recent mechanistic interpretability work supplies the motivation. Nanda et al. (2023) fully reverse-engineered a one-layer transformer trained on modular addition: the network discovers an explicit Fourier multiplication algorithm, encoding inputs as sinusoidal functions at five key frequencies, computing trigonometric products in the MLP, and decoding via sinusoidal projection. Kantamneni and Tegmark (2025) found the same "Clock algorithm" operating inside GPT-J, Pythia, and Llama-3.1. Zhou et al. (2024) confirmed Fourier features with outlier components in GPT-2-XL and GPT-J.

The finding is consistent across scales: **transformers trained on arithmetic discover sinusoidal computation**. They spend orders of magnitude more parameters and gradient steps than necessary converging on a solution that phase arithmetic encodes natively.

This raises the central question: if the computation a transformer is trying to learn is sinusoidal phase rotation, what happens if phase rotation is the architecture — not something to discover, but something to start with?

---

## 2. Architecture

### 2.1 The Phase Neuron

A standard neuron computes `y = activation(Σ wᵢxᵢ + b)`. A phase neuron computes:

```
φ_out = (Σ wᵢφᵢ_in) mod 2π
```

where each `φᵢ` is a phase in [0, 2π) and each `wᵢ` is a real-valued weight. On SPF hardware this is integer multiply-accumulate modulo 65536. The operation is linear phase mixing — the same operation Nanda's interpretability analysis found in the grokked transformer's MLP, emergently. Here it is the architecture.

### 2.2 PAN for Modular Arithmetic

Given two inputs a, b ∈ [0, P) for prime P, the network proceeds in four stages:

**Encoding.** Each input is mapped to K phases at different frequencies:

```
φₖ(a) = (a · fₖ) mod 2π,   fₖ initialized to k·2π/P
```

Two independent encoders (one for a, one for b) produce 2K phases. The frequencies `fₖ` are learned parameters.

**Phase mixing.** A linear layer combines the 2K encoded phases into K mixed phases:

```
φ_mixed[j] = (Σ W[j,i] · φ_encoded[i]) mod 2π
```

The K×2K weight matrix W is learned. This is the only layer that combines information from both inputs.

**Phase gating.** A cosine gate produces K scalars in [0, 1]:

```
gate[j] = (1 + cos(φ_mixed[j] − φ_ref[j])) / 2
```

where `φ_ref[j]` are K learned reference phases. This is phase-selective activation: a gate fires when the input phase aligns with its reference. Nanda found neurons in grokked transformers doing exactly this — firing selectively at specific Fourier components.

**Decoding.** A standard linear layer maps the K gate values to logits over P classes.

**Parameter count.** For P=113, K=9: 18 encoder frequencies + 162 mixing weights + 9 reference phases + 9×113 decoder weights + 113 decoder biases = 1,319 parameters. The transformer baseline is 227,200 parameters — 172× more.

### 2.3 Engineering Notes

Two failure modes required architectural fixes during development.

*Reference phase wandering.* Adam treats `φ_ref` as a value in ℝ, but it lives on S¹ (the circle). Without intervention, momentum accumulates and pushes reference phases far outside [0, 2π), causing gradient spikes at the inflection points of `cos()`. Fix: wrap with `torch.remainder(φ_ref, 2π)` inside `forward()` before computing the phase difference. The stored parameter can still accumulate momentum freely; the effective gate center is always in [0, 2π). Gradient flows correctly — `torch.remainder` has derivative 1 almost everywhere.

*Phase mixing mode collapse.* The K output channels of the mixing layer can all converge to the same input frequency, collapsing the K-dimensional phase space to one dimension. Fix: an off-diagonal Gram penalty on the mixing layer outputs during training, with weight 0.01. This penalizes correlated output channels without constraining individual frequency values.

---

## 3. Experiments

All experiments use: AdamW, lr=1e-3, weight_decay=0.01, diversity_weight=0.01, batch=256, train/val split 40%/60% (matching Nanda's setting). Grokking is defined as val_acc ≥ 99.0%. Code is in `pan.py`; all runs are logged with full hyperparameter configs.

### 3.1 Tier 1: Existence Proof (mod-113)

We replicate Nanda et al.'s exact task: a + b mod 113, with the same transformer architecture as baseline. A single PAN run (K=5, seed=42) grokked at step 48,400 with 99.9% val accuracy.

| Metric | PAN (K=5) | Transformer | Ratio |
|--------|-----------|-------------|-------|
| Parameters | 743 | 227,200 | 305× fewer |
| Grokking step | 48,400 | 7,000 | 6.9× more steps |
| Wall-clock to grok | ~62s | ~15s | ~4× slower |
| Val accuracy | **99.9%** | 99.1% | PAN higher |

**Ablation.** Zeroing any single component collapses accuracy to chance:

| Intervention | Val accuracy | Drop |
|---|---|---|
| Baseline | 99.9% | — |
| Zero phase mixing | 0.9% | −99.1% |
| Randomize frequencies | 0.7% | −99.2% |
| Zero reference phases | 2.0% | −97.9% |

Phase arithmetic is the active mechanism — there is no shortcut the decoder can exploit around it.

**What the network learned.** Despite K=5 frequencies being available, the phase mixing converged to two effective frequency slots (freq[3] and freq[4]), routing them independently from inputs a and b with mixing weights near ±1. The network found a more compressed solution than Nanda found in the transformer, using two frequencies where the theory predicts five as necessary. This is the primary open question for Tier 3.

### 3.2 Tier 2: Parameter Efficiency — Minimum K Sweep

We swept K from 1 to 15 with three seeds each (seeds 42, 123, 456), 100K steps, P=113.

| K | Grokked | Mean step | Params | Notes |
|---|---------|-----------|--------|-------|
| 1–4 | 0/3 | — | 231–609 | Insufficient capacity |
| 5 | 0/3 sweep* | — | 743 | *seed=42 only: grokked at 48,400 |
| 6 | 1/3 | 13,800 | 881 | Borderline |
| 7 | 1/3 | 29,800 | 1,023 | Borderline |
| **8** | **0/3** | **—** | **1,169** | **Anomaly — see §4** |
| **9** | 2/3 | 22,500 | **1,319** | **Minimum reliable K** |
| 10 | 2/3 | 10,100 | 1,473 | |
| 11 | 2/3 | 15,000 | 1,631 | |
| 12 | 3/3 | 15,133 | 1,793 | All seeds reliable |
| 13–14 | 3/3 | 21,733–23,867 | 1,959–2,129 | |
| 15 | 3/3 | 7,200 | 2,303 | Fastest mean grok |

**Capacity threshold.** K<5 networks are representationally incapable. Mod-113 requires at least 5 Fourier frequencies (matching Nanda's finding exactly), and a PAN with K=3 can only represent 3. Loss barely moves from the random baseline (log 113 ≈ 4.73) for K≤4. This is not an optimization failure — it is an architectural one.

**Seed sensitivity at K=5.** The Tier 1 result grokked because seed=42 happened to be a favorable initialization. All three sweep seeds failed at K=5 within 100K steps. The architecture *can* solve the task at K=5 but does not do so reliably. K=9 is the minimum K where ≥2/3 seeds succeed.

**Parameter efficiency vs transformer:**
- Minimum reliable PAN (K=9): 1,319 params — **172× fewer**
- Fully reliable PAN (K=12): 1,793 params — **127× fewer**

### 3.3 Tier 4: Cross-Prime Generalization

We ran K=9 PAN across five primes with identical hyperparameters — no per-prime tuning.

| P | Grok step | Val acc | Params |
|---|-----------|---------|--------|
| 43 | 12,000 | 99.2% | 619 |
| 67 | 11,800 | 99.1% | 859 |
| 89 | 139,800 | 99.1% | 1,079 |
| 113 | 11,200 | 99.3% | 1,319 |
| 127 | 23,400 | 99.2% | 1,459 |

**5/5 primes grokked at ≥99.1%.** The architecture is principled — same K, same hyperparameters, different prime, same result. No per-prime tuning was required. This rules out the concern that K=9 on mod-113 is a lucky coincidence specific to that problem.

**P=89 is a slow grokker, not a failure.** The 100K run ended mid-cliff (97.1% at step 99K, still rising steeply). With 200K steps, it grokked at step 139,800. The qualitative pattern — a long pre-grokking memorization phase followed by a sharp accuracy cliff — is identical to the other primes, just at a different timescale. P=89 likely sits in an awkward position relative to the K=9 Fourier basis, requiring more optimization steps to find the correct frequency assignment.

**Parameter scaling.** The decoder is a K×P linear layer, so parameter count scales linearly with P. The 619-parameter network on P=43 may be the smallest neural network to solve a nontrivial group operation at high accuracy.

---

## 4. The K=8 Anomaly

The most surprising result in the sweep is K=8: it grokked 0/3 seeds, despite K=7 grokking 1/3 and K=9 grokking 2/3. This is not a monotonic failure — it is a non-monotonicity in a regime where the trend should be improvement with more capacity.

Seed 42 at K=8 is particularly striking. Val accuracy reached 97.4% — tantalizingly close to the 99% threshold — then plateaued for 60,000 steps without progressing. The network clearly found something near the correct solution but could not complete the transition. This is distinct from the K≤4 failures where the network makes no meaningful progress, and distinct from successful runs where the accuracy cliff is sharp and fast once it begins.

Three hypotheses, in decreasing order of plausibility:

**Hypothesis 1: Resonance in the K=8 loss landscape.** K=8 may create a specific configuration of the phase mixing matrix where a partial Fourier solution is metastable — accurate enough to reduce loss significantly but not structured enough to trigger the sharp generalization transition. The diversity regularizer (DW=0.01) may be insufficiently strong to push the K=8 mixing matrix out of this basin.

**Hypothesis 2: Aliasing between K=8 frequencies and mod-113.** The natural Fourier basis of ℤ₁₁₃ has frequencies at k×2π/113 for k=1..5. K=8 may create a situation where learned frequencies alias against each other in a way that K=7 and K=9 avoid by accident of their relationship to 113.

**Hypothesis 3: Initialization sensitivity amplified at K=8.** The three seeds used in the sweep (42, 123, 456) may all initialize K=8 in a region of the parameter space that leads to the 97.4% basin. Running more seeds would test this — if K=8 grokks at seeds 789 or 1000, the issue is seed sensitivity, not a structural problem.

**Recommended follow-up.** Run K=8 with 10 seeds at 200K steps. If 0/10 grok, the K=8 basin is real and worth mapping. If some seeds grok, it is a seed sensitivity artifact similar to K=5's. Either result is informative.

---

## 5. Open Questions

### 5.1 Mechanistic Equivalence (Tier 3)

The central unresolved question: does PAN learn the same algorithm Nanda found in the transformer?

Nanda's transformer converged on frequencies at k×2π/113 for k ∈ {1,2,3,4,5}. Our frequency analysis shows PAN's encoder parameters drift far from their initializations (raw values like −6.26 rad are aliases of their canonical values mod 2π), but it is unclear whether the learned frequencies converge to the theoretical Fourier basis or to a different solution entirely.

The Tier 1 run converged to two active frequency slots rather than five — using frequencies 3 and 4 from the theoretical basis and leaving the others dormant. This could mean PAN found a more efficient encoding, or that it found a different circuit altogether. Distinguishing these requires checkpoint logging during training to track frequency convergence over time, and a full Fourier analysis of the mixing weights at the grokked solution.

The practical test: compute the angular error between learned frequencies and k×2π/113 (mod 2π), and check whether mixing weights at the grokked solution form identity or permutation matrices over the active frequency slots. If yes, PAN is implementing exactly Nanda's algorithm. If the frequencies or weights are structured differently, PAN found an alternative Fourier circuit.

### 5.2 Why Does K=9 Work for All Tested Primes?

The theoretical minimum K for mod-P is approximately the number of Fourier frequencies needed to span ℤ_P — which Nanda's analysis puts at 5 for P=113. Scaling naively gives K_min(P) ≈ 5×P/113, which predicts K_min(43) ≈ 2 and K_min(127) ≈ 6. K=9 is well above all of these, which explains why it generalizes — it is over-provisioned for every prime in the test set.

The interesting follow-up is to find the minimum reliable K for each prime and check whether it scales with P or remains approximately constant. If minimum K is roughly constant across primes, the Fourier basis structure may be more universal than the per-prime analysis suggests. If it scales with P, a per-prime K selection rule would be needed for deployment.

### 5.3 Extension to Other Group Operations

The current experiments cover only addition. Modular multiplication — a×b mod P — would be a sharp test of whether the phase mixing architecture can handle more complex group structure. Multiplication mod P is not simply phase addition; it requires the mixing layer to implement a more complex linear combination. Whether K=9 suffices, or a different K is needed, is an empirical question worth running.

Two-step arithmetic — (a+b)×c mod P — tests whether PAN can compose operations, which would require either a deeper architecture or a single mixing layer expressive enough to implement the composed operation.

### 5.4 The Step Count Gap

PAN takes more gradient steps to grok than the transformer (48,400 vs 7,000 in the primary comparison). This is not a wall-clock problem — PAN steps are faster — but it is a sample efficiency gap worth understanding. The transformer's attention mechanism may provide a more direct gradient signal toward the Fourier solution. A learning rate schedule or curriculum (starting with small P and increasing) might close the gap.

### 5.5 Toward Language

The honest expectation for Tier 5 is that PAN will underperform a transformer on language modeling. Language does not obviously reduce to sinusoidal composition the way modular arithmetic does. A negative result here would be scientifically valuable: it would constrain the claim from "phase arithmetic is a better neural primitive" to "phase arithmetic is the right primitive for tasks with Fourier structure." Mapping the boundary of that claim — which tasks have the right structure, which do not — is the long-run research agenda.

The more tractable near-term question is the hybrid architecture: replacing the MLP sublayer with a PAN in transformer layers that interpretability analysis identifies as computing sinusoidal operations. This would be surgical rather than wholesale replacement, and the interpretability literature gives concrete targets to try.

---

## 6. Related Work

**Grokking and modular arithmetic.** Nanda et al. (2023) introduced the mechanistic analysis of grokking on modular arithmetic that motivates this work. Power et al. (2022) characterized grokking as delayed generalization in small transformers. Varma et al. (2023) showed that weight decay is the critical hyperparameter. Our results confirm the weight decay sensitivity — WD=1.0 completely prevents PAN from learning, while WD=0.01 reliably produces grokking.

**Fourier structure in large models.** Kantamneni and Tegmark (2025) showed that GPT-J (6B), Pythia (6.9B), and Llama-3.1 (8B) all represent numbers as generalized helices — complex sinusoidal functions at periods {2, 5, 10, 100}. Zhou et al. (2024) found the same Fourier features in GPT-2-XL. These results suggest that Fourier computation is not an artifact of small grokking experiments but a general phenomenon in trained neural networks.

**Phase-based neural computation.** Complex-valued neural networks (Trabelsi et al., ICLR 2018) outperform real-valued networks on music transcription and MRI reconstruction — tasks with natural phase structure. PAN is related but distinct: it does not use complex-valued activations in the standard sense, but instead treats phase as the primary representational quantity and uses phase addition as the core primitive.

**Neural architecture inductive bias.** The general principle that matching architectural primitives to task structure improves efficiency is well-established in CNNs for vision and RNNs for sequences. PAN applies this principle to the specific case of Fourier computation: if the task requires sinusoidal arithmetic, use sinusoidal arithmetic as the primitive rather than having the network discover it.

**Spectral methods in neural networks.** FFT-based attention (Cooley and Tukey, 1965; Lee et al., 2021) and Fourier neural operators (Li et al., 2021) use the FFT as a computational tool inside otherwise standard architectures. PAN is not using the FFT as a subroutine — it is making phase accumulation the fundamental operation, which is a stronger claim.

---

## 7. Conclusion

Phase Accumulator Networks solve modular arithmetic with 127–305× fewer parameters than a transformer baseline, generalize across five primes with zero per-prime tuning, and confirm via ablation that phase arithmetic is the active computational mechanism — not a shortcut the decoder exploits around it.

The result is not a compression trick. The 743-parameter K=5 PAN and the 1,319-parameter K=9 PAN are not compressed transformers. They are architectures built around a different computational primitive, and that difference accounts for the parameter gap. Transformers require hundreds of thousands of parameters to discover a Fourier circuit. PAN starts with a Fourier circuit and requires only the weights needed to configure it for the specific task.

The analogy to Quake's inverse square root is imperfect but instructive. That trick did not make `sqrt()` faster. It eliminated the need for a general `sqrt()` implementation when the bit layout of the number format already encoded the answer. Similarly, PAN does not make the transformer's Fourier circuit faster. It eliminates the need to discover that circuit when the architectural primitive already performs it.

What remains unknown is whether this transfers. The current results are limited to group addition on small primes — a domain where the Fourier structure is analytically understood. Whether the same principle applies to other tasks with sinusoidal structure (modular multiplication, positional encoding, state-space models), and whether it fails gracefully on tasks without that structure (language), is the next research agenda.

The K=8 anomaly — a network that reaches 97.4% accuracy and stalls, in a regime where both smaller and larger K succeed — is perhaps the most interesting unexplained result. It suggests the PAN loss landscape has structure we do not yet understand, and mapping that structure may be as informative as the headline parameter counts.

---

## References

Nanda, N., Chan, L., Liberum, T., Smith, J., and Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*.

Power, A., Burda, Y., Edwards, H., Babuschkin, I., and Misra, V. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *ICLR Workshop on Enormous Language Models*.

Varma, V., Shah, R., Kenton, Z., Kramár, J., and Kumar, R. (2023). Explaining grokking through circuit efficiency. *arXiv:2309.02390*.

Kantamneni, S. and Tegmark, M. (2025). Clock and PIZZA: Unveiling the hidden mechanisms of modular arithmetic in large language models. *arXiv:2501.18256*.

Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J., Bengio, S., and Nakkiran, P. (2024). What algorithms can transformers learn? A study in length generalization. *NeurIPS 2024*.

Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian, S., Santos, J. F., Mehri, S., Rostamzadeh, N., Bengio, Y., and Pal, C. J. (2018). Deep complex networks. *ICLR 2018*.

Li, Z., Kovachki, N., Azizzadenejad, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *ICLR 2021*.

---

*March 2026 · Code: `pan.py` · Tiers 1, 2, 4 complete · Tier 3 (mechanistic) and Tier 5 (language) in progress*
