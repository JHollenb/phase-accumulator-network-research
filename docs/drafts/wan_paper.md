# Walsh Accumulator Networks: Character Arithmetic on 𝔽₂ⁿ as a Neural Primitive

**Companion draft to the PAN paper — April 2026**
*Code: `pan_lab/models/wan.py`, `pan_lab/data.py`, `experiments/wan_*.yaml`*

---

## Abstract

**Phase Accumulator Networks (PAN)** use sinusoidal phase addition — the
group operation of ℤ/Pℤ carried through the Fourier character map
x ↦ exp(i·2πkx/P) — as their primitive. We ask the obvious
follow-up: if the group operation of ℤ/Pℤ is the right primitive for
modular arithmetic tasks, what is the right primitive for Boolean
tasks? The answer is the group operation of 𝔽₂ⁿ, carried through the
Walsh–Hadamard character map x ↦ (−1)^(s·x).

We introduce **Walsh Accumulator Networks (WAN)**: architecturally a
direct translation of PAN with phases φ ∈ ℝ/2πℤ replaced by Walsh
"phases" v ∈ ℝ/2ℤ, and the complex exponential cos(φ) replaced by the
real cosine cos(π·v). Every stage of PAN has a one-to-one WAN analog.
The stated goal of this draft is twofold: (1) spell out the
architectural translation so it is unambiguous, and (2) predict, in
advance of running the experiments, what the grokking / mechanistic
signatures should look like if the PAN claim generalizes to the Walsh
group. The experiment YAMLs at `experiments/wan_*.yaml` run exactly the
tests described below.

The target claim this paper aims to establish: *character arithmetic on
the task's underlying group is the right neural primitive*. PAN was
the G = ℤ/Pℤ instance. WAN is the G = 𝔽₂ⁿ instance. If both hold, the
next step is G = ℤ/2ⁿℤ (rotations / carries), which ties directly to
the SPF-PA format work. The path to that general claim starts here.

---

## 1. The Primitive

### 1.1 Characters of 𝔽₂ⁿ

For the additive group ℤ/Pℤ, the irreducible characters are

  χ_k(x) = e^(2πi·k·x/P),    k, x ∈ ℤ/Pℤ.

They are orthonormal in ⟨·,·⟩_{ℤ/P}, and multiplication of characters
encodes addition in the group:

  χ_k(x) · χ_k(y) = χ_k(x + y).

For 𝔽₂ⁿ (the group of n-bit strings under XOR), the irreducible
characters are real and take values in {−1, +1}:

  χ_s(x) = (−1)^(s·x),    s, x ∈ 𝔽₂ⁿ,

where s·x = ⊕ᵢ(sᵢ ∧ xᵢ) is the XOR-parity of the bitwise AND. The
identities parallel the Fourier case exactly:

  χ_s(x) · χ_s(y) = χ_s(x ⊕ y)      (character multiplication = group op)
  χ_s(x) · χ_t(x) = χ_{s⊕t}(x)      (character × character = sum of indices)

The Walsh–Hadamard transform is the inner-product expansion of a
function on 𝔽₂ⁿ in this basis. Walsh-sparse targets — those with
support on a small number of characters χ_s — are the Walsh analog of
Fourier-sparse targets on ℤ/Pℤ.

### 1.2 Real vs complex characters

The only structural difference from the Fourier case is that Walsh
characters are real, so there is no separate "sin" partner. A single
channel j already expresses both amplitude and phase through its sign
in {−1, +1}. This simplifies bookkeeping: where PAN's forward pass
computes cos(φ) and implicitly uses the two-dimensional structure of
S¹, WAN's forward pass is scalar.

### 1.3 What does "phase addition" become?

Let l_i = (1 − χ_{e_i}(x))/2 ∈ {0, 1} be the bit-valued character of x
under the i-th standard mask (so l_i is just x_i). Consider a learned
real-weighted combination of these bits:

  v = Σᵢ Wᵢ · lᵢ ∈ ℝ.

Then cos(π·v) takes values in {−1, +1} whenever v is an integer, and
interpolates smoothly between them otherwise. It is the analog of cos(φ)
in PAN, with period 2 (instead of 2π) and discrete target integers
(instead of multiples of 2π/P). The whole WAN forward pass can be
stated in these terms — see §2.

---

## 2. Architecture

Given an input x ∈ 𝔽₂ⁿ (an n-bit integer), WAN proceeds in four stages
that mirror PAN exactly.

    PAN stage (from the v5 paper)       WAN analog
    ────────────────────────────────    ─────────────────────────────────
    encode  φ_k(a) = k·2π·a / P         v_k(x) = Σᵢ m[k,i] · x_i
            mod 2π                      mod 2   (implicit; cos·π is 2-per)
    mix     ψ = W·φ  mod 2π             u = W·v  mod 2
    gate    g = (1+cos(ψ−ψ_ref))/2      g = (1+cos(π·(u−u_ref)))/2
    decode  linear K → P                linear K → C

`pan_lab/models/wan.py` implements this one-for-one. `WalshEncoder`
holds learned logits α_{k,i}; the effective mask is m_{k,i} = σ(α_{k,i})
∈ [0, 1]. Its forward is `bits @ mask.T`, exactly the sum form v_k
above. `WalshMixingLayer` is a bias-free linear layer. `WalshGate`
wraps its reference parameter to [0, 2) at forward time via
`torch.remainder` (PAN wraps to [0, 2π) for the same reason — Adam's
momentum pushes unconstrained real parameters off the period, causing
spikes at cosine inflection points).

### 2.1 Parameter count

For the representative case n_bits=8, K=4, 1 input, 2 output classes:

  masks (1 encoder × 4 × 8)     = 32
  mixing  (4 × 4)               = 16
  gate reference                = 4
  decoder (4 → 2) + bias        = 10
  total                         = 62

This is reproduced exactly by `WalshAccumulatorNetwork(n_bits=8,
k_freqs=4).count_parameters()` in the companion tests. The same
arithmetic for a K=1 parity circuit gives 15 parameters.

At n_bits=16 with K=n_bits=16: ≈ 290 parameters. A transformer
baseline at d_model=128 / d_mlp=512 on the same input set is
≈ 90,000 parameters — a ratio in line with PAN's 140–323× but
expected to be more extreme because Walsh-sparse Boolean targets are
often strictly sparser in characters than Fourier-sparse modular
targets are in their basis.

### 2.2 What the discrete circuit looks like

At convergence the learned masks m_{k,i} should snap to {0, 1}: each
Walsh channel reads a specific subset of the input bits. The mixing
weights W should snap to small integers (0, ±1 — and for popcount-style
targets, possibly ±2). The gate reference u_ref takes values in {0, 1}
mod 2 because that is where cos(π·u) attains ±1. The decoder is a
K→C linear readout.

When all three snap, the forward pass is exact Walsh character
arithmetic: encoders read masked XOR-parities of their inputs; the mix
XORs those parities together per output channel; the gate agrees or
disagrees with a learned reference sign; the decoder classifies the
resulting sign pattern.

---

## 3. Claim and Evidence Plan

We aim to establish three claims, each mirroring a PAN headline result.

### Claim 1: WAN groks Walsh-sparse Boolean tasks under random init.

Randomly-initialized masks and mixing weights converge to discrete
values that implement the target's Walsh decomposition. Evidence:
train-curve pairs with sharp val-accuracy transitions at late training
steps (the grokking signature), reproduced across seeds. Produced by
`experiments/wan_parity.yaml` (parity), `wan_popcount_mod4.yaml`
(non-trivial Walsh spectrum), `wan_xor_two.yaml` (2-input XOR),
`wan_rotl.yaml` (rotation).

The parity task is the low-hanging fruit: the target has a single
Walsh coefficient at s = 1…1, so K=1 suffices. The popcount-mod-4
task is the discriminating case — its Walsh spectrum is not 1-sparse,
so it tests whether gradient descent can discover multiple characters.
The "K=8 anomaly" in the PAN paper (where K=8 runs had qualitatively
different dynamics than their neighbours) has no direct WAN
counterpart, but the minimum-K curve for popcount-mod-4 is the
equivalent empirical question.

### Claim 2: Walsh encoding is doing the work, not the rest of the net.

We swap the learned mask structure for random ±1 masks and measure
validation accuracy. This is the WAN analog of PAN's
"randomize_frequencies" ablation. Expected result: accuracy collapses
to chance. The ablation is wired through `pan_lab/analysis.py` under
`randomize_masks` and is emitted into `ablations.csv` whenever
`options.ablations: true` is set in a WAN YAML.

A complementary ablation — `zero_walsh_mixing` — zeroes the mixing
matrix and confirms that the mixing stage (the "character
multiplication = sum of indices" machinery) is contributing. A third —
`zero_ref_v` — sets the gate reference to zero so every gate becomes
(1 + cos(π·u))/2, removing the learned discriminator between channels.
All three are run by the WAN experiments above.

### Claim 3: After grokking, the circuit is legible.

After a grokked run we dump `get_learned_masks()` and report (a)
whether the continuous masks m_{k,i} = σ(α_{k,i}) have snapped to
within ε of {0, 1}, (b) the binary mask per channel, (c) the popcount
of each binary mask, (d) the integer pattern of the mixing matrix. For
parity we expect all masks = 1…1 and mix = [1]; for popcount-mod-4
we expect a structured set of low-weight masks matching the Walsh
coefficients of (· mod 4); for 2-input XOR we expect channel j to
select bit j from both inputs with mix weights (+1, +1).

Legibility of the learned circuit is the part of the PAN paper that
motivated the architecture in the first place. "You can read the
algorithm off the weights" is the Walsh analog of Nanda's "five key
frequencies" finding.

### Claim 4: Parameter efficiency vs transformer.

The `wan_compare.yaml` YAML runs WAN at K=1 and a 64-dim transformer
on 8-bit parity, matching train_frac and step budget. Expected: WAN
solves parity in ~10³ steps with ~15 parameters; the transformer
either never groks within the step budget or groks later with
thousands of parameters. The resulting `parameter_efficiency.png`
figure is the headline ratio.

This is the same figure-shape as PAN's compare.yaml but on Boolean
rather than modular tasks. If the ratio comes in at O(100×) or more,
the "character arithmetic is the right primitive" claim is supported
at both group instances.

### Failure modes we expect to engineer around

Two predicted failure modes from the original spec, both of which the
current implementation addresses:

1. **Mask wandering.** Adam treats mask logits α_{k,i} as unconstrained
   reals; momentum can push them outside the usable sigmoid range,
   saturating gradients. We use plain `sigmoid(α)` in forward — no
   straight-through estimator — because the gate's cosine keeps
   gradients alive across the whole relaxation range. If mask
   saturation becomes an issue empirically, the fallback is a small
   entropy penalty pushing σ(α) toward {0, 1}; this plugs in as a
   training-loop modification identical to PAN's diversity-reg path.

2. **Gradient collapse at integer W.** At the discrete fixed point the
   WAN forward pass is exact XOR, where cos(π·u) = ±1 has zero
   gradient with respect to W. This is the Walsh analog of PAN's
   "reference phases at cosine inflection points" issue, and the same
   fix works: wrap u to R/2Z via cos(π·u) = cos(π·(u mod 2)).
   Because `torch.cos` is already 2-periodic in π·u, no explicit wrap
   is required in WAN — the PAN version had an explicit mod-2π on the
   mixing layer output only as numerical hygiene.

---

## 4. Experimental Protocol

Five YAMLs carry the experimental burden.

| YAML                                | Purpose                                       | Grid                                    |
| ----                                | ----                                          | ----                                    |
| `wan_parity.yaml`                   | Claim 1, trivial spectrum                     | mask_init × 5 seeds                     |
| `wan_popcount_mod4.yaml`            | Claim 1, non-trivial spectrum (K sweep)       | K ∈ {2,4,8,16} × 3 seeds                |
| `wan_xor_two.yaml`                  | Claim 1, 2-input task                         | K ∈ {4,6,8,12} × 3 seeds                |
| `wan_rotl.yaml`                     | Claim 1, permutation task                     | K ∈ {4,8,12} × 3 seeds                  |
| `wan_mask_init_ablation.yaml`       | Claim 2, is random init sufficient            | mask_init × 5 seeds                     |
| `wan_k_sweep.yaml`                  | K-reliability curve (insuff/transition/plat.) | K ∈ {1..16} × 5 seeds                   |
| `wan_compare.yaml`                  | Claim 4, head-to-head vs transformer          | {WAN, TF} × 3 seeds                     |

Every run writes `runs.csv`, `curves.csv`, `curves_stream.csv`,
`ablations.csv` (when enabled), plus any plots declared in the YAML,
under `results/<yaml-name>/`. Provenance (git SHA, torch version,
device, hostname) is captured in `manifest.json` exactly as for PAN.

Methodology follows PAN's protocol: every run's `seed` deterministically
selects the train/val split and the initial weights. Two runs with
identical configs produce bit-identical training curves on the same
device. Running `uv run python -m pan_lab experiments/wan_<name>.yaml
--dry-run` prints the full plan without training.

---

## 5. What Would Falsify This

The experiments above will falsify the Walsh-arithmetic hypothesis if:

- **Parity never groks under random init** — would mean the claim
  reduces to "if you hand-pick the all-ones mask, parity is easy,"
  which is trivial.
- **Popcount-mod-4 requires K ≫ n_bits** — would mean WAN needs a
  dense, distributed Walsh representation rather than the sparse one
  the mechanistic story predicts.
- **After grokking, masks and mix weights stay non-integer** — would
  mean the gradient descent fixed point is not the character-arithmetic
  circuit we claim, and some other (possibly transformer-like)
  representation is doing the work.
- **The transformer matches WAN at equal parameter count on parity** —
  would undercut the parameter-efficiency claim.

A "mixed" outcome — WAN groks parity cleanly, falters on popcount-mod-4
— would narrow the Walsh-arithmetic claim to targets with genuine
1-sparsity but still preserve the abstract "character arithmetic on
the right group" story. That's a respectable outcome too; it just
means the claim's reach is smaller than the most-extreme version.

---

## 6. Relationship to the SPF-PA Roadmap

The PAN paper ends by pointing at a general pattern:

> *Character arithmetic on the task's underlying group is the right
> neural primitive for tasks whose target function lives sparsely in
> that group's character basis.*

PAN is the G = ℤ/Pℤ instance. WAN is the G = 𝔽₂ⁿ instance. The next
natural instance is **G = ℤ/2ⁿℤ** — the group of n-bit integers under
modular addition, whose characters are complex-valued but whose
"phase" is quantized to {k·2π/2ⁿ : k ∈ ℤ/2ⁿℤ}. That group is exactly
what the SPF-PA (Spectral IEEE 754) format spec is about, and its
hardware story is already worked out in companion documents.

If WAN works — if the same architectural pattern grokks Walsh tasks
with the same parameter-efficiency signature as PAN's — the three-way
table below is the endpoint:

    Group              Characters            Mixing        Relevant tasks
    ─────              ──────────            ──────        ──────────────
    ℤ/Pℤ (PAN)         e^(2πikx/P)           phase add     modular arithmetic
    𝔽₂ⁿ (WAN)          (−1)^(s·x)            XOR           parity, bit hacks
    ℤ/2ⁿℤ (SPF-PA)     e^(2πikx/2ⁿ)         phase add     float add, IEEE 754

All three use the same pipeline: encode, mix, gate, decode. Only the
period differs. A downstream paper would tie this triple together;
this one's job is to ship the second row.

---

## 7. Status

Implementation, experiment YAMLs, and test coverage are in the branch
`claude/create-wan-architecture-Yc53t`. Smoke tests verify the
forward pass, parameter count, dataset generation, and that a
parity-initialised WAN with K=1 drives training loss to ~10⁻³ within
500 steps on CPU. Running the full experimental sweep is the next
step.

---

*Companion code: `pan_lab/models/wan.py`. Tests:
`tests/test_wan.py`. Experiments: `experiments/wan_*.yaml`.
Run plan: `make wan_all` (or `make wan_parity` / `make wan_compare`
for individual targets).*
