# PAN Paper v5 — Verification Report

**Paper reviewed:** `docs/drafts/05_pan_paper_v5.md`
**Verification date:** 2026-04-18
**Verifier:** end-to-end cross-check of every numeric claim against the CSVs consolidated in `data/20260418_paper_results/`.

This report documents (a) where each claim's underlying data now lives, (b) how to reproduce the verification, and (c) the three discrepancies still outstanding between the paper text and the data.

---

## 1. Summary

| Section | Topic | Status |
|---|---|---|
| §3.1 | PAN vs Transformer head-to-head (P=113, K=9) | ✅ verified |
| §3.2 | Seed=42 deep dive (grok step, ablations, metrics at termination) | ✅ verified (with one nuance — see §3.2 note) |
| §3.3 | n=20 slot census at K=9 Fourier | ⚠️ one discrepancy (compliance==1.00 count) |
| §3.4 | K-sweep reliability (K=1..14, both inits) | ✅ verified (one minor step-number nit) |
| §3.5 | Cross-prime generalization, K=10, both inits | ⚠️ one discrepancy (Fourier reliability: 20/24, not 22/24) |
| §3.6 | Decoder analysis (learned vs clock-only vs gate-optimal) | ✅ verified |
| §3.7 | SFP-16 quantization invariance | ✅ verified |
| §3.8 | K≤4 insufficiency | ✅ verified |
| §4.1 | Decoder-swap identifiability | ✅ verified |

Three claims do not match the data exactly. They are itemised in §6 with exact numbers and file pointers.

---

## 2. Where the data lives

Every paper claim is backed by a CSV under `data/20260418_paper_results/`. The relevant subdirectories and what they feed:

```
data/20260418_paper_results/
├── compare/                     §3.1   PAN vs Transformer head-to-head
├── tier3/                       §3.2   seed=42 deep dive (runs, ablations, metrics)
├── k_census_n20/                §3.3/§3.8  base n=20 slot census, K=6..12, Fourier init (source for §3.8 Fourier grok rates)
├── k_census_n20_fourier/        (alias of k_census_n20; kept for back-compat)
├── k_census_n20_random/         §3.4   K-sweep reliability, random init
├── paper_k5_extended/           §3.4   K=5 extended (seeds 20-58) for 11/59 pooled stat
├── paper_k13_fourier/           §3.4   K=13 Fourier reliability (n=21)
├── paper_k13_random/            §3.4   K=13 random reliability (n=21)
├── primes_primary_k/            §3.5   K=10 Fourier across P∈{43..127}
├── paper_cross_primes/          §3.5   K=10 random across P∈{43..127}
├── held_out_primes_p97_long/    §3.5   P=97 long-run addendum
├── decoder_analysis/            §3.6   learned vs clock-only vs gate-optimal decoder
├── sifp16_inference/            §3.7   SFP-16 vs FP32 inference equivalence
├── k4_run/                      §3.8   K=4 random × 6 seeds × 200K steps (0/6 grok)
├── decoder_swap/                §4.1   swap-learned-decoder-for-Fourier identifiability
└── (supporting) freq_init_ablation, held_out_primes, k8_sweep, slot_census,
    random_init_census_20, k_census_n20, init_random_primary_k,
    decoder_analysis_initial
```

Each directory contains at minimum `runs.csv` and `manifest.json` (provenance: git SHA, torch version, device, argv). Per-eval trajectories live in `curves.csv` / `curves_stream.csv`; per-step mechanistic metrics live in `metrics.csv`; ablations in `ablations.csv`; slot censuses in `slots.csv`.

One housekeeping move made during verification: `primes_primary_k/runs.csv` and `slots.csv` were copied in from the top-level `results/runs.csv` / `results/slots.csv` (they had been left behind at the top level during an earlier run).

---

## 3. How to reproduce each check

All commands below are run from repo root and assume `uv run` as the Python entry point (matching CLAUDE.md). Every check is a pure pandas aggregation over the relevant CSV.

### §3.1 — PAN vs Transformer head-to-head

**Data:** `data/20260418_paper_results/compare/runs.csv` (2 rows: `pan`, `tf`).

```python
import pandas as pd
df = pd.read_csv("data/20260418_paper_results/compare/runs.csv")
print(df[["label","model_kind","grok_step","peak_val_acc","elapsed_s","param_count"]])
```

| label | model_kind  | grok_step | peak_val_acc | elapsed_s | param_count |
|-------|-------------|----------:|-------------:|----------:|------------:|
| pan   | pan         |    39,400 |       0.9918 |     48.69 |       1,319 |
| tf    | transformer |     7,200 |       0.9932 |     14.55 |     227,200 |

Paper §3.1 claim: "PAN groks at step 39,400 (48.7s, 1,319 params, 99.18% peak val acc); transformer at step 7,200 (14.6s, 227,200 params, 99.32%)." **Match.** Parameter ratio 227,200 / 1,319 ≈ 172×. **Match.**

### §3.2 — seed=42 deep dive

**Data:** `data/20260418_paper_results/tier3/runs.csv`, `ablations.csv`, `metrics.csv`.

```python
import pandas as pd
pd.read_csv("data/20260418_paper_results/tier3/runs.csv")[["seed","grok_step","peak_val_acc","param_count"]]
pd.read_csv("data/20260418_paper_results/tier3/ablations.csv")
m = pd.read_csv("data/20260418_paper_results/tier3/metrics.csv")
m.iloc[-1][["step","clock_compliance","active_freq_count","active_freq_set",
            "decoder_fourier_peak_mean"]]
m["decoder_fourier_peak_mean"].max(), m.loc[m["decoder_fourier_peak_mean"].idxmax(),"step"]
```

- grok_step = 16,500; peak_val_acc = 1.000; param_count = 1,319 — **match**.
- Ablations at termination: baseline 1.0000, zero_phase_mixing 0.00887, randomize_frequencies 0.00861, zero_ref_phases 0.03589 — **match** (paper rounds to 1.00 / 0.009 / 0.009 / 0.036).
- At termination (step 99,500): clock_compliance = 0.667, active_freq_set = {3, 5, 8, 10}, decoder_fourier_peak_mean = 0.410 — **match**.
- Peak decoder_fourier_peak_mean = **0.872 at step 2,500** (well before grok); paper reports the early peak / late decay arc correctly.

**Nuance:** `gate_linear_acc`, `fp32_acc`, `sifp16_acc`, `quant_delta` are `NaN` on the very last row of `metrics.csv` (evaluated only at specific logging steps, not every eval). Paper's reported `gate_linear_acc = 0.961 @ step 95,000` and `sifp16_acc = fp32_acc = 1.0 @ step 95,000` are taken from the step-95,000 row, which is correct — but a reader rerunning `.iloc[-1]` will see NaN on those columns. Pulling from `m[m.step==95000]` reproduces the paper's numbers.

### §3.3 — n=20 slot census at K=9 Fourier

**Data:** `data/20260418_paper_results/k_census_n20/` (`runs.csv` + `metrics.csv`). This is the base Fourier K-sweep; `k_census_n20_fourier/` is an alias kept for back-compat and resolves to the same CSVs.

```python
import pandas as pd, re
runs = pd.read_csv("data/20260418_paper_results/k_census_n20/runs.csv")
k9 = runs[runs.k_freqs==9]
print("K=9:", len(k9), "grokked:", k9.grokked.sum())

metrics = pd.read_csv("data/20260418_paper_results/k_census_n20/metrics.csv")
last = metrics.sort_values("step").groupby("run_id").tail(1)
grok_ids = set(k9[k9.grokked]["run_id"])
lg = last[last.run_id.isin(grok_ids)]
print("mean compliance:", lg.clock_compliance.mean())
print("N @ compliance==1.0:", (lg.clock_compliance==1.0).sum())
print("mean active_freq_count:", lg.active_freq_count.mean())
```

- K=9 grok rate: **14 / 20** — match.
- Mean clock_compliance across the 14 grokked seeds: **0.9365** (rounds to 0.94) — match.
- Min / max clock_compliance: 0.778 / 1.000 — match.
- Mean active_freq_count across grokked seeds: **3.79** — match.
- **Seeds at compliance == 1.000: {3, 5, 10, 12, 13, 15, 16} — seven seeds.** Paper §3.3 says "five of the fourteen." ⚠️ See §6.

### §3.4 — K-sweep reliability

**Data:** `data/20260418_paper_results/k_census_n20_random/runs.csv` (K=1..14, 20 seeds),
`data/20260418_paper_results/paper_k5_extended/runs.csv` (K=5, seeds 20–58 for n=39),
`data/20260418_paper_results/paper_k13_fourier/runs.csv` (K=13 Fourier, n=21),
`data/20260418_paper_results/paper_k13_random/runs.csv` (K=13 random, n=21).

```python
import pandas as pd
for path in ["data/20260418_paper_results/paper_k13_fourier/runs.csv",
             "data/20260418_paper_results/paper_k13_random/runs.csv"]:
    df = pd.read_csv(path)
    g = df[df.grokked]
    print(path, len(df), "grokked", len(g), "median grok_step", g.grok_step.median())
```

- K=13 Fourier: **19 / 21 = 90.5%**, median grok_step = **8,000** (paper says 8,500 — see §6).
- K=13 random: **20 / 21 = 95.2%**, median grok_step = **13,000** — match.
- K=5 pooled (K5 census 20 seeds + paper_k5_extended 39 seeds): 5 + 6 = **11 / 59 ≈ 18.6%** — match.

### §3.5 — cross-prime generalization at K=10

**Two sub-studies, one for each encoder init.**

**Random init (paper_cross_primes):**

```python
import pandas as pd
df = pd.read_csv("data/20260418_paper_results/paper_cross_primes/runs.csv")
df["grokked"].value_counts()
df.groupby("label")["grok_step"].median()
```

- Total (non-`p97-long`) runs: 24. **All 24 grokked.** — match.
- Median grok_step per P: 43 → 8,000; 59 → 9,500; 67 → 17,500; 71 → 26,500; 89 → 10,500; 97 → 18,000; 113 → 15,500; 127 → 54,000 — match.
- P=97 long-run addendum (seed 0/1/2): 12,500 / 18,000 / 66,500 — match.
- Param counts 703 (P=43) → 1,627 (P=127) — match.

**Fourier init (primes_primary_k):**

```python
import pandas as pd
df = pd.read_csv("data/20260418_paper_results/primes_primary_k/runs.csv")
main = df[~df.label.str.startswith("p97-long")]
print("grokked:", main.grokked.sum(), "/", len(main))
print(main[~main.grokked][["label","peak_val_acc"]])
```

- Total runs (excluding p97-long): 24. **Grokked: 20 / 24.** Paper §3.5 says 22 / 24. ⚠️ See §6.
- Failures: P89-s0 (peak 0.398), P113-s0 (peak 0.652), P127-s0 (peak 0.310), P127-s1 (peak 0.901).

### §3.6 — decoder analysis

**Data:** `data/20260418_paper_results/decoder_analysis/decoder_analysis.csv`.

```python
import pandas as pd
df = pd.read_csv("data/20260418_paper_results/decoder_analysis/decoder_analysis.csv")
df[df.grokked][["seed","acc_learned","acc_clock_only","gate_optimal_acc"]]
```

| seed | acc_learned | acc_clock_only | gate_optimal_acc |
|----:|-----------:|--------------:|----------------:|
|  42 |      0.9923 |         0.0717 |          0.9991 |
| 123 |      0.9914 |         0.6430 |          1.0000 |
| 789 |      0.9935 |         0.3446 |          1.0000 |

All three grokked seeds: gate_optimal_acc ≥ 0.999 while acc_clock_only is well below the paper's threshold — **match** the paper's "the clock alone isn't enough; the mixer's extras carry real signal; a hand-built gate-only decoder saturates" narrative. Seeds 456 and 999 are the non-grokked controls (both show NaN for the clock-only and gate-optimal rows because those analyses short-circuit when the model never groks).

### §3.7 — SFP-16 quantization invariance

**Data:** `data/20260418_paper_results/sifp16_inference/quant_eval.csv`.

> *Terminology note:* this report refers to the 16-bit integer phase quantization as **SFP** (the paper-facing name). The codebase still uses the internal symbol `SIFP16_QUANT_ERR` (`pan_lab/config.py:38-40`) and the column name `val_acc_sifp16` in `quant_eval.csv`. These are the same quantity — SFP is the external name for what the code calls SIFP-16.

```python
import pandas as pd
pd.read_csv("data/20260418_paper_results/sifp16_inference/quant_eval.csv")
```

| seed | val_acc_fp32 | val_acc_sfp (was val_acc_sifp16) | delta |
|----:|-------------:|---------------------------------:|------:|
|  42 |       0.9923 |                           0.9923 |  0.000 |
| 123 |       0.9914 |                           0.9914 |  0.000 |
| 456 |       0.9436 |                           0.9406 | -0.003 |

Two of three seeds show **zero** drop from FP32 → SFP-16 inference; the worst-case drop is 0.3pp on a non-grokked seed (456 was below threshold already). **Match** the paper's §3.7 claim that SFP-16 preserves accuracy.

### §3.8 — K ≤ 4 insufficiency

**Data:** `data/20260418_paper_results/k4_run/` (K=4, 6 seeds, 200K steps) plus the K=1..4 columns of `data/20260418_paper_results/k_census_n20_random/runs.csv`.

```python
import pandas as pd
df = pd.read_csv("data/20260418_paper_results/k4_run/runs.csv")
print("K=4 long: grokked", df.grokked.sum(), "/", len(df))

df2 = pd.read_csv("data/20260418_paper_results/k_census_n20_random/runs.csv")
df2[df2.k_freqs<=4].groupby("k_freqs")["grokked"].sum()
```

- K=4 × 6 seeds × 200K steps: **0 / 6 grokked** — match.
- K=1..4 census: all 0 / 20 — match.

### §4.1 — decoder-swap identifiability

**Data:** `data/20260418_paper_results/decoder_swap/decoder_swap.csv`.

```python
import pandas as pd
pd.read_csv("data/20260418_paper_results/decoder_swap/decoder_swap.csv")
```

| seed | val_acc_pan_decoder | val_acc_fourier_decoder | delta  |
|----:|--------------------:|------------------------:|-------:|
|  42 |               0.992 |                   0.017 | -0.975 |
| 123 |               0.991 |                   0.003 | -0.988 |
| 456 |               0.944 |                   0.011 | -0.932 |

Swapping the learned decoder for an analytically-constructed Fourier decoder collapses accuracy to chance — **match** the §4.1 claim that the learned decoder is not trivially a Fourier readout.

---

## 4. Provenance & reproducibility harness

Every directory has a `manifest.json` with the full provenance block (`git_sha`, `torch`, `device`, `cwd`, `argv`). A paper-submission consumer can re-run the relevant YAML from the same SHA to reproduce. The YAMLs that produced each dataset:

| Directory | Driving YAML |
|---|---|
| `compare/`                   | `experiments/compare.yaml` |
| `tier3/`                     | `experiments/tier3.yaml` |
| `k_census_n20/`              | `experiments/k_census_n20.yaml` (base sweep, Fourier init — source of §3.8 Fourier K=6..12 rates) |
| `k_census_n20_fourier/`      | alias of `k_census_n20/` (same CSVs) |
| `k_census_n20_random/`       | `experiments/k_census_n20.yaml` with `freq_init: random` |
| `paper_k5_extended/`         | `experiments/paper_k5_extended.yaml` |
| `paper_k13_fourier/`         | `experiments/paper_k13_fourier.yaml` |
| `paper_k13_random/`          | `experiments/paper_k13_random.yaml` |
| `primes_primary_k/`          | `experiments/primes_primary_k.yaml` |
| `paper_cross_primes/`        | `experiments/paper_cross_primes.yaml` |
| `decoder_analysis/`          | `experiments/decoder_analysis.yaml` (bespoke analyzer) |
| `sifp16_inference/`          | `experiments/sifp16_inference.yaml` (bespoke analyzer) |
| `decoder_swap/`              | `experiments/decoder_swap.yaml` (bespoke analyzer) |
| `k4_run/`                    | `experiments/k4_run.yaml` |

---

## 5. Figures

All paper figures are produced from the CSVs above by the registered plot functions in `pan_lab/plots.py` (dispatched via `PLOT_REGISTRY` in `pan_lab/grid_sweep.py`). `uv run python -m pan_lab --replot data/20260418_paper_results/<dir>` regenerates every figure from its CSVs without retraining. This has been smoke-tested for `compare/`, `tier3/`, `paper_cross_primes/`, and both `paper_k13_*/` dirs during verification.

---

## 6. Discrepancies between paper and data

Three numbers in the paper were found not to match what the CSVs show. Each is small, but each should be either corrected in the paper or substantiated from a different dataset if one was intended. §6.1 has since been reconciled in the paper prose; §6.2 and §6.3 remain open.

### 6.1 §3.5 — Fourier K=10 cross-prime reliability ✅ reconciled

- **Paper now says:** 20 / 24 grokked at K=10 Fourier (+16.7pp random-init advantage). Prose and Figure 6 caption agree.
- **Data says:** 20 / 24 grokked. Source: `data/20260418_paper_results/primes_primary_k/runs.csv`.
- **Failing seeds:** P89-s0 (peak_val_acc 0.398), P113-s0 (0.652), P127-s0 (0.310), P127-s1 (0.901).

### 6.2 §3.3 — "five of fourteen" at compliance == 1.00

- **Paper says:** "five of the fourteen grokked seeds have clock_compliance == 1.000."
- **Data says:** seven do (seeds 3, 5, 10, 12, 13, 15, 16). Source: last-step row of `data/20260418_paper_results/k_census_n20/metrics.csv`, filtered to the 14 K=9 grokked runs.
- **Implied correction:** "seven of the fourteen."

### 6.3 §3.4 — K=13 Fourier median grok step

- **Paper says:** median grok step 8,500.
- **Data says:** 8,000. Source: `data/20260418_paper_results/paper_k13_fourier/runs.csv`, `grok_step` median over `grokked==True`.
- **Implied correction:** 8,000.

(The K=13 random median of 13,000, the K=5 pooled 11/59, and every cross-prime median are exact.)

---

## 7. Artifact checklist

Everything referenced in the paper now sits under `data/20260418_paper_results/`, with 22 subdirectories covering:

- Every §3.1–§4.1 numeric claim (with the three discrepancies in §6).
- Per-run provenance manifests.
- The post-training analyzers' CSVs (decoder_analysis, decoder_swap, sifp16_inference).
- Supplementary sweeps (`freq_init_ablation`, `held_out_primes`, `k8_sweep`, etc.) that the paper doesn't cite but which back up the narrative.
- The handful of model `.pt` checkpoints the mechanistic analyses load.

A reviewer with the repo at the recorded `git_sha` and `uv sync` can reproduce every number in this report by running the pandas snippets in §3 against the CSVs in-place.
