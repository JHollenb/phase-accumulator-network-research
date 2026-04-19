# Data for `02_nanda_grokking.ipynb`

Self-contained snapshot of the transformer head-to-head run used by
`notebooks/02_nanda_grokking.ipynb`.

## Provenance

- **Source dir:** `results/compare/` at repo root.
- **Source YAML:** `experiments/compare.yaml` (PAN vs transformer,
  `mod_add`, p=113, seed=42, 50K-step budget with early-stop).
- **Source commit:** `cacd8fa` (per `manifest.json`).
- **Produced:** 2026-04-17 on MPS (Apple Silicon).

## Files

| File | Source | Purpose |
|------|--------|---------|
| `runs.csv`      | `results/compare/runs.csv`      | Per-run metadata (2 rows: PAN + TF). |
| `curves.csv`    | `results/compare/curves.csv`    | Train/val loss + acc per eval step (125 rows total). |
| `manifest.json` | `results/compare/manifest.json` | Git SHA, torch version, device, argv. |

Byte-identical copies — no transforms applied.

## Filter rule

The notebook filters every DataFrame to
`run_id == "tf-88648a7fb8"` (the transformer row). The PAN row
(`pan-0862ac9440`) is present in the CSVs but unused here; the
PAN-side story lives in `notebooks/01_diffusion_investigation.ipynb`.

## Not included

- `ablations.csv`, `metrics.csv`, `metrics_peaks.csv`,
  `metrics_spectra.csv` — all PAN-specific; the transformer run's
  rows in them are mostly empty.
- No `.pt` model snapshots were saved for this run (the YAML does not
  set `save_model`), so Nanda's four progress measures (restricted /
  excluded loss, trigonometric similarity, Gini) cannot be recomputed
  from these files. The notebook's §5 explains this.
