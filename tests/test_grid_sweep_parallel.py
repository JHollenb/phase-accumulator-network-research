"""
Equivalence tests for grid_sweep workers >= 2 vs workers == 1.

Spawns real subprocesses, so each test takes ~10s on macOS to pay the
torch-import cost in the worker. Kept to one tiny grid (4 cfgs) so the
total cost stays bounded.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pan_lab.experiments import run_experiment


def _run_tiny_sweep(cfg, out_dir, workers):
    return run_experiment(
        "grid_sweep",
        cfg,
        str(out_dir),
        dry_run=False,
        grid={"seed": [0, 1, 2, 3]},
        options={
            "ablations": False,
            "slots":     False,
            "metrics":   False,
            "workers":   workers,
        },
    )


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [c for c in ("run_id", "step", "intervention", "k") if c in df.columns]
    return (df.sort_values(sort_cols)
              .reset_index(drop=True)
              .reindex(sorted(df.columns), axis=1))


@pytest.mark.parametrize("workers", [2, 3])
def test_parallel_matches_sequential(tmp_outdir, tiny_cfg, workers):
    seq_dir = tmp_outdir / "seq"
    par_dir = tmp_outdir / "par"
    seq_dir.mkdir()
    par_dir.mkdir()

    base = tiny_cfg.with_overrides(early_stop=False)

    rep_seq = _run_tiny_sweep(base, seq_dir, workers=1)
    rep_par = _run_tiny_sweep(base, par_dir, workers=workers)

    # Bit-equal modulo row order. Drop fields that legitimately differ:
    #   elapsed_s — wall-clock timing
    #   experiment — derived from out_dir basename ("seq" vs "par")
    drop_cols = ("elapsed_s", "experiment")
    for getter in ("runs_df", "curves_df"):
        seq = _canon(getattr(rep_seq, getter)())
        par = _canon(getattr(rep_par, getter)())
        for c in drop_cols:
            if c in seq.columns:
                seq = seq.drop(columns=[c])
            if c in par.columns:
                par = par.drop(columns=[c])
        pd.testing.assert_frame_equal(
            seq, par, check_dtype=False, check_exact=False, atol=0, rtol=0,
            obj=getter,
        )


def test_parallel_curves_stream_csv_concatenated(tmp_outdir, tiny_cfg):
    base = tiny_cfg.with_overrides(early_stop=False)
    _run_tiny_sweep(base, tmp_outdir, workers=2)

    stream = tmp_outdir / "curves_stream.csv"
    assert stream.exists(), "parallel run should still produce curves_stream.csv"
    df = pd.read_csv(stream)
    assert {"run_id", "step", "val_acc"}.issubset(df.columns)
    # 4 seeds × 200 steps / 50 log_every = 4 × 4 = 16 expected eval rows
    assert df["run_id"].nunique() == 4
