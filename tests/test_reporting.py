"""Tests for pan_lab.reporting — CSV schema and manifest integrity."""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from pan_lab import RunConfig, make_model, train
from pan_lab.data      import make_modular_dataset
from pan_lab.reporting import ExperimentReporter


@pytest.fixture
def one_result(tiny_cfg):
    cfg = tiny_cfg
    tx, ty, vx, vy = make_modular_dataset(
        p=cfg.p, task_kind=cfg.task_kind, seed=cfg.seed, device="cpu")
    m = make_model(cfg)
    r = train(m, cfg, tx, ty, vx, vy, verbose=False)
    return r, vx, vy


def test_reporter_writes_runs_csv(tmp_outdir, one_result):
    r, vx, vy = one_result
    rep = ExperimentReporter("test_exp", str(tmp_outdir))
    rep.add_run(r, val_x=vx, val_y=vy, ablations=True)
    paths = rep.write_all()

    assert "runs.csv" in paths
    assert os.path.exists(paths["runs.csv"])
    df = pd.read_csv(paths["runs.csv"])
    assert len(df) == 1
    for col in ("run_id", "experiment", "p", "k_freqs", "seed",
                "grok_step", "grokked", "final_val_acc", "peak_val_acc",
                "param_count", "mode_collapsed"):
        assert col in df.columns


def test_reporter_writes_curves_csv(tmp_outdir, one_result):
    r, vx, vy = one_result
    rep = ExperimentReporter("test_exp", str(tmp_outdir))
    rep.add_run(r, val_x=vx, val_y=vy)
    paths = rep.write_all()

    assert "curves.csv" in paths
    df = pd.read_csv(paths["curves.csv"])
    for col in ("run_id", "step", "train_loss", "val_loss", "val_acc"):
        assert col in df.columns
    # One row per eval step
    assert len(df) == len(r.history.steps)


def test_reporter_writes_ablations_csv(tmp_outdir, one_result):
    r, vx, vy = one_result
    rep = ExperimentReporter("test_exp", str(tmp_outdir))
    rep.add_run(r, val_x=vx, val_y=vy, ablations=True)
    paths = rep.write_all()

    assert "ablations.csv" in paths
    df = pd.read_csv(paths["ablations.csv"])
    interventions = set(df["intervention"])
    assert "baseline" in interventions
    assert "zero_phase_mixing" in interventions


def test_manifest_has_provenance(tmp_outdir, one_result):
    r, vx, vy = one_result
    rep = ExperimentReporter("test_exp", str(tmp_outdir))
    rep.add_run(r, val_x=vx, val_y=vy)
    rep.write_all()
    mp = rep.write_manifest()

    assert os.path.exists(mp)
    with open(mp) as f:
        manifest = json.load(f)
    assert manifest["experiment"] == "test_exp"
    assert manifest["n_runs"] == 1
    assert "provenance" in manifest
    assert manifest["provenance"]["torch"] is not None
    assert manifest["provenance"]["device"] in ("cpu", "cuda", "mps")


def test_manifest_catalogs_files_by_kind(tmp_outdir, one_result):
    r, vx, vy = one_result
    rep = ExperimentReporter("test_exp", str(tmp_outdir))
    rep.add_run(r, val_x=vx, val_y=vy)
    rep.write_all()

    # Drop a fake plot + model that the reporter didn't write itself —
    # they should still show up after the directory rescan.
    fake_png = os.path.join(str(tmp_outdir), "fake_plot.png")
    fake_pt  = os.path.join(str(tmp_outdir), "model_foo.pt")
    with open(fake_png, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
    with open(fake_pt, "wb") as f:
        f.write(b"fake-weights")

    mp = rep.write_manifest()
    with open(mp) as f:
        manifest = json.load(f)

    files = manifest["files"]
    assert "csvs" in files and "plots" in files and "models" in files
    plot_names  = {e["name"] for e in files["plots"]}
    model_names = {e["name"] for e in files["models"]}
    csv_names   = {e["name"] for e in files["csvs"]}
    assert "fake_plot.png" in plot_names
    assert "model_foo.pt"  in model_names
    assert "runs.csv"      in csv_names
    # Entries carry a size.
    for entry in files["plots"] + files["models"] + files["csvs"]:
        assert entry["size_bytes"] >= 0
    # manifest.json itself is excluded.
    all_names = plot_names | model_names | csv_names | {
        e["name"] for e in files["other"]}
    assert "manifest.json" not in all_names


def test_manifest_is_idempotent(tmp_outdir, one_result):
    r, vx, vy = one_result
    rep = ExperimentReporter("test_exp", str(tmp_outdir))
    rep.add_run(r, val_x=vx, val_y=vy)
    rep.write_all()

    mp1 = rep.write_manifest()
    with open(mp1) as f:
        first = json.load(f)
    mp2 = rep.write_manifest()
    with open(mp2) as f:
        second = json.load(f)
    assert first == second


def test_summary_is_non_empty(tmp_outdir, one_result):
    r, vx, vy = one_result
    rep = ExperimentReporter("test_exp", str(tmp_outdir))
    rep.add_run(r, val_x=vx, val_y=vy)
    s = rep.summary()
    # Single run: falls back to group-by seed
    assert not s.empty
    assert "grok_rate" in s.columns


def test_summary_groups_by_varying_column(tmp_outdir, tiny_cfg):
    """Multiple runs with varying K should group by k_freqs."""
    rep = ExperimentReporter("k_test", str(tmp_outdir))
    for k in (2, 3):
        cfg = tiny_cfg.with_overrides(k_freqs=k, label=f"K{k}")
        tx, ty, vx, vy = make_modular_dataset(
            p=cfg.p, task_kind=cfg.task_kind, seed=cfg.seed, device="cpu")
        m = make_model(cfg)
        r = train(m, cfg, tx, ty, vx, vy, verbose=False)
        rep.add_run(r, val_x=vx, val_y=vy)
    s = rep.summary()
    assert "k_freqs" in s.columns
    assert set(s["k_freqs"]) == {2, 3}
