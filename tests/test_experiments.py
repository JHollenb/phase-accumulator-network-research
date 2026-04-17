"""Tests for pan_lab.experiments — registry, YAML loading, dry-run."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from pan_lab import RunConfig
from pan_lab.experiments import (
    EXPERIMENT_REGISTRY,
    load_experiment_yaml,
    run_experiment,
    run_from_yaml,
)


EXPECTED_EXPERIMENTS = {
    "grid_sweep", "sifp16_inference", "decoder_swap", "decoder_analysis",
}


def test_registry_contains_all_expected_experiments():
    missing = EXPECTED_EXPERIMENTS - set(EXPERIMENT_REGISTRY)
    assert not missing, f"Missing registered experiments: {missing}"


def test_all_registry_values_are_callable():
    for name, fn in EXPERIMENT_REGISTRY.items():
        assert callable(fn), f"Experiment {name!r} is not callable"


def test_dry_run_grid_sweep_writes_no_files(tmp_outdir, tiny_cfg):
    cfg = tiny_cfg.with_overrides(p=11, k_freqs=3, n_steps=100)
    rep = run_experiment("grid_sweep", cfg, str(tmp_outdir), dry_run=True,
                          grid={"seed": [0, 1]})
    # Dry-run: no CSVs written, reporter collected no runs
    assert len(rep._runs) == 0
    assert not (tmp_outdir / "runs.csv").exists()


def test_dry_run_k_sweep_yaml_writes_no_files(tmp_outdir):
    """k_sweep.yaml (rewritten as grid_sweep) must parse and dry-run."""
    here = Path(__file__).parent.parent / "experiments" / "k_sweep.yaml"
    rep = run_from_yaml(str(here), force_dry_run=True)
    assert len(rep._runs) == 0


def test_load_yaml_round_trip(tmp_outdir):
    spec = {
        "experiment": "grid_sweep",
        "out_dir":    str(tmp_outdir),
        "dry_run":    True,
        "base": {"p": 11, "k_freqs": 3, "n_steps": 100,
                  "seed": 42, "log_every": 50},
        "grid": {"seed": [42]},
    }
    p = tmp_outdir / "test.yaml"
    with open(p, "w") as f:
        yaml.dump(spec, f)

    name, base, out_dir, dry_run, args = load_experiment_yaml(str(p))
    assert name == "grid_sweep"
    assert dry_run is True
    assert isinstance(base, RunConfig)
    assert base.p == 11
    assert args["grid"] == {"seed": [42]}


def test_run_from_yaml_dry_run(tmp_outdir):
    spec = {
        "experiment": "grid_sweep",
        "out_dir":    str(tmp_outdir),
        "dry_run":    False,        # overridden below
        "base": {"p": 11, "k_freqs": 3, "n_steps": 100,
                  "seed": 42, "log_every": 50},
        "grid": {"seed": [42, 123]},
    }
    p = tmp_outdir / "test.yaml"
    with open(p, "w") as f:
        yaml.dump(spec, f)
    rep = run_from_yaml(str(p), force_dry_run=True)
    assert len(rep._runs) == 0


def test_unknown_yaml_keys_are_ignored_not_fatal():
    cfg = RunConfig.from_dict(
        {"p": 11, "k_freqs": 3, "nonexistent_field": "whatever"})
    assert cfg.p == 11


def test_unknown_experiment_name_raises():
    with pytest.raises(KeyError):
        run_experiment("not_a_real_experiment", RunConfig(), "/tmp", dry_run=True)


def test_all_yaml_files_load_without_error():
    """Every YAML in experiments/ must parse into a valid (name, cfg)."""
    here = Path(__file__).parent.parent / "experiments"
    yamls = list(here.glob("*.yaml"))
    assert len(yamls) > 0, "No experiment YAMLs found"
    for y in yamls:
        name, base, out_dir, dry_run, args = load_experiment_yaml(str(y))
        assert name in EXPERIMENT_REGISTRY, (
            f"{y.name}: references unknown experiment {name!r}")
        assert isinstance(base, RunConfig)
