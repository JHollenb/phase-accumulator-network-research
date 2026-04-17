"""Tests for pan_lab.experiments — registry, YAML loading, dry-run."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from pan_lab import RunConfig
from pan_lab.experiments import (
    EXPERIMENT_REGISTRY,
    ExperimentSpecValidationError,
    build_execution_manifest,
    load_experiment_yaml,
    run_experiment,
    run_from_yaml,
)


EXPECTED_EXPERIMENTS = {
    "compare", "k_sweep", "k8_sweep", "dw_sweep", "wd_sweep",
    "primes", "held_out_primes", "tier3", "slot_census",
    "freq_init_ablation", "sifp16_inference", "decoder_swap",
    "mod_mul", "mod_two_step", "tf_sweep",
}


def test_registry_contains_all_expected_experiments():
    missing = EXPECTED_EXPERIMENTS - set(EXPERIMENT_REGISTRY)
    assert not missing, f"Missing registered experiments: {missing}"


def test_all_registry_values_are_callable():
    for name, fn in EXPERIMENT_REGISTRY.items():
        assert callable(fn), f"Experiment {name!r} is not callable"


def test_dry_run_compare_writes_no_files(tmp_outdir, tiny_cfg):
    cfg = tiny_cfg.with_overrides(p=11, k_freqs=3, n_steps=100)
    rep = run_experiment("compare", cfg, str(tmp_outdir), dry_run=True)
    # Dry-run: no CSVs written, reporter collected no runs
    assert len(rep._runs) == 0
    assert not (tmp_outdir / "runs.csv").exists()


def test_dry_run_k_sweep_writes_no_files(tmp_outdir, tiny_cfg):
    cfg = tiny_cfg
    rep = run_experiment("k_sweep", cfg, str(tmp_outdir), dry_run=True,
                          ks=[2, 3], seeds=[0, 1])
    assert len(rep._runs) == 0


def test_load_yaml_round_trip(tmp_outdir):
    spec = {
        "experiment": "compare",
        "out_dir":    str(tmp_outdir),
        "dry_run":    True,
        "base": {"p": 11, "k_freqs": 3, "n_steps": 100,
                  "seed": 42, "log_every": 50},
    }
    p = tmp_outdir / "test.yaml"
    with open(p, "w") as f:
        yaml.dump(spec, f)

    name, base, out_dir, dry_run, args = load_experiment_yaml(str(p))
    assert name == "compare"
    assert dry_run is True
    assert isinstance(base, RunConfig)
    assert base.p == 11


def test_run_from_yaml_dry_run(tmp_outdir):
    spec = {
        "experiment": "compare",
        "out_dir":    str(tmp_outdir),
        "dry_run":    False,        # overridden below
        "base": {"p": 11, "k_freqs": 3, "n_steps": 100,
                  "seed": 42, "log_every": 50},
    }
    p = tmp_outdir / "test.yaml"
    with open(p, "w") as f:
        yaml.dump(spec, f)
    rep = run_from_yaml(str(p), force_dry_run=True)
    assert len(rep._runs) == 0


def test_load_yaml_v2_schema_sections_map_to_exp_args(tmp_outdir):
    spec = {
        "schema_version": 2,
        "experiment": "wd_sweep",
        "out_dir": str(tmp_outdir),
        "base": {"p": 11, "k_freqs": 3, "n_steps": 100, "seed": 42},
        "grid": {"wds": [0.01, 0.02], "seeds": [1, 2], "k_freqs": 7},
        "capture": {"ablations": True, "slots": False},
        "analyses": ["decoder_swap", "sifp16_eval", "decoder_harmonics"],
        "plots": [{"type": "reliability", "group_by": "weight_decay"}],
        "experiment_args": {"legacy_only": 123},
    }
    p = tmp_outdir / "test_v2.yaml"
    with open(p, "w") as f:
        yaml.dump(spec, f)

    name, base, out_dir, dry_run, args = load_experiment_yaml(str(p))
    assert name == "wd_sweep"
    assert isinstance(base, RunConfig)
    assert args["wds"] == [0.01, 0.02]
    assert args["seeds"] == [1, 2]
    assert args["k_freqs"] == 7
    assert args["capture"]["ablations"] is True
    assert args["analyses"] == ["decoder_swap", "sifp16_eval", "decoder_harmonics"]
    assert args["plots"][0]["type"] == "reliability"
    assert args["legacy_only"] == 123




def test_factory_backed_runner_for_v2_sweeps_dry_run(tmp_outdir, tiny_cfg):
    rep = run_experiment(
        "k_sweep",
        tiny_cfg,
        str(tmp_outdir),
        dry_run=True,
        _schema_version=2,
        ks=[2, 3],
        seeds=[0, 1],
        plots=[],
        analyses=[],
        capture={},
    )
    assert len(rep._runs) == 0


def test_factory_and_legacy_schema_paths_generate_identical_manifests(tmp_outdir, tiny_cfg):
    v1_manifest = build_execution_manifest(
        "k_sweep",
        tiny_cfg,
        str(tmp_outdir),
        {"ks": [2, 3], "seeds": [0, 1]},
    )
    v2_manifest = build_execution_manifest(
        "k_sweep",
        tiny_cfg,
        str(tmp_outdir),
        {
            "_schema_version": 2,
            "ks": [2, 3],
            "seeds": [0, 1],
            "capture": {},
            "analyses": [],
            "plots": [],
        },
    )
    assert v1_manifest["expanded_runs"] == v2_manifest["expanded_runs"]
    assert v1_manifest["capture"] == v2_manifest["capture"]
    assert v1_manifest["expected_outputs"] == v2_manifest["expected_outputs"]


def test_callable_registry_entry_uses_compatibility_adapter(tmp_outdir, tiny_cfg):
    sentinel = {"called": False}

    def _legacy_callable(base, out_dir, dry_run=False, **_):
        sentinel["called"] = True
        from pan_lab.reporting import ExperimentReporter

        return ExperimentReporter("legacy_callable", out_dir)

    EXPERIMENT_REGISTRY["legacy_callable"] = _legacy_callable
    try:
        rep = run_experiment("legacy_callable", tiny_cfg, str(tmp_outdir), dry_run=True)
    finally:
        del EXPERIMENT_REGISTRY["legacy_callable"]

    assert sentinel["called"] is True
    assert rep.name == "legacy_callable"


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


def test_yaml_validation_fails_on_unknown_top_level_key(tmp_outdir):
    p = tmp_outdir / "invalid_top_level.yaml"
    p.write_text(
        yaml.dump(
            {
                "schema_version": 2,
                "experiment": "k_sweep",
                "base": {"p": 11, "k_freqs": 3},
                "grid": {"ks": [2], "seeds": [42]},
                "extra": {"nope": True},
            }
        )
    )
    with pytest.raises(ExperimentSpecValidationError, match="Unknown top-level YAML keys"):
        load_experiment_yaml(str(p))


def test_yaml_validation_fails_on_unknown_plugin_names(tmp_outdir):
    p = tmp_outdir / "invalid_plugins.yaml"
    p.write_text(
        yaml.dump(
            {
                "schema_version": 2,
                "experiment": "k_sweep",
                "base": {"p": 11, "k_freqs": 3},
                "grid": {"ks": [2], "seeds": [42]},
                "analyses": ["does_not_exist"],
                "plots": [{"type": "nope_plot"}],
            }
        )
    )
    with pytest.raises(ExperimentSpecValidationError, match="Unknown analyzer plugin names"):
        load_experiment_yaml(str(p))
