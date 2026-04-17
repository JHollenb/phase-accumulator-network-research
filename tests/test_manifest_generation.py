from __future__ import annotations

import json
from pathlib import Path

from pan_lab.experiments import build_execution_manifest, load_experiment_yaml


def _snapshot_view(manifest: dict) -> dict:
    runs = manifest["expanded_runs"]
    picked = runs[:3]
    if len(runs) > 3:
        picked = picked + [runs[-1]]
    return {
        "experiment": manifest["experiment"],
        "out_dir": manifest["out_dir"],
        "run_count": len(runs),
        "run_preview": [
            {
                "run_id": row["run_id"],
                "seed": row["config"]["seed"],
                "k_freqs": row["config"]["k_freqs"],
                "label": row["config"]["label"],
                "save_model": row["config"]["save_model"],
            }
            for row in picked
        ],
        "capture": manifest["capture"],
        "analyses": manifest["analyses"],
        "plots": manifest["plots"],
        "expected_outputs": manifest["expected_outputs"],
    }


def test_manifest_generation_snapshots_for_baseline_experiments():
    expected_path = Path("tests/fixtures/manifest_snapshots.json")
    expected = json.loads(expected_path.read_text())
    actual: dict[str, dict] = {}

    for yaml_path in ("experiments/k_sweep.yaml", "experiments/decoder_analysis.yaml"):
        name, base, out_dir, _dry_run, exp_args = load_experiment_yaml(yaml_path)
        manifest = build_execution_manifest(name, base, out_dir, exp_args)
        actual[yaml_path] = _snapshot_view(manifest)

    assert actual == expected


def test_manifest_metadata_matches_for_factory_and_legacy_style_inputs(tmp_path):
    from pan_lab import RunConfig

    base = RunConfig.from_dict({"p": 31, "k_freqs": 5, "n_steps": 50, "seed": 7})
    out_dir = str(tmp_path / "k_sweep")
    legacy_like_args = {"ks": [2, 3], "seeds": [1, 2]}
    v2_like_args = {
        "_schema_version": 2,
        "ks": [2, 3],
        "seeds": [1, 2],
        "capture": {"ablations": True},
        "analyses": [],
        "plots": [],
    }

    legacy_manifest = build_execution_manifest("k_sweep", base, out_dir, legacy_like_args)
    v2_manifest = build_execution_manifest("k_sweep", base, out_dir, v2_like_args)

    assert legacy_manifest["expanded_runs"] == v2_manifest["expanded_runs"]
    assert legacy_manifest["capture"]["ablations"] is False
    assert v2_manifest["capture"]["ablations"] is True
    assert "ablations.csv" not in legacy_manifest["expected_outputs"]
    assert "ablations.csv" in v2_manifest["expected_outputs"]
