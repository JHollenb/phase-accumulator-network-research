"""
pan_lab.experiments — registered experiments, driven by YAML configs.

Most experiments are *grid sweeps* — they enumerate a Cartesian product of
(parameter, seed), train each, and plot a few figures. All of those go
through the single `grid_sweep` entry (see pan_lab.grid_sweep), so adding
a new sweep is a YAML file, not Python.

Three experiments remain as bespoke functions because they do custom
post-training analysis that doesn't fit a grid:

    sifp16_inference   — train a PAN, then eval with SIFP-16 fake-quant
    decoder_swap       — train a PAN, then swap in a theoretical Fourier
                         decoder and measure the accuracy delta
    decoder_analysis   — train a PAN, then decompose the learned decoder
                         onto a Clock + harmonic basis

YAML schema (used by grid_sweep):

    experiment: grid_sweep
    out_dir:    results/k8_sweep
    base:
      p: 113
      k_freqs: 8
      model_kind: pan
      n_steps: 200000
      weight_decay: 0.01
    grid:                       # dict → Cartesian product; or list-of-dicts
      seed: [42, 123, 456, ...]
    options:
      ablations: false
      slots:     false
      hooks:     []             # e.g. [checkpoint_logger]
    plots:
      - {type: training_curves, title: "K=8 — all seeds"}
"""
from __future__ import annotations

import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import yaml
import pandas as pd

from pan_lab.config     import DEVICE, RunConfig, TWO_PI
from pan_lab.data       import make_modular_dataset, make_dataset_from_cfg
from pan_lab.hooks      import CSVStreamLogger
from pan_lab.models     import make_model
from pan_lab.models.quantize import apply_sifp16_to_pan
from pan_lab.reporting import ExperimentReporter, save_model_weights
from pan_lab.trainer    import train

EXPERIMENT_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    """Decorator that registers an experiment function under a given name."""
    def _wrap(fn):
        EXPERIMENT_REGISTRY[name] = fn
        return fn
    return _wrap


# ─────────────────────────────────────────────────────────────────────────────
def _move(tensors, device):
    return tuple(t.to(device) for t in tensors)


def _print_plan(cfgs: List[RunConfig], name: str) -> None:
    print(f"\n══ {name} — dry-run plan ({len(cfgs)} sub-runs) ══")
    print(f"  device: {DEVICE}")
    for c in cfgs:
        print(f"  - {c.display_id():<22} "
              f"p={c.p} k={c.k_freqs} task={c.task_kind} model={c.model_kind} "
              f"seed={c.seed} steps={c.n_steps:,} wd={c.weight_decay} "
              f"dw={c.diversity_weight} freq_init={c.freq_init}")
    total_steps = sum(c.n_steps for c in cfgs)
    print(f"  total planned steps: {total_steps:,}")


def _run_cfgs(
    cfgs, name, out_dir, dry_run,
    hook_factory=None, ablations=True, slots=False,
    metrics=True, metrics_expensive_every=5000,
    metrics_gate_decode_max_rows=4000,
    metrics_logit_spectrum=False,
    metrics_logit_spectrum_classes=None,
    post_run_hook=None,
    workers: int = 1,
    hook_names: Optional[List[str]] = None,
):
    """
    Shared engine for every registered experiment: take a list of
    RunConfigs, train each, and collect results into one ExperimentReporter.

    Dry-runs print the plan and return an empty reporter. Otherwise each
    cfg: builds its dataset, constructs the model, attaches any hooks
    requested by `hook_factory(cfg)` plus a CSVStreamLogger, trains, and
    adds the result (with optional ablations/slots/save_model) to the
    reporter. At the end, the reporter writes all CSVs and prints a
    summary.

    `workers > 1` runs cfgs in a ProcessPoolExecutor (spawn). Each cfg's
    seed already pins per-run determinism, so parallel and sequential
    runs produce identical CSVs (modulo row ordering, which the reporter
    sorts canonically before write). Hook construction in workers uses
    `hook_names` because the sequential `hook_factory` lambda is not
    picklable across spawned processes.
    """
    rep = ExperimentReporter(name=name, out_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if dry_run:
        _print_plan(cfgs, name)
        return rep

    if workers <= 1:
        stream_path = os.path.join(out_dir, "curves_stream.csv")
        if os.path.exists(stream_path):
            os.remove(stream_path)

        for cfg in cfgs:
            tx, ty, vx, vy = make_dataset_from_cfg(cfg)
            model  = make_model(cfg).to(DEVICE)
            hooks  = list(hook_factory(cfg)) if hook_factory else []
            hooks.append(CSVStreamLogger(stream_path, run_id=cfg.display_id()))
            if metrics and cfg.model_kind == "pan":
                from pan_lab.metrics import MetricsLogger
                hooks.append(MetricsLogger(
                    val_x=vx, val_y=vy,
                    expensive_every=metrics_expensive_every,
                    gate_decode_max_rows=metrics_gate_decode_max_rows,
                    logit_spectrum=metrics_logit_spectrum,
                    logit_spectrum_classes=metrics_logit_spectrum_classes,
                ))
            result = train(model, cfg, tx, ty, vx, vy, hooks=hooks, verbose=True)
            rep.add_run(
                result, val_x=vx, val_y=vy,
                ablations=ablations and cfg.model_kind == "pan",
                slots=slots and cfg.model_kind == "pan",
            )

            # ★ NEW: honor cfg.save_model
            if cfg.save_model:
                path = save_model_weights(result, out_dir)
                print(f"  saved model weights: {path}")

            # Stream the accumulated CSVs + manifest to disk after each run
            # so a crash or ^C mid-sweep doesn't discard completed work.
            rep.flush()
            if post_run_hook is not None:
                post_run_hook(rep)

        rep.print_summary()
        return rep

    # ── Parallel path ────────────────────────────────────────────────
    workers_root = os.path.join(out_dir, "_workers")
    if os.path.isdir(workers_root):
        shutil.rmtree(workers_root)

    canonical_stream = os.path.join(out_dir, "curves_stream.csv")
    if os.path.exists(canonical_stream):
        os.remove(canonical_stream)

    hook_names = list(hook_names or [])
    print(f"[{name}] running {len(cfgs)} configs across {workers} workers (spawn)")

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {
            pool.submit(
                _run_one_cfg,
                cfg, name, out_dir,
                hook_names,
                ablations, slots, metrics,
                metrics_expensive_every,
                metrics_gate_decode_max_rows,
                metrics_logit_spectrum,
                metrics_logit_spectrum_classes,
            ): cfg
            for cfg in cfgs
        }
        for future in as_completed(futures):
            cfg = futures[future]
            buffers = future.result()
            _merge_into_reporter(rep, buffers)
            if buffers.get("saved_model_path"):
                print(f"  saved model weights: {buffers['saved_model_path']}")
            rep.flush()

    _merge_worker_streams(out_dir)

    if post_run_hook is not None:
        post_run_hook(rep)

    rep.print_summary()
    return rep


def _run_one_cfg(
    cfg, name, out_dir,
    hook_names,
    ablations, slots, metrics,
    metrics_expensive_every,
    metrics_gate_decode_max_rows,
    metrics_logit_spectrum,
    metrics_logit_spectrum_classes,
) -> dict:
    """
    Worker target (must be importable / picklable for spawn). Trains one
    cfg in isolation, runs per-cfg analyses (ablations / slots / metrics)
    inside this process so the parent only does cheap merging, and
    returns a buffers dict the parent splices into its reporter.
    """
    from pan_lab.grid_sweep import HOOK_REGISTRY

    worker_dir  = os.path.join(out_dir, "_workers", cfg.display_id())
    os.makedirs(worker_dir, exist_ok=True)
    stream_path = os.path.join(worker_dir, "curves_stream.csv")
    if os.path.exists(stream_path):
        os.remove(stream_path)

    tx, ty, vx, vy = make_dataset_from_cfg(cfg)
    model = make_model(cfg).to(DEVICE)
    hooks = [HOOK_REGISTRY[h]() for h in hook_names]
    hooks.append(CSVStreamLogger(stream_path, run_id=cfg.display_id()))
    if metrics and cfg.model_kind == "pan":
        from pan_lab.metrics import MetricsLogger
        hooks.append(MetricsLogger(
            val_x=vx, val_y=vy,
            expensive_every=metrics_expensive_every,
            gate_decode_max_rows=metrics_gate_decode_max_rows,
            logit_spectrum=metrics_logit_spectrum,
            logit_spectrum_classes=metrics_logit_spectrum_classes,
        ))

    result = train(model, cfg, tx, ty, vx, vy, hooks=hooks, verbose=True)

    mini_rep = ExperimentReporter(name=name, out_dir=worker_dir)
    mini_rep.add_run(
        result, val_x=vx, val_y=vy,
        ablations=ablations and cfg.model_kind == "pan",
        slots=slots and cfg.model_kind == "pan",
    )

    saved_model_path = None
    if cfg.save_model:
        saved_model_path = save_model_weights(result, out_dir)

    return {
        "display_id":       cfg.display_id(),
        "stream_path":      stream_path,
        "saved_model_path": saved_model_path,
        "runs":         mini_rep._runs,
        "curves":       mini_rep._curves,
        "checkpoints":  mini_rep._checkpoints,
        "ablations":    mini_rep._ablations,
        "slots":        mini_rep._slots,
        "metrics":      mini_rep._metrics,
        "provenance":   mini_rep._provenance,
    }


def _merge_into_reporter(rep: ExperimentReporter, buffers: dict) -> None:
    rep._runs.extend(buffers["runs"])
    rep._curves.extend(buffers["curves"])
    rep._checkpoints.extend(buffers["checkpoints"])
    rep._ablations.extend(buffers["ablations"])
    rep._slots.extend(buffers["slots"])
    rep._metrics.extend(buffers["metrics"])
    if rep._provenance is None and buffers.get("provenance") is not None:
        rep._provenance = buffers["provenance"]


def _merge_worker_streams(out_dir: str) -> None:
    """Concatenate per-worker curves_stream.csv files into one canonical file."""
    workers_root = os.path.join(out_dir, "_workers")
    if not os.path.isdir(workers_root):
        return
    streams = sorted(
        os.path.join(workers_root, d, "curves_stream.csv")
        for d in os.listdir(workers_root)
        if os.path.exists(os.path.join(workers_root, d, "curves_stream.csv"))
    )
    if not streams:
        return

    import csv
    with open(os.path.join(out_dir, "curves_stream.csv"), "w", newline="") as out:
        writer = csv.writer(out)
        wrote_header = False
        for path in streams:
            with open(path, newline="") as inp:
                reader = csv.reader(inp)
                header = next(reader, None)
                if header is None:
                    continue
                if not wrote_header:
                    writer.writerow(header)
                    wrote_header = True
                writer.writerows(reader)

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

# grid_sweep subsumes what used to be 13 near-identical sweep functions
# (compare, k_sweep, dw_sweep, wd_sweep, k8_sweep, primes, held_out_primes,
# tier3, slot_census, freq_init_ablation, tf_sweep, mod_mul, mod_two_step).
# Each of those is now a YAML that dispatches here.
from pan_lab.grid_sweep import run_grid_sweep
register("grid_sweep")(run_grid_sweep)


# ─────────────────────────────────────────────────────────────────────────────
@register("sifp16_inference")
def exp_sifp16_inference(base: RunConfig, out_dir: str,
                         dry_run: bool = False,
                         seeds: Optional[List[int]] = None,
                         **_) -> ExperimentReporter:
    """
    Experiment E — SIFP-16 quantization at inference.

    Train a normal PAN, then evaluate it both un-quantized and with
    SIFP-16 fake-quant applied to every phase computation. The
    accuracy delta is the hardware-relevance number.
    """
    import pandas as pd
    seeds = seeds or [42, 123, 456]
    cfgs = [
        base.with_overrides(model_kind="pan", seed=s, weight_decay=0.01,
                             label=f"sifp-s{s}")
        for s in seeds
    ]

    if dry_run:
        _print_plan(cfgs, "sifp16_inference")
        return ExperimentReporter("sifp16_inference", out_dir)

    rep = ExperimentReporter("sifp16_inference", out_dir)
    quant_rows = []
    for cfg in cfgs:
        tx, ty, vx, vy = make_modular_dataset(
            p=cfg.p, task_kind=cfg.task_kind,
            train_frac=cfg.train_frac, seed=cfg.seed)
        model = make_model(cfg).to(DEVICE)
        result = train(model, cfg, tx, ty, vx, vy, verbose=True)
        rep.add_run(result, val_x=vx, val_y=vy, ablations=True)

        # Apply SIFP-16 to a copy of the trained model and eval.
        import copy
        qmodel = copy.deepcopy(result.model)
        qmodel.eval()
        apply_sifp16_to_pan(qmodel)
        with torch.no_grad():
            logits = qmodel(vx)
            acc_q  = float((logits.argmax(-1) == vy).float().mean().item())
        acc_fp = result.history.val_acc[-1] if result.history.val_acc else 0.0
        quant_rows.append({
            "run_id": cfg.display_id(),
            "seed":   cfg.seed,
            "val_acc_fp32":    acc_fp,
            "val_acc_sifp16":  acc_q,
            "delta":           acc_q - acc_fp,
        })

        # Stream per-seed: rewrite the bespoke CSV + flush standard ones
        # so a crash after seed N preserves seeds 0..N.
        pd.DataFrame(quant_rows).to_csv(
            os.path.join(out_dir, "quant_eval.csv"), index=False)
        rep.flush()

    quant_df = pd.DataFrame(quant_rows)
    print("\n── SIFP-16 quantization eval ──")
    print(quant_df.to_string(index=False))
    rep.write_manifest()
    return rep


# ─────────────────────────────────────────────────────────────────────────────
@register("decoder_swap")
def exp_decoder_swap(base: RunConfig, out_dir: str,
                     dry_run: bool = False,
                     seeds: Optional[List[int]] = None,
                     **_) -> ExperimentReporter:
    """
    Experiment I — at grokking, swap the learned decoder for a fixed
    theoretical Fourier decoder and measure accuracy. If accuracy
    holds, the rest of the network has converged to a canonical
    Fourier circuit (and the decoder is just reading it out).
    """
    import pandas as pd
    seeds = seeds or [42, 123, 456]
    cfgs = [base.with_overrides(model_kind="pan", seed=s,
                                 weight_decay=0.01, label=f"swap-s{s}")
            for s in seeds]

    if dry_run:
        _print_plan(cfgs, "decoder_swap")
        return ExperimentReporter("decoder_swap", out_dir)

    rep = ExperimentReporter("decoder_swap", out_dir)
    swap_rows = []
    for cfg in cfgs:
        tx, ty, vx, vy = make_modular_dataset(
            p=cfg.p, task_kind=cfg.task_kind,
            train_frac=cfg.train_frac, seed=cfg.seed)
        model = make_model(cfg).to(DEVICE)
        result = train(model, cfg, tx, ty, vx, vy, verbose=True)
        rep.add_run(result, val_x=vx, val_y=vy, ablations=False)
        pan = result.model

        # Build the theoretical Fourier decoder: for each output class c
        # and each learned frequency k, the ideal log-weight is
        # cos(c * f_k). We L2-normalize each row for scale invariance.
        with torch.no_grad():
            theta = torch.arange(cfg.p, device=DEVICE).float().unsqueeze(-1)   # (P, 1)
            f     = pan.encoders[0].freq.detach()                              # (K,)
            decoder_fourier = torch.cos(theta * f.unsqueeze(0))                # (P, K)
            # Normalize each row
            n = decoder_fourier.norm(dim=1, keepdim=True).clamp(min=1e-8)
            decoder_fourier = decoder_fourier / n

            # Swap in
            saved_w = pan.decoder.weight.data.clone()
            saved_b = pan.decoder.bias.data.clone()
            pan.decoder.weight.data = decoder_fourier
            pan.decoder.bias.data.zero_()

            logits = pan(vx)
            acc_swap = float((logits.argmax(-1) == vy).float().mean().item())
            pan.decoder.weight.data.copy_(saved_w)
            pan.decoder.bias.data.copy_(saved_b)

        acc_learned = result.history.val_acc[-1] if result.history.val_acc else 0.0
        swap_rows.append({
            "run_id": cfg.display_id(),
            "seed":   cfg.seed,
            "val_acc_learned_decoder": acc_learned,
            "val_acc_fourier_decoder": acc_swap,
            "delta":                   acc_swap - acc_learned,
        })

        # Stream per-seed: rewrite the bespoke CSV + flush standard ones
        # so a crash after seed N preserves seeds 0..N.
        pd.DataFrame(swap_rows).to_csv(
            os.path.join(out_dir, "decoder_swap.csv"), index=False)
        rep.flush()

    swap_df = pd.DataFrame(swap_rows)
    print("\n── Decoder-swap eval ──")
    print(swap_df.to_string(index=False))
    rep.write_manifest()
    return rep


# ─────────────────────────────────────────────────────────────────────────
# Decoder Experiment Helpers
# ─────────────────────────────────────────────────────────────────────────

"""
patch_decoder_analysis_v2.py

Critical update to decoder_analysis.py based on the following finding:

  Even an optimal linear decoder reading from K=9 pure-Clock-pair
  gate channels (cos-only) cannot exceed ~66% accuracy on P=113.

  But real grokked PANs achieve 99%. So the circuit is NOT pure Clock.

The residual spectrum analysis confirms this: seed 42 has clock_explained
= 0.66 (energy) but acc_clock_only = 0.07 (classification). The 33% of
unexplained energy contains the discriminative information.

Two hypotheses for what the "extra" structure is:

  (A) HARMONICS: when the mixing matrix has multiple channels at the
      same frequency f with different phi_ref, the decoder can
      combine them to reconstruct harmonics cos(2f*c), cos(3f*c), ...
      via trigonometric identities. This would explain why seed 42
      converges to 4 channels at k=3 and 2 channels at k=1 — it's
      building a 4-term Fourier series at k=3.

  (B) ALIASING: the mod-2pi operation inside phase_mix can produce
      frequency content at integer multiples of the input frequencies
      when weights aren't strictly +1/-1. The learned mixing weights
      are near +1 but not exactly +1, so small aliasing contributions
      could account for discriminative accuracy.

This patch extends decoder_analysis with TWO new tests:

  1. HARMONIC_BASIS: for each effective frequency f_eff_j, add
     cos/sin columns not just at f_eff_j but at 2*f_eff_j, 3*f_eff_j,
     up to harmonic_order=4. Re-project. If harmonic basis fully
     explains the decoder, (A) wins.

  2. SAMPLE_GATE_SPACE: compute the actual gate outputs across all
     P*P input pairs. Fit a linear decoder via least squares directly
     on the gate activations. This gives an upper bound on what any
     decoder with perfect knowledge could do — and if it matches
     acc_learned, the gate output itself is sufficient. If it doesn't,
     the decoder has access to information the gate abstraction hides.

Add the contents of this file as a new function in decoder_analysis.py
and call it alongside the existing basis projection.
"""

def _build_harmonic_basis(p: int, f_channels: np.ndarray,
                           phi_ref: np.ndarray,
                           harmonic_order: int = 4) -> np.ndarray:
    """
    Like _build_clock_basis but includes harmonics 1, 2, ..., H at each
    effective channel frequency. Returns (P, 2*K*H).

    For each channel j with (f_j, phi_j):
        cos(h*f_j*c - phi_j), sin(h*f_j*c - phi_j)  for h = 1..H
    """
    P = p
    K = len(f_channels)
    c = np.arange(P, dtype=np.float64).reshape(P, 1)
    cols = []
    for h in range(1, harmonic_order + 1):
        f   = (h * f_channels).reshape(1, K)
        phi = phi_ref.reshape(1, K)
        arg = f * c - phi
        cols.append(np.cos(arg))
        cols.append(np.sin(arg))
    return np.concatenate(cols, axis=1)


def analyze_harmonics(pan_model, val_x: torch.Tensor, val_y: torch.Tensor,
                       harmonic_order: int = 4) -> dict:
    """
    Fit the learned decoder to a harmonic basis at each effective
    channel frequency. If val_acc using the projection matches the
    learned baseline, the circuit is Clock-plus-harmonics.
    """

    with torch.no_grad():
        W_mix     = pan_model.phase_mix.weight.detach().cpu().numpy()
        f0        = pan_model.encoders[0].freq.detach().cpu().numpy()
        f1        = pan_model.encoders[1].freq.detach().cpu().numpy()
        phi       = (pan_model.phase_gate.ref_phase.detach()
                     .cpu().numpy() % TWO_PI)
        W_learned = pan_model.decoder.weight.detach().cpu().numpy()

    f_eff = _channel_effective_frequency(W_mix, f0, f1)

    out = {"f_eff": f_eff.tolist(),
            "harmonic_order": harmonic_order,
            "baseline_learned": None,
            "clock_only": None,
            "harmonic_fits": {}}

    with torch.no_grad():
        out["baseline_learned"] = float(
            (pan_model(val_x).argmax(-1) == val_y).float().mean().item())

    # Clock-only (h=1 only), for reference
    B1 = _build_clock_basis(pan_model.p, f_eff, phi)
    W_fit, resid = _project_onto_basis(W_learned, B1)
    acc = _evaluate_decoder(pan_model, W_fit, val_x, val_y)
    explained = 1 - (resid**2).sum() / (W_learned**2).sum()
    out["clock_only"] = {"acc": acc, "explained_frac": float(explained),
                          "n_basis_cols": B1.shape[1]}

    # Incrementally add harmonics
    for H in range(2, harmonic_order + 1):
        BH = _build_harmonic_basis(pan_model.p, f_eff, phi, harmonic_order=H)
        W_fit, resid = _project_onto_basis(W_learned, BH)
        acc = _evaluate_decoder(pan_model, W_fit, val_x, val_y)
        explained = 1 - (resid**2).sum() / (W_learned**2).sum()
        out["harmonic_fits"][f"H={H}"] = {
            "acc": acc,
            "explained_frac": float(explained),
            "n_basis_cols": BH.shape[1],
        }
    return out


"""
patch_gate_upper_bound_logreg.py

BUG FOUND: analyze_gate_space_upper_bound used OLS on one-hot targets
to compute the "optimal linear decoder." OLS minimizes MSE, not
argmax accuracy. For the K=9 gate representation on mod-113, OLS
decoding achieves ~39% while logistic regression on the SAME
features achieves ~100%. The "upper bound" was not an upper bound —
it was OLS-specific.

This leads to false conclusions like seed 42's gate_optimal=0.262
suggesting the gate representation lacks discriminative info, when
in fact the gate is fully linearly-decodable.

FIX: use sklearn's LogisticRegression (multinomial, high C for
near-unregularized fit) to compute the true linear argmax upper bound.

APPLY BY replacing analyze_gate_space_upper_bound in
patch_decoder_analysis_harmonics.py (or in the main decoder_analysis.py
if already merged) with this version.
"""

def analyze_gate_space_upper_bound(pan_model, val_x, val_y, p: int) -> dict:
    """
    Measure the TRUE linear-decoder upper bound on the gate
    representation.

    Uses multinomial logistic regression (softmax cross-entropy),
    which is the optimal linear argmax fit. NOT OLS — OLS on
    one-hot targets minimizes MSE, which is a weak proxy for
    argmax accuracy.

    Returns:
      gate_optimal_acc   — accuracy of true optimal linear decoder
      gate_ols_acc       — accuracy via OLS (for comparison; shows
                            how much the decoder's discriminative
                            power is obscured by the MSE objective)
      learned_acc        — the trained network's accuracy
    """
    from sklearn.linear_model import LogisticRegression

    P = p
    device = next(pan_model.parameters()).device

    a_all = torch.arange(P, device=device).repeat_interleave(P)
    b_all = torch.arange(P, device=device).repeat(P)
    x_all = torch.stack([a_all, b_all], dim=1)
    y_all = (a_all + b_all) % P

    with torch.no_grad():
        phases = [enc(x_all[:, i]) for i, enc in enumerate(pan_model.encoders)]
        concat = torch.cat(phases, dim=-1)
        mixed  = pan_model.phase_mix(concat)
        gates  = pan_model.phase_gate(mixed)

    G = gates.cpu().numpy()
    y = y_all.cpu().numpy()

    # OLS for comparison with prior numbers
    T = np.eye(P)[y]
    W_ols, *_ = np.linalg.lstsq(G, T, rcond=None)
    ols_acc = (np.argmax(G @ W_ols, axis=1) == y).mean()

    # Logistic regression — the true optimal linear decoder for argmax
    # C=1e6 means essentially unregularized (large inverse-strength).
    # max_iter raised to ensure convergence.
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        C=1e6,
        fit_intercept=True,
    )
    lr.fit(G, y)
    lr_acc = lr.score(G, y)

    with torch.no_grad():
        learned_acc = float(
            (pan_model(val_x).argmax(-1) == val_y).float().mean().item())

    return {
        "gate_optimal_acc":  float(lr_acc),
        "gate_ols_acc":      float(ols_acc),
        "learned_acc":       float(learned_acc),
        "gap_from_optimal":  float(learned_acc - lr_acc),
    }

def _channel_effective_frequency(W_mix: np.ndarray,
                                  enc0_freq: np.ndarray,
                                  enc1_freq: np.ndarray) -> np.ndarray:
    """
    For each output channel j in the phase-mixing matrix, estimate the
    effective frequency as the circular mean of the top-1 enc0 slot's
    frequency and the top-1 enc1 slot's frequency.

    Returns (K,) float array in [0, 2pi).
    """
    K = W_mix.shape[0]
    out = np.zeros(K, dtype=np.float64)
    for j in range(K):
        abs_row = np.abs(W_mix[j])
        top_e0  = int(np.argmax(abs_row[:K]))
        top_e1  = int(np.argmax(abs_row[K:]))
        f0 = float(enc0_freq[top_e0] % TWO_PI)
        f1 = float(enc1_freq[top_e1] % TWO_PI)
        s  = (np.sin(f0) + np.sin(f1)) / 2
        c  = (np.cos(f0) + np.cos(f1)) / 2
        out[j] = float(np.arctan2(s, c) % TWO_PI)
    return out


def _build_clock_basis(p: int, f_channels: np.ndarray,
                       phi_ref: np.ndarray) -> np.ndarray:
    """
    Build the Clock basis matrix B of shape (P, 2K).
    Columns 0..K-1:  cos(f_j * c - phi_j)
    Columns K..2K-1: sin(f_j * c - phi_j)

    Using both cos and sin lets us match any phase offset the learned
    decoder may have absorbed — without sin we'd only capture decoders
    aligned to exactly the learned phi_ref, which is overly strict.
    """
    P = p
    K = len(f_channels)
    c = np.arange(P, dtype=np.float64).reshape(P, 1)        # (P, 1)
    f = f_channels.reshape(1, K)                            # (1, K)
    phi = phi_ref.reshape(1, K)                             # (1, K)
    arg = f * c - phi                                        # (P, K)
    B = np.concatenate([np.cos(arg), np.sin(arg)], axis=1)  # (P, 2K)
    return B


def _project_onto_basis(W: np.ndarray, B: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Column-space projection: for each learned decoder column w_j (one
    per channel), find the best fit in the column span of B.

    Returns (W_proj, residual), same shape as W.
    """
    # We fit per-column: for column j of W, solve
    #   min_alpha  || B @ alpha - W[:, j] ||^2
    # in closed form via lstsq.
    alpha, *_ = np.linalg.lstsq(B, W, rcond=None)   # (2K, K)
    W_proj    = B @ alpha                            # (P, K)
    residual  = W - W_proj
    return W_proj, residual


def _residual_spectrum(residual: np.ndarray, p: int
                       ) -> dict[str, np.ndarray]:
    """
    FFT each column of the residual, take magnitudes, average across
    columns. Returns keyed arrays for csv export.
    """
    # residual shape (P, K); FFT along axis=0 gives (P, K) complex
    F = np.fft.fft(residual, axis=0)
    mag = np.abs(F).mean(axis=1)                     # (P,)
    # FFT bins correspond to k = 0..P-1 on integers; the "angular"
    # frequency at bin k is 2*pi*k/P
    return {
        "k":         np.arange(p),
        "magnitude": mag,
    }


def _evaluate_decoder(pan, W_new: np.ndarray,
                       val_x: torch.Tensor, val_y: torch.Tensor
                       ) -> float:
    """Swap W_new into pan.decoder.weight, zero bias, eval, restore."""
    saved_w = pan.decoder.weight.data.clone()
    saved_b = pan.decoder.bias.data.clone()
    with torch.no_grad():
        pan.decoder.weight.data = torch.tensor(
            W_new, dtype=torch.float32, device=DEVICE)
        pan.decoder.bias.data.zero_()
        logits = pan(val_x)
        acc = float((logits.argmax(-1) == val_y).float().mean().item())
        pan.decoder.weight.data.copy_(saved_w)
        pan.decoder.bias.data.copy_(saved_b)
    return acc


# ─────────────────────────────────────────────────────────────────────────
# The experiment
# ─────────────────────────────────────────────────────────────────────────
@register("decoder_analysis")
def exp_decoder_analysis(base: RunConfig, out_dir: str,
                          dry_run: bool = False,
                          seeds: Optional[List[int]] = None,
                          max_extra_freqs: int = 20,
                          **_) -> ExperimentReporter:
    """
    For each seed, train PAN, then decompose the learned decoder:

      0. Measure learned-decoder val_acc (the baseline).
      1. Build Clock basis from the mixing matrix and gate phases.
      2. Project learned decoder onto Clock basis.
         - explained_energy_fraction = 1 - ||R||^2 / ||W||^2
         - val_acc_clock_only       = eval(W_proj)
      3. Analyze residual spectrum (FFT per column, average magnitudes).
      4. Basis expansion: iteratively add (cos, sin) pairs at the
         top-residual-magnitude k values. For each step, refit and
         re-evaluate val_acc. Stops when val_acc within 1% of learned
         OR max_extra_freqs reached.

    Writes:
      decoder_analysis.csv          — one row per seed: summary metrics
      decoder_recovery_curve.csv    — long: (seed, n_extras, val_acc)
      decoder_residual_spectrum.csv — long: (seed, k, magnitude)
    """
    seeds = seeds or [42, 123, 456]
    cfgs  = [base.with_overrides(model_kind="pan", seed=s,
                                  weight_decay=0.01, save_model=True,
                                  label=f"danal-s{s}") for s in seeds]

    if dry_run:
        _print_plan(cfgs, "decoder_analysis")
        return ExperimentReporter("decoder_analysis", out_dir)

    rep = ExperimentReporter("decoder_analysis", out_dir)
    summary_rows:  list[dict] = []
    recovery_rows: list[dict] = []
    spectrum_rows: list[dict] = []

    def _flush_bespoke() -> None:
        """Rewrite the three per-seed CSVs + flush ExperimentReporter."""
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "decoder_analysis.csv"), index=False)
        pd.DataFrame(recovery_rows).to_csv(
            os.path.join(out_dir, "decoder_recovery_curve.csv"), index=False)
        pd.DataFrame(spectrum_rows).to_csv(
            os.path.join(out_dir, "decoder_residual_spectrum.csv"), index=False)
        rep.flush()

    for cfg in cfgs:
        tx, ty, vx, vy = make_modular_dataset(
            p=cfg.p, task_kind=cfg.task_kind,
            train_frac=cfg.train_frac, seed=cfg.seed)
        model  = make_model(cfg).to(DEVICE)
        result = train(model, cfg, tx, ty, vx, vy, verbose=True)
        rep.add_run(result, val_x=vx, val_y=vy, ablations=False)
        pan = result.model

        # ── 0. Baseline ──
        with torch.no_grad():
            acc_learned = float(
                (pan(vx).argmax(-1) == vy).float().mean().item())
            W_learned = pan.decoder.weight.detach().cpu().numpy()  # (P, K)

        if acc_learned < 0.95:
            print(f"  [{cfg.display_id()}] skipped — not grokked "
                  f"(learned={acc_learned:.3f})")
            summary_rows.append({
                "run_id":     cfg.display_id(),
                "seed":       cfg.seed,
                "grokked":    False,
                "acc_learned": acc_learned,
            })
            _flush_bespoke()
            continue

        # ── 1. Extract circuit parameters ──
        W_mix = pan.phase_mix.weight.detach().cpu().numpy()         # (K, 2K)
        f0    = pan.encoders[0].freq.detach().cpu().numpy()
        f1    = pan.encoders[1].freq.detach().cpu().numpy()
        phi   = (pan.phase_gate.ref_phase.detach().cpu().numpy()
                  % TWO_PI)
        f_eff = _channel_effective_frequency(W_mix, f0, f1)         # (K,)
        K     = cfg.k_freqs
        P     = cfg.p

        # ── 2. Clock projection ──
        B_clock = _build_clock_basis(P, f_eff, phi)                 # (P, 2K)
        W_proj, residual = _project_onto_basis(W_learned, B_clock)
        acc_clock = _evaluate_decoder(pan, W_proj, vx, vy)

        energy_total    = float((W_learned ** 2).sum())
        energy_residual = float((residual ** 2).sum())
        explained_frac  = 1.0 - energy_residual / max(energy_total, 1e-12)

        # ── 3. Residual spectrum ──
        spec = _residual_spectrum(residual, P)
        for k, mag in zip(spec["k"], spec["magnitude"]):
            spectrum_rows.append({
                "run_id":    cfg.display_id(),
                "seed":      cfg.seed,
                "k":         int(k),
                "magnitude": float(mag),
            })

        # ── 4. Basis expansion ──
        # Sort FFT bins by magnitude (exclude DC and bins already close
        # to channel frequencies). Angular frequency at bin k is
        # 2*pi*k/P. We add cos/sin pairs at those angular freqs with
        # phi=0 (we'll refit phases via projection).
        mag = spec["magnitude"].copy()
        mag[0] = 0.0                              # exclude DC
        # Exclude bins that already correspond to an existing channel freq
        for fj in f_eff:
            # Find nearest FFT bin
            nearest_k = int(round(fj * P / TWO_PI)) % P
            mag[nearest_k] = 0.0
            # Also suppress the conjugate bin (FFT symmetry)
            conj = (-nearest_k) % P
            mag[conj] = 0.0

        # ── 5. Harmonic fit ──
        harm = analyze_harmonics(pan, vx, vy, harmonic_order=4)
        print(f"  [{cfg.display_id()}] harmonic fit:")
        for key, v in sorted(harm["harmonic_fits"].items()):
            print(f"    {key}: acc={v['acc']:.3f}  "
                  f"explained={v['explained_frac']:.3f}  "
                  f"basis_cols={v['n_basis_cols']}")
        clock = harm["clock_only"]
        print(f"    Clock-only (H=1): acc={clock['acc']:.3f}  "
              f"explained={clock['explained_frac']:.3f}")

        # ── 6. Gate-space upper bound ──
        gate_ub = analyze_gate_space_upper_bound(pan, vx, vy, p=cfg.p)
        print(f"  [{cfg.display_id()}] optimal linear decoder on gates: "
              f"{gate_ub['gate_optimal_acc']:.3f}  "
              f"{gate_ub['gate_ols_acc']:.3f}  "
              f"(learned: {gate_ub['learned_acc']:.3f}, "
              f"gap: {gate_ub['gap_from_optimal']:+.3f})")

        # Top-residual frequencies in descending order
        candidate_order = np.argsort(mag)[::-1]
        # Take only bins in the first half (real signal -> symmetric
        # spectrum, avoid duplicating conjugates)
        candidate_order = [int(k) for k in candidate_order
                           if 0 < int(k) <= P // 2]

        # Record n_extras = 0 first (Clock-only)
        recovery_rows.append({
            "run_id":     cfg.display_id(),
            "seed":       cfg.seed,
            "n_extras":   0,
            "added_ks":   "",
            "val_acc":    acc_clock,
            "explained":  explained_frac,
        })

        added_ks: List[int] = []
        # Start basis as just Clock
        current_basis = B_clock.copy()

        n_extras_needed = None
        for step in range(1, min(max_extra_freqs, len(candidate_order)) + 1):
            k_new = candidate_order[step - 1]
            added_ks.append(k_new)
            freq_new = TWO_PI * k_new / P
            c  = np.arange(P, dtype=np.float64).reshape(P, 1)
            cos_col = np.cos(freq_new * c)
            sin_col = np.sin(freq_new * c)
            current_basis = np.concatenate(
                [current_basis, cos_col, sin_col], axis=1)

            # Refit and evaluate
            W_fit, _ = _project_onto_basis(W_learned, current_basis)
            acc_now  = _evaluate_decoder(pan, W_fit, vx, vy)

            energy_resid_now = float(((W_learned - W_fit) ** 2).sum())
            explained_now    = 1.0 - energy_resid_now / max(energy_total, 1e-12)

            recovery_rows.append({
                "run_id":     cfg.display_id(),
                "seed":       cfg.seed,
                "n_extras":   step,
                "added_ks":   ",".join(str(k) for k in added_ks),
                "val_acc":    acc_now,
                "explained":  explained_now,
            })

            if n_extras_needed is None and acc_now >= acc_learned - 0.01:
                n_extras_needed = step
                # Continue a few more for completeness
                if step >= n_extras_needed + 3:
                    break

        # ── Summary ──
        summary_rows.append({
            "run_id":              cfg.display_id(),
            "seed":                cfg.seed,
            "grokked":             True,
            "acc_learned":         acc_learned,
            "acc_clock_only":      acc_clock,
            "gap_clock":           acc_learned - acc_clock,
            "clock_explained_frac": explained_frac,
            "n_extras_for_1pct":   n_extras_needed if n_extras_needed else -1,
            "first_3_extras":      ",".join(
                str(k) for k in candidate_order[:3]),
            "f_eff_channels":      ",".join(
                f"{f:.4f}" for f in f_eff),
        })

        # Add to summary row:
        summary_rows[-1].update({
            "harmonic_H2_acc":  harm["harmonic_fits"].get("H=2",{}).get("acc"),
            "harmonic_H3_acc":  harm["harmonic_fits"].get("H=3",{}).get("acc"),
            "harmonic_H4_acc":  harm["harmonic_fits"].get("H=4",{}).get("acc"),
            "gate_optimal_acc": gate_ub["gate_optimal_acc"],
        })


        print(f"  [{cfg.display_id()}] "
              f"learned={acc_learned:.3f}  "
              f"clock_only={acc_clock:.3f}  "
              f"explained={explained_frac:.2%}  "
              f"n_extras={n_extras_needed}")

        _flush_bespoke()

    summary_df = pd.DataFrame(summary_rows)
    print("\n── Decoder analysis summary ──")
    if len(summary_df):
        print(summary_df.to_string(index=False))
    rep.write_manifest()
    return rep



# ═════════════════════════════════════════════════════════════════════════════
# YAML LOADER
# ═════════════════════════════════════════════════════════════════════════════
def load_experiment_yaml(path: str) -> tuple:
    """
    Parse a YAML experiment spec into (name, base_cfg, out_dir,
    dry_run, experiment_args).

    Top-level keys `grid`, `options`, and `plots` (used by grid_sweep) are
    merged into experiment_args so they flow through to the experiment
    function as keyword arguments. Legacy YAMLs with `experiment_args:`
    continue to work.
    """
    with open(path, "r") as f:
        spec = yaml.safe_load(f)

    name    = spec["experiment"]
    out_dir = spec.get("out_dir", f"results/{name}")
    dry_run = bool(spec.get("dry_run", False))

    base_dict = spec.get("base", {})
    base_cfg  = RunConfig.from_dict(base_dict)

    exp_args = dict(spec.get("experiment_args", {}) or {})
    for key in ("grid", "options", "plots"):
        if key in spec:
            exp_args[key] = spec[key]

    return name, base_cfg, out_dir, dry_run, exp_args


def run_experiment(
    name:     str,
    base:     RunConfig,
    out_dir:  str,
    dry_run:  bool  = False,
    **exp_args,
) -> ExperimentReporter:
    """Dispatch to the named experiment function."""
    if name not in EXPERIMENT_REGISTRY:
        raise KeyError(f"Unknown experiment: {name!r}. "
                        f"Available: {sorted(EXPERIMENT_REGISTRY)}")
    fn = EXPERIMENT_REGISTRY[name]
    return fn(base=base, out_dir=out_dir, dry_run=dry_run, **exp_args)


def run_from_yaml(
    path: str,
    force_dry_run: Optional[bool] = None,
    workers_override: Optional[int] = None,
):
    name, base, out_dir, dry_run, exp_args = load_experiment_yaml(path)
    if force_dry_run is not None:
        dry_run = force_dry_run
    if workers_override is not None:
        opts = dict(exp_args.get("options") or {})
        opts["workers"] = workers_override
        exp_args["options"] = opts
    print(f"\n▶ loading {path}: experiment={name!r} out={out_dir} "
          f"dry_run={dry_run}")
    return run_experiment(name, base, out_dir, dry_run, **exp_args)
