# Common operations for pan_lab.
#   make install   one-time pip -e install
#   make test      run the pytest suite
#   make plan      dry-run all shipped YAMLs
#   make tier3     overnight tier3 mechanistic run
#   make paper     the full set of runs needed for the paper submission

.PHONY: install test plan smoke \
        compare tier3 slot_census k8_sweep dw_sweep wd_sweep primes \
        held_out_primes freq_init decoder_swap sifp16 mod_mul mod_two_step \
        tf_sweep paper clean

install:
	v run pip install -e .

test:
	uv run pytest

# Dry-run every YAML to verify plans look right without training
plan:
	@for f in experiments/*.yaml; do \
	  echo "═══ $$f ═══"; \
	  python -m pan_lab "$$f" --dry-run 2>&1 \
	    | grep -E "^(══|  total|  device)" ; \
	done

# Fast end-to-end smoke test (~10s): pan vs tf at P=11, K=3, 500 steps.
smoke:
	@python -c "import yaml; open('/tmp/panlab_smoke.yaml','w').write(yaml.dump({\
	  'experiment':'grid_sweep','out_dir':'/tmp/panlab_smoke',\
	  'base':{'p':11,'k_freqs':3,'n_steps':500,'batch_size':64,\
	          'log_every':100,'early_stop':False,'use_compile':False},\
	  'grid':[{'model_kind':'pan','weight_decay':0.01,'label':'pan'},\
	          {'model_kind':'transformer','weight_decay':1.0,'label':'tf'}],\
	  'plots':['metric_formation_curves','metric_spectra','metric_peak_timescales']}))"
	python -m pan_lab /tmp/panlab_smoke.yaml
	@echo "smoke output in /tmp/panlab_smoke/"

compare:          ;  python -m pan_lab experiments/compare.yaml
tier3:            ;  python -m pan_lab experiments/tier3.yaml
slot_census:      ;  python -m pan_lab experiments/slot_census.yaml
k8_sweep:         ;  python -m pan_lab experiments/k8_sweep.yaml
dw_sweep:         ;  python -m pan_lab experiments/dw_sweep.yaml
wd_sweep:         ;  python -m pan_lab experiments/wd_sweep.yaml
primes:           ;  python -m pan_lab experiments/primes.yaml
held_out_primes:  ;  python -m pan_lab experiments/held_out_primes.yaml
freq_init:        ;  python -m pan_lab experiments/freq_init_ablation.yaml
decoder_swap:     ;  python -m pan_lab experiments/decoder_swap.yaml
sifp16:           ;  python -m pan_lab experiments/sifp16_inference.yaml
mod_mul:          ;  python -m pan_lab experiments/mod_mul.yaml
mod_two_step:     ;  python -m pan_lab experiments/mod_two_step.yaml
tf_sweep:         ;  python -m pan_lab experiments/tf_sweep.yaml

# The paper submission checklist: everything needed to close Phase 0.
#   tier3          — Task 0.1 mechanistic equivalence
#   slot_census    — Exp A, direct answer to §5.1
#   k8_sweep       — Task 0.2, the K=8 anomaly
#   freq_init      — Exp H, the cleanest mechanistic test
#   decoder_swap   — Exp I, canonical-Fourier-circuit proof
#   sifp16         — Exp E, hardware relevance
#   held_out_primes — Reviewer robustness
paper: tier3 slot_census k8_sweep freq_init decoder_swap sifp16 held_out_primes
	@echo "Paper-submission experiment set complete. Output in results/"

paper-followup:
	python -m pan_lab experiments/random_init_census_20.yaml     # ~10 min
	python -m pan_lab experiments/held_out_p97_long.yaml

clean:
	rm -rf results/ /tmp/panlab_smoke* /tmp/smoke*.yaml
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
