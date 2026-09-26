# tools/

Orchestration and analysis scripts used during development (copied from `tmp/scratch/`). None of them is needed
to reproduce the submission — `code/business_entity_resolution/README.md` covers that — but they document how
every experiment was run and measured.

- Orchestration: `chain.py` (train -> predict -> CE train/apply), `variant.py` / `apply_model.py` (retrain / rescore
  from saved features), `evening_queue.py`, `late_queue.py` (unattended queues + FINAL_REPORT), `ens.py`
  (probability averaging), `prior_match.py` (unlabelled-country decision prototype)
- Analysis: `err_analysis.py`, `residuals.py`, `loss_final.py` (loss breakdowns), `blocking_debug.py`,
  `cand_prune.py`, `rerank_proto.py` (blocking), `loco*.py` (unseen-country proxy), `shift.py`, `shape.py`,
  `fr_*.py` (France diagnostics), `verify_prep.py`, `verify_sub.py`, `leak_check.py` (data / submission checks)
- Cross-encoder experiments: `ce_experiment.py`, `ce_stack.py`, `ce_france.py`, `stack_exp.py`
