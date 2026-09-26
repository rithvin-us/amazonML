# Improvement plan after v5 (2026-09-26)

Baseline: v2 val 0.9498 -> LB 0.9357 · v4 val 0.9708 (LB pending) · v5 (re-ranker) val pending.
Top of LB 0.988. Deadline ~2026-09-28 00:00 IST, 5 submissions/day.

## Where F0.5 is lost (v4 val, 40k S1, total loss 0.029)

| bucket | F lost | evidence |
|---|---|---|
| blocking misses | 0.0097 | true pairs ranked ~150th by IDF-sum; **v5 re-ranker fixes most** (block recall 0.9645 -> 0.9875) |
| model FN (cand present, p < 0.7) | 0.0073 | 4.6k pairs, p spread evenly 0.05-0.7; 97% of them have a confident true sibling |
| S1 with zero TP | 0.0062 | 246 S1 (mostly n_true=1): 128 no positive in cands, 116 positive at p~0.3 |
| false positives | 0.0048 | 892 pairs: 486 pure distractors (mean p 0.89), 394 owned by an S1 outside val, 12 by val S1 |
| singleton FP | 0.0012 | |

Distractors are generated as **a copy of the entity with one field nudged**: house number 1030->1031,
1305->1318; one name word medical->media, capital->capaol. Pairwise features cannot tell these from
typo-positives (8250->8252 is a true match). Only group context can: the true copies agree with each
other, the nudged copy disagrees with all of them.

---

## Segment 1 — Group-consensus features  (target: FP 0.0048 + FN 0.0073; est. +0.003..0.006)
Anchors per S1 = top-3 candidates by re-ranker score `rr` (known before the model; no leakage).
Per candidate, computed in `features.py` group block (polars window ops + one anchor-text join):
- `hn_vote`: share of other anchors whose house number equals this candidate's; `hn_agree_s1_or_anchor`
- `name_vote` / `addr_vote`: mean and max token-set sim of this candidate to the other anchors
- `twin_better`: another candidate of the same S1 has the same name (nc_tset >= 95) but exact house
  number while this one does not -> flags nudged copies
- within-S1 ranks of `ad_tset`, `hn_sim`, `ncc_ratio` (only `nf_tset_rank` exists today)
Measure: val F0.5, FP count, FN count vs v5.

## Segment 2 — Sharper pair features  (target: model FN; est. +0.002..0.004)
- `addr_core` sim: address with high-df tokens removed (city/state/"road"), so same-city distractors
  stop scoring 90+ on `ad_tset`
- gibberish/OOV name: share of candidate name tokens never seen in the country's S1 vocabulary
  (`zephtavo`, `rizaxylo` are renamed copies -> trust the address)
- legal-type mismatch flag (pvt ltd vs llc/llp, sarl vs sas)
- token alignment: share of S1 name tokens with a JW >= 0.9 partner in the candidate
Files: `features.py`, per-record columns attached next to `add_name_counts`.

## Segment 3 — Validation fidelity + decision  (est. +0.001..0.002, better LB prediction)
- Val today holds 40k S1; 44% of FPs belong to an S1 that is not in val, so exclusivity is weaker on
  val than on test (all 1.73M S1 compete). Add each val candidate's GT owner S1 as a competitor
  (scored, used only for exclusivity, not counted in the metric), then tune the decision.
- Per-source (S2/S3) thresholds; keep top-1 rescue for n_true=1 S1s.
Files: `pipeline.py` train_stage val block, `tune_decision`.

## Segment 4 — Model capacity  (est. +0.001..0.003)
- 400k train S1 (re-ranker makes ~35 pairs/S1, so ~14M rows; the 450k OOM was at 53 pairs/S1)
- depth 10, eta 0.03, up to 6000 rounds; optional LightGBM (CPU) blend averaged with XGB
- pagefile now has room on C: (30GB free)

## Segment 5 — Blocking extras  (est. +0.001..0.003)
- wider retrieval 400/80/80 (wide recall caps at ~0.988)
- 28% of remaining misses share no kept key (max_df pruning / renamed + empty address):
  2-hop expansion — query the index with the S1's top anchor's text and add its neighbours

## Segment 6 — Cross-encoder on the uncertain band  (optional, highest ceiling, needs install)
- Only pairs with p in [0.02, 0.995): ~1 pair per S1 (1.7M on test, ~10 min on the RTX 4050)
- MiniLM-L6 (Apache-2.0, 22M params) fine-tuned on (S1 text, candidate text) from non-val train S1,
  hard negatives from the re-ranker; final p = small stacker(p1, ce)
- Needs `torch` (CUDA) + `transformers` installed into `.venv311` (~3GB on D:) and a ~90MB model
  download. **Needs your OK before installing.** Gate: integrate only if OOF val gain >= +0.003.

## Segment 7 — France / generalization
- After the v4/v5 LB scores: if the val->LB gap stays >> 0.01, run label-free France diagnostics
  (`tmp/scratch/france_check.py`); features are country-agnostic already.

---

## Execution order
1. v5 done -> measure val + residuals (`tmp/scratch/err_analysis.py`, `residuals.py`) -> submit if better than v4.
2. **v6 = Segments 1 + 2 + 3** in one run (~60 min, one chain).
3. **v7 = v6 + Segment 4** (capacity), if v6 RAM headroom allows.
4. Segment 6 experiment in parallel (after install OK), merge only if gated gain holds.
5. Segment 5 if time remains. Keep one submission spare for the final pick.
