# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Techiva  
**Team Members:** Rithvin U S, Preethika Kumaravel, Shreenithi N  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
A two-stage *learned blocking* pipeline feeds a gradient-boosted matcher. Blocking retrieves a wide candidate
pool from an IDF-weighted inverted index over hashed name and address keys, then a small XGBoost re-ranker
keeps only a handful of candidates per Source 1 record (11.6 on average on test, down from ~350 retrieved)
while keeping ~98.5% of true matches. The matcher combines ~70 pairwise and **group-consensus** features (does
this candidate agree with the S1's other strongest candidates?); a fine-tuned MiniLM cross-encoder re-scores
only the uncertain pairs, and a decision rule is tuned for macro F0.5 under the one-owner-per-record structure
of the ground truth.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** 2.2M train / 1.73M test S1; ~10M Source 2+3 records per split. 16 GB RAM machine, so every stage
  is streamed per country and per S1 chunk, with parquet spills on disk.
- **Ground truth is exclusive:** each S2/S3 record matches at most one S1 (7.64M train pairs, none shared).
  An S1 has 3.46 matches on average; ~5% of S1 are singletons.
- **Noise in true matches:** typos (`sager`/`5ager`), word transpositions, legal-suffix swaps, added words
  (`services`, `center`), handle/domain forms (`lcprivate` for *Lex Communication Pvt*, `creativeinternational`),
  names replaced by pseudo-words with an intact address (`zephtavo`), house-number typos or truncation
  (`8250→8252`, `27724→2772`), dropped/reordered address parts, Devanagari names, empty addresses.
- **Hard negatives:** unmatched records are mostly *copies of a real entity with one field nudged*: house number
  `1030→1031`, one name word `medical→media`. At the pair level they look like typo-positives; they only stand
  out against the entity's other copies.
- **France** appears only in test: nothing country-specific is learned; the index, vocabularies and stop tokens
  are built per country label from the (unlabelled) data of the split itself.

### 2.2 Solution Strategy
**Approach Type:** Learned blocking + gradient-boosted classifier + exclusivity-aware decision (hybrid)  
**Core Innovation:** (1) a learned re-ranker inside blocking, which lifts candidate recall from 0.965 to 0.985
while cutting candidates per S1 from 40 to ~10; (2) group-consensus features that expose "one field nudged"
distractors; (3) a cross-encoder applied only to the uncertain band (~1.5 pairs per S1); (4) decision tuning
with *competitor* S1s so validation sees the same record-ownership competition as the full test set.

---

## 3. Candidate Generation (Blocking)
1. **Normalisation** (`normalize.py`): Unicode NFKC + transliteration (Devanagari via a dictionary learned from
   train ground truth, accents stripped), legal-form canonicalisation (`private limited→pvt ltd`,
   `sasu→sas`), address abbreviation maps (`road→rd`, `r./rue`, `allée→all`), state/region codes, postcode and
   house-number extraction. Output: `name_full`, `name_core` (no legal/stop words), `name_skel` (consonant
   skeleton), `addr`, `postcode`, `addr_nums`.
2. **Inverted index per country** (`blocking.py`): 10 hashed key types — name tokens, name bigrams, 4-char
   prefixes, consonant-skeleton tokens, compact-name prefix and exact compact name, address tokens, adjacent
   address token pairs, postcode, first-name-token × house number. Keys with document frequency > 600 are
   dropped; score = Σ key weight × IDF. Stored as memory-mapped CSR arrays.
3. **Wide retrieval:** top 300 by total score + top 60 by name-only and 60 by address-only score (~350 per S1).
4. **Learned re-ranker:** XGBoost on the key scores/ranks plus five cheap fuzzy similarities (compact-name
   ratio, full-name partial ratio, name/address token-set ratio, house-number equality), fit on train S1 that
   are in neither the matcher's training nor its validation set.
5. **Final candidate set** (`candidate_pairs.tsv`): top-3 by re-ranker score always, plus any candidate with
   re-ranker probability ≥ 0.002, capped at 40. This is exactly the set the matcher scores.

- **Blocking keys used:** name tokens / bigrams / prefixes / skeleton, compact name, address tokens and token
  pairs, postcode, name-token × house number; learned re-ranking on top.
- **Candidate pairs generated:** 20,063,811 on test = 11.58 per S1 (was 69.2M / 39.9 per S1 with a fixed
  top-40); 9.3 per S1 on validation.
- **How true matches were not lost:** two independent channels (name-only and address-only extras) so records
  with an empty address or a renamed business still enter the wide pool; the re-ranker is judged on recall of
  the wide pool; validation recall is tracked for every run (0.9645 plain IDF top-40 → 0.9875 re-ranked top-40
  → 0.9853 with the adaptive candidate set at 9.3 per S1).

---

## 4. Matching Model

**Features used (~70):**
- Name: token-set / token-sort / partial / plain ratio and Jaro-Winkler on full and core name; compact-name
  ratio and partial; consonant-skeleton ratios; first-token equality; initials-as-handle prefix; share of
  candidate name tokens never seen in the country's S1 vocabulary (renamed copies); legal-form agreement and
  conflict; name frequencies in pool and S1 (chains / generic names).
- Address: token-set / partial / plain ratio; the same after removing city/state-level tokens (tokens in
  ≥0.2% of the country's S1 addresses); postcode equality; number-set Jaccard; house-number equality, fuzzy
  similarity and prefix (truncation) relation; empty-address flags; lengths.
- Blocking: key scores (total / name / address, raw and normalised), ranks, re-ranker score and rank.
- **Group consensus:** similarity of the candidate to the S1's other top-3 re-ranked candidates (name and
  address mean/max, house-number vote), `twin_better` (a near-identical sibling has the exact house number
  and this one does not), candidates per S1, gaps to the S1's best, within-S1 ranks of key similarities.

**Model type:** XGBoost (`hist`, CUDA), depth 8, learning rate 0.05, early stopping on a 5% holdout of the
training S1 by log-loss (the decision relies on calibrated probabilities), trained on 700k S1 (6.2M pairs).
Training data is streamed from per-chunk parquet files into a `QuantileDMatrix`.

**Stage 3 — cross-encoder on the uncertain band:** `cross-encoder/ms-marco-MiniLM-L6-v2` (Apache-2.0, 22M
parameters) fine-tuned for 2 epochs on 600k training pairs ("name | address" of S1 vs candidate, balanced
positives / hard negatives). It re-scores only pairs with stage-1 probability in [0.02, 0.995) — 2.7M of the
20.1M test pairs — and a monotone depth-3 XGBoost stacks [logit p, cross-encoder logit] (fit on the validation
band with out-of-fold estimates by block). Band AUC: stage-1 0.952, cross-encoder 0.944, stacked 0.969.

**Threshold selection method:** grid search on validation macro F0.5 over three rules — global threshold,
threshold plus "top-1 rescue" for S1 with nothing above threshold, and per-S1 expected-F0.5 optimisation — each
with and without **exclusivity** (every pool record goes only to the S1 that scores it highest, matching the
ground truth structure). Validation S1 are sampled as whole blocks (country | city | name prefix) and the
ground-truth owners of their candidates are scored as *competitors* (never counted in the metric), so
exclusivity acts as it does on the full test set.

---

## 5. Results & Error Analysis

| version | change | val F0.5 | public LB |
|---|---|---|---|
| v2 | IDF blocking + pairwise XGBoost | 0.9498 | 0.9357 |
| v4 | Indic transliteration, two-channel blocking, exclusivity | 0.9708 | – |
| v5 | learned re-ranker in blocking | 0.9764 | 0.9685 |
| v6 | group-consensus + sharper pair features, competitor-aware tuning | 0.9812 | – |
| v6 + CE | cross-encoder on the uncertain band | 0.9839 | _pending_ |
| v7 | adaptive candidates (11.6/S1 on test), 700k training S1 | 0.9822 | – |
| **v7 + CE** | **final pipeline** | **0.9846** | _pending_ |

- **F_0.5 Score (macro):** 0.9846 on validation (40k block-sampled train S1 never used for training, decision
  tuned with competitor S1); v6 onwards numbers include competitor-aware tuning.
- **Common false positives (wrong merges):** "nudged copies" — same name with the house number moved by a few
  units (`1305` vs `1318 pacific ave`) or one name word swapped (`medical`/`media`), usually in the same city.
- **Common false negatives (missed matches):** renamed records (pseudo-word or handle names) with a partial or
  empty address; heavy address truncation combined with a name typo; S1 with a single true match.

---

## 6. Conclusion
Most of the gain came from treating blocking as a learning problem: a small re-ranker recovers true matches
that pure IDF ranking buried and lets the final candidate set shrink to single digits per S1. Group features
that compare a candidate with the entity's other copies are what separate genuine typos from deliberately
nudged distractors. Everything runs on a 16 GB laptop by streaming per country and per chunk.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `README.md` (exact commands), `requirements.txt` (pinned), `src/`:
`pipeline.py` (entry point: `prep`, `train`, `predict`, `decide`, `rescore`, `submit`), `prep.py`,
`normalize.py`, `indic.py`, `blocking.py`, `features.py`, `stage2.py`, `config.py`, `io_utils.py`,
`tracking.py`, `hwmon.py`. `prep` → `train` → `predict --run <train_run_id>` regenerates both output files.

### B. Additional Results
- Final candidate set on test: 20,063,811 pairs (11.58 per S1; 2 S1 with no candidates).
- Validation blocking recall by stage: IDF top-40 0.9645 · re-ranked top-40 0.9875 · adaptive (~9) ~0.984.
- Loss breakdown (v6 validation, F0.5 points lost): model misses 0.0077, S1 with zero correct matches
  0.0044, false positives 0.0039, blocking misses 0.0034, singleton false positives 0.0009.
