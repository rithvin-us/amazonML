# Evening run report
generated Sat Sep 26 16:36:40 2026

**Recommended file:** `output/SUBMIT_THIS/matching_results.tsv` = **v9_ce**

| file | what | validator | val F0.5 (US/India) | test shape by country |
|---|---|---|---|---|
| v9_ce | v9 (v5 norm, 1.6M S1, L12 CE on 1M pairs) + France shape rule  [MAIN] | PASS | 0.98585 | France: 3.355/S1 empty 0.0532; India: 3.336/S1 empty 0.0578; US: 3.379/S1 empty 0.0551 |
| v9_plain | v9, val-tuned rule for France too | PASS | None | France: 3.437/S1 empty 0.0484; India: 3.336/S1 empty 0.0578; US: 3.379/S1 empty 0.0551 |
| v9_cefr | v9 + France-adapted CE + France shape rule | PASS | 0.98531 | France: 3.351/S1 empty 0.0538; India: 3.331/S1 empty 0.0577; US: 3.377/S1 empty 0.0547 |
| v8bigce_ce | v8big (1.3M S1) + L12 CE + France shape rule  [FALLBACK] | PASS | 0.98522 | France: 3.344/S1 empty 0.0538; India: 3.323/S1 empty 0.0574; US: 3.367/S1 empty 0.0543 |
| v8bigce_plain | v8big + L12 CE, val-tuned rule for France too | PASS | None | France: 3.408/S1 empty 0.0476; India: 3.323/S1 empty 0.0574; US: 3.367/S1 empty 0.0543 |

Reference: v7+CE LB 0.973901 (val 0.98457); v7fr+L12 LB 0.973905 (val 0.98499).
Stable copies of every passing file: output/submissions/<file>_matching_results.tsv

Queue log: runs/evening_queue.log
## Late queue (depth-10 variant + ensemble)
- v9d10 stage-1 val F0.5: 0.98331 (v9 stage-1: 0.98317)
- ensemble(v9, v9d10) + L12 CE + France rule val F0.5: 0.98588 (v9_ce: 0.98585)
- SUBMIT_THIS unchanged (ensemble gain < 0.0003 or failed)
