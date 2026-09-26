"""Patch pipeline.py: factor decide+write+validate into decide_stage + `decide` command."""
from pathlib import Path

p = Path(r"D:\amazon-ml\code\business_entity_resolution\src\pipeline.py")
s = p.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub('''
    run.start_stage("decide+write_test")
    s1_map = s1.select(pl.col("idx").alias("s1_idx"), pl.col("entity_id").alias("source1_entity_id"))
    n_s1 = s1_map.height
    del s1
    gc.collect()
    out_dir = run.dir / "output"''', '''    del s1
    gc.collect()
    decide_stage(cfg, run, spill, dec, n_pairs)


def decide_stage(cfg: Config, run: Run, spill: Path, dec: dict, n_pairs: int | None = None) -> None:
    """Decision + TSV writing + validation from a run's pred/ spill (scored-*.parquet, cand-*.parquet).

    Also used standalone (`pipeline.py decide --run <id>`) to re-decide without re-blocking. s1_idx is the
    test S1 row number in raw source order (the norm caches assign idx the same way).
    """
    run.start_stage("decide+write_test")
    s1_map = (pl.read_parquet(CACHE_DIR / "raw_test_s1.parquet", columns=["entity_id"]).with_row_index("s1_idx")
              .select(pl.col("s1_idx").cast(pl.Int64), pl.col("entity_id").alias("source1_entity_id")))
    n_s1 = s1_map.height
    out_dir = run.dir / "output"''')
sub('''    run.set_metrics(test_decision=f"{dec['mode']}:{dec['param']} excl={dec.get('excl')}", test_s1=n_s1,
                    test_nonempty=nonempty, test_pred_pairs=sel.height, test_pred_pairs_alt=alt.height,
                    test_avg_candidates=round(n_pairs / max(n_s1, 1), 2))
    run.log(f"metric test nonempty={nonempty:,}/{n_s1:,} pred_pairs={sel.height:,} alt_pairs={alt.height:,} "
            f"avg_cands={n_pairs / max(n_s1, 1):.1f}")''', '''    run.set_metrics(test_decision=f"{dec['mode']}:{dec['param']} excl={dec.get('excl')}", test_s1=n_s1,
                    test_nonempty=nonempty, test_pred_pairs=sel.height, test_pred_pairs_alt=alt.height)
    if n_pairs is not None:
        run.set_metrics(test_avg_candidates=round(n_pairs / max(n_s1, 1), 2))
    run.log(f"metric test nonempty={nonempty:,}/{n_s1:,} pred_pairs={sel.height:,} alt_pairs={alt.height:,}")''')
sub('''    ap.add_argument("cmd", choices=["prep", "train", "predict", "all", "submit", "lb", "stage2"])''',
    '''    ap.add_argument("cmd", choices=["prep", "train", "predict", "all", "submit", "lb", "stage2", "decide"])''')
sub('''            if a.cmd == "stage2":''', '''            if a.cmd == "decide":
                src = RUNS_DIR / a.run
                m = json.loads((src / "metrics.json").read_text())
                dec = {"mode": m.get("decision_mode", "threshold"),
                       "param": m.get("decision_param", m.get("threshold", 0.5)), "excl": m.get("decision_excl", False)}
                if a.threshold:
                    dec = {"mode": "threshold", "param": a.threshold, "excl": dec["excl"]}
                run.log(f"re-deciding {src.name} pred/ with {dec}")
                decide_stage(cfg, run, src / "pred", dec)
            if a.cmd == "stage2":''')
s = s.replace('''    cfg = Config(run_name=a.name or (f"s2_{a.run}" if a.cmd == "stage2" else''',
              '''    cfg = Config(run_name=a.name or (f"{a.cmd}_{a.run}" if a.cmd in ("stage2", "decide") else''')
p.write_text(s, encoding="utf-8")
print("pipeline.py patched")
