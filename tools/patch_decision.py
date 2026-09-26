"""Patch pipeline.py: joint (rule x exclusivity) decision tuning + top-1 rescue + alt variant output."""
from pathlib import Path

p = Path(r"D:\amazon-ml\code\business_entity_resolution\src\pipeline.py")
s = p.read_text(encoding="utf-8")


def sub(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


sub('''ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
''', '''ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
TOP1_GRID = [0.2, 0.3, 0.4, 0.5, 0.6]  # "top-1 rescue" floors for S1s with nothing above the main threshold
''')

sub('''def apply_decision(scored: pl.DataFrame, dec: dict, floor: float) -> pl.DataFrame:
    if dec["mode"] == "threshold":
        return scored.filter(pl.col("p") >= dec["param"])
    return select_expected_f(scored.filter(pl.col("p") >= floor), dec["param"])


def exclusive(sel: pl.DataFrame) -> pl.DataFrame:
    """GT is one-to-many: a pool record matches at most one S1. Keep each cand only for its best-p S1."""
    return sel.filter(pl.col("p").rank("ordinal", descending=True).over("cand_idx") == 1)


def tune_decision(scored: pl.DataFrame, truth_counts: pl.DataFrame, floor: float, run: Run) -> dict:
    res = {}
    for t in THRESHOLD_GRID:
        m = eval_selection(apply_decision(scored, {"mode": "threshold", "param": float(t)}, floor), truth_counts)
        if m["f05"] > res.get("threshold", {"f05": -1})["f05"]:
            res["threshold"] = {"mode": "threshold", "param": float(t), **m}
    for a in ALPHA_GRID:
        m = eval_selection(apply_decision(scored, {"mode": "expected_f", "param": a}, floor), truth_counts)
        if m["f05"] > res.get("expected_f", {"f05": -1})["f05"]:
            res["expected_f"] = {"mode": "expected_f", "param": a, **m}
    for k, v in res.items():
        run.log(f"decision {k}: param={v['param']} f05={v['f05']:.5f} P={v['precision']:.4f} R={v['recall']:.4f}")
    return res''', '''def exclusive(sel: pl.DataFrame) -> pl.DataFrame:
    """GT is one-to-many: a pool record matches at most one S1. Keep each cand only for its best-p S1."""
    return sel.filter(pl.col("p").rank("ordinal", descending=True).over("cand_idx") == 1)


def apply_decision(scored: pl.DataFrame, dec: dict, floor: float) -> pl.DataFrame:
    """dec: {mode, param, excl}. excl -> each pool record is first assigned to its best-p S1 only (GT is
    exclusive), then the per-S1 rule runs. Modes: threshold (p >= t); thr_top1 (p >= t, plus the S1's best
    candidate when it has p >= t1 < t); expected_f (per-S1 plug-in expected-F0.5 optimum)."""
    if dec.get("excl"):
        scored = exclusive(scored.filter(pl.col("p") >= floor))
    if dec["mode"] == "threshold":
        return scored.filter(pl.col("p") >= dec["param"])
    if dec["mode"] == "thr_top1":
        t, t1 = dec["param"]
        top1 = pl.col("p").rank("ordinal", descending=True).over("s1_idx") == 1
        return scored.filter((pl.col("p") >= t) | (top1 & (pl.col("p") >= t1)))
    return select_expected_f(scored.filter(pl.col("p") >= floor), dec["param"])


def tune_decision(scored: pl.DataFrame, truth_counts: pl.DataFrame, floor: float, run: Run) -> dict:
    """Best param per (mode, excl) on labelled pairs -> {name: {mode, param, excl, f05, precision, recall}}."""
    res = {}
    cands = ([("threshold", float(t)) for t in THRESHOLD_GRID] + [("expected_f", a) for a in ALPHA_GRID]
             + [("thr_top1", [float(t), t1]) for t in THRESHOLD_GRID if t >= 0.5 for t1 in TOP1_GRID if t1 < t])
    for excl in (False, True):
        base = exclusive(scored.filter(pl.col("p") >= floor)) if excl else scored
        for mode, param in cands:
            m = eval_selection(apply_decision(base, {"mode": mode, "param": param}, floor), truth_counts)
            name = mode + ("+excl" if excl else "")
            if m["f05"] > res.get(name, {"f05": -1})["f05"]:
                res[name] = {"mode": mode, "param": param, "excl": excl, **m}
    for k, v in res.items():
        run.log(f"decision {k}: param={v['param']} f05={v['f05']:.5f} P={v['precision']:.4f} R={v['recall']:.4f}")
    return res''')

sub('''    res = tune_decision(va, tvc, cfg.p_floor, run)
    dec = max(res.values(), key=lambda d: d["f05"])
    ex = eval_selection(exclusive(apply_decision(va, dec, cfg.p_floor)), tvc)
    run.set_metrics(decision_mode=dec["mode"], decision_param=dec["param"],
                    threshold=res["threshold"]["param"], val_f05=round(dec["f05"], 5),
                    val_precision=round(dec["precision"], 4), val_recall=round(dec["recall"], 4),
                    val_f05_threshold=round(res["threshold"]["f05"], 5),
                    val_f05_expected_f=round(res["expected_f"]["f05"], 5),
                    val_f05_excl=round(ex["f05"], 5),''', '''    res = tune_decision(va, tvc, cfg.p_floor, run)
    dec = max(res.values(), key=lambda d: d["f05"])
    run.set_metrics(decision_mode=dec["mode"], decision_param=dec["param"], decision_excl=dec["excl"],
                    threshold=res["threshold"]["param"], val_f05=round(dec["f05"], 5),
                    val_precision=round(dec["precision"], 4), val_recall=round(dec["recall"], 4),
                    val_f05_by_decision={k: round(v["f05"], 5) for k, v in res.items()},
                    val_f05_threshold=round(res["threshold"]["f05"], 5),
                    val_f05_excl=round(res["threshold+excl"]["f05"], 5),''')

sub('''    run.log(f"metric val_f05={dec['f05']:.5f} P={dec['precision']:.4f} R={dec['recall']:.4f} "
            f"decision={dec['mode']}:{dec['param']} by_country={per_c}")''', '''    run.log(f"metric val_f05={dec['f05']:.5f} P={dec['precision']:.4f} R={dec['recall']:.4f} "
            f"decision={dec['mode']}:{dec['param']} excl={dec['excl']} by_country={per_c}")''')

sub('''    dec = {"mode": metrics.get("decision_mode", "threshold"),
           "param": metrics.get("decision_param", metrics.get("threshold", 0.5))}''', '''    dec = {"mode": metrics.get("decision_mode", "threshold"),
           "param": metrics.get("decision_param", metrics.get("threshold", 0.5)),
           "excl": metrics.get("decision_excl", False)}''')

sub('''    run.log(f"model {model.kind} on {model.device}, decision {dec['mode']}:{dec['param']}")''',
    '''    run.log(f"model {model.kind} on {model.device}, decision {dec['mode']}:{dec['param']} excl={dec.get('excl')}")''')

sub('''    sel = apply_decision(scored, dec, cfg.p_floor)
    excl = exclusive(sel)
    del scored
    _write_ids(s1_map, _join_ids(sel), "matched_entity_ids", out_dir / "matching_results.tsv")
    _write_ids(s1_map, _join_ids(excl), "matched_entity_ids", out_dir / "matching_results_excl.tsv")
    for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(out_dir / fn, OUTPUT_DIR / fn)
    (OUTPUT_DIR / "variants").mkdir(exist_ok=True)
    shutil.copy(out_dir / "matching_results_excl.tsv", OUTPUT_DIR / "variants" / "matching_results_excl.tsv")
    nonempty = sel["s1_idx"].n_unique()
    run.set_metrics(test_decision=f"{dec['mode']}:{dec['param']}", test_s1=n_s1, test_nonempty=nonempty,
                    test_pred_pairs=sel.height, test_pred_pairs_excl=excl.height,
                    test_excl_dropped=sel.height - excl.height,
                    test_avg_candidates=round(n_pairs / max(n_s1, 1), 2))
    run.log(f"metric test nonempty={nonempty:,}/{n_s1:,} pred_pairs={sel.height:,} "
            f"excl_dropped={sel.height - excl.height:,} avg_cands={n_pairs / max(n_s1, 1):.1f}")
    del sel, excl''', '''    # main = the val-best decision (incl. its exclusivity choice); alt = same rule with exclusivity flipped
    sel = apply_decision(scored, dec, cfg.p_floor)
    alt = apply_decision(scored, {**dec, "excl": not dec.get("excl")}, cfg.p_floor)
    del scored
    _write_ids(s1_map, _join_ids(sel), "matched_entity_ids", out_dir / "matching_results.tsv")
    _write_ids(s1_map, _join_ids(alt), "matched_entity_ids", out_dir / "matching_results_alt.tsv")
    for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copy(out_dir / fn, OUTPUT_DIR / fn)
    (OUTPUT_DIR / "variants").mkdir(exist_ok=True)
    shutil.copy(out_dir / "matching_results_alt.tsv", OUTPUT_DIR / "variants" / "matching_results_alt.tsv")
    nonempty = sel["s1_idx"].n_unique()
    run.set_metrics(test_decision=f"{dec['mode']}:{dec['param']} excl={dec.get('excl')}", test_s1=n_s1,
                    test_nonempty=nonempty, test_pred_pairs=sel.height, test_pred_pairs_alt=alt.height,
                    test_avg_candidates=round(n_pairs / max(n_s1, 1), 2))
    run.log(f"metric test nonempty={nonempty:,}/{n_s1:,} pred_pairs={sel.height:,} alt_pairs={alt.height:,} "
            f"avg_cands={n_pairs / max(n_s1, 1):.1f}")
    del sel, alt''')

sub('''    for fn in ("matching_results.tsv", "variants/matching_results_excl.tsv"):
        r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(OUTPUT_DIR / fn),
                            "--test-dir", str(test_dir)],''', '''    for fn in ("matching_results.tsv", "variants/matching_results_alt.tsv"):
        # --candidate at a missing path: the official validator holds ~70M candidate ids in Python sets (OOM
        # here); candidate_pairs.tsv gets the streamed polars check below instead
        r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", str(OUTPUT_DIR / fn),
                            "--candidate", str(OUTPUT_DIR / "__skip__.tsv"), "--test-dir", str(test_dir)],''')

p.write_text(s, encoding="utf-8")
print("pipeline.py patched")
