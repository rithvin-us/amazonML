"""Macro F0.5 per Source1 entity, exactly as the challenge README defines it."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


def f05_single(pred: set, true: set, beta: float = 0.5) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(pred: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]],
              s1_ids: Iterable[str]) -> dict:
    """Average over s1_ids (must include singletons). Also reports micro P/R."""
    tot = tp = npred = ntrue = 0.0
    n = 0
    for s1 in s1_ids:
        ps, ts = set(pred.get(s1, ())), set(truth.get(s1, ()))
        tot += f05_single(ps, ts)
        tp += len(ps & ts)
        npred += len(ps)
        ntrue += len(ts)
        n += 1
    return {
        "f05": tot / max(n, 1),
        "precision": tp / npred if npred else 1.0,
        "recall": tp / ntrue if ntrue else 1.0,
        "n": n,
    }


def blocking_recall(cands: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]) -> dict:
    tp = ntrue = ncand = 0
    for s1, ts in truth.items():
        cs = set(cands.get(s1, ()))
        ts = set(ts)
        tp += len(cs & ts)
        ntrue += len(ts)
    for cs in cands.values():
        ncand += len(set(cs))
    return {"block_recall": tp / ntrue if ntrue else 1.0,
            "avg_candidates": ncand / max(len(cands), 1)}


if __name__ == "__main__":
    s = f05_single({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(s - 0.714) < 1e-3, s
    assert f05_single(set(), set()) == 1.0
    assert f05_single({"x"}, set()) == 0.0
    assert f05_single(set(), {"x"}) == 0.0
    print("evaluate self-test OK", round(s, 4))
