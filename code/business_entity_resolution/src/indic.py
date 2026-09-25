"""Indic-script -> Latin token dictionary learned from train ground truth (no external data).

Pool names in Devanagari/Tamil/Telugu/... are token-by-token renderings of the Latin name
(99.998% of matched pairs have equal token counts), so positional alignment of
(S1 Latin token, pool Indic token) over train GT pairs gives an exact dictionary for
frequent words ("लिमिटेड" -> "limited"). unidecode stays the fallback for unseen tokens.

  python indic.py        # build cache/indic_dict.json (prep.py calls ensure_dict())
"""
from __future__ import annotations

import json
import re
import unicodedata

from config import CACHE_DIR

DICT_PATH = CACHE_DIR / "indic_dict.json"
INDIC_RE = re.compile(r"[ऀ-෿]")
_EDGE_PUNCT = ".,;:!?()[]{}\"'`-/&|।॥"  # incl. danda / double danda
_LATIN_CLEAN = re.compile(r"[^a-z0-9]")
MIN_COUNT = 2
MIN_SHARE = 0.5

_dict: dict[str, str] | None = None


def _indic_tok(t: str) -> str:
    return t.strip(_EDGE_PUNCT)


def _latin_tok(t: str) -> str:
    from unidecode import unidecode
    return _LATIN_CLEAN.sub("", unidecode(t).lower())


def build(min_count: int = MIN_COUNT, min_share: float = MIN_SHARE) -> dict[str, str]:
    import polars as pl
    pat = r"[\x{0900}-\x{0DFF}]"
    gt = pl.scan_parquet(CACHE_DIR / "gt_long.parquet").drop_nulls()
    pool = (pl.concat([pl.scan_parquet(CACHE_DIR / "raw_train_s2.parquet"),
                       pl.scan_parquet(CACHE_DIR / "raw_train_s3.parquet")])
            .filter(pl.col("business_name").str.contains(pat))
            .select(pl.col("entity_id").alias("match_id"), pl.col("business_name").alias("pn")))
    s1 = pl.scan_parquet(CACHE_DIR / "raw_train_s1.parquet").select(
        pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("sn"))
    d = pool.join(gt, on="match_id").join(s1, on="s1_id").select("pn", "sn").collect()
    counts: dict[tuple[str, str], int] = {}
    for pn, sn in zip(d["pn"].to_list(), d["sn"].to_list()):
        pt = unicodedata.normalize("NFKC", pn).split()
        st = sn.split()
        if len(pt) != len(st):
            continue
        for a, b in zip(pt, st):
            if not INDIC_RE.search(a):
                continue
            a, b = _indic_tok(a), _latin_tok(b)
            if a and b:
                counts[(a, b)] = counts.get((a, b), 0) + 1
    best: dict[str, tuple[str, int]] = {}
    tot: dict[str, int] = {}
    for (a, b), n in counts.items():
        tot[a] = tot.get(a, 0) + n
        if n > best.get(a, ("", 0))[1]:
            best[a] = (b, n)
    return {a: b for a, (b, n) in best.items() if n >= min_count and n / tot[a] >= min_share}


def ensure_dict() -> None:
    if DICT_PATH.exists():
        return
    d = build()
    DICT_PATH.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    print(f"indic dict: {len(d):,} tokens -> {DICT_PATH.name}", flush=True)


def _load() -> dict[str, str]:
    global _dict
    if _dict is None:
        _dict = json.loads(DICT_PATH.read_text(encoding="utf-8")) if DICT_PATH.exists() else {}
    return _dict


def to_latin(s: str) -> str:
    """Replace known Indic tokens by their learned Latin form; others pass through (unidecode later)."""
    if not INDIC_RE.search(s):
        return s
    d = _load()
    out = []
    for t in s.split():
        if INDIC_RE.search(t):
            k = _indic_tok(t)
            t = d.get(k, t)
        out.append(t)
    return " ".join(out)


if __name__ == "__main__":
    ensure_dict()
