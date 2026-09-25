"""TSV readers (tab-separated, no quoting) with parquet caching."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from config import CACHE_DIR, DATA_DIR

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,       # everything as string
        null_values=[""],
        truncate_ragged_lines=True,
    )


def load_source(split: str, n: int, use_cache: bool = True) -> pl.DataFrame:
    """split in {train,test}, n in {1,2,3}. Cached as raw parquet."""
    cache = CACHE_DIR / f"raw_{split}_s{n}.parquet"
    if use_cache and cache.exists():
        return pl.read_parquet(cache)
    df = read_tsv(DATA_DIR / split / f"{split}_source{n}.tsv").select(SOURCE_COLS)
    df = df.with_columns(pl.col(c).fill_null("") for c in SOURCE_COLS[1:])
    df.write_parquet(cache)
    return df


def load_ground_truth(use_cache: bool = True) -> pl.DataFrame:
    """Long format: s1_id, match_id (one row per positive pair) + s1 ids with no matches (match_id null)."""
    cache = CACHE_DIR / "gt_long.parquet"
    if use_cache and cache.exists():
        return pl.read_parquet(cache)
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = gt.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "m"})
    long = (
        gt.with_columns(pl.col("m").fill_null("").str.split(","))
        .explode("m")
        .with_columns(pl.when(pl.col("m") == "").then(None).otherwise(pl.col("m")).alias("match_id"))
        .select("s1_id", "match_id")
        .unique()
    )
    long.write_parquet(cache)
    return long


def write_id_lists(mapping: dict[str, list[str]], all_s1: list[str], path: Path, col: str) -> None:
    """Write submission-format TSV: one row per S1 id, comma-joined, deduped, order kept."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in all_s1:
            ids = list(dict.fromkeys(mapping.get(s1, ())))
            f.write(f"{s1}\t{','.join(ids)}\n")
