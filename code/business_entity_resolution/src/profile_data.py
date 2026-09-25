"""Aggregate-only data profile -> runs/profile.json (safe for Claude to read).

Memory-lean: everything via lazy scan_parquet + streaming collect.
"""
from __future__ import annotations

import json

import polars as pl

from config import CACHE_DIR, RUNS_DIR
from io_utils import load_ground_truth, load_source


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def main() -> None:
    prof: dict = {}
    for split in ("train", "test"):
        for n in (1, 2, 3):
            path = CACHE_DIR / f"raw_{split}_s{n}.parquet"
            if not path.exists():
                load_source(split, n)  # builds cache
            lf = pl.scan_parquet(path)
            agg = _collect(lf.select(
                pl.len().alias("rows"),
                (pl.col("business_name") == "").sum().alias("empty_name"),
                (pl.col("business_address") == "").sum().alias("empty_addr"),
                pl.col("business_name").str.len_chars().mean().round(1).alias("name_len_mean"),
                pl.col("business_address").str.len_chars().mean().round(1).alias("addr_len_mean"),
                pl.col("business_name").str.contains(r"[^\x00-\x7F]").mean().round(4).alias("non_ascii_name_frac"),
            )).to_dicts()[0]
            agg["country"] = dict(_collect(lf.group_by("country").len().sort("len", descending=True)
                                           .head(10)).iter_rows())
            prof[f"{split}_s{n}"] = agg
            print(split, n, agg, flush=True)

    load_ground_truth()  # ensure cache
    gt = pl.scan_parquet(CACHE_DIR / "gt_long.parquet")
    per = _collect(gt.group_by("s1_id").agg(
        pl.col("match_id").drop_nulls().len().alias("k"),
        pl.col("match_id").drop_nulls().str.starts_with("S2-").sum().alias("k2")))
    prof["gt"] = {
        "s1_entities": per.height,
        "singleton_frac": round((per["k"] == 0).mean(), 4),
        "matches_mean": round(per["k"].mean(), 3),
        "matches_p50_p90_p99_max": [int(per["k"].quantile(q)) for q in (0.5, 0.9, 0.99)] + [int(per["k"].max())],
        "s2_share_of_matches": round(per["k2"].sum() / max(per["k"].sum(), 1), 4),
        "k_hist": {str(k): v for k, v in per.group_by(pl.col("k").clip(0, 10)).len().sort("k").iter_rows()},
    }
    del per
    # 5 example pairs for eyeballing normalisation
    ex_pairs = _collect(gt.drop_nulls().filter(pl.col("s1_id").hash(0) % 200_000 == 7).head(5))
    s1 = pl.scan_parquet(CACHE_DIR / "raw_train_s1.parquet")
    oth = pl.concat([pl.scan_parquet(CACHE_DIR / "raw_train_s2.parquet"),
                     pl.scan_parquet(CACHE_DIR / "raw_train_s3.parquet")])
    a = _collect(s1.filter(pl.col("entity_id").is_in(ex_pairs["s1_id"].implode())))
    b = _collect(oth.filter(pl.col("entity_id").is_in(ex_pairs["match_id"].implode())))
    ex = ex_pairs.join(a.rename({"entity_id": "s1_id"}), on="s1_id").join(
        b.rename({"entity_id": "match_id"}), on="match_id", suffix="_m")
    prof["examples"] = ex.drop("s1_id", "match_id").to_dicts()
    out = RUNS_DIR / "profile.json"
    out.write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in prof.items() if k != "examples"}, indent=1, ensure_ascii=False))
    print(json.dumps(prof["examples"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
