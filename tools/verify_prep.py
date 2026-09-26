"""Hard data checks on preprocessing: raw TSV -> raw parquet -> norm caches (counts, idx alignment, IDs, nulls),
plus a few normalised samples per country."""
import sys

import polars as pl

sys.path.insert(0, r"D:\amazon-ml\code\business_entity_resolution\src")
import pipeline as pp  # noqa: E402

pl.Config.set_tbl_rows(20)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(220)
DATA = pp.ROOT / "student_resource" / "dataset"
ok = True


def check(cond: bool, msg: str) -> None:
    global ok
    ok &= bool(cond)
    print(("OK   " if cond else "FAIL ") + msg)


for split in ("train", "test"):
    # 1. TSV line counts == raw parquet rows (raw files are read without loading them into Python objects)
    raws = {}
    for n in (1, 2, 3):
        f = DATA / split / f"{split}_source{n}.tsv"
        with open(f, "rb") as fh:
            lines = sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 24), b""))
        raw = pl.read_parquet(pp.CACHE_DIR / f"raw_{split}_s{n}.parquet", columns=["entity_id", "country"])
        raws[n] = raw
        check(raw.height == lines - 1, f"{split} S{n}: tsv data lines {lines - 1:,} == raw parquet rows {raw.height:,}")
        check(raw["entity_id"].n_unique() == raw.height, f"{split} S{n}: entity_id unique")
        check(raw["entity_id"].str.starts_with(f"S{n}-").all(), f"{split} S{n}: all ids prefixed S{n}-")
    # 2. norm s1: idx == raw row order, same ids, same rows
    ns1 = pp.scan_norm(split, "s1").select("idx", "entity_id", "country_n", "name_full", "addr").collect().sort("idx")
    check(ns1.height == raws[1].height, f"{split} norm S1 rows {ns1.height:,} == raw {raws[1].height:,}")
    check((ns1["idx"] == pl.int_range(0, ns1.height, eager=True)).all(), f"{split} norm S1 idx is 0..n-1")
    check((ns1["entity_id"] == raws[1]["entity_id"]).all(), f"{split} norm S1 row idx -> same entity as raw row (order preserved)")
    check((ns1["country_n"] == raws[1]["country"].str.strip_chars().str.to_lowercase()).all(), f"{split} norm S1 country matches raw")
    # 3. norm pool: S2 then S3, idx 0..n-1, ids in raw order
    npool = pp.scan_norm(split, "pool").select("idx", "entity_id", "src").collect().sort("idx")
    rawpool = pl.concat([raws[2]["entity_id"], raws[3]["entity_id"]])
    check(npool.height == rawpool.len(), f"{split} norm pool rows {npool.height:,} == raw S2+S3 {rawpool.len():,}")
    check((npool["entity_id"] == rawpool).all(), f"{split} norm pool order == raw S2 then S3")
    check((npool["src"] == npool["entity_id"].str.slice(0, 2)).all(), f"{split} norm pool src tag matches id prefix")
    # 4. empties after normalisation
    print(f"     {split} S1 empty name_full {int((ns1['name_full'].fill_null('') == '').sum()):,}  "
          f"empty addr {int((ns1['addr'].fill_null('') == '').sum()):,}")

# 5. ground truth ids exist and are exclusive
gt = pl.read_parquet(pp.CACHE_DIR / "gt_long.parquet")
s1ids = pl.read_parquet(pp.CACHE_DIR / "raw_train_s1.parquet", columns=["entity_id"])["entity_id"]
poolids = pl.concat([pl.read_parquet(pp.CACHE_DIR / f"raw_train_s{n}.parquet", columns=["entity_id"])["entity_id"] for n in (2, 3)])
g = gt.drop_nulls()
check(gt["s1_id"].is_in(s1ids.implode()).all(), "GT S1 ids all exist in train S1")
check(g["match_id"].is_in(poolids.implode()).all(), "GT match ids all exist in train S2/S3")
check(g["match_id"].n_unique() == g.height, f"GT exclusive: {g.height:,} pairs, each pool id once")
check(gt["s1_id"].n_unique() == s1ids.len(), f"GT covers every train S1 ({gt['s1_id'].n_unique():,})")

# 6. normalised samples per country (raw vs normalised)
for split, c in (("train", "india"), ("train", "us"), ("test", "france")):
    raw = pl.read_parquet(pp.CACHE_DIR / f"raw_{split}_s1.parquet").with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int64))
    n = pp.scan_norm(split, "s1").filter(pl.col("country_n") == c).select("idx", "name_full", "name_core", "addr", "postcode").collect().sample(4, seed=5)
    print(f"--- {split} {c}")
    print(n.join(raw.select("idx", "business_name", "business_address"), on="idx").select(
        "business_name", "name_core", "business_address", "addr", "postcode"))
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
