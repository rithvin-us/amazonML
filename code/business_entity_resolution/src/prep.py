"""Streamed normalisation: raw parquet -> cache/norm_<split>_<kind>/part-*.parquet.

Kept deliberately light on imports: on Windows every multiprocessing worker re-imports
the __main__ module, so heavy libs here would be loaded once per worker.

  python prep.py <split> [--workers N]
"""
from __future__ import annotations

import argparse
import shutil
from multiprocessing import Pool

import polars as pl
import pyarrow.parquet as pq

import indic
import normalize as nz
from config import CACHE_DIR, NORM_VERSION
from io_utils import load_source


def _norm_chunk(args):
    names, addrs = args
    nn = [nz.norm_name(x) for x in names]
    addrs = [x or "" for x in addrs]
    aa = [nz.norm_addr(x) for x in addrs]
    return ([n[0] for n in nn], [n[1] for n in nn], [n[2] for n in nn], aa,
            [nz.postcode(x) for x in addrs], [nz.numbers(a) for a in aa])


def norm_dir(split: str, kind: str):
    return CACHE_DIR / f"norm_{split}_{kind}_{NORM_VERSION}"


def prep(split: str, workers: int = 6, batch: int = 200_000) -> None:
    indic.ensure_dict()  # learned from train GT; must exist before workers import normalize
    for kind in ("s1", "pool"):
        out = norm_dir(split, kind)
        if (out / "_DONE").exists():
            print(f"cache hit {out.name}", flush=True)
            continue
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        srcs = [("S1", 1)] if kind == "s1" else [("S2", 2), ("S3", 3)]
        for _, n in srcs:
            if not (CACHE_DIR / f"raw_{split}_s{n}.parquet").exists():
                load_source(split, n)
        total = sum(pq.ParquetFile(CACHE_DIR / f"raw_{split}_s{n}.parquet").metadata.num_rows for _, n in srcs)
        offset = part = 0
        with Pool(workers, maxtasksperchild=200) as p:
            for tag, n in srcs:
                pf = pq.ParquetFile(CACHE_DIR / f"raw_{split}_s{n}.parquet")
                for rb in pf.iter_batches(batch_size=batch):
                    df = pl.from_arrow(rb)
                    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
                    step = 10_000
                    jobs = [(names[i:i + step], addrs[i:i + step]) for i in range(0, len(names), step)]
                    cols = [[], [], [], [], [], []]
                    for res in p.imap(_norm_chunk, jobs):
                        for c, r in zip(cols, res):
                            c.extend(r)
                    df = df.select(
                        (pl.int_range(pl.len(), dtype=pl.Int64) + offset).alias("idx"), "entity_id",
                        pl.Series("name_full", cols[0]), pl.Series("name_core", cols[1]),
                        pl.Series("name_skel", cols[2]), pl.Series("addr", cols[3]),
                        pl.Series("postcode", cols[4]), pl.Series("addr_nums", cols[5]),
                        pl.col("country").str.strip_chars().str.to_lowercase().alias("country_n"),
                        pl.lit(tag).alias("src"))
                    df.write_parquet(out / f"part-{part:04d}.parquet")
                    offset += df.height
                    part += 1
                    del df, names, addrs, cols, jobs
                    print(f"PROGRESS {offset / total:.4f} {split}/{kind} {offset:,}/{total:,}", flush=True)
        (out / "_DONE").write_text(str(offset))
        print(f"wrote {out.name}: {offset:,} rows in {part} parts", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    prep(a.split, a.workers)
