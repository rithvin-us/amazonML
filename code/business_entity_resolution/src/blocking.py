"""Candidate generation via IDF-weighted inverted index over hashed keys (memory-lean CSR).

The index is built per country label (pipeline iterates countries), so keys carry no country.
Keys per record, each type hashed with its own seed -> u64 (no string concatenation):
  n  name_core tokens (len>=2)               weight 1.0
  b  consecutive name_core token bigrams     weight 1.0  (rescues names made of common words)
  p  name_core token prefixes [:4] (len>=5)  weight 0.5  (typo / suffix robustness)
  a  address tokens (len>=3)                 weight 0.4
  z  postcode                                weight 1.5
Keys with pool doc-freq > `max_df` are dropped (too common to be useful). Postings are CSR
arrays (sorted kept keys -> offsets -> u32 candidate idx) filled in two chunked passes, so peak
RAM is ~ kept postings x 4 bytes + kept keys x 20 bytes instead of one big string frame.
Score(s1, cand) = sum over shared keys of weight * idf(key). Top-K per S1 kept.
Unknown/unseen countries work unchanged — they just get their own index.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

KEY_WEIGHTS = {"n": 1.0, "b": 1.0, "p": 0.5, "a": 0.4, "z": 1.5}
_SEED = {"n": 11, "b": 23, "p": 37, "a": 53, "z": 71}


def _mk(frame: pl.DataFrame, expr: pl.Expr, typ: str) -> pl.DataFrame:
    return frame.select("idx", expr.hash(_SEED[typ]).alias("key"),
                        pl.lit(KEY_WEIGHTS[typ], pl.Float32).alias("w"))


def build_keys(df: pl.DataFrame, id_col: str = "idx") -> pl.DataFrame:
    """df needs id_col, name_core, addr, postcode -> (idx:u32, key:u64, w:f32), one row per (idx, key)."""
    base = df.select(pl.col(id_col).cast(pl.UInt32).alias("idx"),
                     *[pl.col(c).fill_null("") for c in ("name_core", "addr", "postcode")])
    tok = (base.select("idx", pl.col("name_core").str.split(" ").alias("t")).explode("t")
           .filter(pl.col("t").str.len_chars() > 0)
           .with_columns(pl.when(pl.col("idx").shift(-1) == pl.col("idx"))
                         .then(pl.col("t").shift(-1)).alias("t2")))
    addr = (base.select("idx", pl.col("addr").str.split(" ").alias("t")).explode("t")
            .filter(pl.col("t").str.len_chars() >= 3))
    parts = [
        _mk(tok.filter(pl.col("t").str.len_chars() >= 2), pl.col("t"), "n"),
        _mk(tok.filter(pl.col("t2").is_not_null()), pl.concat_str(["t", "t2"], separator=" "), "b"),
        _mk(tok.filter(pl.col("t").str.len_chars() >= 5), pl.col("t").str.slice(0, 4), "p"),
        _mk(addr, pl.col("t"), "a"),
        _mk(base.filter(pl.col("postcode") != ""), pl.col("postcode"), "z"),
    ]
    # a key counted once per record; keep max weight if duplicated
    return pl.concat(parts).group_by("idx", "key").agg(pl.col("w").max())


class BlockIndex:
    """Build once over one country's pool; query with S1 chunks."""

    def __init__(self, pool: pl.DataFrame, max_df: int = 150, chunk: int = 250_000):
        n = pool.height
        self.idx_dtype = pool.schema["idx"]
        # pass 1: doc-freq per key
        dfc = None
        for i in range(0, n, chunk):
            c = build_keys(pool.slice(i, chunk)).group_by("key").len()
            dfc = c if dfc is None else pl.concat([dfc, c]).group_by("key").agg(pl.col("len").sum())
        if dfc is None:
            dfc = pl.DataFrame(schema={"key": pl.UInt64, "len": pl.UInt32})
        self.n_keys_total = dfc.height
        kept = (dfc.filter(pl.col("len") <= max_df).sort("key")
                .with_columns((math.log(n + 1) - (pl.col("len").cast(pl.Float64) + 1).log())
                              .cast(pl.Float32).alias("idf")))
        del dfc
        self.keys = kept["key"].to_numpy()
        self.idf = kept["idf"].to_numpy()
        self.offsets = np.zeros(kept.height + 1, np.int64)
        np.cumsum(kept["len"].to_numpy().astype(np.int64), out=self.offsets[1:])
        del kept
        # pass 2: fill postings, grouped by key id
        self.cands = np.empty(int(self.offsets[-1]), np.uint32)
        fill = self.offsets[:-1].copy()
        for i in range(0, n, chunk):
            p = build_keys(pool.slice(i, chunk)).select("idx", "key")
            kid, hit = self._lookup(p["key"].to_numpy())
            kid, idx = kid[hit], p["idx"].to_numpy()[hit]
            order = np.argsort(kid, kind="stable")
            kid, idx = kid[order], idx[order]
            if not len(kid):
                continue
            first = np.flatnonzero(np.r_[True, kid[1:] != kid[:-1]])
            run_len = np.diff(np.r_[first, len(kid)])
            pos_in_run = np.arange(len(kid)) - np.repeat(first, run_len)
            self.cands[fill[kid] + pos_in_run] = idx
            fill[kid[first]] += run_len
        self.n_postings = len(self.cands)

    def _lookup(self, qk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """hashed keys -> (key id, found mask) via binary search over the sorted kept keys."""
        if not len(self.keys):
            return np.zeros(len(qk), np.int64), np.zeros(len(qk), bool)
        pos = np.searchsorted(self.keys, qk)
        pos[pos >= len(self.keys)] = 0
        return pos, self.keys[pos] == qk

    def query(self, s1: pl.DataFrame, top_k: int = 50, chunk: int = 4000, progress=None) -> pl.DataFrame:
        """s1 normalised frame with int idx -> (s1_idx, cand_idx, bscore, brank, bscore_norm)."""
        s1_dtype = s1.schema["idx"]
        q = build_keys(s1)
        kid, hit = self._lookup(q["key"].to_numpy())
        s1i, kid = q["idx"].to_numpy()[hit], kid[hit]
        qw = q["w"].to_numpy()[hit] * self.idf[kid]
        order = np.argsort(s1i, kind="stable")
        s1i, kid, qw = s1i[order], kid[order], qw[order]
        out = []
        if len(s1i):
            uniq, first = np.unique(s1i, return_index=True)
            qmax = pl.DataFrame({"s1_idx": uniq, "qmax": np.add.reduceat(qw, first)})
            bounds = np.r_[first, len(s1i)]
            for ci in range(0, len(uniq), chunk):
                lo, hi = bounds[ci], bounds[min(ci + chunk, len(uniq))]
                st = self.offsets[kid[lo:hi]]
                cnt = self.offsets[kid[lo:hi] + 1] - st
                tot = int(cnt.sum())
                if tot:
                    ramp = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    pairs = (pl.DataFrame({"s1_idx": np.repeat(s1i[lo:hi], cnt),
                                           "cand_idx": self.cands[np.repeat(st, cnt) + ramp],
                                           "qw": np.repeat(qw[lo:hi], cnt)})
                             .group_by("s1_idx", "cand_idx").agg(pl.col("qw").sum().alias("bscore"))
                             .with_columns(pl.col("bscore").rank("ordinal", descending=True)
                                           .over("s1_idx").alias("brank"))
                             .filter(pl.col("brank") <= top_k))
                    out.append(pairs)
                if progress:
                    progress(min(1.0, (ci + chunk) / len(uniq)))
        if not out:
            return pl.DataFrame(schema={"s1_idx": s1_dtype, "cand_idx": self.idx_dtype, "bscore": pl.Float32,
                                        "brank": pl.UInt32, "bscore_norm": pl.Float32})
        res = pl.concat(out).join(qmax, on="s1_idx")
        return res.select(pl.col("s1_idx").cast(s1_dtype), pl.col("cand_idx").cast(self.idx_dtype),
                          pl.col("bscore").cast(pl.Float32), "brank",
                          (pl.col("bscore") / pl.col("qmax")).cast(pl.Float32).alias("bscore_norm"))
