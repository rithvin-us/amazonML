"""Candidate generation via IDF-weighted inverted index over hashed keys (memory-lean CSR).

The index is built per country label (pipeline iterates countries), so keys carry no country.
Keys per record, each type hashed with its own seed -> u64 (no string concatenation):
  n  name_core tokens (len>=2)               weight 1.0
  b  consecutive name_core token bigrams     weight 1.0  (rescues names made of common words)
  p  name_core token prefixes [:4] (len>=5)  weight 0.5  (typo / suffix robustness)
  a  address tokens (len>=3)                 weight 0.4
  z  postcode                                weight 1.5
  c  adjacent address token pairs            weight 0.8  (order-robust street/number combos)
  x  first name token x address number       weight 1.0
  k  compact name_core prefix [:8] (len>=6)  weight 0.8  (names written as one word / domain / handle)
  e  exact compact name_core                 weight 1.5
  s  name_skel tokens (len>=3)               weight 0.5  (vowel / transliteration noise)
Two channels: name keys (n b p k e s) and address keys (a z c x). A candidate is kept if it is in
the top-K by total score OR top-k_name by name score OR top-k_addr by address score, so records
with an empty address (name keys only) are no longer crowded out by address-only neighbours.
Keys with pool doc-freq > `max_df` are dropped (too common to be useful). Postings are CSR
arrays (sorted kept keys -> offsets -> u32 candidate idx) filled in two chunked passes, so peak
RAM is ~ kept postings x 4 bytes + kept keys x 20 bytes instead of one big string frame.
Score(s1, cand) = sum over shared keys of weight * idf(key). Top-K per S1 kept.
Unknown/unseen countries work unchanged — they just get their own index.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import polars as pl

KEY_WEIGHTS = {"n": 1.0, "b": 1.0, "p": 0.5, "a": 0.4, "z": 1.5, "c": 0.8, "x": 1.0, "k": 0.8, "e": 1.5, "s": 0.5}
_SEED = {"n": 11, "b": 23, "p": 37, "a": 53, "z": 71, "c": 89, "x": 97, "k": 101, "e": 103, "s": 107}
CHANNEL = {**{t: 0 for t in "nbpkes"}, **{t: 1 for t in "azcx"}}  # 0 = name, 1 = address
BLOCK_VERSION = 3  # bump whenever build_keys key definitions change (invalidates cache/index/*)
_ARRAYS = ("keys", "idf", "offsets", "cands")
NUM = r"^\d+[a-z]?$"


def _mk(frame: pl.DataFrame, expr: pl.Expr, typ: str) -> pl.DataFrame:
    return frame.select("idx", expr.hash(_SEED[typ]).alias("key"),
                        pl.lit(KEY_WEIGHTS[typ], pl.Float32).alias("w"), pl.lit(CHANNEL[typ], pl.UInt8).alias("ch"))


def build_keys(df: pl.DataFrame, id_col: str = "idx") -> pl.DataFrame:
    """df needs id_col, name_core, name_skel, addr, postcode -> (idx:u32, key:u64, w:f32, ch:u8), one row per (idx, key)."""
    base = df.select(pl.col(id_col).cast(pl.UInt32).alias("idx"),
                     *[pl.col(c).fill_null("") for c in ("name_core", "name_skel", "addr", "postcode")])
    base = base.with_columns(pl.col("name_core").str.replace_all(" ", "", literal=True).alias("cc"))
    skel = (base.select("idx", pl.col("name_skel").str.split(" ").alias("t")).explode("t")
            .filter(pl.col("t").str.len_chars() >= 3))
    tok = (base.select("idx", pl.col("name_core").str.split(" ").alias("t")).explode("t")
           .filter(pl.col("t").str.len_chars() > 0)
           .with_columns(pl.when(pl.col("idx").shift(-1) == pl.col("idx"))
                         .then(pl.col("t").shift(-1)).alias("t2")))
    addr = (base.select("idx", pl.col("addr").str.split(" ").alias("t")).explode("t")
            .filter(pl.col("t").str.len_chars() >= 3))
    addr_all = (base.select("idx", pl.col("addr").str.split(" ").alias("t")).explode("t")
                .filter(pl.col("t").str.len_chars() > 0)
                .with_columns(pl.when(pl.col("idx").shift(-1) == pl.col("idx"))
                              .then(pl.col("t").shift(-1)).alias("t2")))
    first = tok.group_by("idx", maintain_order=True).agg(pl.col("t").first().alias("f"))
    nums = addr_all.filter(pl.col("t").str.contains(NUM)).join(first, on="idx")
    parts = [
        _mk(tok.filter(pl.col("t").str.len_chars() >= 2), pl.col("t"), "n"),
        _mk(tok.filter(pl.col("t2").is_not_null()), pl.concat_str(["t", "t2"], separator=" "), "b"),
        _mk(tok.filter(pl.col("t").str.len_chars() >= 5), pl.col("t").str.slice(0, 4), "p"),
        _mk(addr, pl.col("t"), "a"),
        _mk(base.filter(pl.col("postcode") != ""), pl.col("postcode"), "z"),
        _mk(addr_all.filter(pl.col("t2").is_not_null()), pl.concat_str(["t", "t2"], separator=" "), "c"),
        _mk(nums, pl.concat_str(["f", "t"], separator=" "), "x"),
        _mk(base.filter(pl.col("cc").str.len_chars() >= 6), pl.col("cc").str.slice(0, 8), "k"),
        _mk(base.filter(pl.col("cc") != ""), pl.col("cc"), "e"),
        _mk(skel, pl.col("t"), "s"),
    ]
    # a key counted once per record; keep max weight if duplicated
    return pl.concat(parts).group_by("idx", "key").agg(pl.col("w").max(), pl.col("ch").first())


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

    def save(self, d: Path) -> None:
        tmp = d.with_name(d.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for a in _ARRAYS:
            np.save(tmp / f"{a}.npy", getattr(self, a))
        (tmp / "meta.json").write_text(json.dumps({"n_keys_total": self.n_keys_total, "idx_dtype": str(self.idx_dtype)}))
        tmp.rename(d)

    @classmethod
    def load(cls, d: Path) -> "BlockIndex":
        """Memory-mapped arrays: file-backed pages, so they do not count against the RAM commit limit."""
        self = cls.__new__(cls)
        for a in _ARRAYS:
            setattr(self, a, np.load(d / f"{a}.npy", mmap_mode="r"))
        meta = json.loads((d / "meta.json").read_text())
        self.n_keys_total, self.idx_dtype = meta["n_keys_total"], getattr(pl, meta["idx_dtype"])
        self.n_postings = len(self.cands)
        return self

    @classmethod
    def cached(cls, pool: pl.DataFrame, d: Path, max_df: int, chunk: int) -> tuple["BlockIndex", bool]:
        """Load the index from d if present, else build, save and reload it memory-mapped. -> (index, cache_hit)."""
        if not (d / "meta.json").exists():
            ix = cls(pool, max_df=max_df, chunk=chunk)
            ix.save(d)
            del ix
            return cls.load(d), False
        return cls.load(d), True

    def _lookup(self, qk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """hashed keys -> (key id, found mask) via binary search over the sorted kept keys."""
        if not len(self.keys):
            return np.zeros(len(qk), np.int64), np.zeros(len(qk), bool)
        # sorted needles -> numpy's binary search reuses the previous bound (cache friendly); random-order
        # needles into a multi-million key array were ~10x slower and dominated index build time
        order = np.argsort(qk, kind="stable")
        pos = np.empty(len(qk), np.int64)
        pos[order] = np.searchsorted(self.keys, qk[order])
        pos[pos >= len(self.keys)] = 0
        return pos, self.keys[pos] == qk

    def query(self, s1: pl.DataFrame, top_k: int = 50, chunk: int = 4000, progress=None,
              k_name: int = 10, k_addr: int = 5, max_rows: int = 6_000_000) -> pl.DataFrame:
        """s1 normalised frame with int idx -> (s1_idx, cand_idx, bscore, brank, bscore_norm, bname, baddr,
        bname_norm, baddr_norm, brank_name, brank_addr).

        Kept: top_k by total score, plus up to k_name / k_addr extras ranked by name / address score among
        the candidates outside that top_k. brank_name / brank_addr are global channel ranks (null when the
        channel score is 0). S1 are processed in slices of <= `chunk` S1 and <= ~max_rows posting hits.
        """
        s1_dtype = s1.schema["idx"]
        q = build_keys(s1)
        kid, hit = self._lookup(q["key"].to_numpy())
        s1i, kid = q["idx"].to_numpy()[hit], kid[hit]
        qw = q["w"].to_numpy()[hit] * self.idf[kid]
        qc = q["ch"].to_numpy()[hit]
        order = np.argsort(s1i, kind="stable")
        s1i, kid, qw, qc = s1i[order], kid[order], qw[order], qc[order]
        qn = np.where(qc == 0, qw, 0).astype(np.float32)  # name-channel share of each key weight
        qa = (qw - qn).astype(np.float32)
        out = []
        if len(s1i):
            uniq, first = np.unique(s1i, return_index=True)
            qmax = pl.DataFrame({"s1_idx": uniq, "qmax": np.add.reduceat(qw, first),
                                 "qmax_n": np.add.reduceat(qn, first), "qmax_a": np.add.reduceat(qa, first)})
            bounds = np.r_[first, len(s1i)]
            hits = np.add.reduceat((self.offsets[kid + 1] - self.offsets[kid]).astype(np.int64), first)
            # slice ids: new slice when the posting budget or the S1 count per slice is exceeded
            sl = np.maximum(np.cumsum(hits) // max_rows, np.arange(len(uniq)) // chunk)
            starts = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1]])
            for a, b in zip(starts, np.r_[starts[1:], len(uniq)]):
                lo, hi = bounds[a], bounds[b]
                st = self.offsets[kid[lo:hi]]
                cnt = self.offsets[kid[lo:hi] + 1] - st
                tot = int(cnt.sum())
                if tot:
                    ramp = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    pairs = (pl.DataFrame({"s1_idx": np.repeat(s1i[lo:hi], cnt),
                                           "cand_idx": self.cands[np.repeat(st, cnt) + ramp],
                                           "qw": np.repeat(qw[lo:hi], cnt), "qn": np.repeat(qn[lo:hi], cnt),
                                           "qa": np.repeat(qa[lo:hi], cnt)})
                             .group_by("s1_idx", "cand_idx").agg(pl.col("qw").sum().alias("bscore"),
                                                                 pl.col("qn").sum().alias("bname"),
                                                                 pl.col("qa").sum().alias("baddr"))
                             .sort("s1_idx", "cand_idx")  # deterministic tie-breaks in the ordinal ranks
                             .with_columns(
                                 pl.col("bscore").rank("ordinal", descending=True).over("s1_idx").alias("brank"),
                                 *[pl.when(pl.col(c) > 0).then(pl.col(c)).rank("ordinal", descending=True)
                                   .over("s1_idx").alias(r) for c, r in (("bname", "brank_name"), ("baddr", "brank_addr"))])
                             .with_columns(
                                 *[pl.when((pl.col("brank") > top_k) & (pl.col(c) > 0)).then(pl.col(c))
                                   .rank("ordinal", descending=True).over("s1_idx").alias(r)
                                   for c, r in (("bname", "_rn"), ("baddr", "_ra"))])
                             .filter((pl.col("brank") <= top_k) | (pl.col("_rn") <= k_name) | (pl.col("_ra") <= k_addr))
                             .drop("_rn", "_ra"))
                    out.append(pairs)
                if progress:
                    progress(min(1.0, b / len(uniq)))
        if not out:
            return pl.DataFrame(schema={"s1_idx": s1_dtype, "cand_idx": self.idx_dtype, "bscore": pl.Float32,
                                        "brank": pl.UInt32, "bscore_norm": pl.Float32, "bname": pl.Float32,
                                        "baddr": pl.Float32, "bname_norm": pl.Float32, "baddr_norm": pl.Float32,
                                        "brank_name": pl.UInt32, "brank_addr": pl.UInt32})
        res = pl.concat(out).join(qmax, on="s1_idx")
        nz = lambda c: pl.when(pl.col(c) > 0).then(pl.col(c))  # noqa: E731
        return res.select(pl.col("s1_idx").cast(s1_dtype), pl.col("cand_idx").cast(self.idx_dtype),
                          pl.col("bscore").cast(pl.Float32), "brank",
                          (pl.col("bscore") / pl.col("qmax")).cast(pl.Float32).alias("bscore_norm"),
                          pl.col("bname").cast(pl.Float32), pl.col("baddr").cast(pl.Float32),
                          (pl.col("bname") / nz("qmax_n")).cast(pl.Float32).alias("bname_norm"),
                          (pl.col("baddr") / nz("qmax_a")).cast(pl.Float32).alias("baddr_norm"),
                          pl.col("brank_name").cast(pl.UInt32), pl.col("brank_addr").cast(pl.UInt32))
