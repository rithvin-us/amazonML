"""Candidate generation via IDF-weighted inverted index over hashed keys.

Keys per record (country-scoped so records only meet within same country label):
  n:<token>        name_core tokens            weight 1.0
  p:<tok[:4]>      name_core token prefixes    weight 0.5  (typo / suffix robustness)
  a:<token>        address tokens (len>=3)     weight 0.4
  z:<postcode>     postcode                    weight 1.5
Score(s1, cand) = sum over shared keys of weight * idf(key). Keys with pool doc-freq above
`max_df` are dropped (too common to be useful). Top-K per S1 kept.
Unknown/unseen countries work unchanged — country is just part of the key.
"""
from __future__ import annotations

import math

import polars as pl

KEY_WEIGHTS = {"n": 1.0, "p": 0.5, "a": 0.4, "z": 1.5}


def build_keys(df: pl.DataFrame, id_col: str = "idx") -> pl.DataFrame:
    """df needs idx(int), country_n, name_core, addr, postcode -> (idx, key:u64, w:f32)."""
    c = pl.col("country_n")
    name_tok = pl.col("name_core").str.split(" ")
    addr_tok = pl.col("addr").str.split(" ")
    names = df.select(pl.col(id_col), c, name_tok.alias("t")).explode("t")
    parts = [
        names.filter(pl.col("t").str.len_chars() >= 2)
             .select(id_col, pl.concat_str([pl.lit("n:"), c, pl.lit("|"), pl.col("t")]).alias("k"),
                     pl.lit(KEY_WEIGHTS["n"]).alias("w")),
        names.filter(pl.col("t").str.len_chars() >= 5)
             .select(id_col, pl.concat_str([pl.lit("p:"), c, pl.lit("|"), pl.col("t").str.slice(0, 4)]).alias("k"),
                     pl.lit(KEY_WEIGHTS["p"]).alias("w")),
        df.select(pl.col(id_col), c, addr_tok.alias("t")).explode("t")
          .filter(pl.col("t").str.len_chars() >= 3)
          .select(id_col, pl.concat_str([pl.lit("a:"), c, pl.lit("|"), pl.col("t")]).alias("k"),
                  pl.lit(KEY_WEIGHTS["a"]).alias("w")),
        df.filter(pl.col("postcode") != "")
          .select(id_col, pl.concat_str([pl.lit("z:"), c, pl.lit("|"), pl.col("postcode")]).alias("k"),
                  pl.lit(KEY_WEIGHTS["z"]).alias("w")),
    ]
    keys = pl.concat(parts).with_columns(pl.col("k").hash().alias("key"), pl.col("w").cast(pl.Float32))
    # a key counted once per record; keep max weight if duplicated
    return keys.group_by(id_col, "key").agg(pl.col("w").max())


class BlockIndex:
    """Build once over the pool; query with S1 chunks."""

    def __init__(self, pool: pl.DataFrame, max_df: int = 300):
        pk = build_keys(pool)
        df_ = pk.group_by("key").len().rename({"len": "df"})
        self.n_keys_total = df_.height
        df_ = df_.filter(pl.col("df") <= max_df).with_columns(
            (pl.lit(math.log(pool.height + 1)) - (pl.col("df") + 1).log()).cast(pl.Float32).alias("idf"))
        self.idf = df_.select("key", "idf")
        self.pk = pk.join(self.idf, on="key", how="inner").select(pl.col("idx").alias("cand_idx"), "key")

    def query(self, s1: pl.DataFrame, top_k: int = 50, chunk: int = 50_000, progress=None) -> pl.DataFrame:
        """s1 normalised frame with int idx -> (s1_idx, cand_idx, bscore, brank, bscore_norm)."""
        q = build_keys(s1).join(self.idf, on="key", how="inner")
        q = q.with_columns((pl.col("w") * pl.col("idf")).alias("qw")).select(
            pl.col("idx").alias("s1_idx"), "key", "qw")
        selfw = q.group_by("s1_idx").agg(pl.col("qw").sum().alias("qmax"))
        ids = q["s1_idx"].unique().sort()
        out = []
        for i in range(0, len(ids), chunk):
            qc = q.filter(pl.col("s1_idx").is_in(ids.slice(i, chunk).implode()))
            pairs = (qc.join(self.pk, on="key", how="inner")
                     .group_by("s1_idx", "cand_idx").agg(pl.col("qw").sum().alias("bscore")))
            pairs = (pairs.with_columns(pl.col("bscore").rank("ordinal", descending=True)
                                        .over("s1_idx").alias("brank"))
                     .filter(pl.col("brank") <= top_k))
            out.append(pairs)
            if progress:
                progress(min(1.0, (i + chunk) / max(len(ids), 1)))
        if not out:
            return pl.DataFrame(schema={"s1_idx": pl.Int64, "cand_idx": pl.Int64, "bscore": pl.Float32,
                                        "brank": pl.UInt32, "bscore_norm": pl.Float32})
        res = pl.concat(out).join(selfw, on="s1_idx")
        return res.with_columns((pl.col("bscore") / pl.col("qmax")).alias("bscore_norm")).drop("qmax")
