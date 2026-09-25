"""Pairwise features for (S1, candidate) pairs. Vectorised with rapidfuzz.cpdist."""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

TEXT_COLS = ["name_full", "name_core", "addr", "postcode", "addr_nums", "country_n", "src"]

FEATURES = [
    "bscore", "bscore_norm", "brank",
    "nf_ratio", "nf_tset", "nf_tsort", "nf_partial", "nf_jw",
    "nc_ratio", "nc_tset", "nc_jw", "nc_first_eq",
    "ad_ratio", "ad_tset", "ad_partial",
    "pc_eq", "pc_both", "num_jacc", "num_both",
    "len_name_s1", "len_name_c", "len_addr_s1", "len_addr_c", "addr_empty_any",
    "is_s3", "same_country",
    "n_cands", "bscore_gap_top", "nf_tset_rank",
]


def _pair_scores(a: list[str], b: list[str], scorer, workers: int) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _jacc(a: list[str], b: list[str]) -> tuple[np.ndarray, np.ndarray]:
    j = np.zeros(len(a), np.float32)
    both = np.zeros(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            j[i] = len(sx & sy) / len(sx | sy)
            both[i] = 1
    return j, both


def build_features(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame,
                   workers: int = -1) -> pl.DataFrame:
    """pairs: s1_idx,cand_idx,bscore,bscore_norm,brank. s1/pool: idx + TEXT_COLS."""
    a = pairs.join(s1.select(["idx"] + TEXT_COLS).rename({c: c + "_1" for c in ["idx"] + TEXT_COLS}),
                   left_on="s1_idx", right_on="idx_1", how="left")
    a = a.join(pool.select(["idx"] + TEXT_COLS).rename({c: c + "_2" for c in ["idx"] + TEXT_COLS}),
               left_on="cand_idx", right_on="idx_2", how="left")
    g = lambda c: a[c].fill_null("").to_list()  # noqa: E731
    nf1, nf2, nc1, nc2 = g("name_full_1"), g("name_full_2"), g("name_core_1"), g("name_core_2")
    ad1, ad2 = g("addr_1"), g("addr_2")
    f = {
        "nf_ratio": _pair_scores(nf1, nf2, fuzz.ratio, workers),
        "nf_tset": _pair_scores(nf1, nf2, fuzz.token_set_ratio, workers),
        "nf_tsort": _pair_scores(nf1, nf2, fuzz.token_sort_ratio, workers),
        "nf_partial": _pair_scores(nf1, nf2, fuzz.partial_ratio, workers),
        "nf_jw": _pair_scores(nf1, nf2, distance.JaroWinkler.normalized_similarity, workers),
        "nc_ratio": _pair_scores(nc1, nc2, fuzz.ratio, workers),
        "nc_tset": _pair_scores(nc1, nc2, fuzz.token_set_ratio, workers),
        "nc_jw": _pair_scores(nc1, nc2, distance.JaroWinkler.normalized_similarity, workers),
        "ad_ratio": _pair_scores(ad1, ad2, fuzz.ratio, workers),
        "ad_tset": _pair_scores(ad1, ad2, fuzz.token_set_ratio, workers),
        "ad_partial": _pair_scores(ad1, ad2, fuzz.partial_ratio, workers),
    }
    num_j, num_b = _jacc(g("addr_nums_1"), g("addr_nums_2"))
    f["num_jacc"], f["num_both"] = num_j, num_b
    feats = a.select("s1_idx", "cand_idx", "bscore", "bscore_norm", "brank").with_columns(
        [pl.Series(k, v) for k, v in f.items()])
    feats = feats.with_columns(
        (a["name_core_1"].str.split(" ").list.first() == a["name_core_2"].str.split(" ").list.first())
        .cast(pl.Float32).alias("nc_first_eq"),
        ((a["postcode_1"] == a["postcode_2"]) & (a["postcode_1"] != "")).cast(pl.Float32).alias("pc_eq"),
        ((a["postcode_1"] != "") & (a["postcode_2"] != "")).cast(pl.Float32).alias("pc_both"),
        a["name_full_1"].str.len_chars().alias("len_name_s1"),
        a["name_full_2"].str.len_chars().alias("len_name_c"),
        a["addr_1"].str.len_chars().alias("len_addr_s1"),
        a["addr_2"].str.len_chars().alias("len_addr_c"),
        ((a["addr_1"] == "") | (a["addr_2"] == "")).cast(pl.Float32).alias("addr_empty_any"),
        (a["src_2"] == "S3").cast(pl.Float32).alias("is_s3"),
        (a["country_n_1"] == a["country_n_2"]).cast(pl.Float32).alias("same_country"),
    )
    # group context: how crowded / how far from best candidate
    feats = feats.with_columns(
        pl.len().over("s1_idx").alias("n_cands"),
        (pl.col("bscore").max().over("s1_idx") - pl.col("bscore")).alias("bscore_gap_top"),
        pl.col("nf_tset").rank("average", descending=True).over("s1_idx").alias("nf_tset_rank"),
    )
    return feats.with_columns(pl.col(FEATURES).cast(pl.Float32))
