"""Pairwise features for (S1, candidate) pairs. Vectorised with rapidfuzz.cpdist."""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

TEXT_COLS = ["name_full", "name_core", "name_skel", "addr", "postcode", "addr_nums", "country_n", "src"]

FEATURES = [
    "bscore", "bscore_norm", "brank",
    "nf_ratio", "nf_tset", "nf_tsort", "nf_partial", "nf_jw",
    "nc_ratio", "nc_tset", "nc_jw", "nc_first_eq",
    "ad_ratio", "ad_tset", "ad_partial",
    "pc_eq", "pc_both", "num_jacc", "num_both",
    "len_name_s1", "len_name_c", "len_addr_s1", "len_addr_c", "addr_empty_any",
    "is_s3", "same_country",
    "n_cands", "bscore_gap_top", "nf_tset_rank",
    # v3
    "bname", "baddr", "bname_norm", "baddr_norm", "brank_name", "brank_addr",
    "ncc_ratio", "ncc_partial", "nsk_ratio", "nsk_tset", "hn_eq", "hn_both",
    "c_name_cnt", "s1_name_cnt", "s1_same_name", "c_name_ratio", "nc_tset_gap", "ad_tset_gap",
    # v5: blocking re-ranker score/rank, fuzzy house number (8250~8252 is a typo, 33 vs 46 a different
    # building), S1 initials as prefix of a handle-style candidate ("lex communication" -> "lcprivate")
    "rr", "rr_rank", "hn_sim", "hn_prefix", "init_pref",
]
# re-ranker inputs: blocking scores + three cheap fuzzy sims (see rerank_sims)
RR_FEATURES = ["bscore", "bscore_norm", "bname", "baddr", "bname_norm", "baddr_norm", "brank", "brank_name",
               "brank_addr", "r_cc_ratio", "r_nf_part", "r_ad_tset"]
# per-record name frequencies within the country, attached by add_name_counts (pool / S1 side)
CNT_COLS = ["ncnt_pool", "ncnt_s1"]
_HOUSE_NO = r"\b(\d+)"


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


def add_name_counts(frames: list[pl.DataFrame], pool: pl.DataFrame, s1_all: pl.LazyFrame) -> list[pl.DataFrame]:
    """Attach ncnt_pool (#pool records with this name_core in the country) and ncnt_s1 (#S1 records in the
    full S1 population of the country, not the sample) to each frame. Chains / generic names count high."""
    pc = pool.group_by("name_core").agg(pl.len().cast(pl.Float32).alias("ncnt_pool"))
    sc = s1_all.group_by("name_core").agg(pl.len().cast(pl.Float32).alias("ncnt_s1")).collect()
    return [f.join(pc, on="name_core", how="left", maintain_order="left")
            .join(sc, on="name_core", how="left", maintain_order="left")
            .with_columns(pl.col(CNT_COLS).fill_null(0.0)) for f in frames]


def rerank_sims(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, workers: int = -1) -> pl.DataFrame:
    """Adds r_cc_ratio (compact name), r_nf_part (full name partial), r_ad_tset (address token set) to
    wide blocking output. Cheap enough for ~350 candidates per S1."""
    cols = ["idx", "name_core", "name_full", "addr"]
    a = pairs.join(s1.select(cols).rename({c: c + "_1" for c in cols}),
                   left_on="s1_idx", right_on="idx_1", how="left", maintain_order="left")
    a = a.join(pool.select(cols).rename({c: c + "_2" for c in cols}),
               left_on="cand_idx", right_on="idx_2", how="left", maintain_order="left")
    g = lambda c: a[c].fill_null("").to_list()  # noqa: E731
    cc = lambda c: a[c].fill_null("").str.replace_all(" ", "", literal=True).to_list()  # noqa: E731
    return pairs.with_columns(
        pl.Series("r_cc_ratio", _pair_scores(cc("name_core_1"), cc("name_core_2"), fuzz.ratio, workers)),
        pl.Series("r_nf_part", _pair_scores(g("name_full_1"), g("name_full_2"), fuzz.partial_ratio, workers)),
        pl.Series("r_ad_tset", _pair_scores(g("addr_1"), g("addr_2"), fuzz.token_set_ratio, workers)))


def build_features(pairs: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, workers: int = -1) -> pl.DataFrame:
    """pairs: blocking output (s1_idx, cand_idx, bscore, ...). s1/pool: idx + TEXT_COLS + CNT_COLS."""
    cols = ["idx"] + TEXT_COLS + CNT_COLS
    a = pairs.join(s1.select(cols).rename({c: c + "_1" for c in cols}),
                   left_on="s1_idx", right_on="idx_1", how="left", maintain_order="left")
    a = a.join(pool.select(cols).rename({c: c + "_2" for c in cols}),
               left_on="cand_idx", right_on="idx_2", how="left", maintain_order="left")
    g = lambda c: a[c].fill_null("").to_list()  # noqa: E731
    nf1, nf2, nc1, nc2 = g("name_full_1"), g("name_full_2"), g("name_core_1"), g("name_core_2")
    ad1, ad2 = g("addr_1"), g("addr_2")
    nk1, nk2 = g("name_skel_1"), g("name_skel_2")
    cc1, cc2 = [x.replace(" ", "") for x in nc1], [x.replace(" ", "") for x in nc2]
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
        "ncc_ratio": _pair_scores(cc1, cc2, fuzz.ratio, workers),
        "ncc_partial": _pair_scores(cc1, cc2, fuzz.partial_ratio, workers),
        "nsk_ratio": _pair_scores(nk1, nk2, fuzz.ratio, workers),
        "nsk_tset": _pair_scores(nk1, nk2, fuzz.token_set_ratio, workers),
    }
    num_j, num_b = _jacc(g("addr_nums_1"), g("addr_nums_2"))
    f["num_jacc"], f["num_both"] = num_j, num_b
    hn1, hn2 = a["addr_1"].str.extract(_HOUSE_NO, 1), a["addr_2"].str.extract(_HOUSE_NO, 1)
    f["hn_sim"] = _pair_scores(hn1.fill_null("").to_list(), hn2.fill_null("").to_list(), fuzz.ratio, workers)
    feats = a.select("s1_idx", "cand_idx", "bscore", "bscore_norm", "brank", "bname", "baddr", "bname_norm",
                     "baddr_norm", "brank_name", "brank_addr", "rr", "rr_rank"
                     ).with_columns([pl.Series(k, v) for k, v in f.items()])
    h = pl.DataFrame({"h1": hn1, "h2": hn2, "n1": a["name_core_1"].fill_null(""),
                      "c2": a["name_core_2"].fill_null("").str.replace_all(" ", "", literal=True)})
    h = h.with_columns(pl.col("n1").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("i1"))
    extra = h.select(
        # one house number is a prefix of the other (27724 -> 2772, 175 -> 17): truncation noise
        (pl.col("h1").is_not_null() & pl.col("h2").is_not_null() & (pl.col("h1") != pl.col("h2"))
         & (pl.col("h1").str.starts_with(pl.col("h2")) | pl.col("h2").str.starts_with(pl.col("h1"))))
        .fill_null(False).cast(pl.Float32).alias("hn_prefix"),
        ((pl.col("i1").str.len_chars() >= 2) & (pl.col("i1").str.len_chars() < pl.col("n1").str.len_chars())
         & pl.col("c2").str.starts_with(pl.col("i1"))).fill_null(False).cast(pl.Float32).alias("init_pref"))
    feats = feats.with_columns(extra["hn_prefix"], extra["init_pref"])
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
        (hn1 == hn2).fill_null(False).cast(pl.Float32).alias("hn_eq"),
        (hn1.is_not_null() & hn2.is_not_null()).cast(pl.Float32).alias("hn_both"),
        a["ncnt_pool_1"].alias("s1_name_cnt"),
        a["ncnt_pool_2"].alias("c_name_cnt"),
        a["ncnt_s1_1"].alias("s1_same_name"),
        (a["ncnt_pool_2"] / a["ncnt_s1_2"].clip(lower_bound=1.0)).alias("c_name_ratio"),
    )
    # group context: how crowded / how far from best candidate
    feats = feats.with_columns(
        pl.len().over("s1_idx").alias("n_cands"),
        (pl.col("bscore").max().over("s1_idx") - pl.col("bscore")).alias("bscore_gap_top"),
        pl.col("nf_tset").rank("average", descending=True).over("s1_idx").alias("nf_tset_rank"),
        (pl.col("nc_tset").max().over("s1_idx") - pl.col("nc_tset")).alias("nc_tset_gap"),
        (pl.col("ad_tset").max().over("s1_idx") - pl.col("ad_tset")).alias("ad_tset_gap"),
    )
    return feats.with_columns(pl.col(FEATURES).cast(pl.Float32))
