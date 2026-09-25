"""Name / address canonicalisation. Pure functions + a polars-wide apply.

Country-agnostic: maps cover common English/Indian/French variants but nothing is
filtered by country, so unseen countries still get generic cleaning.
"""
from __future__ import annotations

import re
import unicodedata

import polars as pl
from unidecode import unidecode

LEGAL = {
    # english / india
    "private": "pvt", "pvt": "pvt", "pvt.": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltd.": "ltd", "llp": "llp", "llc": "llc", "l.l.c": "llc",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp", "co": "co",
    "company": "co", "compny": "co", "plc": "plc", "lp": "lp",
    # french / eu
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "societe": "ste", "ste": "ste", "gmbh": "gmbh", "bv": "bv", "srl": "srl",
}
LEGAL_TOKENS = set(LEGAL.values())
NAME_STOP = {"the", "and", "of", "de", "la", "le", "les", "du", "des", "et", "m/s", "ms", "dba"}

ADDR = {
    "road": "rd", "rd": "rd", "street": "st", "st": "st", "str": "st", "avenue": "ave", "ave": "ave",
    "av": "ave", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "suite": "ste",
    "ste": "ste", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy",
    "north": "n", "south": "s", "east": "e", "west": "w", "n": "n", "s": "s", "e": "e", "w": "w",
    "apartment": "apt", "apt": "apt", "building": "bldg", "bldg": "bldg", "floor": "fl", "fl": "fl",
    "nagar": "ngr", "ngr": "ngr", "colony": "col", "sector": "sec", "sec": "sec", "near": "nr",
    "nr": "nr", "opposite": "opp", "opp": "opp", "district": "dist", "dist": "dist",
    "rue": "rue", "r": "rue", "chemin": "ch", "ch": "ch", "allee": "all", "impasse": "imp",
    "route": "rte", "rte": "rte", "cedex": "", "unit": "unit", "#": "",
}
# Region/state full names -> short code (applied as phrase replacement before tokenising).
# Generic lookup table over common US/Indian/French regions; unknown regions pass through.
STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od",
    "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk",
    "west bengal": "wb", "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "ile de france": "idf", "ile-de-france": "idf", "hauts de france": "hdf", "hauts-de-france": "hdf",
    "provence alpes cote d azur": "paca", "auvergne rhone alpes": "ara", "nouvelle aquitaine": "naq",
    "occitanie": "occ", "grand est": "ges", "bretagne": "bre", "normandie": "nor",
    "pays de la loire": "pdl", "centre val de loire": "cvl", "bourgogne franche comte": "bfc",
}
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, STATES), key=len, reverse=True)) + r")\b")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")
_PIN6 = re.compile(r"\b\d{6}\b")
_ZIP5 = re.compile(r"\b\d{5}\b")


def to_ascii(s: str) -> str:
    """NFKC, transliterate (Devanagari/accents -> latin), lowercase."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    return unidecode(s).lower()


def _tokens(s: str) -> list[str]:
    s = s.replace("&", " and ").replace("@", " at ")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip().split()


def norm_name(raw: str) -> tuple[str, str]:
    """-> (full normalised name, core name without legal suffixes/stopwords)."""
    toks = [LEGAL.get(t, t) for t in _tokens(to_ascii(raw))]
    full = " ".join(toks)
    core = " ".join(t for t in toks if t not in LEGAL_TOKENS and t not in NAME_STOP)
    return full, core or full


def norm_addr(raw: str) -> str:
    s = _WS.sub(" ", _PUNCT.sub(" ", to_ascii(raw)))
    s = _STATE_RE.sub(lambda m: STATES[m.group(1)], s)
    toks = [ADDR.get(t, t) for t in _tokens(s)]
    return " ".join(t for t in toks if t)


def postcode(raw: str) -> str:
    m = _PIN6.search(raw) or _ZIP5.search(raw)
    return m.group(0) if m else ""


def numbers(s: str) -> str:
    return " ".join(sorted(set(_NUM.findall(s))))


def normalize_df(df: pl.DataFrame) -> pl.DataFrame:
    """Adds name_full, name_core, addr, postcode, addr_nums, country_n columns."""
    names = [norm_name(x) for x in df["business_name"].to_list()]
    addrs_raw = df["business_address"].to_list()
    addrs = [norm_addr(x) for x in addrs_raw]
    return df.with_columns(
        pl.Series("name_full", [n[0] for n in names]),
        pl.Series("name_core", [n[1] for n in names]),
        pl.Series("addr", addrs),
        pl.Series("postcode", [postcode(x) for x in addrs_raw]),
        pl.Series("addr_nums", [numbers(a) for a in addrs]),
        pl.col("country").str.strip_chars().str.to_lowercase().alias("country_n"),
    )
