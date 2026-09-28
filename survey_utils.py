"""Shared helpers for the processing scripts (text normalisation, programme registry)."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRIVATE_DIR = PROJECT_ROOT / "data" / "private"
DOCS_DIR = PROJECT_ROOT / "docs"
REGISTRY_PATH = REFERENCE_DIR / "programs_registry.csv"

ACADEMIC_YEAR = "2026-27"

METADATA_COLS = [
    "Номер ответа",
    "Дата ответа",
    "Время выполнения",
    "IP - адрес",
    "Источник распространения",
    "Операционная система",
    "Браузер",
    "Устройство",
]


def clean_text(value) -> str | float:
    """NFKC + collapse whitespace + strip. Survey exports mix NBSP, stray '·' and
    decomposed Cyrillic; NaN/empty stays NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = unicodedata.normalize("NFKC", str(value)).replace("·", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else np.nan


def normalize_key(value) -> str:
    """Case/ё-insensitive key for matching programme names."""
    s = clean_text(value)
    if not isinstance(s, str):
        return ""
    return s.lower().replace("ё", "е")


def normalize_ranges(value) -> str | float:
    """'6-10' / '6 — 10' -> '6–10' (en dash) so answer buckets match across blocks."""
    s = clean_text(value)
    if not isinstance(s, str):
        return s
    return re.sub(r"(\d)\s*[-–—]\s*(\d)", r"\1–\2", s)


def strip_qnum(col: str) -> str:
    return re.sub(r"^\d+\.\s*", "", str(col)).strip()


def load_registry(path: Path = REGISTRY_PATH) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: registry not found: {path}")
        return pd.DataFrame(columns=["Программа", "Школа", "Уровень", "Курсы"])
    reg = pd.read_csv(path, sep=";", dtype=str)
    reg["program_key"] = reg["Программа"].map(normalize_key)
    return reg


def enrich_from_registry(df: pd.DataFrame, program_col: str = "program", year_col: str = "year") -> pd.DataFrame:
    """Add `school` and `level` by programme name; fill an empty `year` when the
    registry lists exactly one course for the programme."""
    reg = load_registry()
    out = df.copy()
    key = out[program_col].map(normalize_key)
    school = reg.set_index("program_key")["Школа"].to_dict()
    level = reg.set_index("program_key")["Уровень"].to_dict()
    out["school"] = key.map(school)
    out["level"] = key.map(level)

    if year_col in out.columns and "Курсы" in reg.columns:
        single = {
            r.program_key: f"{r.Курсы.strip()} курс"
            for r in reg.itertuples()
            if isinstance(r.Курсы, str) and r.Курсы.strip().isdigit()
        }
        missing = out[year_col].isna() | (out[year_col].astype(str).str.strip() == "")
        out.loc[missing, year_col] = key[missing].map(single)

    unknown = sorted(out.loc[out["school"].isna(), program_col].dropna().unique())
    if unknown:
        print("WARNING: programmes missing from the registry (school is empty):")
        for p in unknown:
            print(f"  - {p}")
    return out
