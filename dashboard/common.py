"""Helpers shared by the autumn and winter/spring (SFQ) parts of the dashboard."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
DOCS_DIR = PROJECT_ROOT / "docs"

ROUND_DECIMALS = 2

# Categorical order is fixed (never cycled) and CVD-validated for adjacent pairs.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT = PALETTE[0]
MUTED = "#94A3B8"
SEQUENTIAL = "Blues"
DIVERGING = "RdBu"

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 1.0rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 1400px;
}
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
}
</style>
"""


# ── Generic helpers ───────────────────────────────────────────────────────────

def num_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def round_df(df: pd.DataFrame, decimals: int = ROUND_DECIMALS) -> pd.DataFrame:
    out = df.copy()
    num = out.select_dtypes(include=[np.number]).columns
    out[num] = out[num].round(decimals)
    return out


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_unicode_columns(
    df: pd.DataFrame, cols: Iterable[str] = ("program", "year", "school", "teacher")
) -> pd.DataFrame:
    """Normalize key text columns to NFC.

    Survey exports mix precomposed (NFC) and decomposed (NFD) Unicode for some
    Cyrillic letters (e.g. «й» as и+◌̆). Without this, identical-looking program
    names are treated as distinct in concat/groupby/filters and fail to match the
    enrollment table."""
    for c in cols:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].map(lambda v: unicodedata.normalize("NFC", v) if isinstance(v, str) else v)
    return df


def load_codebook(path: Path | str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def shorten_program_name(name: str, max_len: int = 34) -> str:
    if pd.isna(name):
        return name
    s = str(name).strip()
    replacements = {
        "Менеджмент в креативных индустриях (сетевая программа)": "Менеджмент в КИ (сетевая)",
        "Двухмерная графика: анимация, концепт-арт, комиксы": "2D-графика: анимация, концепт-арт",
        "Стратегические коммуникации и управление репутацией": "Стратег. коммуникации и репутация",
        "Архитектурное проектирование и реконструкция зданий": "Арх. проектирование и реконструкция",
        "в креативных индустриях": "в КИ",
        "и городских общественных пространств": "и городских пространств",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def add_program_display(df: pd.DataFrame, short_labels: bool) -> pd.DataFrame:
    out = df.copy()
    if "program" in out.columns:
        out["program_display"] = out["program"].map(lambda x: shorten_program_name(x) if short_labels else x)
    return out


def render_interp(title: str, rules: list[str]) -> None:
    with st.expander(title, expanded=False):
        for rule in rules:
            st.markdown(f"- {rule}")


# ── Statistics ───────────────────────────────────────────────────────────────

def bootstrap_mean_ci(x: pd.Series, n_boot: int = 3000, ci: int = 95, seed: int = 42) -> tuple[float, float, float, int]:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    n = vals.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (100 - ci) / 2
    return vals.mean(), np.percentile(draws, alpha), np.percentile(draws, 100 - alpha), n


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion (in %); stable for small n and p near 0/1."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def chi2_summary(table: pd.DataFrame) -> dict:
    """Chi-square test of independence + Cramér's V for a counts table."""
    t = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return {"chi2": np.nan, "p": np.nan, "dof": 0, "cramers_v": np.nan, "low_expected_share": np.nan}
    chi2, p, dof, expected = stats.chi2_contingency(t.to_numpy())
    n = t.to_numpy().sum()
    k = min(t.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else np.nan
    return {
        "chi2": chi2,
        "p": p,
        "dof": dof,
        "cramers_v": v,
        "low_expected_share": float((expected < 5).mean()),
    }


# ── Filters ──────────────────────────────────────────────────────────────────

@dataclass
class FilterState:
    programs: list[str]
    years: list[str]
    schools: list[str]
    short_program_labels: bool


def build_filters(df: pd.DataFrame, key_prefix: str) -> FilterState:
    st.sidebar.header("Фильтры")
    st.sidebar.caption("Фильтры применяются ко всем вкладкам дашборда.")

    programs = sorted(df["program"].dropna().unique().tolist()) if "program" in df.columns else []
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    schools = sorted(df["school"].dropna().unique().tolist()) if "school" in df.columns else []

    selected_schools = st.sidebar.multiselect("Школа", schools, default=schools, key=f"{key_prefix}_schools")
    selected_years = st.sidebar.multiselect("Курс", years, default=years, key=f"{key_prefix}_years")
    selected_programs = st.sidebar.multiselect("Программа", programs, default=programs, key=f"{key_prefix}_programs")
    short_program_labels = st.sidebar.checkbox(
        "Сокращать длинные названия программ",
        value=True,
        help="Укорачивает подписи в графиках/легендах для лучшей читаемости, особенно на телефоне.",
        key=f"{key_prefix}_short",
    )
    return FilterState(
        programs=selected_programs,
        years=selected_years,
        schools=selected_schools,
        short_program_labels=short_program_labels,
    )


def apply_filters(df: pd.DataFrame, f: FilterState) -> pd.DataFrame:
    out = df.copy()
    if "program" in out.columns and f.programs:
        out = out[out["program"].isin(f.programs)]
    if "year" in out.columns and f.years:
        out = out[out["year"].isin(f.years)]
    if "school" in out.columns and f.schools:
        out = out[out["school"].isin(f.schools)]
    return out


# ── Enrollment / response rate ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_contingent(path: str = str(REFERENCE_DIR / "contingent.csv")) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["program", "year", "school", "contingent"])
    df = pd.read_csv(p)
    df["contingent"] = pd.to_numeric(df["contingent"], errors="coerce")
    return normalize_unicode_columns(df)


def filter_contingent(cont: pd.DataFrame, f: FilterState) -> pd.DataFrame:
    """Filter the enrollment table to match a FilterState.

    Rows with an empty `year` belong to single-course programmes and are kept
    regardless of the year filter (the programme has no course split)."""
    out = cont.copy()
    if out.empty:
        return out
    if f.programs:
        out = out[out["program"].isin(f.programs)]
    if f.schools:
        out = out[out["school"].isin(f.schools) | out["school"].isna()]
    if f.years:
        out = out[out["year"].isin(f.years) | out["year"].isna()]
    return out


def margin_of_error_pct(n: int, N: float, z: float = 1.96) -> float:
    """95% margin of error (%) for a proportion at worst case p=0.5,
    with finite population correction. Returns 0 for a census (n>=N)."""
    if n <= 0 or not np.isfinite(N) or N <= 0:
        return float("nan")
    if n >= N:
        return 0.0
    fpc = (N - n) / (N - 1) if N > 1 else 0.0
    return float(z * np.sqrt(0.25 / n) * np.sqrt(fpc) * 100.0)


def response_rate_table(df_slice: pd.DataFrame, cont_slice: pd.DataFrame) -> pd.DataFrame:
    """Per-programme response rate / margin of error for the current slice."""
    n_by_prog = df_slice.groupby("program").size().rename("n")
    N_by_prog = cont_slice.groupby("program")["contingent"].sum().rename("N")
    rr = pd.concat([n_by_prog, N_by_prog], axis=1)
    rr["n"] = rr["n"].fillna(0).astype(int)
    rr["RR, %"] = np.where(rr["N"] > 0, rr["n"] / rr["N"] * 100.0, np.nan)
    rr["Погрешность ±%"] = [margin_of_error_pct(int(r.n), r.N) for r in rr.itertuples()]
    return rr.reset_index().rename(columns={"program": "Программа", "N": "Контингент"})


def render_response_rate(df: pd.DataFrame, cont_slice: pd.DataFrame | None, combined_note: str | None = None) -> None:
    """Response rate + margin of error for the current slice, using enrollment."""
    st.markdown("### Отклик и достоверность")
    if combined_note:
        st.info(combined_note)
        return
    if cont_slice is None or cont_slice.empty:
        st.caption(
            "Контингент не загружен: положите `Контингент.xlsx` в `data/reference/` и запустите "
            "`python process_contingent.py` — здесь появятся response rate и погрешность."
        )
        return

    n = len(df)
    N = float(cont_slice["contingent"].sum())
    if N <= 0:
        return
    rr = n / N * 100.0
    moe = margin_of_error_pct(n, N)

    st.caption(
        "Response rate и предельная погрешность оценивают, насколько выборка "
        "представляет генеральную совокупность (контингент)."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Контингент (N)", f"{int(round(N)):,}", help="Число студентов в выбранных программах/курсах (из `Контингент.xlsx`).")
    k2.metric("Ответов (n)", f"{n:,}", help="Число анкет в текущем срезе.")
    k3.metric("Response rate", f"{min(rr, 100):.1f}%", help="Доля контингента, заполнившая анкету (n / N).")
    k4.metric(
        "Погрешность ±",
        f"{moe:.1f}%" if np.isfinite(moe) else "н/д",
        help="Предельная погрешность 95% (p=0.5, с поправкой на конечную совокупность). Чем меньше, тем надёжнее.",
    )
    if rr > 100:
        st.caption("⚠ По части программ ответов больше, чем в контингенте — вероятно, расхождение в данных по малым группам.")

    render_interp(
        "Как читать отклик и погрешность",
        [
            "Если response rate высокий (>50%), то выборка хорошо представляет контингент.",
            "Если предельная погрешность мала (например, ±5%), то оценкам долей/средних можно доверять.",
            "Если по программе мало ответов и большой контингент, то её отдельные выводы менее надёжны.",
            "Погрешность считается для доли при наихудшем случае p=0.5, поэтому это консервативная (верхняя) оценка.",
        ],
    )

    rr_tbl = response_rate_table(df, cont_slice)
    if rr_tbl.empty:
        return
    if "program_display" in df.columns:
        disp = df.groupby("program")["program_display"].first()
        rr_tbl = rr_tbl.merge(disp.rename("Короткое имя"), left_on="Программа", right_index=True, how="left")
    rr_tbl = round_df(rr_tbl.sort_values("RR, %", ascending=False))
    st.caption("Таблица: response rate и погрешность по программам.")
    st.dataframe(rr_tbl, width="stretch", hide_index=True)

    plot_df = rr_tbl.copy()
    plot_df["__y"] = plot_df.get("Короткое имя", plot_df["Программа"]).fillna(plot_df["Программа"])
    plot_df["RR_disp"] = plot_df["RR, %"].clip(upper=100)
    fig = px.bar(
        plot_df.sort_values("RR, %"),
        x="RR_disp",
        y="__y",
        orientation="h",
        title="Response rate по программам",
        labels={"RR_disp": "Response rate, %", "__y": "Программа"},
        hover_data={"Программа": True, "__y": False, "Контингент": True, "n": True, "Погрешность ±%": ":.1f"},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(height=max(320, 28 * len(plot_df) + 120), yaxis_title="")
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(fig, width="stretch")
