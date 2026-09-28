"""Helpers shared by the autumn and winter/spring (SFQ) parts of the dashboard."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
DOCS_DIR = PROJECT_ROOT / "docs"

ROUND_DECIMALS = 2

# The 2025-26 hues, reordered so that adjacent series stay distinguishable
# (checked for colour-vision deficiency). Order is fixed, never cycled.
PALETTE = ["#1D4ED8", "#F59E0B", "#14B8A6", "#EF4444", "#A855F7", "#0EA5E9", "#22C55E"]
ACCENT = PALETTE[0]
MUTED = "#94A3B8"

# Russian number format in charts: decimal comma, thin space for thousands.
pio.templates["sfq"] = go.layout.Template(layout=go.Layout(separators=", "))
px.defaults.template = "plotly_white+sfq"
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


# ── Formatting ───────────────────────────────────────────────────────────────

def fmt(x, decimals: int = 2) -> str:
    """Number with a decimal comma; «н/д» for missing values."""
    try:
        if x is None or not np.isfinite(float(x)):
            return "н/д"
    except (TypeError, ValueError):
        return "н/д"
    return f"{float(x):,.{decimals}f}".replace(",", " ").replace(".", ",")


def fmt_p(p) -> str:
    """p-value: «< 0,001» instead of a row of zeros."""
    try:
        if p is not None and np.isfinite(float(p)) and float(p) < 0.001:
            return "< 0,001"
    except (TypeError, ValueError):
        pass
    return fmt(p, 3)


def fmt_pct(x, decimals: int = 0) -> str:
    s = fmt(x, decimals)
    return s if s == "н/д" else f"{s}%"


def show_table(df: pd.DataFrame, decimals: int = ROUND_DECIMALS, **kwargs) -> None:
    """st.dataframe with Russian number format (decimal comma)."""
    kwargs.setdefault("width", "stretch")
    num = df.select_dtypes(include=[np.number]).columns
    if len(num) == 0 or df.empty:
        st.dataframe(df, **kwargs)
        return
    styler = df.style.format(
        {c: ("{:,.0f}" if pd.api.types.is_integer_dtype(df[c]) else f"{{:,.{decimals}f}}") for c in num},
        decimal=",",
        thousands=" ",
        na_rep="—",
    )
    st.dataframe(styler, **kwargs)


# ── Generic helpers ──────────────────────────────────────────────────────────

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
        "Двухмерная графика: анимация, концепт-арт, комиксы": "Двухмерная графика",
        "Стратегические коммуникации и управление репутацией": "Стратегические коммуникации",
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
    st.sidebar.caption("Фильтры применяются ко всем вкладкам.")

    programs = sorted(df["program"].dropna().unique().tolist()) if "program" in df.columns else []
    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    schools = sorted(df["school"].dropna().unique().tolist()) if "school" in df.columns else []

    selected_programs = st.sidebar.multiselect("Программа", programs, default=programs, key=f"{key_prefix}_programs")
    selected_years = st.sidebar.multiselect("Курс", years, default=years, key=f"{key_prefix}_years")
    selected_schools = st.sidebar.multiselect("Школа", schools, default=schools, key=f"{key_prefix}_schools")
    short_program_labels = st.sidebar.checkbox(
        "Сокращать длинные названия программ",
        value=True,
        help="Сокращает подписи на графиках и в легендах, чтобы их было удобнее читать, особенно с телефона.",
        key=f"{key_prefix}_short",
    )
    # A full selection means "no filter", so rows with an unknown school/course are kept.
    return FilterState(
        programs=[] if set(selected_programs) == set(programs) else selected_programs,
        years=[] if set(selected_years) == set(years) else selected_years,
        schools=[] if set(selected_schools) == set(schools) else selected_schools,
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
    n_by_prog = df_slice.groupby("program").size().rename("Ответов")
    N_by_prog = cont_slice.groupby("program")["contingent"].sum().rename("Контингент")
    rr = pd.concat([n_by_prog, N_by_prog], axis=1)
    rr["Ответов"] = rr["Ответов"].fillna(0).astype(int)
    rr["Отклик, %"] = np.where(rr["Контингент"] > 0, rr["Ответов"] / rr["Контингент"] * 100.0, np.nan)
    rr["Погрешность, ±%"] = [margin_of_error_pct(int(n), N) for n, N in zip(rr["Ответов"], rr["Контингент"])]
    return rr.reset_index().rename(columns={"program": "Программа"})


def render_response_rate(df: pd.DataFrame, cont_slice: pd.DataFrame | None, combined_note: str | None = None) -> None:
    """Response rate + margin of error for the current slice, using enrollment."""
    st.markdown("### Отклик и достоверность")
    if combined_note:
        st.info(combined_note)
        return
    if cont_slice is None or cont_slice.empty:
        st.caption(
            "Данные о контингенте не загружены. Чтобы рассчитать отклик и погрешность, положите файл "
            "`Контингент.xlsx` в папку `data/reference/` и запустите `python process_contingent.py`."
        )
        return

    n = len(df)
    N = float(cont_slice["contingent"].sum())
    if N <= 0:
        return
    rr = n / N * 100.0
    moe = margin_of_error_pct(n, N)

    st.caption(
        "Отклик и предельная погрешность показывают, насколько выборка представляет "
        "генеральную совокупность — весь контингент."
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Контингент (N)", fmt(N, 0), help="Число студентов выбранных программ и курсов (по файлу `Контингент.xlsx`).")
    k2.metric("Ответов (n)", fmt(n, 0), help="Число анкет в текущем срезе.")
    k3.metric("Отклик", fmt_pct(min(rr, 100), 1), help="Доля контингента, заполнившая анкету (n / N).")
    k4.metric(
        "Погрешность",
        f"±{fmt_pct(moe, 1)}" if np.isfinite(moe) else "н/д",
        help="Предельная погрешность при 95%-й доверительной вероятности (p = 0,5, с поправкой на конечную совокупность). "
        "Чем она меньше, тем надёжнее оценки.",
    )
    if rr > 100:
        st.caption("⚠ По части программ ответов больше, чем студентов в контингенте: вероятно, данные по небольшим группам расходятся.")

    render_interp(
        "Как интерпретировать отклик и погрешность",
        [
            "Если отклик высокий (больше 50%), то выборка хорошо представляет контингент.",
            "Если предельная погрешность мала (например, ±5%), то оценкам долей и средних можно доверять.",
            "Если по программе мало ответов при большом контингенте, то выводы по этой программе менее надёжны.",
            "Погрешность рассчитана для доли в худшем случае (p = 0,5), поэтому это консервативная, то есть завышенная, оценка.",
        ],
    )

    rr_tbl = response_rate_table(df, cont_slice)
    if rr_tbl.empty:
        return
    if "program_display" in df.columns:
        disp = df.groupby("program")["program_display"].first()
        rr_tbl = rr_tbl.merge(disp.rename("Краткое название"), left_on="Программа", right_index=True, how="left")
    rr_tbl = rr_tbl.sort_values("Отклик, %", ascending=False)
    st.caption("Таблица: отклик и погрешность по программам.")
    show_table(rr_tbl, decimals=1, hide_index=True)

    plot_df = rr_tbl.copy()
    plot_df["__y"] = plot_df.get("Краткое название", plot_df["Программа"]).fillna(plot_df["Программа"])
    plot_df["__rr"] = plot_df["Отклик, %"].clip(upper=100)
    st.caption("График: отклик по программам.")
    fig = px.bar(
        plot_df.sort_values("Отклик, %"),
        x="__rr",
        y="__y",
        orientation="h",
        color="__rr",
        color_continuous_scale="Tealgrn",
        title="Отклик по программам",
        labels={"__rr": "Отклик, %", "__y": "Программа"},
        hover_data={"Программа": True, "__y": False, "Контингент": True, "Ответов": True, "Погрешность, ±%": ":.1f"},
    )
    fig.update_layout(height=max(360, 28 * len(plot_df) + 140), yaxis_title="", coloraxis_showscale=False)
    fig.update_xaxes(range=[0, 100])
    st.plotly_chart(fig, width="stretch")
