"""Autumn start-of-year survey dashboard (+ year-over-year comparison of autumn waves).

Data: data/processed/autumn_<YYYY-YY>.csv (+ _meta.json), built by process_autumn.py.
Every file matching that name is treated as one academic year, so dropping next
year's (or last year's) autumn file into data/processed/ enables the «Год к году» tab.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scikit_posthocs as sp
import streamlit as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

from dashboard.common import (
    ACCENT,
    DOCS_DIR,
    MUTED,
    PALETTE,
    PROCESSED_DIR,
    SEQUENTIAL,
    FilterState,
    add_program_display,
    apply_filters,
    bootstrap_mean_ci,
    build_filters,
    chi2_summary,
    filter_contingent,
    load_codebook,
    load_contingent,
    normalize_unicode_columns,
    render_interp,
    render_response_rate,
    round_df,
    wilson_ci,
)

AUTUMN_FILE_RE = re.compile(r"^autumn_(\d{4}-\d{2})\.csv$")

VAR_LABELS = {
    "year": "Курс",
    "school": "Школа",
    "program": "Программа",
    "program_display": "Программа",
    "decision": "Кто принимал решение о выборе программы",
    "olympiads": "Участие в олимпиадах",
    "english_level": "Английский в профессиональной сфере",
    "difficulties": "Что будет самым трудным в этом году",
    "hours_planned": "Самоподготовка в неделю, план (ч)",
    "hours_last_year": "Самоподготовка в неделю в прошлом году (ч)",
    "work_plans": "Планы на работу в этом году",
    "work_related": "Связь работы с профилем программы",
    "travel_time": "Дорога до кампуса в одну сторону",
    "expectations": "Ожидания от учёбы в этом году",
    "priorities": "Приоритеты на учебный год",
    "confidence": "Уверенность в достижении целей (1–5)",
    "thoughts_transfer": "Мысли о переводе / отчислении / академе в прошлом году",
    "transfer_influences": "Что повлияло на мысли об уходе",
    "missing": "Чего не хватает в программе",
    "about_you": "Что из этого про вас",
    "games_team": "Командные игры",
    "games_solo": "Одиночные игры",
    "games_mobile": "Мобильные игры",
    "games_intellectual": "Интеллектуальные игры",
    "tournament_interest": "Интерес к университетскому турниру",
    "extracurricular": "Внеучебные активности (свободный ответ)",
    "special_needs": "Потребность в специальных условиях",
    "retention_status": "Мысли об уходе: итог",
    "retention_risk": "Доля думавших об уходе/переводе",
    "hours_change": "Самоподготовка: план vs прошлый год",
    "plans_to_work": "Планирует ли работать",
    "hours_planned_num": "Самоподготовка, план (ч/нед, середина интервала)",
    "hours_last_year_num": "Самоподготовка в прошлом году (ч/нед, середина интервала)",
    "travel_min_num": "Дорога до кампуса (мин, середина интервала)",
    "difficulties_other": "Трудности: свой вариант",
    "priorities_other": "Приоритеты: свой вариант",
    "missing_other": "Чего не хватает: свой вариант",
}

GROUP_DIMS = {"year": "Курс", "school": "Школа", "program_display": "Программа"}

SINGLE_VARS = [
    "decision", "olympiads", "english_level", "hours_planned", "hours_last_year", "hours_change",
    "work_plans", "plans_to_work", "work_related", "travel_time", "retention_status",
    "tournament_interest", "special_needs",
]
MULTI_VARS = [
    "priorities", "difficulties", "missing", "thoughts_transfer", "about_you",
    "games_team", "games_solo", "games_mobile", "games_intellectual",
]
NUMERIC_VARS = ["confidence", "hours_planned_num", "hours_last_year_num", "travel_min_num", "retention_risk"]
OPEN_TEXT = {
    "expectations": "Ожидания от учёбы",
    "transfer_influences": "Что повлияло на мысли об уходе",
    "extracurricular": "Внеучебные активности",
    "difficulties_other": "Трудности — свой вариант",
    "priorities_other": "Приоритеты — свой вариант",
    "missing_other": "Чего не хватает — свой вариант",
}
GAME_OTHER = ["games_team_other", "games_solo_other", "games_mobile_other", "games_intellectual_other"]


def vlabel(var: str) -> str:
    return VAR_LABELS.get(var, var)


# ── Data loading ─────────────────────────────────────────────────────────────

def autumn_files() -> dict[str, Path]:
    files = {}
    for p in sorted(PROCESSED_DIR.glob("autumn_*.csv")):
        m = AUTUMN_FILE_RE.match(p.name)
        if m:
            files[m.group(1)] = p
    return files


@st.cache_data(show_spinner=False)
def load_autumn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_unicode_columns(df)
    for c in df.columns:
        if c in MULTI_VARS or c.endswith("_other"):
            df[c] = df[c].fillna("").astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_meta(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"variables": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def question_text(meta: dict, var: str) -> str:
    return meta.get("variables", {}).get(var, {}).get("question", "")


def options_of(meta: dict, var: str) -> list[str]:
    return list(meta.get("variables", {}).get(var, {}).get("options", []))


def courses_of(meta: dict, var: str) -> list[str]:
    return list(meta.get("variables", {}).get(var, {}).get("courses", []))


# ── Counting ─────────────────────────────────────────────────────────────────

def answered(df: pd.DataFrame, var: str) -> pd.DataFrame:
    s = df[var]
    if var in MULTI_VARS:
        return df[s.fillna("").astype(str) != ""]
    return df[s.notna()]


def explode_multi(df: pd.DataFrame, var: str, keep: list[str] | None = None) -> pd.DataFrame:
    keep = keep or []
    d = answered(df, var)[[var] + keep].copy()
    d[var] = d[var].str.split("; ")
    return d.explode(var)


def single_counts(df: pd.DataFrame, var: str, order: list[str]) -> pd.DataFrame:
    s = df[var].dropna().astype(str)
    if s.empty:
        return pd.DataFrame(columns=["answer", "n", "pct"])
    vc = s.value_counts()
    known = [o for o in order if o in vc.index]
    rest = [o for o in vc.index if o not in known]
    out = pd.DataFrame({"answer": known + rest})
    out["n"] = out["answer"].map(vc).astype(int)
    out["pct"] = out["n"] / out["n"].sum() * 100
    out.attrs["ordered"] = bool(known) and bool(order)
    out.attrs["base"] = int(len(s))
    return out


def multi_counts(df: pd.DataFrame, var: str, order: list[str]) -> pd.DataFrame:
    base = answered(df, var)
    if base.empty:
        return pd.DataFrame(columns=["answer", "n", "pct"])
    vc = explode_multi(df, var)[var].value_counts()
    answers = [o for o in order if o in vc.index] + [o for o in vc.index if o not in order]
    out = pd.DataFrame({"answer": answers})
    out["n"] = out["answer"].map(vc).fillna(0).astype(int)
    out["pct"] = out["n"] / len(base) * 100
    out = out.sort_values("n", ascending=False)
    out.attrs["ordered"] = False
    out.attrs["base"] = int(len(base))
    return out


def group_sizes(df: pd.DataFrame, var: str, dim: str) -> pd.Series:
    return answered(df, var).groupby(dim).size()


def crosstab_pct(df: pd.DataFrame, var: str, dim: str, order: list[str], min_n: int) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """% of each group choosing each answer (answers × groups); small groups dropped."""
    sizes = group_sizes(df, var, dim)
    keep_groups = sizes[sizes >= min_n].index.tolist()
    hidden = sizes[sizes < min_n].index.tolist()
    if var in MULTI_VARS:
        ex = explode_multi(df, var, keep=[dim])
        counts = pd.crosstab(ex[var], ex[dim])
    else:
        d = answered(df, var)
        counts = pd.crosstab(d[var].astype(str), d[dim])
    counts = counts.reindex(columns=[g for g in counts.columns if g in keep_groups])
    if counts.empty:
        return counts, sizes, hidden
    pct = counts.div(sizes.reindex(counts.columns), axis=1) * 100
    if var in MULTI_VARS:
        pct = pct.loc[pct.mean(axis=1).sort_values(ascending=False).index]
    else:
        idx = [o for o in order if o in pct.index] + [o for o in pct.index if o not in order]
        pct = pct.loc[idx]
    if dim == "year":
        pct = pct[sorted(pct.columns)]
    return pct, sizes, hidden


# ── Charts ───────────────────────────────────────────────────────────────────

def bar_pct(counts: pd.DataFrame, title: str, xlabel: str = "% ответивших") -> go.Figure:
    d = counts.copy()
    d["label"] = d.apply(lambda r: f"{r.pct:.0f}% ({int(r.n)})", axis=1)
    order = d["answer"].tolist()
    fig = px.bar(
        d,
        x="pct",
        y="answer",
        orientation="h",
        text="label",
        title=title,
        labels={"pct": xlabel, "answer": ""},
        hover_data={"n": True, "pct": ":.1f", "answer": True, "label": False},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)), automargin=True)
    fig.update_xaxes(range=[0, min(100, max(10.0, d["pct"].max() * 1.25))], showgrid=True, gridcolor="#EEF0F3")
    fig.update_layout(height=max(260, 34 * len(d) + 110), margin=dict(l=10, r=30, t=50, b=30), bargap=0.25)
    return fig


def heatmap_pct(pct: pd.DataFrame, title: str, dim: str) -> go.Figure:
    fig = px.imshow(
        pct.round(0),
        text_auto=".0f",
        color_continuous_scale=SEQUENTIAL,
        zmin=0,
        zmax=max(10.0, float(np.nanmax(pct.to_numpy()))),  # scale to the data so cells stay distinguishable
        aspect="auto",
        labels={"x": GROUP_DIMS.get(dim, dim), "y": "", "color": "%"},
        title=title,
    )
    fig.update_xaxes(side="top", tickangle=-35 if dim == "program_display" else 0)
    fig.update_yaxes(automargin=True)
    fig.update_layout(height=max(320, 34 * len(pct) + 180), coloraxis_colorbar=dict(title="% группы"))
    fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>")
    return fig


def base_caption(counts: pd.DataFrame, var: str, meta: dict, multi: bool) -> str:
    base = counts.attrs.get("base", 0)
    courses = courses_of(meta, var)
    who = f" Вопрос задавался: {', '.join(courses)}." if courses and len(courses) < 4 else ""
    kind = " Можно было выбрать несколько вариантов, поэтому сумма > 100%." if multi else ""
    return f"База: {base} ответивших.{who}{kind}"


def render_question(df: pd.DataFrame, var: str, meta: dict, dim: str | None, min_n: int, key: str) -> None:
    """Overall distribution of a single/multi question + optional breakdown heatmap."""
    multi = var in MULTI_VARS
    order = options_of(meta, var)
    counts = multi_counts(df, var, order) if multi else single_counts(df, var, order)
    st.markdown(f"#### {vlabel(var)}")
    q = question_text(meta, var)
    if q and var not in ("retention_status", "hours_change", "plans_to_work"):
        st.caption(f"Вопрос: «{q}»")
    if counts.empty:
        st.info("В текущем срезе нет ответов на этот вопрос.")
        return
    st.plotly_chart(bar_pct(counts, ""), width="stretch", key=f"{key}_bar")
    st.caption(base_caption(counts, var, meta, multi))

    if dim:
        pct, sizes, hidden = crosstab_pct(df, var, dim, order, min_n)
        if pct.empty or pct.shape[1] < 2:
            st.caption(f"Разбивка «{GROUP_DIMS[dim]}»: меньше двух групп с n ≥ {min_n}.")
        else:
            st.plotly_chart(
                heatmap_pct(pct, f"{vlabel(var)} — по группам «{GROUP_DIMS[dim]}», % группы", dim),
                width="stretch",
                key=f"{key}_heat",
            )
            size_txt = ", ".join(f"{g}: {int(sizes[g])}" for g in pct.columns)
            st.caption(f"Размер групп (n): {size_txt}.")
        if hidden:
            st.caption(f"Скрыто групп с n < {min_n}: {len(hidden)} ({', '.join(map(str, hidden))}).")


def dim_selector(key: str, default: str | None = "year") -> str | None:
    options = [None, "year", "school", "program_display"]
    return st.selectbox(
        "Разбивка по группам",
        options,
        index=options.index(default),
        format_func=lambda d: "Без разбивки" if d is None else GROUP_DIMS[d],
        key=key,
    )


def mean_by_group(df: pd.DataFrame, var: str, dim: str, min_n: int) -> pd.DataFrame:
    rows = []
    for g, part in df.groupby(dim):
        mean, lo, hi, n = bootstrap_mean_ci(part[var])
        if n >= min_n:
            rows.append({"group": g, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def share_by_group(df: pd.DataFrame, var: str, dim: str, min_n: int) -> pd.DataFrame:
    rows = []
    for g, part in df.groupby(dim):
        s = part[var].dropna()
        n, k = len(s), int(s.sum())
        if n >= min_n:
            lo, hi = wilson_ci(k, n)
            rows.append({"group": g, "n": n, "k": k, "pct": k / n * 100, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def dot_ci(d: pd.DataFrame, x: str, title: str, xlabel: str, ref: float | None = None, xrange=None) -> go.Figure:
    d = d.sort_values(x)
    fig = px.scatter(
        d,
        x=x,
        y="group",
        error_x=d["ci_high"] - d[x],
        error_x_minus=d[x] - d["ci_low"],
        title=title,
        labels={x: xlabel, "group": ""},
        hover_data={"n": True, x: ":.2f", "ci_low": ":.2f", "ci_high": ":.2f"},
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(marker=dict(size=10))
    if ref is not None and np.isfinite(ref):
        fig.add_vline(ref, line_dash="dash", line_color=MUTED, annotation_text="все", annotation_position="top")
    if xrange:
        fig.update_xaxes(range=xrange)
    fig.update_yaxes(automargin=True, categoryorder="array", categoryarray=d["group"].tolist())
    fig.update_layout(height=max(280, 34 * len(d) + 120))
    return fig


def pct_str(x: float) -> str:
    return f"{x:.0f}%" if np.isfinite(x) else "н/д"


# ── Tabs ─────────────────────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame, cont_slice: pd.DataFrame, year_label: str) -> None:
    st.subheader("Обзор")
    st.info("Что здесь: объём и структура выборки, отклик и главные показатели стартовой анкеты.")

    upper = df[df["year"] != "1 курс"]
    risk = upper["retention_risk"].dropna()
    now = upper["retention_status"].dropna()
    works = df["plans_to_work"].dropna()
    travel = df["travel_min_num"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ответов", f"{len(df):,}", help="Анкет после фильтров (тестовые исключены при обработке).")
    c2.metric("Программ", f"{df['program'].nunique()}")
    c3.metric("Школ", f"{df['school'].nunique()}")
    c4.metric(
        "Уверенность в целях",
        f"{df['confidence'].mean():.2f} / 5" if df["confidence"].notna().any() else "н/д",
        help="Среднее по вопросу «Насколько вы уверены, что достигнете этих целей?» (1–5).",
    )
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Думали об уходе (2+ курс)",
        pct_str(risk.mean() * 100 if len(risk) else np.nan),
        help="Доля 2–4 курса, отметивших мысли о переводе, отчислении или академе (в прошлом году или сейчас). "
        "База — ответившие, кроме «Предпочитаю не отвечать».",
    )
    c6.metric(
        "Думают об этом сейчас",
        pct_str((now == "Думает об этом сейчас").mean() * 100 if len(now) else np.nan),
        help="Доля 2–4 курса, выбравших «Думаю об этом сейчас».",
    )
    c7.metric(
        "Работают или планируют",
        pct_str((works == "Работает / планирует").mean() * 100 if len(works) else np.nan),
        help="Доля тех, кто уже работает или планирует работать в этом учебном году.",
    )
    c8.metric(
        "Дорога > 60 мин",
        pct_str((travel > 60).mean() * 100 if len(travel) else np.nan),
        help="Доля тех, у кого дорога до кампуса в одну сторону занимает больше часа.",
    )
    render_interp(
        "Как читать обзор",
        [
            "Анкета ветвится по курсу: у 1 курса — блок о поступлении, у 2+ курса — вопросы о прошлом годе и удержании.",
            "Доли в карточках считаются от ответивших на конкретный вопрос, а не от всей выборки.",
            "Если по программе мало ответов, её отдельные доли неустойчивы — смотрите на размер групп и доверительные интервалы.",
        ],
    )

    render_response_rate(df, cont_slice)

    st.markdown("### Структура выборки")
    left, right = st.columns(2)
    with left:
        by_year = df["year"].value_counts().sort_index().rename_axis("Курс").reset_index(name="n")
        fig = px.bar(by_year, x="Курс", y="n", text="n", title="Ответов по курсам", color_discrete_sequence=[ACCENT])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=340, yaxis_title="Ответов", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with right:
        by_school = df["school"].fillna("не указана").value_counts().rename_axis("Школа").reset_index(name="n")
        fig = px.bar(by_school, x="Школа", y="n", text="n", title="Ответов по школам", color_discrete_sequence=[ACCENT])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=340, yaxis_title="Ответов", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    by_prog = df.groupby(["program_display", "year"]).size().rename("n").reset_index()
    totals = by_prog.groupby("program_display")["n"].sum().sort_values()
    fig = px.bar(
        by_prog,
        x="n",
        y="program_display",
        color="year",
        orientation="h",
        title="Ответов по программам и курсам",
        labels={"n": "Ответов", "program_display": "", "year": "Курс"},
        category_orders={"year": sorted(by_prog["year"].unique()), "program_display": totals.index[::-1].tolist()},
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(barmode="stack", height=max(360, 30 * len(totals) + 140), legend_title_text="Курс")
    st.plotly_chart(fig, width="stretch")

    by_day = df.dropna(subset=["date"]).assign(day=lambda d: d["date"].dt.date).groupby("day").size().rename("n").reset_index()
    if not by_day.empty:
        fig = px.bar(by_day, x="day", y="n", title="Ответы по дням сбора", labels={"day": "", "n": "Ответов"},
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=300)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Период сбора: {by_day['day'].min():%d.%m.%Y} — {by_day['day'].max():%d.%m.%Y}.")

    st.markdown("### Навигатор по вкладкам")
    st.dataframe(
        pd.DataFrame(
            [
                {"Вкладка": "Первокурсники", "Что внутри": "Кто выбирал программу, олимпиады, английский (только 1 курс)."},
                {"Вкладка": "Цели и трудности", "Что внутри": "Приоритеты года, ожидаемые трудности, уверенность в достижении целей."},
                {"Вкладка": "Нагрузка и быт", "Что внутри": "Часы самоподготовки (план и прошлый год), работа, дорога до кампуса."},
                {"Вкладка": "Удержание и поддержка", "Что внутри": "Мысли о переводе/отчислении (2+ курс), чего не хватает в программе, спец. условия (агрегировано)."},
                {"Вкладка": "Досуг и активности", "Что внутри": "Спорт, творчество, игры, интерес к турниру."},
                {"Вкладка": "Сравнение групп", "Что внутри": "Любой вопрос по курсам/школам/программам со статистическими тестами."},
                {"Вкладка": "Год к году", "Что внутри": f"Сравнение осенних анкет разных лет (сейчас: {year_label})."},
                {"Вкладка": "Открытые ответы", "Что внутри": "Свободные ответы и «свой вариант» с поиском."},
                {"Вкладка": "Кодбук", "Что внутри": "Поля датасета, вопросы, варианты ответов и правила обработки."},
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def render_freshmen(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Первокурсники")
    st.info("Что здесь: вопросы, которые задавались только 1 курсу, — как выбирали программу, олимпиады, английский.")
    d = df[df["year"] == "1 курс"]
    if d.empty:
        st.warning("В текущем срезе нет ответов 1 курса (проверьте фильтр «Курс»).")
        return
    st.caption(f"Ответов 1 курса в срезе: {len(d)}.")
    dim = st.selectbox(
        "Разбивка по группам",
        [None, "school", "program_display"],
        index=2,
        format_func=lambda x: "Без разбивки" if x is None else GROUP_DIMS[x],
        key="fr_dim",
    )
    for var in ["decision", "olympiads", "english_level"]:
        render_question(d, var, meta, dim, min_n, key=f"fr_{var}")

    st.markdown("#### Английский как ожидаемая трудность")
    st.caption("Доля выбравших «Английский язык» среди трудностей года — в зависимости от самооценки уровня.")
    rows = []
    for level, part in d.groupby("english_level"):
        n = len(part)
        k = int(part["difficulties"].str.contains("Английский язык", regex=False).sum())
        if n >= min_n:
            lo, hi = wilson_ci(k, n)
            rows.append({"group": level, "n": n, "pct": k / n * 100, "ci_low": lo, "ci_high": hi})
    if rows:
        st.plotly_chart(
            dot_ci(pd.DataFrame(rows), "pct", "Считают английский трудностью, % (95% CI)", "%", xrange=[0, 100]),
            width="stretch",
        )
    else:
        st.caption(f"Недостаточно групп с n ≥ {min_n}.")


def render_goals(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Цели и трудности")
    st.info("Что здесь: приоритеты на год, ожидаемые трудности и уверенность в достижении целей.")
    render_interp(
        "Как читать",
        [
            "В вопросах с несколькими вариантами доля считается от ответивших на вопрос; сумма по вариантам больше 100%.",
            "Тепловая карта показывает долю внутри каждой группы — так сравниваются группы разного размера.",
            "Уверенность — шкала 1–5; точки со «спицами» — среднее и 95% bootstrap-интервал.",
        ],
    )
    dim = dim_selector("goals_dim")
    for var in ["priorities", "difficulties"]:
        render_question(df, var, meta, dim, min_n, key=f"goals_{var}")

    st.markdown(f"#### {vlabel('confidence')}")
    st.caption(f"Вопрос: «{question_text(meta, 'confidence')}»")
    conf = single_counts(df.assign(c=df["confidence"].astype("Int64").astype(str).replace("<NA>", np.nan)), "c", ["1", "2", "3", "4", "5"])
    if not conf.empty:
        st.plotly_chart(bar_pct(conf, "Распределение оценок уверенности"), width="stretch", key="conf_bar")
        overall = df["confidence"].mean()
        st.caption(f"Среднее: {overall:.2f}, медиана: {df['confidence'].median():.0f}, n = {df['confidence'].notna().sum()}.")
        if dim:
            g = mean_by_group(df, "confidence", dim, min_n)
            if len(g) >= 2:
                st.plotly_chart(
                    dot_ci(g, "mean", f"Уверенность по группам «{GROUP_DIMS[dim]}»", "Среднее (1–5)", ref=overall, xrange=[1, 5]),
                    width="stretch",
                    key="conf_dim",
                )

        st.markdown("#### Уверенность и ожидаемые трудности")
        st.caption(
            "Средняя уверенность у тех, кто отметил каждую трудность. Трудности, связанные с заметно более низкой "
            "уверенностью, — кандидаты для адресной поддержки."
        )
        ex = explode_multi(df, "difficulties", keep=["confidence"]).dropna(subset=["confidence"])
        rows = []
        for opt, part in ex.groupby("difficulties"):
            mean, lo, hi, n = bootstrap_mean_ci(part["confidence"])
            if n >= min_n:
                rows.append({"group": opt, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi})
        if rows:
            st.plotly_chart(
                dot_ci(pd.DataFrame(rows), "mean", "Уверенность у выбравших трудность", "Среднее (1–5)", ref=overall, xrange=[1, 5]),
                width="stretch",
                key="conf_diff",
            )
    n_exp = int(df["expectations"].notna().sum())
    st.caption(f"Свободных ответов об ожиданиях: {n_exp} — см. вкладку «Открытые ответы».")


def render_load(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Нагрузка и быт")
    st.info("Что здесь: сколько времени студенты готовы тратить на самоподготовку, работа и дорога до кампуса.")
    render_interp(
        "Как читать",
        [
            "Часы указаны интервалами; для средних используется середина интервала (0, 3, 8, 13, 18, 23, 28 ч) — это оценка.",
            "Сравнение «план vs прошлый год» есть только у 2+ курса: 1 курс не отвечал про прошлый год.",
            "Если студент планирует больше часов, чем тратил, — это ожидание роста нагрузки; меньше — возможный сигнал выгорания или работы.",
        ],
    )
    dim = dim_selector("load_dim")
    render_question(df, "hours_planned", meta, dim, min_n, key="load_hp")
    if dim:
        g = mean_by_group(df, "hours_planned_num", dim, min_n)
        if len(g) >= 2:
            st.plotly_chart(
                dot_ci(g, "mean", "Самоподготовка (план), ч/нед — среднее по группам", "ч/нед (оценка)",
                       ref=df["hours_planned_num"].mean()),
                width="stretch",
            )

    st.markdown("#### План на этот год vs прошлый год (2+ курс)")
    both = df.dropna(subset=["hours_planned", "hours_last_year"])
    if both.empty:
        st.caption("В срезе нет ответов 2+ курса на оба вопроса.")
    else:
        order = options_of(meta, "hours_planned")
        ct = pd.crosstab(both["hours_last_year"], both["hours_planned"]).reindex(index=order, columns=order).fillna(0).astype(int)
        ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
        fig = px.imshow(
            ct,
            text_auto=True,
            color_continuous_scale=SEQUENTIAL,
            aspect="auto",
            labels={"x": "План на этот год, ч/нед", "y": "Прошлый год, ч/нед", "color": "Студентов"},
            title="Сколько часов тратили vs сколько планируют (число студентов)",
        )
        fig.update_layout(height=460)
        st.plotly_chart(fig, width="stretch")
        delta = (both["hours_planned_num"] - both["hours_last_year_num"]).mean()
        st.caption(
            f"Клетки на диагонали — «столько же». n = {len(both)}. "
            f"Средний сдвиг плана относительно прошлого года: {delta:+.1f} ч/нед (оценка по серединам интервалов)."
        )
        render_question(df, "hours_change", meta, dim, min_n, key="load_hc")

    for var in ["work_plans", "work_related", "travel_time"]:
        render_question(df, var, meta, dim, min_n, key=f"load_{var}")

    st.markdown("#### Работа и ожидаемая трудность «Совмещение учёбы с работой»")
    rows = []
    for status, part in df.dropna(subset=["plans_to_work"]).groupby("plans_to_work"):
        n = len(part)
        k = int(part["difficulties"].str.contains("Совмещение учёбы с работой", regex=False).sum())
        if n >= min_n:
            lo, hi = wilson_ci(k, n)
            rows.append({"group": status, "n": n, "pct": k / n * 100, "ci_low": lo, "ci_high": hi})
    if rows:
        st.plotly_chart(
            dot_ci(pd.DataFrame(rows), "pct", "Считают совмещение с работой трудностью, % (95% CI)", "%", xrange=[0, 100]),
            width="stretch",
        )


def render_retention(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Удержание и поддержка")
    st.info(
        "Что здесь: мысли о переводе, отчислении и академическом отпуске (2+ курс), чего не хватает в программе "
        "и агрегированная потребность в специальных условиях."
    )
    render_interp(
        "Как читать",
        [
            "«Думали об уходе» = отметили любой вариант «Думал(а) о …» или «Думаю об этом сейчас». "
            "«Предпочитаю не отвечать» исключается из базы.",
            "Интервал 95% (Уилсон) показывает неопределённость доли: при малом n он широкий.",
            "Сравнение «чего не хватает» у думавших об уходе и остальных подсказывает, какие дефициты сильнее связаны с риском ухода "
            "(это связь, а не доказанная причина).",
        ],
    )
    up = df[df["year"] != "1 курс"]
    if up.empty:
        st.warning("В текущем срезе нет ответов 2+ курса.")
        return
    risk = up["retention_risk"].dropna()
    status = up["retention_status"].dropna()
    k, n = int(risk.sum()), len(risk)
    lo, hi = wilson_ci(k, n)
    c1, c2, c3 = st.columns(3)
    c1.metric("Думали об уходе", pct_str(k / n * 100 if n else np.nan), help=f"{k} из {n}; 95% CI: {lo:.0f}–{hi:.0f}%.")
    c2.metric("Думают сейчас", pct_str((status == "Думает об этом сейчас").mean() * 100 if len(status) else np.nan))
    c3.metric("Уже перевелись", pct_str((status == "Уже перевелся(-ась)").mean() * 100 if len(status) else np.nan))

    dim = dim_selector("ret_dim")
    render_question(up, "thoughts_transfer", meta, None, min_n, key="ret_tt")
    render_question(up, "retention_status", meta, dim, min_n, key="ret_status")
    if dim:
        g = share_by_group(up, "retention_risk", dim, min_n)
        if len(g) >= 2:
            st.plotly_chart(
                dot_ci(g, "pct", f"Думали об уходе — по группам «{GROUP_DIMS[dim]}», % (95% CI)", "%",
                       ref=k / n * 100 if n else None, xrange=[0, 100]),
                width="stretch",
            )
    n_infl = int(up["transfer_influences"].notna().sum())
    st.caption(f"Свободных ответов «Что на это повлияло?»: {n_infl} — см. вкладку «Открытые ответы».")

    render_question(up, "missing", meta, dim, min_n, key="ret_missing")

    st.markdown("#### Чего не хватает: думавшие об уходе vs остальные")
    d = up.dropna(subset=["retention_risk"])
    d = d[d["missing"] != ""]
    sizes = d.groupby("retention_risk").size()
    if len(sizes) < 2 or sizes.min() < min_n:
        st.caption(f"Недостаточно ответов в одной из групп (нужно n ≥ {min_n}).")
    else:
        ex = explode_multi(d, "missing", keep=["retention_risk"])
        ct = pd.crosstab(ex["missing"], ex["retention_risk"]).reindex(columns=[0.0, 1.0], fill_value=0)
        pct = ct.div(sizes.reindex([0.0, 1.0]), axis=1) * 100
        pvals = []
        for opt in ct.index:
            table = np.array([[ct.loc[opt, 0.0], sizes[0.0] - ct.loc[opt, 0.0]], [ct.loc[opt, 1.0], sizes[1.0] - ct.loc[opt, 1.0]]])
            pvals.append(stats.fisher_exact(table)[1])
        p_adj = multipletests(pvals, method="holm")[1]
        res = pd.DataFrame(
            {
                "Чего не хватает": ct.index,
                "Не думали, %": pct[0.0].values,
                "Думали об уходе, %": pct[1.0].values,
                "Разница, п.п.": (pct[1.0] - pct[0.0]).values,
                "p (Fisher, Holm)": p_adj,
            }
        ).sort_values("Разница, п.п.", ascending=False)
        long = res.melt(id_vars="Чего не хватает", value_vars=["Не думали, %", "Думали об уходе, %"], var_name="Группа", value_name="%")
        long["Группа"] = long["Группа"].str.replace(", %", "", regex=False)
        fig = px.bar(
            long,
            x="%",
            y="Чего не хватает",
            color="Группа",
            barmode="group",
            orientation="h",
            color_discrete_map={"Не думали": MUTED, "Думали об уходе": PALETTE[1]},
            category_orders={"Чего не хватает": res["Чего не хватает"].tolist()},
            title="Доля отметивших дефицит, % группы",
        )
        fig.update_layout(height=max(360, 44 * len(res) + 140), yaxis_title="", legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Группы: не думали — {int(sizes[0.0])}, думали об уходе — {int(sizes[1.0])} (из ответивших на оба вопроса).")
        st.dataframe(round_df(res), width="stretch", hide_index=True)

    st.markdown("### Специальные условия обучения")
    st.caption(
        "В анкете обещан ограниченный доступ к этим ответам, поэтому здесь только агрегированные доли; "
        "типы потребностей, описания и контакты хранятся отдельно (`data/private/`, не публикуются). "
        "Группы меньше порога скрыты."
    )
    sn_dim = st.selectbox(
        "Разбивка", [None, "year", "school"], index=1,
        format_func=lambda x: "Без разбивки" if x is None else GROUP_DIMS[x], key="sn_dim",
    )
    render_question(df, "special_needs", meta, sn_dim, max(min_n, 5), key="ret_sn")


def render_leisure(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Досуг и активности")
    st.info("Что здесь: чем студенты занимаются вне учёбы, во что играют и хотят ли участвовать в турнире.")
    dim = dim_selector("leis_dim")
    render_question(df, "about_you", meta, dim, min_n, key="leis_about")
    render_question(df, "tournament_interest", meta, dim, min_n, key="leis_tour")

    st.markdown("### Игры хотя бы раз в неделю")
    st.caption(
        "Вопросы про игры задавались тем, кто отметил соответствующий пункт в «Что из этого про вас». "
        "«Не играю в игры этой категории» — объединённые ответы «не играю / ни во что»."
    )
    for var in ["games_team", "games_solo", "games_mobile", "games_intellectual"]:
        render_question(df, var, meta, None, min_n, key=f"leis_{var}")
        other = df[f"{var}_other"]
        items = other[other != ""].str.split("; ").explode().str.strip()
        if not items.empty:
            top = (
                items.to_frame("title")
                .assign(key=lambda d: d["title"].str.lower())
                .groupby("key")
                .agg(Игра=("title", "first"), Упоминаний=("title", "size"))
                .sort_values("Упоминаний", ascending=False)
                .head(15)
            )
            with st.expander(f"Свой вариант — топ упоминаний ({len(items)})"):
                st.dataframe(top, width="stretch", hide_index=True)
    n_extra = int(df["extracurricular"].notna().sum())
    st.caption(f"Свободных ответов о внеучебных активностях: {n_extra} — см. вкладку «Открытые ответы».")


def render_group_comparison(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Сравнение групп")
    st.info(
        "Что здесь: любой вопрос анкеты в разрезе курсов, школ или программ со статистической проверкой различий. "
        "Для вопросов с одним ответом — χ² и V Крамера; для множественного выбора — χ² по каждому варианту с поправкой Холма; "
        "для числовых — Kruskal-Wallis и post-hoc Dunn."
    )
    kinds = {v: "single" for v in SINGLE_VARS} | {v: "multi" for v in MULTI_VARS} | {v: "numeric" for v in NUMERIC_VARS}
    c1, c2, c3 = st.columns([3, 2, 2])
    var = c1.selectbox("Вопрос", list(kinds), index=list(kinds).index("priorities"), format_func=vlabel, key="cmp_var")
    dim = c2.selectbox("Группы", list(GROUP_DIMS), format_func=GROUP_DIMS.get, key="cmp_dim")
    min_g = c3.slider("Мин. размер группы", 3, 30, max(min_n, 10), key="cmp_min")
    render_interp(
        "Как интерпретировать",
        [
            "Если p < 0.05, различия между группами вряд ли случайны.",
            "V Крамера ≈ 0.1 / 0.3 / 0.5 — слабая / умеренная / сильная связь.",
            "Если много клеток с ожидаемой частотой < 5, χ² ненадёжен: укрупните группы (школа вместо программы) или поднимите порог.",
            "Kruskal-Wallis сравнивает распределения; Dunn показывает, какие именно пары групп различаются (p с поправкой Холма).",
        ],
    )
    kind = kinds[var]
    d = df.copy()
    if kind == "numeric":
        d = d.dropna(subset=[var])
    else:
        d = answered(d, var)
    sizes = d.groupby(dim).size()
    keep = sizes[sizes >= min_g].index
    d = d[d[dim].isin(keep)]
    st.caption(f"В анализе групп: {len(keep)}, наблюдений: {len(d)}. Исключено групп с n < {min_g}: {int((sizes < min_g).sum())}.")
    if len(keep) < 2:
        st.info("Для сравнения нужно минимум две группы достаточного размера.")
        return

    if kind == "numeric":
        groups = [g[var].to_numpy() for _, g in d.groupby(dim)]
        h, p = stats.kruskal(*groups)
        k1, k2, k3 = st.columns(3)
        k1.metric("Kruskal-Wallis p", f"{p:.3f}")
        k2.metric("Эффект ε²", f"{(h - len(groups) + 1) / (len(d) - len(groups)):.3f}",
                  help="Доля вариации рангов, объяснённая группой (0.01 / 0.08 / 0.26 — малый / средний / крупный).")
        k3.metric("Групп", len(groups))
        g = mean_by_group(d, var, dim, min_g)
        st.plotly_chart(dot_ci(g, "mean", f"{vlabel(var)} — среднее по группам", "Среднее", ref=d[var].mean()), width="stretch")
        fig = px.box(d, x=var, y=dim, orientation="h", points="all", labels={var: vlabel(var), dim: ""},
                     color_discrete_sequence=[ACCENT], title="Распределение по группам")
        fig.update_layout(height=max(320, 40 * len(keep) + 120))
        st.plotly_chart(fig, width="stretch")
        dunn = sp.posthoc_dunn(d, val_col=var, group_col=dim, p_adjust="holm")
        st.caption("Dunn post-hoc, p-value с поправкой Холма.")
        st.dataframe(round_df(dunn, 3), width="stretch")
        return

    order = options_of(meta, var)
    pct, sizes2, _ = crosstab_pct(d, var, dim, order, min_g)
    st.plotly_chart(heatmap_pct(pct, f"{vlabel(var)}, % группы", dim), width="stretch")
    if kind == "single":
        counts = pd.crosstab(d[var].astype(str), d[dim])
        res = chi2_summary(counts)
        k1, k2, k3 = st.columns(3)
        k1.metric("χ² p-value", f"{res['p']:.3f}" if np.isfinite(res["p"]) else "н/д")
        k2.metric("V Крамера", f"{res['cramers_v']:.2f}" if np.isfinite(res["cramers_v"]) else "н/д")
        k3.metric("Клеток с ожидаемой < 5", f"{res['low_expected_share'] * 100:.0f}%" if np.isfinite(res["low_expected_share"]) else "н/д")
        if np.isfinite(res["low_expected_share"]) and res["low_expected_share"] > 0.2:
            st.warning("Больше 20% клеток с ожидаемой частотой < 5 — результат χ² ориентировочный.")
        st.dataframe(counts, width="stretch")
    else:
        ex = explode_multi(d, var, keep=[dim])
        chosen = pd.crosstab(ex[var], ex[dim])
        n_g = d.groupby(dim).size().reindex(chosen.columns)
        rows = []
        for opt in chosen.index:
            table = pd.DataFrame({"да": chosen.loc[opt], "нет": n_g - chosen.loc[opt]}).T
            r = chi2_summary(table)
            rows.append({"Вариант": opt, "Выбрали всего": int(chosen.loc[opt].sum()), "χ² p": r["p"], "V Крамера": r["cramers_v"]})
        res = pd.DataFrame(rows)
        valid = res["χ² p"].notna()
        res.loc[valid, "p (Holm)"] = multipletests(res.loc[valid, "χ² p"], method="holm")[1]
        res = res.sort_values("p (Holm)")
        st.caption("Для каждого варианта: различается ли доля выбравших между группами (χ², поправка Холма на число вариантов).")
        st.dataframe(round_df(res, 3), width="stretch", hide_index=True)


# ── Year over year ───────────────────────────────────────────────────────────

YOY_KPIS = [
    ("Ответов", lambda d: len(d), "{:.0f}"),
    ("Уверенность в целях (1–5)", lambda d: d["confidence"].mean(), "{:.2f}"),
    ("Самоподготовка, план (ч/нед)", lambda d: d["hours_planned_num"].mean(), "{:.1f}"),
    ("Работают / планируют, %", lambda d: (d["plans_to_work"].dropna() == "Работает / планирует").mean() * 100, "{:.0f}"),
    ("Дорога > 60 мин, %", lambda d: (d["travel_min_num"].dropna() > 60).mean() * 100, "{:.0f}"),
    ("Думали об уходе (2+ курс), %", lambda d: d.loc[d["year"] != "1 курс", "retention_risk"].mean() * 100, "{:.0f}"),
    ("Нуждаются в спец. условиях, %", lambda d: (d["special_needs"].dropna() == "Да").mean() * 100, "{:.0f}"),
]


def render_yoy(filters: FilterState, current: str, min_n: int) -> None:
    st.subheader("Год к году: осенние анкеты")
    files = autumn_files()
    st.info(
        "Что здесь: сравнение стартовых (осенних) анкет разных учебных лет. Зимняя и весенняя анкеты — другой "
        "опросник (SFQ), их динамика смотрится внутри года в режиме «Зима + Весна»."
    )
    if len(files) < 2:
        st.warning(
            f"Сейчас загружен только один год ({', '.join(files) or '—'}). Чтобы сравнить, положите в `data/processed/` "
            "файл другого года в том же формате — `autumn_<ГГГГ-ГГ>.csv` и `autumn_<ГГГГ-ГГ>_meta.json` "
            "(их создаёт `process_autumn.py --raw <выгрузка> --year <ГГГГ-ГГ>`)."
        )
        return

    years = st.multiselect("Учебные годы", list(files), default=list(files), key="yoy_years")
    if len(years) < 2:
        st.info("Выберите минимум два года.")
        return
    frames = []
    for y in years:
        d = load_autumn(str(files[y]))
        # Programme names change between years, so only course/school filters apply here.
        if filters.years:
            d = d[d["year"].isin(filters.years)]
        if filters.schools:
            d = d[d["school"].isin(filters.schools)]
        frames.append(d.assign(academic_year=y))
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    for c in MULTI_VARS:
        if c in all_df.columns:
            all_df[c] = all_df[c].fillna("")
    progs = sorted(all_df["program"].dropna().unique())
    sel = st.multiselect("Программы (названия могут меняться между годами)", progs, default=progs, key="yoy_progs")
    all_df = all_df[all_df["program"].isin(sel)]
    st.caption("Фильтры «Курс» и «Школа» из сайдбара применены; фильтр программ — выше.")

    rows = []
    for name, fn, fmt in YOY_KPIS:
        row = {"Показатель": name}
        for y in years:
            part = all_df[all_df["academic_year"] == y]
            try:
                val = fn(part)
            except KeyError:
                val = np.nan
            row[y] = fmt.format(val) if pd.notna(val) and np.isfinite(val) else "н/д"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    meta = load_meta(str(files[current].with_name(f"autumn_{current}_meta.json")))
    candidates = [v for v in SINGLE_VARS + MULTI_VARS if v in all_df.columns]
    var = st.selectbox("Вопрос для сравнения", candidates, format_func=vlabel, key="yoy_var")
    order = options_of(meta, var)
    pct, sizes, hidden = crosstab_pct(all_df, var, "academic_year", order, min_n)
    if pct.empty or pct.shape[1] < 2:
        st.info("Недостаточно ответов в выбранных годах.")
        return
    long = pct.reset_index().melt(id_vars=pct.index.name or "index", var_name="Год", value_name="%")
    long = long.rename(columns={pct.index.name or "index": "Ответ"})
    fig = px.bar(
        long,
        x="%",
        y="Ответ",
        color="Год",
        barmode="group",
        orientation="h",
        category_orders={"Ответ": pct.index.tolist(), "Год": list(pct.columns)},
        color_discrete_sequence=PALETTE,
        title=f"{vlabel(var)}: % ответивших по годам",
    )
    fig.update_layout(height=max(360, 30 * len(pct) * len(pct.columns) + 140), yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Ответивших: " + ", ".join(f"{y}: {int(sizes[y])}" for y in pct.columns) + ".")
    if var in SINGLE_VARS:
        d = answered(all_df, var)
        res = chi2_summary(pd.crosstab(d[var].astype(str), d["academic_year"]))
        st.caption(f"χ² между годами: p = {res['p']:.3f}, V Крамера = {res['cramers_v']:.2f}.")
    st.caption("Если формулировки или варианты ответа между годами менялись, сравнение по такому вопросу некорректно — сверяйтесь с кодбуками.")


def render_open_answers(df: pd.DataFrame) -> None:
    st.subheader("Открытые ответы")
    st.info("Что здесь: свободные ответы и «свой вариант» по вопросам с выбором. Можно фильтровать по разделу и искать по слову.")
    available = {c: lbl for c, lbl in OPEN_TEXT.items() if c in df.columns}
    counts = {}
    for col, lbl in available.items():
        s = df[col].fillna("").astype(str).str.strip()
        counts[lbl] = int((s.str.contains(r"[A-Za-zА-Яа-яЁё]", regex=True)).sum())
    cols = st.columns(len(counts))
    for w, (lbl, n) in zip(cols, counts.items()):
        w.metric(lbl, n)

    c1, c2 = st.columns([2, 3])
    sections = c1.multiselect("Раздел", list(available.values()), default=list(available.values()), key="open_sections")
    search = c2.text_input("Поиск по тексту", placeholder="Ключевое слово…", key="open_search")

    rows = []
    for col, lbl in available.items():
        if lbl not in sections:
            continue
        sub = df[["program_display", "school", "year", col]].copy()
        sub[col] = sub[col].fillna("").astype(str).str.strip()
        # "-", "." and similar non-answers carry nothing
        sub = sub[sub[col].str.contains(r"[A-Za-zА-Яа-яЁё]", regex=True)]
        sub = sub.rename(columns={"program_display": "Программа", "school": "Школа", "year": "Курс", col: "Ответ"})
        sub.insert(0, "Раздел", lbl)
        rows.append(sub)
    if not rows:
        st.info("Нет ответов по выбранным разделам.")
        return
    out = pd.concat(rows, ignore_index=True)
    if search.strip():
        out = out[out["Ответ"].str.contains(search.strip(), case=False, na=False, regex=False)]
    st.caption(f"Показано: **{len(out)}** ответов.")
    st.dataframe(
        out,
        width="stretch",
        hide_index=True,
        column_config={
            "Раздел": st.column_config.TextColumn("Раздел", width="small"),
            "Программа": st.column_config.TextColumn("Программа", width="medium"),
            "Школа": st.column_config.TextColumn("Школа", width="small"),
            "Курс": st.column_config.TextColumn("Курс", width="small"),
            "Ответ": st.column_config.TextColumn("Ответ", width="large"),
        },
    )


def render_codebook() -> None:
    st.subheader("Кодбук")
    st.info("Что здесь: поля датасета осенней анкеты, формулировки вопросов, варианты ответа и правила обработки.")
    text = load_codebook(DOCS_DIR / "codebook_autumn.md")
    if not text:
        st.warning("Кодбук не найден: запустите `python process_autumn.py`.")
        return
    st.markdown(text)


# ── Entry point ──────────────────────────────────────────────────────────────

def render_autumn() -> None:
    files = autumn_files()
    if not files:
        st.title("Осенняя анкета")
        st.error("Нет файла `data/processed/autumn_<ГГГГ-ГГ>.csv`. Запустите `python process_autumn.py`.")
        return
    current = st.sidebar.selectbox("Учебный год", list(files)[::-1], key="autumn_year") if len(files) > 1 else list(files)[0]
    st.title(f"Осенняя анкета {current}: старт учебного года")
    st.caption(f"Источник данных: `{files[current].name}`")

    df = load_autumn(str(files[current]))
    meta = load_meta(str(files[current].with_name(f"autumn_{current}_meta.json")))

    filters = build_filters(df, key_prefix=f"autumn_{current}")
    min_n = st.sidebar.slider(
        "Мин. размер группы в разбивках",
        min_value=1,
        max_value=20,
        value=5,
        help="Группы меньше порога скрываются в разбивках: доли по 2–3 ответам неустойчивы и могут раскрывать конкретных студентов.",
        key="autumn_min_n",
    )
    dff = apply_filters(df, filters)
    if dff.empty:
        st.error("После применения фильтров данных не осталось.")
        return
    dff = add_program_display(dff, short_labels=filters.short_program_labels)
    cont_slice = filter_contingent(load_contingent(), filters)

    tabs = st.tabs(
        ["Обзор", "Первокурсники", "Цели и трудности", "Нагрузка и быт", "Удержание и поддержка",
         "Досуг и активности", "Сравнение групп", "Год к году", "Открытые ответы", "Кодбук"]
    )
    with tabs[0]:
        render_overview(dff, cont_slice, current)
    with tabs[1]:
        render_freshmen(dff, meta, min_n)
    with tabs[2]:
        render_goals(dff, meta, min_n)
    with tabs[3]:
        render_load(dff, meta, min_n)
    with tabs[4]:
        render_retention(dff, meta, min_n)
    with tabs[5]:
        render_leisure(dff, meta, min_n)
    with tabs[6]:
        render_group_comparison(dff, meta, min_n)
    with tabs[7]:
        render_yoy(filters, current, min_n)
    with tabs[8]:
        render_open_answers(dff)
    with tabs[9]:
        render_codebook()
