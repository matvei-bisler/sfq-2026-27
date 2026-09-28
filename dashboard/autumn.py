"""Autumn start-of-year survey dashboard (+ year-over-year comparison of autumn waves).

Data: data/processed/autumn_<YYYY-YY>.csv (+ _meta.json), built by process_autumn.py.
Every file matching that name is treated as one academic year, so dropping next
year's (or last year's) autumn file into data/processed/ enables the year-over-year tab.
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
    PROCESSED_DIR,
    FilterState,
    add_program_display,
    apply_filters,
    bootstrap_mean_ci,
    build_filters,
    chi2_summary,
    filter_contingent,
    fmt,
    fmt_p,
    fmt_pct,
    load_codebook,
    load_contingent,
    normalize_unicode_columns,
    render_interp,
    render_response_rate,
    show_table,
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
    "english_level": "Уровень английского языка в профессиональной сфере",
    "difficulties": "Что будет самым трудным в этом учебном году",
    "hours_planned": "Самоподготовка в неделю: план на этот год",
    "hours_last_year": "Самоподготовка в неделю в прошлом году",
    "work_plans": "Планы на работу в этом учебном году",
    "work_related": "Связь работы с профилем программы",
    "travel_time": "Дорога до кампуса в одну сторону",
    "expectations": "Ожидания от учёбы в этом году",
    "priorities": "Приоритеты на учебный год",
    "confidence": "Уверенность в достижении целей (1–5)",
    "thoughts_transfer": "Мысли о переводе, отчислении или академическом отпуске",
    "transfer_influences": "Что повлияло на мысли об уходе",
    "missing": "Чего не хватает в программе",
    "about_you": "Чем студенты занимаются вне учёбы",
    "games_team": "Командные компьютерные игры",
    "games_solo": "Одиночные компьютерные игры",
    "games_mobile": "Мобильные игры",
    "games_intellectual": "Интеллектуальные игры",
    "tournament_interest": "Интерес к турниру, организованному университетом",
    "extracurricular": "Внеучебные активности, в которых хотели бы участвовать",
    "special_needs": "Потребность в специальных условиях обучения",
    "retention_status": "Мысли об уходе: итоговая категория",
    "retention_risk": "Доля думавших об уходе или переводе",
    "hours_change": "Самоподготовка: план по сравнению с прошлым годом",
    "plans_to_work": "Планы на работу (укрупнённо)",
    "hours_planned_num": "Самоподготовка по плану, ч в неделю",
    "hours_last_year_num": "Самоподготовка в прошлом году, ч в неделю",
    "travel_min_num": "Дорога до кампуса, мин",
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
DERIVED_VARS = {"retention_status", "hours_change", "plans_to_work", "special_needs"}
GAME_VARS = ["games_team", "games_solo", "games_mobile", "games_intellectual"]
NO_GAMES = "Не играю в игры этой категории"
OTHER_OPTION = "Другое (свой вариант)"

OPEN_TEXT = {
    "expectations": "Ожидания от учёбы",
    "transfer_influences": "Что повлияло на мысли об уходе",
    "extracurricular": "Внеучебные активности",
    "difficulties_other": "Трудности: собственный ответ",
    "priorities_other": "Приоритеты: собственный ответ",
    "missing_other": "Чего не хватает: собственный ответ",
}

# Written-in game titles: spelling variants -> one name.
GAME_ALIASES = {
    "genshin": "Genshin Impact",
    "genshin impact": "Genshin Impact",
    "геншин": "Genshin Impact",
    "roblox": "Roblox",
    "roblex": "Roblox",
    "роблокс": "Roblox",
    "minecraft": "Minecraft",
    "maincraft": "Minecraft",
    "майнкрафт": "Minecraft",
    "block blast": "Block Blast",
    "блокбаст": "Block Blast",
    "brawl stars": "Brawl Stars",
    "brawl star": "Brawl Stars",
    "судоку": "Судоку",
    "sims 4": "The Sims",
    "sims": "The Sims",
    "repo": "R.E.P.O.",
    "r.e.p.o": "R.E.P.O.",
}

# Heuristic themes for the free-text question about extracurricular activities.
ACTIVITY_THEMES = {
    "Студенческий совет, организация мероприятий": r"студсовет|студ\.?\s*совет|студенческ\w*\s+совет|организац|волонт",
    "Кино: показы, клубы, фестивали": r"кино|фильм|мульт",
    "Музыка и концерты": r"музык|концерт|джем|диджей|звукорежисс|саундтрек",
    "Театр, стендап, импровизация": r"театр|стендап|импров|скетч|актёр|актер",
    "Спорт и активный отдых": r"спорт|йог|футбол|волейбол|баскетбол|теннис|танц",
    "Квизы, настольные и интеллектуальные игры": r"квиз|шахмат|мафи|настол|что\?\s*где",
    "Выставки, показы, ярмарки": r"выставк|ярмарк|недел\w*\s+моды|fashion|показ(?!\w*\s+(фильм|мульт))",
    "Лекции, мастер-классы, встречи с практиками": r"лекци|мастер-?класс|встреч|спикер|нетворк|экскурс",
    "Стажировки и практика": r"стажир|практик",
    "Клубы и кружки по интересам": r"клуб|кружк|кружок|сообществ",
    "Поездки и выезды": r"поездк|выезд|путешеств",
    "Не планируют участвовать": r"^\s*(ни\s+в\s+как|никак|нет\b)",
}


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


def has_option(df: pd.DataFrame, var: str, option: str) -> pd.Series:
    return df[var].fillna("").astype(str).str.split("; ").map(lambda items: option in items)


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
    out.attrs["base"] = int(len(base))
    return out


def crosstab_pct(df: pd.DataFrame, var: str, dim: str, order: list[str], min_n: int) -> tuple[pd.DataFrame, pd.Series, list]:
    """% of each group choosing each answer (answers × groups); small groups are dropped."""
    sizes = answered(df, var).groupby(dim).size()
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

def bar_pct(counts: pd.DataFrame, title: str, xlabel: str = "Доля ответивших, %") -> go.Figure:
    d = counts.copy()
    d["label"] = [f"{fmt_pct(p)} ({int(n)})" for p, n in zip(d["pct"], d["n"])]
    order = d["answer"].tolist()
    fig = px.bar(
        d,
        x="pct",
        y="answer",
        orientation="h",
        text="label",
        color="pct",
        color_continuous_scale="Teal",
        title=title,
        labels={"pct": xlabel, "answer": "Ответ", "n": "Число ответов"},
        hover_data={"n": True, "pct": ":.1f", "label": False},
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(order)), automargin=True, title="")
    fig.update_xaxes(range=[0, min(100, max(10.0, d["pct"].max() * 1.25))])
    fig.update_layout(height=max(280, 34 * len(d) + 120), coloraxis_showscale=False)
    return fig


def heatmap_pct(pct: pd.DataFrame, title: str, dim: str) -> go.Figure:
    fig = px.imshow(
        pct.round(0),
        text_auto=".0f",
        color_continuous_scale="Teal",
        zmin=0,
        zmax=max(10.0, float(np.nanmax(pct.to_numpy()))),  # scale to the data so cells stay distinguishable
        aspect="auto",
        labels={"x": GROUP_DIMS.get(dim, dim), "y": "", "color": "% группы"},
        title=title,
    )
    fig.update_xaxes(side="top", tickangle=-35 if dim == "program_display" else 0)
    fig.update_yaxes(automargin=True)
    fig.update_layout(height=max(340, 34 * len(pct) + 190))
    fig.update_traces(hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>")
    return fig


def dot_ci(d: pd.DataFrame, x: str, title: str, xlabel: str, ref: float | None = None, xrange=None) -> go.Figure:
    """Group estimates with 95% intervals (same look as the 2025-26 «mean by programme» charts)."""
    d = d.sort_values(x)
    fig = px.scatter(
        d,
        x=x,
        y="group",
        error_x=d["ci_high"] - d[x],
        error_x_minus=d[x] - d["ci_low"],
        size="n",
        color=x,
        color_continuous_scale="Viridis",
        title=title,
        labels={x: xlabel, "group": "Группа", "n": "Число ответов", "ci_low": "Нижняя граница ДИ", "ci_high": "Верхняя граница ДИ"},
        hover_data={"n": True, x: ":.2f", "ci_low": ":.2f", "ci_high": ":.2f"},
    )
    if ref is not None and np.isfinite(ref):
        fig.add_vline(ref, line_dash="dash", line_color="gray", annotation_text="в целом", annotation_position="top")
    if xrange:
        fig.update_xaxes(range=xrange)
    fig.update_yaxes(automargin=True, categoryorder="array", categoryarray=d["group"].tolist(), title="")
    fig.update_layout(height=max(300, 34 * len(d) + 130), coloraxis_showscale=False)
    return fig


def base_caption(counts: pd.DataFrame, var: str, meta: dict, multi: bool) -> str:
    base = counts.attrs.get("base", 0)
    courses = courses_of(meta, var)
    who = f" Кому задавался вопрос: {', '.join(courses)}." if courses and len(courses) < 4 else ""
    kind = " Можно было выбрать несколько вариантов, поэтому сумма долей больше 100%." if multi else ""
    return f"Ответили на вопрос: {base}.{who}{kind}"


def render_question(df: pd.DataFrame, var: str, meta: dict, dim: str | None, min_n: int, key: str) -> None:
    """Distribution of answers to a single- or multiple-choice question + optional breakdown."""
    multi = var in MULTI_VARS
    order = options_of(meta, var)
    counts = multi_counts(df, var, order) if multi else single_counts(df, var, order)
    st.markdown(f"### {vlabel(var)}")
    q = question_text(meta, var)
    if q and var not in DERIVED_VARS:
        st.caption(f"Вопрос анкеты: «{q}»")
    if counts.empty:
        st.info("В текущем срезе нет ответов на этот вопрос.")
        return
    st.caption("График: распределение ответов, % ответивших.")
    st.plotly_chart(bar_pct(counts, ""), width="stretch", key=f"{key}_bar")
    st.caption(base_caption(counts, var, meta, multi))

    if not dim:
        return
    pct, sizes, hidden = crosstab_pct(df, var, dim, order, min_n)
    if pct.empty or pct.shape[1] < 2:
        st.caption(f"Разбивка по признаку «{GROUP_DIMS[dim]}» недоступна: меньше двух групп, в которых не менее {min_n} ответов.")
    else:
        st.caption(f"График: доля каждого ответа внутри групп (признак «{GROUP_DIMS[dim]}»), %.")
        st.plotly_chart(
            heatmap_pct(pct, f"{vlabel(var)}: разбивка по признаку «{GROUP_DIMS[dim]}», % группы", dim),
            width="stretch",
            key=f"{key}_heat",
        )
        st.caption("Размер групп: " + "; ".join(f"{g} — {int(sizes[g])}" for g in pct.columns) + ".")
    if hidden:
        st.caption(f"Не показаны группы, в которых меньше {min_n} ответов: {', '.join(map(str, hidden))}.")


def dim_selector(key: str, default: str | None = "year", options: list | None = None) -> str | None:
    options = options or [None, "year", "school", "program_display"]
    return st.selectbox(
        "Разбивка",
        options,
        index=options.index(default),
        format_func=lambda d: "Без разбивки" if d is None else GROUP_DIMS[d],
        key=key,
        help="Признак, по которому ответы сравниваются между группами студентов.",
    )


def mean_by_group(df: pd.DataFrame, var: str, dim: str, min_n: int) -> pd.DataFrame:
    rows = []
    for g, part in df.groupby(dim):
        mean, lo, hi, n = bootstrap_mean_ci(part[var])
        if n >= min_n:
            rows.append({"group": g, "n": n, "mean": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def share_by_group(df: pd.DataFrame, flag: pd.Series, dim: str, min_n: int) -> pd.DataFrame:
    """Share of True in `flag` within each group, with Wilson intervals."""
    rows = []
    d = df.assign(__flag=flag).dropna(subset=["__flag"])
    for g, part in d.groupby(dim):
        n, k = len(part), int(part["__flag"].astype(bool).sum())
        if n >= min_n:
            lo, hi = wilson_ci(k, n)
            rows.append({"group": g, "n": n, "k": k, "pct": k / n * 100, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def share(flag: pd.Series) -> float:
    flag = flag.dropna()
    return float(flag.astype(bool).mean() * 100) if len(flag) else np.nan


# ── Tabs ─────────────────────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame, cont_slice: pd.DataFrame, year_label: str) -> None:
    st.subheader("Обзор")
    st.info("Что здесь: объём и структура выборки, отклик и ключевые показатели анкеты.")

    upper = df[df["year"] != "1 курс"]
    status = upper["retention_status"].dropna()
    works = df["plans_to_work"].dropna()
    travel = df["travel_min_num"].dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ответов", fmt(len(df), 0), help="Число анкет после применения фильтров (тестовые ответы исключены при обработке).")
    c2.metric("Программ", fmt(df["program"].nunique(), 0), help="Число программ в текущем срезе.")
    c3.metric("Школ", fmt(df["school"].nunique(), 0), help="Число школ в текущем срезе.")
    c4.metric(
        "Уверенность в целях",
        f"{fmt(df['confidence'].mean())} из 5",
        help="Средняя оценка по вопросу «Насколько вы уверены, что достигнете этих целей?» (шкала от 1 до 5).",
    )
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Думали об уходе",
        fmt_pct(share(upper["retention_risk"])),
        help="Доля студентов 2-го курса и старше, которые думали о переводе, отчислении или академическом отпуске "
        "в прошлом году или думают об этом сейчас. Ответы «Предпочитаю не отвечать» не учитываются.",
    )
    c6.metric(
        "Думают об уходе сейчас",
        fmt_pct(share(status == "Думают сейчас") if len(status) else np.nan),
        help="Доля студентов 2-го курса и старше, выбравших вариант «Думаю об этом сейчас».",
    )
    c7.metric(
        "Работают или планируют",
        fmt_pct(share(works == "Работают или планируют работать") if len(works) else np.nan),
        help="Доля тех, кто уже работает или собирается работать в этом учебном году.",
    )
    c8.metric(
        "Дорога дольше часа",
        fmt_pct(share(travel > 60) if len(travel) else np.nan),
        help="Доля тех, у кого дорога до кампуса в одну сторону занимает больше 60 минут.",
    )
    render_interp(
        "Как интерпретировать обзор",
        [
            "Вопросы анкеты зависят от курса: первокурсникам задавались вопросы о поступлении, "
            "студентам 2-го курса и старше — о прошлом учебном годе и мыслях об уходе.",
            "Доли в карточках рассчитаны от числа ответивших на конкретный вопрос, а не от всей выборки.",
            "Если по программе мало ответов, то доли по ней неустойчивы: обращайте внимание на размер групп и доверительные интервалы.",
        ],
    )

    render_response_rate(df, cont_slice)

    st.markdown("### Структура выборки")
    left, right = st.columns(2)
    with left:
        by_year = df["year"].value_counts().sort_index().rename_axis("Курс").reset_index(name="Ответов")
        st.caption("График: число ответов по курсам.")
        fig = px.bar(by_year, x="Курс", y="Ответов", text="Ответов", color="Ответов",
                     color_continuous_scale="Teal", title="Ответы по курсам")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=360, xaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        by_school = df["school"].fillna("Не указана").value_counts().rename_axis("Школа").reset_index(name="Ответов")
        st.caption("График: число ответов по школам.")
        fig = px.bar(by_school, x="Школа", y="Ответов", text="Ответов", color="Ответов",
                     color_continuous_scale="Teal", title="Ответы по школам")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=360, xaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    by_prog = df.groupby(["program_display", "year"]).size().rename("Ответов").reset_index()
    totals = by_prog.groupby("program_display")["Ответов"].sum().sort_values()
    st.caption("График: размер выборки по программам с разбивкой по курсам.")
    fig = px.bar(
        by_prog,
        x="Ответов",
        y="program_display",
        color="year",
        orientation="h",
        title="Размер выборки по программам",
        labels={"program_display": "Программа", "year": "Курс"},
        category_orders={"year": sorted(by_prog["year"].unique()), "program_display": totals.index[::-1].tolist()},
    )
    # one offset group, otherwise plotly draws the courses side by side instead of stacking them
    fig.update_traces(marker_line_color="white", marker_line_width=1, offsetgroup="0", alignmentgroup="0")
    fig.update_layout(barmode="stack", height=max(380, 30 * len(totals) + 150), yaxis_title="", legend_title_text="Курс")
    st.plotly_chart(fig, width="stretch")

    by_day = (
        df.dropna(subset=["date"]).assign(day=lambda d: d["date"].dt.normalize())
        .groupby("day").size().rename("Ответов").reset_index()
    )
    if not by_day.empty:
        st.caption("График: число ответов по дням сбора.")
        fig = px.bar(by_day, x="day", y="Ответов", title="Ответы по дням", labels={"day": "Дата"},
                     color_discrete_sequence=[ACCENT])
        fig.update_xaxes(tickformat="%d.%m")
        fig.update_layout(height=320, xaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Период сбора ответов: {by_day['day'].min():%d.%m.%Y} — {by_day['day'].max():%d.%m.%Y}.")

    st.markdown("### Навигатор по вкладкам")
    show_table(
        pd.DataFrame(
            [
                {"Вкладка": "Первый курс", "Что внутри": "Как выбирали программу, участие в олимпиадах, уровень английского языка."},
                {"Вкладка": "Цели и трудности", "Что внутри": "Приоритеты на год, ожидаемые трудности, уверенность в достижении целей."},
                {"Вкладка": "Нагрузка и быт", "Что внутри": "Время на самоподготовку (план и прошлый год), работа, дорога до кампуса."},
                {"Вкладка": "Удержание и поддержка", "Что внутри": "Мысли об уходе (2-й курс и старше), чего не хватает в программе, потребность в специальных условиях."},
                {"Вкладка": "Внеучебная жизнь", "Что внутри": "Спорт и проекты, игры, интерес к турниру, пожелания по внеучебным активностям."},
                {"Вкладка": "Сравнение групп", "Что внутри": "Любой вопрос по курсам, школам или программам со статистической проверкой различий."},
                {"Вкладка": "Сравнение по годам", "Что внутри": f"Осенние анкеты разных учебных лет (сейчас загружен {year_label})."},
                {"Вкладка": "Открытые ответы", "Что внутри": "Свободные ответы и собственные варианты с поиском по тексту."},
                {"Вкладка": "Кодификатор", "Что внутри": "Переменные набора данных, формулировки вопросов, варианты ответов и правила обработки."},
            ]
        ),
        hide_index=True,
    )


def render_freshmen(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Первый курс")
    st.info("Что здесь: вопросы, которые задавались только первокурсникам, — как выбирали программу, участие в олимпиадах, английский язык.")
    d = df[df["year"] == "1 курс"]
    if d.empty:
        st.warning("В текущем срезе нет ответов первокурсников: проверьте фильтр «Курс».")
        return
    st.caption(f"Ответов первокурсников в срезе: {len(d)}.")
    render_interp(
        "Как интерпретировать",
        [
            "Если большинство принимали решение самостоятельно, то коммуникацию о программе стоит адресовать прежде всего абитуриентам, а не родителям.",
            "Если на программе много призёров олимпиад, то можно предлагать им более сложные задания и проектные треки.",
            "Если студенты с низкой самооценкой английского чаще называют его трудностью, то им стоит заранее предложить поддержку.",
        ],
    )
    dim = dim_selector("fr_dim", default="program_display", options=[None, "school", "program_display"])
    for var in ["decision", "olympiads", "english_level"]:
        render_question(d, var, meta, dim, min_n, key=f"fr_{var}")

    st.markdown("### Английский язык как ожидаемая трудность")
    st.caption(
        "График: доля первокурсников, назвавших английский язык одной из главных трудностей года, "
        "в зависимости от самооценки уровня (с 95%-м доверительным интервалом)."
    )
    g = share_by_group(d, has_option(d, "difficulties", "Английский язык"), "english_level", min_n)
    if len(g):
        st.plotly_chart(
            dot_ci(g, "pct", "Считают английский язык трудностью, %", "Доля, %", xrange=[0, 100]),
            width="stretch",
        )
    else:
        st.caption(f"Недостаточно групп, в которых не менее {min_n} ответов.")


def render_goals(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Цели и трудности")
    st.info("Что здесь: приоритеты на учебный год, ожидаемые трудности и уверенность в достижении целей.")
    render_interp(
        "Как интерпретировать",
        [
            "В вопросах с несколькими вариантами ответа доля рассчитана от числа ответивших, поэтому сумма долей больше 100%.",
            "Тепловая карта показывает доли внутри каждой группы: так можно сравнивать группы разного размера.",
            "Если вариант заметно чаще выбирают в одной группе, то это повод обсудить её особенности с кураторами.",
            "Уверенность оценивается по шкале от 1 до 5; точка — среднее, отрезок — 95%-й доверительный интервал (бутстрэп).",
        ],
    )
    dim = dim_selector("goals_dim")
    for var in ["priorities", "difficulties"]:
        render_question(df, var, meta, dim, min_n, key=f"goals_{var}")

    st.markdown(f"### {vlabel('confidence')}")
    st.caption(f"Вопрос анкеты: «{question_text(meta, 'confidence')}»")
    conf_df = df.assign(c=df["confidence"].map(lambda v: str(int(v)) if pd.notna(v) else np.nan))
    conf = single_counts(conf_df, "c", ["1", "2", "3", "4", "5"])
    if conf.empty:
        return
    overall = df["confidence"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Среднее", fmt(overall), help="Средняя оценка уверенности (от 1 до 5).")
    c2.metric("Медиана", fmt(df["confidence"].median(), 0))
    c3.metric("Уверены (4–5 баллов)", fmt_pct(share(df["confidence"].dropna() >= 4)))
    st.caption("График: распределение оценок уверенности, % ответивших.")
    st.plotly_chart(bar_pct(conf, "Распределение оценок уверенности"), width="stretch", key="conf_bar")
    if dim:
        g = mean_by_group(df, "confidence", dim, min_n)
        if len(g) >= 2:
            st.caption(f"График: средняя уверенность по признаку «{GROUP_DIMS[dim]}» с 95%-м доверительным интервалом.")
            st.plotly_chart(
                dot_ci(g, "mean", f"Уверенность в достижении целей: разбивка по признаку «{GROUP_DIMS[dim]}»",
                       "Средняя оценка (1–5)", ref=overall, xrange=[1, 5]),
                width="stretch",
                key="conf_dim",
            )

    st.markdown("### Уверенность и ожидаемые трудности")
    st.caption(
        "График: средняя уверенность студентов, отметивших ту или иную трудность. Трудности, при которых уверенность "
        "заметно ниже средней, указывают, где студентам может понадобиться поддержка."
    )
    ex = explode_multi(df, "difficulties", keep=["confidence"]).dropna(subset=["confidence"])
    g = mean_by_group(ex.rename(columns={"difficulties": "__g"}), "confidence", "__g", min_n)
    if len(g):
        st.plotly_chart(
            dot_ci(g, "mean", "Уверенность у отметивших трудность", "Средняя оценка (1–5)", ref=overall, xrange=[1, 5]),
            width="stretch",
            key="conf_diff",
        )
    st.caption(f"Ответов на открытый вопрос об ожиданиях: {int(df['expectations'].notna().sum())}. Их можно прочитать на вкладке «Открытые ответы».")


def render_load(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Нагрузка и быт")
    st.info("Что здесь: сколько времени студенты планируют тратить на самоподготовку, работа во время учёбы и дорога до кампуса.")
    render_interp(
        "Как интерпретировать",
        [
            "Время на самоподготовку указано интервалами; для средних берётся середина интервала (0, 3, 8, 13, 18, 23 и 28 ч), поэтому средние приблизительные.",
            "Сравнение с прошлым годом доступно только для студентов 2-го курса и старше: первокурсникам этот вопрос не задавался.",
            "Если студент планирует тратить больше времени, чем в прошлом году, то он ожидает роста нагрузки; "
            "если меньше, то причиной может быть работа или усталость.",
            "Если доля работающих в группе высокая, то стоит учитывать это при составлении расписания и сроков сдачи.",
        ],
    )
    dim = dim_selector("load_dim")
    render_question(df, "hours_planned", meta, dim, min_n, key="load_hp")
    if dim:
        g = mean_by_group(df, "hours_planned_num", dim, min_n)
        if len(g) >= 2:
            st.caption(f"График: среднее запланированное время на самоподготовку по признаку «{GROUP_DIMS[dim]}», ч в неделю.")
            st.plotly_chart(
                dot_ci(g, "mean", "Самоподготовка по плану, ч в неделю", "Часов в неделю (оценка)",
                       ref=df["hours_planned_num"].mean()),
                width="stretch",
            )

    st.markdown("### Самоподготовка: план на этот год и прошлый год")
    both = df.dropna(subset=["hours_planned", "hours_last_year"])
    if both.empty:
        st.caption("В текущем срезе нет студентов 2-го курса и старше, ответивших на оба вопроса.")
    else:
        order = options_of(meta, "hours_planned")
        ct = (
            pd.crosstab(both["hours_last_year"], both["hours_planned"])
            .reindex(index=order, columns=order).fillna(0).astype(int)
        )
        ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
        st.caption("График: число студентов по сочетанию «сколько часов тратили в прошлом году» и «сколько планируют сейчас».")
        fig = px.imshow(
            ct,
            text_auto=True,
            color_continuous_scale="Teal",
            aspect="auto",
            labels={"x": "План на этот год, ч в неделю", "y": "Прошлый год, ч в неделю", "color": "Студентов"},
            title="Самоподготовка: прошлый год и план на этот год",
        )
        fig.update_layout(height=480)
        st.plotly_chart(fig, width="stretch")
        delta = (both["hours_planned_num"] - both["hours_last_year_num"]).mean()
        st.caption(
            f"Клетки на диагонали — студенты, которые планируют тратить столько же времени. Ответили на оба вопроса: {len(both)}. "
            f"В среднем план отличается от прошлого года на {fmt(delta, 1)} ч в неделю (оценка по серединам интервалов)."
        )
        render_question(df, "hours_change", meta, dim, min_n, key="load_hc")

    for var in ["work_plans", "work_related", "travel_time"]:
        render_question(df, var, meta, dim, min_n, key=f"load_{var}")

    st.markdown("### Работа и совмещение с учёбой")
    st.caption(
        "График: доля студентов, назвавших трудностью совмещение учёбы с работой, в зависимости от планов на работу "
        "(с 95%-м доверительным интервалом)."
    )
    d = df.dropna(subset=["plans_to_work"])
    g = share_by_group(d, has_option(d, "difficulties", "Совмещение учёбы с работой"), "plans_to_work", min_n)
    if len(g):
        st.plotly_chart(
            dot_ci(g, "pct", "Считают трудностью совмещение учёбы с работой, %", "Доля, %", xrange=[0, 100]),
            width="stretch",
        )


def render_retention(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Удержание и поддержка")
    st.info(
        "Что здесь: мысли о переводе, отчислении и академическом отпуске (2-й курс и старше), "
        "чего не хватает в программе, а также потребность в специальных условиях обучения."
    )
    render_interp(
        "Как интерпретировать",
        [
            "«Думали об уходе» — студенты, отметившие любой вариант «Думал(а) о…» или «Думаю об этом сейчас». "
            "Ответы «Предпочитаю не отвечать» не учитываются.",
            "Доверительный интервал (метод Уилсона) показывает неопределённость доли: чем меньше ответов, тем он шире.",
            "Если какой-то дефицит заметно чаще отмечают те, кто думал об уходе, то он связан с риском ухода. "
            "Это связь, а не доказанная причина.",
        ],
    )
    up = df[df["year"] != "1 курс"]
    if up.empty:
        st.warning("В текущем срезе нет ответов студентов 2-го курса и старше.")
        return
    risk = up["retention_risk"].dropna()
    status = up["retention_status"].dropna()
    k, n = int(risk.sum()), len(risk)
    lo, hi = wilson_ci(k, n)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Думали об уходе",
        fmt_pct(k / n * 100 if n else np.nan),
        help=f"{k} из {n}; 95%-й доверительный интервал: {fmt(lo, 0)}–{fmt(hi, 0)}%.",
    )
    c2.metric("Думают об уходе сейчас", fmt_pct(share(status == "Думают сейчас") if len(status) else np.nan))
    c3.metric("Уже перевелись", fmt_pct(share(status == "Уже перевелись") if len(status) else np.nan))

    dim = dim_selector("ret_dim")
    render_question(up, "thoughts_transfer", meta, None, min_n, key="ret_tt")
    render_question(up, "retention_status", meta, dim, min_n, key="ret_status")
    if dim:
        g = share_by_group(up, up["retention_risk"], dim, min_n)
        if len(g) >= 2:
            st.caption(f"График: доля думавших об уходе по признаку «{GROUP_DIMS[dim]}» с 95%-м доверительным интервалом.")
            st.plotly_chart(
                dot_ci(g, "pct", f"Думали об уходе: разбивка по признаку «{GROUP_DIMS[dim]}», %", "Доля, %",
                       ref=k / n * 100 if n else None, xrange=[0, 100]),
                width="stretch",
            )
    st.caption(
        f"Ответов на вопрос «Что на это повлияло?»: {int(up['transfer_influences'].notna().sum())}. "
        "Их можно прочитать на вкладке «Открытые ответы»."
    )

    render_question(up, "missing", meta, dim, min_n, key="ret_missing")

    st.markdown("### Чего не хватает: думавшие об уходе и остальные")
    d = up.dropna(subset=["retention_risk"])
    d = d[d["missing"] != ""]
    sizes = d.groupby("retention_risk").size()
    if len(sizes) < 2 or sizes.min() < min_n:
        st.caption(f"Недостаточно ответов в одной из групп (нужно не менее {min_n}).")
    else:
        ex = explode_multi(d, "missing", keep=["retention_risk"])
        ct = pd.crosstab(ex["missing"], ex["retention_risk"]).reindex(columns=[0.0, 1.0], fill_value=0)
        pct = ct.div(sizes.reindex([0.0, 1.0]), axis=1) * 100
        pvals = []
        for opt in ct.index:
            table = np.array(
                [[ct.loc[opt, 0.0], sizes[0.0] - ct.loc[opt, 0.0]], [ct.loc[opt, 1.0], sizes[1.0] - ct.loc[opt, 1.0]]]
            )
            pvals.append(stats.fisher_exact(table)[1])
        res = pd.DataFrame(
            {
                "Чего не хватает": ct.index,
                "Не думали об уходе, %": pct[0.0].values,
                "Думали об уходе, %": pct[1.0].values,
                "Разница, п. п.": (pct[1.0] - pct[0.0]).values,
                "p-значение": multipletests(pvals, method="holm")[1],
            }
        ).sort_values("Разница, п. п.", ascending=False)
        long = res.melt(
            id_vars="Чего не хватает", value_vars=["Не думали об уходе, %", "Думали об уходе, %"],
            var_name="Группа", value_name="Доля, %",
        )
        long["Группа"] = long["Группа"].str.replace(", %", "", regex=False)
        st.caption("График: доля отметивших каждый дефицит в двух группах, %.")
        fig = px.bar(
            long,
            x="Доля, %",
            y="Чего не хватает",
            color="Группа",
            barmode="group",
            orientation="h",
            color_discrete_map={"Не думали об уходе": MUTED, "Думали об уходе": ACCENT},
            category_orders={"Чего не хватает": res["Чего не хватает"].tolist()},
            title="Чего не хватает в программе: думавшие об уходе и остальные",
        )
        fig.update_layout(height=max(380, 44 * len(res) + 150), yaxis_title="", legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Не думали об уходе: {int(sizes[0.0])}; думали об уходе: {int(sizes[1.0])} "
            "(учтены студенты, ответившие на оба вопроса)."
        )
        st.caption("Таблица: сравнение групп; p-значение — точный критерий Фишера с поправкой Холма на число вариантов.")
        show_table(res, hide_index=True)

    st.markdown("### Специальные условия обучения")
    st.caption(
        "В анкете обещан ограниченный доступ к этим ответам, поэтому здесь приведены только обобщённые доли. "
        "Типы потребностей, описания и контакты хранятся отдельно (`data/private/`) и не публикуются. "
        "Группы, в которых меньше 5 ответов, не показываются."
    )
    sn_dim = dim_selector("sn_dim", default="year", options=[None, "year", "school"])
    render_question(df, "special_needs", meta, sn_dim, max(min_n, 5), key="ret_sn")


def normalize_game(title: str) -> str:
    key = re.sub(r"\s+", " ", title.strip().lower())
    return GAME_ALIASES.get(key, title.strip())


def games_long(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """One row per (respondent, game): listed options plus written-in titles."""
    rows = []
    for var in GAME_VARS:
        listed = set(options_of(meta, var)) - {NO_GAMES, OTHER_OPTION}
        for rid, items, other in zip(df.index, df[var], df[f"{var}_other"]):
            for g in [i for i in items.split("; ") if i in listed]:
                rows.append({"rid": rid, "Категория": vlabel(var), "Игра": g})
            for g in [i for i in other.split("; ") if i]:
                rows.append({"rid": rid, "Категория": vlabel(var), "Игра": normalize_game(g)})
    out = pd.DataFrame(rows, columns=["rid", "Категория", "Игра"])
    # the questionnaire lists some titles in lower case («шахматы», «квизы»)
    out["Игра"] = out["Игра"].map(lambda g: g[:1].upper() + g[1:])
    return out.drop_duplicates()


def activity_themes(texts: pd.Series) -> pd.DataFrame:
    rows = []
    for idx, text in texts.items():
        low = str(text).lower()
        for theme, pattern in ACTIVITY_THEMES.items():
            if re.search(pattern, low):
                rows.append({"idx": idx, "Тема": theme})
    return pd.DataFrame(rows, columns=["idx", "Тема"])


def render_leisure(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Внеучебная жизнь")
    st.info(
        "Что здесь: чем студенты занимаются вне учёбы, во что играют, интересен ли им университетский турнир "
        "и в каких внеучебных активностях они хотели бы участвовать."
    )
    render_interp(
        "Как интерпретировать",
        [
            "Вопросы об играх задавались только тем, кто отметил соответствующий пункт в вопросе «Что из этого про вас?».",
            "Если группа часто отмечает спорт или творческие проекты, то анонсы клубов и мероприятий стоит адресовать именно ей.",
            "Если заметная доля готова помочь с организацией турнира, то на них можно опереться при подготовке.",
            "Темы пожеланий определены автоматически по ключевым словам, поэтому возможны неточности; для выводов читайте сами ответы.",
        ],
    )

    about = answered(df, "about_you")
    gamer_opts = [o for o in options_of(meta, "about_you") if o.startswith("Играю в") and "шахматы" not in o]
    is_gamer = about["about_you"].str.split("; ").map(lambda items: any(o in items for o in gamer_opts))
    tour = df["tournament_interest"].dropna()
    ready = tour.isin(["Да, участвовать", "Готов(а) помочь с организацией или стать капитаном команды"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Занимаются спортом", fmt_pct(share(has_option(about, "about_you", "Занимаюсь спортом"))),
              help="Доля от ответивших на вопрос «Что из этого про вас?».")
    c2.metric("Участвуют в проектах",
              fmt_pct(share(has_option(about, "about_you", "Участвую в творческих или общественных проектах"))),
              help="Доля участвующих в творческих или общественных проектах.")
    c3.metric("Играют в видеоигры", fmt_pct(share(is_gamer)),
              help="Отметили командные, одиночные компьютерные или мобильные игры.")
    c4.metric("Готовы участвовать в турнире", fmt_pct(share(ready) if len(tour) else np.nan),
              help="Выбрали «Да, участвовать» или «Готов(а) помочь с организацией или стать капитаном команды».")

    dim = dim_selector("leis_dim")
    render_question(df, "about_you", meta, dim, min_n, key="leis_about")
    render_question(df, "tournament_interest", meta, dim, min_n, key="leis_tour")

    st.markdown("### Интерес к турниру среди играющих и не играющих")
    d = about.assign(__g=np.where(is_gamer, "Играют в видеоигры", "Не играют в видеоигры")).dropna(subset=["tournament_interest"])
    if d["__g"].nunique() == 2 and d.groupby("__g").size().min() >= min_n:
        pct, sizes, _ = crosstab_pct(d, "tournament_interest", "__g", options_of(meta, "tournament_interest"), min_n)
        long = pct.reset_index().melt(id_vars=pct.index.name or "index", var_name="Группа", value_name="Доля, %")
        long = long.rename(columns={pct.index.name or "index": "Ответ"})
        st.caption("График: распределение ответов о турнире в двух группах, % группы.")
        fig = px.bar(
            long, x="Доля, %", y="Ответ", color="Группа", barmode="group", orientation="h",
            color_discrete_map={"Не играют в видеоигры": MUTED, "Играют в видеоигры": ACCENT},
            category_orders={"Ответ": pct.index.tolist()},
            title="Интерес к турниру: играющие и не играющие в видеоигры",
        )
        fig.update_layout(height=380, yaxis_title="", legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Размер групп: " + "; ".join(f"{g} — {int(sizes[g])}" for g in pct.columns) + ".")
    else:
        st.caption(f"Недостаточно ответов в одной из групп (нужно не менее {min_n}).")

    st.markdown("### Во что играют хотя бы раз в неделю")
    games = games_long(df, meta)
    if games.empty:
        st.caption("В текущем срезе нет ответов об играх.")
    else:
        top_n = st.slider("Сколько игр показать", 5, 40, 20, key="leis_top")
        totals = games.groupby("Игра")["rid"].nunique().sort_values(ascending=False)
        top = totals.head(top_n).index
        by_cat = games[games["Игра"].isin(top)].groupby(["Игра", "Категория"])["rid"].nunique().rename("Игроков").reset_index()
        st.caption(
            "График: самые популярные игры по всем категориям — варианты из анкеты вместе с собственными ответами "
            "(написания одной игры объединены)."
        )
        fig = px.bar(
            by_cat, x="Игроков", y="Игра", color="Категория", orientation="h",
            category_orders={"Игра": top.tolist(), "Категория": [vlabel(v) for v in GAME_VARS]},
            title="Самые популярные игры",
            labels={"Игроков": "Число студентов"},
        )
        fig.update_traces(marker_line_color="white", marker_line_width=1, offsetgroup="0", alignmentgroup="0")
        fig.update_layout(barmode="stack", height=max(380, 26 * len(top) + 150), yaxis_title="", legend_title_text="Категория")
        st.plotly_chart(fig, width="stretch")

        cat = st.radio("Категория игр", GAME_VARS, format_func=vlabel, horizontal=True, key="leis_cat")
        render_question(df, cat, meta, None, min_n, key=f"leis_{cat}")
        other = games[(games["Категория"] == vlabel(cat))]
        listed = set(options_of(meta, cat))
        other = other[~other["Игра"].isin(listed)]
        if not other.empty:
            with st.expander(f"Игры из собственных ответов ({other['Игра'].nunique()})"):
                st.caption("Таблица: игры, которые студенты вписали сами, по числу упоминаний.")
                show_table(
                    other.groupby("Игра")["rid"].nunique().rename("Упоминаний").sort_values(ascending=False).reset_index(),
                    hide_index=True,
                )

    st.markdown("### Какие внеучебные активности хотели бы видеть")
    st.caption(f"Вопрос анкеты: «{question_text(meta, 'extracurricular')}»")
    texts = df["extracurricular"].dropna().astype(str).str.strip()
    texts = texts[texts.str.contains(r"[A-Za-zА-Яа-яЁё]", regex=True)]
    if texts.empty:
        st.caption("В текущем срезе нет ответов на этот вопрос.")
        return
    themes = activity_themes(texts)
    if not themes.empty:
        counts = themes.groupby("Тема").size().rename("n").reset_index().rename(columns={"Тема": "answer"})
        counts["pct"] = counts["n"] / len(texts) * 100
        counts = counts.sort_values("n", ascending=False)
        st.caption("График: темы ответов (определены по ключевым словам), % ответивших; один ответ может относиться к нескольким темам.")
        st.plotly_chart(bar_pct(counts, "Темы пожеланий по внеучебным активностям"), width="stretch", key="leis_themes")
        st.caption(f"Ответили на вопрос: {len(texts)}; не отнесены ни к одной теме: {len(texts) - themes['idx'].nunique()}.")

    c1, c2 = st.columns([2, 3])
    theme_pick = c1.selectbox("Тема", ["Все ответы"] + list(ACTIVITY_THEMES), key="leis_theme")
    search = c2.text_input("Поиск по тексту", placeholder="Ключевое слово…", key="leis_search")
    shown = df.loc[texts.index, ["program_display", "school", "year"]].assign(Ответ=texts)
    if theme_pick != "Все ответы":
        shown = shown.loc[themes.loc[themes["Тема"] == theme_pick, "idx"].unique()]
    if search.strip():
        shown = shown[shown["Ответ"].str.contains(search.strip(), case=False, regex=False)]
    st.caption(f"Таблица: ответы студентов. Показано: {len(shown)}.")
    st.dataframe(
        shown.rename(columns={"program_display": "Программа", "school": "Школа", "year": "Курс"}),
        width="stretch",
        hide_index=True,
        column_config={"Ответ": st.column_config.TextColumn("Ответ", width="large")},
    )


def render_group_comparison(df: pd.DataFrame, meta: dict, min_n: int) -> None:
    st.subheader("Сравнение групп")
    st.info(
        "Что здесь: любой вопрос анкеты в разбивке по курсам, школам или программам со статистической проверкой различий. "
        "Для вопросов с одним вариантом ответа — критерий χ² и коэффициент V Крамера; для вопросов с несколькими "
        "вариантами — критерий χ² по каждому варианту с поправкой Холма; для числовых показателей — критерий "
        "Краскела — Уоллиса и попарные сравнения по критерию Данна."
    )
    kinds = {v: "single" for v in SINGLE_VARS} | {v: "multi" for v in MULTI_VARS} | {v: "numeric" for v in NUMERIC_VARS}
    c1, c2, c3 = st.columns([3, 2, 2])
    var = c1.selectbox("Вопрос", list(kinds), index=list(kinds).index("priorities"), format_func=vlabel, key="cmp_var")
    dim = c2.selectbox("Группы", list(GROUP_DIMS), format_func=GROUP_DIMS.get, key="cmp_dim")
    min_g = c3.slider(
        "Минимальный размер группы для тестов", 3, 30, max(min_n, 10), key="cmp_min",
        help="Группы, в которых меньше ответов, исключаются из статистических тестов.",
    )
    render_interp(
        "Как интерпретировать сравнение групп",
        [
            "Если p-значение меньше 0,05, то различия между группами вряд ли случайны.",
            "Если V Крамера около 0,1, 0,3 или 0,5 и выше, то связь обычно считают слабой, умеренной или сильной соответственно.",
            "Если во многих клетках ожидаемая частота меньше 5, то критерий χ² ненадёжен: укрупните группы (например, школы вместо программ) или повысьте порог.",
            "Если критерий Краскела — Уоллиса показывает различия, то попарные сравнения Данна покажут, какие именно группы различаются.",
        ],
    )
    kind = kinds[var]
    d = df.dropna(subset=[var]) if kind == "numeric" else answered(df, var)
    sizes = d.groupby(dim).size()
    keep = sizes[sizes >= min_g].index
    d = d[d[dim].isin(keep)]
    st.caption(
        f"В анализе: групп — {len(keep)}, ответов — {len(d)}. "
        f"Исключено групп, в которых меньше {min_g} ответов: {int((sizes < min_g).sum())}."
    )
    if len(keep) < 2:
        st.info("Для сравнения нужны как минимум две группы достаточного размера.")
        return

    if kind == "numeric":
        groups = [g[var].to_numpy() for _, g in d.groupby(dim)]
        h, p = stats.kruskal(*groups)
        eps2 = (h - len(groups) + 1) / (len(d) - len(groups))
        k1, k2, k3 = st.columns(3)
        k1.metric("p-значение (Краскел — Уоллис)", fmt_p(p), help="Проверка того, различаются ли распределения между группами.")
        k2.metric("Размер эффекта ε²", fmt(eps2, 3),
                  help="Доля вариации рангов, объяснённая группой: около 0,01 — малый эффект, 0,08 — средний, 0,26 — крупный.")
        k3.metric("Групп", fmt(len(groups), 0))
        g = mean_by_group(d, var, dim, min_g)
        st.caption("График: средние по группам с 95%-м доверительным интервалом (бутстрэп).")
        st.plotly_chart(dot_ci(g, "mean", f"{vlabel(var)}: среднее по группам", "Среднее", ref=d[var].mean()), width="stretch")
        st.caption("График: распределение значений по группам.")
        fig = px.box(d, x=var, y=dim, orientation="h", points="all", color=dim,
                     labels={var: vlabel(var), dim: GROUP_DIMS[dim]}, title=f"{vlabel(var)}: распределение по группам")
        fig.update_layout(showlegend=False, height=max(340, 40 * len(keep) + 130), yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Таблица: попарные сравнения по критерию Данна, p-значения с поправкой Холма.")
        show_table(sp.posthoc_dunn(d, val_col=var, group_col=dim, p_adjust="holm"), decimals=3)
        return

    order = options_of(meta, var)
    pct, _, _ = crosstab_pct(d, var, dim, order, min_g)
    st.caption("График: доля каждого ответа внутри групп, %.")
    st.plotly_chart(heatmap_pct(pct, f"{vlabel(var)}, % группы", dim), width="stretch")
    if kind == "single":
        counts = pd.crosstab(d[var].astype(str), d[dim])
        res = chi2_summary(counts)
        k1, k2, k3 = st.columns(3)
        k1.metric("p-значение (χ²)", fmt_p(res["p"]))
        k2.metric("V Крамера", fmt(res["cramers_v"]))
        k3.metric("Клеток с ожидаемой частотой меньше 5", fmt_pct(res["low_expected_share"] * 100))
        if np.isfinite(res["low_expected_share"]) and res["low_expected_share"] > 0.2:
            st.warning("Больше 20% клеток имеют ожидаемую частоту меньше 5, поэтому результат критерия χ² ориентировочный.")
        st.caption("Таблица: число ответов по группам.")
        show_table(counts)
    else:
        ex = explode_multi(d, var, keep=[dim])
        chosen = pd.crosstab(ex[var], ex[dim])
        n_g = d.groupby(dim).size().reindex(chosen.columns)
        rows = []
        for opt in chosen.index:
            r = chi2_summary(pd.DataFrame({"да": chosen.loc[opt], "нет": n_g - chosen.loc[opt]}).T)
            rows.append({"Вариант ответа": opt, "Выбрали": int(chosen.loc[opt].sum()), "p (χ²)": r["p"], "V Крамера": r["cramers_v"]})
        res = pd.DataFrame(rows)
        valid = res["p (χ²)"].notna()
        res.loc[valid, "p с поправкой Холма"] = multipletests(res.loc[valid, "p (χ²)"], method="holm")[1]
        res = res.sort_values("p с поправкой Холма")
        st.caption("Таблица: различается ли доля выбравших вариант между группами (критерий χ², поправка Холма на число вариантов).")
        show_table(res, decimals=3, hide_index=True)


# ── Year over year ───────────────────────────────────────────────────────────

def _kpi(fn):
    def wrapped(d: pd.DataFrame) -> float:
        try:
            return fn(d)
        except KeyError:
            return np.nan
    return wrapped


YOY_KPIS = [
    ("Ответов", _kpi(lambda d: len(d)), 0),
    ("Уверенность в достижении целей (1–5)", _kpi(lambda d: d["confidence"].mean()), 2),
    ("Самоподготовка по плану, ч в неделю", _kpi(lambda d: d["hours_planned_num"].mean()), 1),
    ("Работают или планируют работать, %", _kpi(lambda d: share(d["plans_to_work"].dropna() == "Работают или планируют работать")), 0),
    ("Дорога дольше часа, %", _kpi(lambda d: share(d["travel_min_num"].dropna() > 60)), 0),
    ("Думали об уходе (2-й курс и старше), %", _kpi(lambda d: share(d.loc[d["year"] != "1 курс", "retention_risk"])), 0),
    ("Нуждаются в специальных условиях, %", _kpi(lambda d: share(d["special_needs"].dropna() == "Да")), 0),
]


def render_yoy(filters: FilterState, current: str, min_n: int) -> None:
    st.subheader("Сравнение по годам")
    files = autumn_files()
    st.info(
        "Что здесь: сравнение осенних анкет разных учебных лет. Зимняя и весенняя анкеты — другой опросник (SFQ): "
        "их изменения внутри года смотрите в режиме «Зима и весна»."
    )
    if len(files) < 2:
        st.warning(
            f"Сейчас загружены данные только за один год ({', '.join(files) or '—'}). Чтобы сравнить годы, положите "
            "в папку `data/processed/` файлы другого года в том же формате: `autumn_<ГГГГ-ГГ>.csv` и "
            "`autumn_<ГГГГ-ГГ>_meta.json`. Их создаёт команда `python process_autumn.py --raw <выгрузка> --year <ГГГГ-ГГ>`."
        )
        return

    years = st.multiselect("Учебные годы", list(files), default=list(files), key="yoy_years")
    if len(years) < 2:
        st.info("Выберите не менее двух лет.")
        return
    frames = []
    for y in years:
        d = load_autumn(str(files[y]))
        # Programme names change between years, so only course and school filters apply here.
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
    sel = st.multiselect("Программы (названия могут меняться от года к году)", progs, default=progs, key="yoy_progs")
    all_df = all_df[all_df["program"].isin(sel)]
    st.caption("Фильтры «Курс» и «Школа» на боковой панели применяются; программы выбираются в поле выше.")
    render_interp(
        "Как интерпретировать сравнение по годам",
        [
            "Если формулировка вопроса или варианты ответа менялись между годами, то сравнивать ответы на него некорректно: сверяйтесь с кодификаторами.",
            "Если набор программ в разные годы различается, то изменения могут объясняться составом выборки, а не динамикой.",
            "Если p-значение критерия χ² меньше 0,05, то распределения ответов в разные годы различаются статистически значимо.",
        ],
    )

    rows = []
    for name, fn, dec in YOY_KPIS:
        row = {"Показатель": name}
        for y in years:
            row[y] = fmt(fn(all_df[all_df["academic_year"] == y]), dec)
        rows.append(row)
    st.caption("Таблица: ключевые показатели по годам.")
    show_table(pd.DataFrame(rows), hide_index=True)

    meta = load_meta(str(files[current].with_name(f"autumn_{current}_meta.json")))
    candidates = [v for v in SINGLE_VARS + MULTI_VARS if v in all_df.columns]
    var = st.selectbox("Вопрос для сравнения", candidates, format_func=vlabel, key="yoy_var")
    pct, sizes, _ = crosstab_pct(all_df, var, "academic_year", options_of(meta, var), min_n)
    if pct.empty or pct.shape[1] < 2:
        st.info("В выбранных годах недостаточно ответов.")
        return
    long = pct.reset_index().melt(id_vars=pct.index.name or "index", var_name="Учебный год", value_name="Доля, %")
    long = long.rename(columns={pct.index.name or "index": "Ответ"})
    st.caption("График: распределение ответов по годам, % ответивших.")
    fig = px.bar(
        long, x="Доля, %", y="Ответ", color="Учебный год", barmode="group", orientation="h",
        category_orders={"Ответ": pct.index.tolist(), "Учебный год": list(pct.columns)},
        title=f"{vlabel(var)}: сравнение по годам",
    )
    fig.update_layout(height=max(380, 30 * len(pct) * len(pct.columns) + 150), yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.caption("Ответили на вопрос: " + "; ".join(f"{y} — {int(sizes[y])}" for y in pct.columns) + ".")
    if var in SINGLE_VARS:
        d = answered(all_df, var)
        res = chi2_summary(pd.crosstab(d[var].astype(str), d["academic_year"]))
        p_txt = fmt_p(res["p"])
        p_txt = p_txt if p_txt.startswith("<") else f"= {p_txt}"
        st.caption(f"Критерий χ² для различий между годами: p {p_txt}, V Крамера = {fmt(res['cramers_v'])}.")


def render_open_answers(df: pd.DataFrame) -> None:
    st.subheader("Открытые ответы")
    st.info("Что здесь: ответы на открытые вопросы и собственные варианты в вопросах с выбором. Можно отфильтровать ответы по разделу и найти их по слову.")
    render_interp(
        "Как читать открытые ответы",
        [
            "Каждая строка — ответ одного студента на один вопрос анкеты.",
            "Пустые и бессодержательные ответы (например, «-») скрыты; фильтры по программе, курсу и школе применяются.",
            "Поле поиска помогает найти ответы по ключевому слову.",
        ],
    )
    available = {c: lbl for c, lbl in OPEN_TEXT.items() if c in df.columns}
    has_text = {c: df[c].fillna("").astype(str).str.contains(r"[A-Za-zА-Яа-яЁё]", regex=True) for c in available}
    counts = {lbl: int(has_text[c].sum()) for c, lbl in available.items()}

    c_total, *cols = st.columns(1 + len(counts))
    c_total.metric("Всего ответов", fmt(sum(counts.values()), 0))
    for w, (lbl, n) in zip(cols, counts.items()):
        w.metric(lbl, fmt(n, 0))

    counts_df = pd.DataFrame(list(counts.items()), columns=["Раздел", "Ответов"]).sort_values("Ответов")
    st.caption("График: число содержательных ответов по разделам.")
    fig = px.bar(counts_df, x="Ответов", y="Раздел", orientation="h", color="Ответов",
                 color_continuous_scale="Blues", title="Ответы по разделам")
    fig.update_layout(height=320, coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### Просмотр ответов")
    c1, c2 = st.columns([2, 3])
    sections = c1.multiselect("Раздел анкеты", list(available.values()), default=list(available.values()), key="open_sections")
    search = c2.text_input("Поиск по тексту", placeholder="Ключевое слово…", key="open_search")

    rows = []
    for col, lbl in available.items():
        if lbl not in sections:
            continue
        sub = df.loc[has_text[col], ["program_display", "school", "year", col]].copy()
        sub[col] = sub[col].astype(str).str.strip()
        sub = sub.rename(columns={"program_display": "Программа", "school": "Школа", "year": "Курс", col: "Ответ"})
        sub.insert(0, "Раздел", lbl)
        rows.append(sub)
    if not rows:
        st.info("По выбранным разделам ответов нет.")
        return
    out = pd.concat(rows, ignore_index=True)
    if search.strip():
        out = out[out["Ответ"].str.contains(search.strip(), case=False, na=False, regex=False)]
    st.caption(f"Показано ответов: **{len(out)}**.")
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
    st.subheader("Кодификатор")
    st.info("Что здесь: переменные набора данных осенней анкеты, формулировки вопросов, варианты ответов и правила обработки.")
    text = load_codebook(DOCS_DIR / "codebook_autumn.md")
    if not text:
        st.warning("Кодификатор не найден: запустите `python process_autumn.py`.")
        return
    st.markdown(text)


# ── Entry point ──────────────────────────────────────────────────────────────

def render_autumn() -> None:
    files = autumn_files()
    if not files:
        st.title("SFQ 2026-27: осенняя анкета")
        st.error("Не найден файл `data/processed/autumn_<ГГГГ-ГГ>.csv`. Запустите `python process_autumn.py`.")
        return
    years = list(files)[::-1]
    current = st.sidebar.selectbox("Учебный год", years, key="autumn_year") if len(years) > 1 else years[0]
    st.title(f"SFQ {current}: осенняя анкета")
    st.caption(f"Источник данных: `{files[current].name}`")

    df = load_autumn(str(files[current]))
    meta = load_meta(str(files[current].with_name(f"autumn_{current}_meta.json")))

    filters = build_filters(df, key_prefix=f"autumn_{current}")
    min_n = st.sidebar.slider(
        "Минимальный размер группы",
        min_value=1,
        max_value=20,
        value=5,
        help="Группы, в которых меньше ответов, не показываются в разбивках: доли по двум-трём ответам неустойчивы "
        "и могут указывать на конкретных студентов.",
        key="autumn_min_n",
    )
    dff = apply_filters(df, filters)
    if dff.empty:
        st.error("После применения фильтров данных не осталось.")
        return
    dff = add_program_display(dff, short_labels=filters.short_program_labels)
    cont_slice = filter_contingent(load_contingent(), filters)

    tabs = st.tabs(
        ["Обзор", "Первый курс", "Цели и трудности", "Нагрузка и быт", "Удержание и поддержка",
         "Внеучебная жизнь", "Сравнение групп", "Сравнение по годам", "Открытые ответы", "Кодификатор"]
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
