"""Winter / spring SFQ dashboard (same questionnaire as SFQ 2025-26).

Ported from the 2025-26 dashboard: «1 семестр» → «Зима», «2 семестр» → «Весна».
Data: data/processed/sfq_<wave>.csv and sfq_<wave>_teachers.csv (see process_sfq.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import scikit_posthocs as sp
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.formula.api import ols

from dashboard.common import (
    ACCENT,
    DOCS_DIR,
    MUTED,
    PROCESSED_DIR,
    ROUND_DECIMALS,
    FilterState,
    add_program_display,
    apply_filters,
    bootstrap_mean_ci,
    build_filters,
    filter_contingent,
    fmt,
    fmt_p,
    load_codebook,
    load_contingent,
    margin_of_error_pct,
    normalize_unicode_columns,
    num_cols,
    render_interp,
    render_response_rate,
    round_df,
    safe_numeric,
    show_table,
)

WAVES = {
    "winter": {
        "label": "Зима",
        "general": PROCESSED_DIR / "sfq_winter.csv",
        "teachers": PROCESSED_DIR / "sfq_winter_teachers.csv",
        "codebook": DOCS_DIR / "codebook_sfq.md",
    },
    "spring": {
        "label": "Весна",
        "general": PROCESSED_DIR / "sfq_spring.csv",
        "teachers": PROCESSED_DIR / "sfq_spring_teachers.csv",
        "codebook": DOCS_DIR / "codebook_sfq.md",
    },
}


def wave_available(wave: str) -> bool:
    return Path(WAVES[wave]["general"]).exists()


SCORE_IMP_PAIRS = [
    ("fac_support_score", "fac_support_imp"),
    ("fac_clarity_score", "fac_clarity_imp"),
    ("cur_timely_score", "cur_timely_imp"),
    ("cur_help_score", "cur_help_imp"),
    ("prog_clarity_score", "prog_clarity_imp"),
    ("prog_deadlines_score", "prog_deadlines_imp"),
    ("prog_relevance_score", "prog_relevance_imp"),
    ("prog_workload_score", "prog_workload_imp"),
    ("coord_respect_score", "coord_respect_imp"),
    ("coord_results_score", "coord_results_imp"),
    ("coord_timely_score", "coord_timely_imp"),
    ("coord_help_score", "coord_help_imp"),
]


CSI_BLOCKS = {
    "Куратор": (["cur_timely_score", "cur_help_score"], ["cur_timely_imp", "cur_help_imp"]),
    "Преподавательский состав": (["fac_support_score", "fac_clarity_score"], ["fac_support_imp", "fac_clarity_imp"]),
    "Программа": (
        ["prog_clarity_score", "prog_deadlines_score", "prog_relevance_score", "prog_workload_score"],
        ["prog_clarity_imp", "prog_deadlines_imp", "prog_relevance_imp", "prog_workload_imp"],
    ),
    "Учебный отдел": (
        ["coord_respect_score", "coord_results_score", "coord_timely_score", "coord_help_score"],
        ["coord_respect_imp", "coord_results_imp", "coord_timely_imp", "coord_help_imp"],
    ),
}


BLOCKS = {
    "faculty": ["fac_support_score", "fac_clarity_score"],
    "curator": ["cur_timely_score", "cur_help_score"],
    "program": ["prog_clarity_score", "prog_deadlines_score", "prog_relevance_score", "prog_workload_score"],
    "coordinator": ["coord_respect_score", "coord_results_score", "coord_timely_score", "coord_help_score"],
    "assessment": ["assess_criteria_timely", "assess_order_clear", "assess_consistent"],
    "infrastructure": [
        "infra_library",
        "infra_wellbeing",
        "infra_food",
        "infra_software",
        "infra_equipment",
        "infra_classrooms",
        "infra_workshops",
    ],
}


KEY_METRICS = [
    "satisf_overall",
    "satisf_teachers",
    "expect_match",
    "assess_criteria_timely",
    "assess_order_clear",
    "assess_consistent",
    "nps",
]


TECHNICAL_NUMERIC_EXCLUDE = {"resp_id", "duration_sec"}


COL_LABELS = {
    "program": "Программа",
    "year": "Курс",
    "school": "Школа",
    "nps": "NPS (0–10)",
    "satisf_overall": "Удовлетворённость образовательным процессом",
    "satisf_teachers": "Удовлетворённость преподавателями",
    "expect_match": "Совпадение ожиданий с опытом",
    "assess_criteria_timely": "Критерии оценивания представлены вовремя",
    "assess_order_clear": "Порядок сдачи и оценивания понятен",
    "assess_consistent": "Оценивание соответствует критериям",
    "fac_support_score": "Поддержка преподавательской команды (оценка)",
    "fac_clarity_score": "Понятность объяснений преподавателей (оценка)",
    "cur_timely_score": "Своевременность информирования куратором (оценка)",
    "cur_help_score": "Готовность куратора помочь (оценка)",
    "prog_clarity_score": "Понятность содержания дисциплин (оценка)",
    "prog_deadlines_score": "Реалистичность сроков сдачи (оценка)",
    "prog_relevance_score": "Связь дисциплин с профессией (оценка)",
    "prog_workload_score": "Оптимальность учебной нагрузки (оценка)",
    "coord_respect_score": "Уважительное отношение координатора (оценка)",
    "coord_results_score": "Результативность обращений к координатору (оценка)",
    "coord_timely_score": "Своевременность информирования координатором (оценка)",
    "coord_help_score": "Готовность координатора помочь (оценка)",
    "fac_support_imp": "Поддержка преподавательской команды (важность)",
    "fac_clarity_imp": "Понятность объяснений преподавателей (важность)",
    "cur_timely_imp": "Своевременность информирования куратором (важность)",
    "cur_help_imp": "Готовность куратора помочь (важность)",
    "prog_clarity_imp": "Понятность содержания дисциплин (важность)",
    "prog_deadlines_imp": "Реалистичность сроков сдачи (важность)",
    "prog_relevance_imp": "Связь дисциплин с профессией (важность)",
    "prog_workload_imp": "Оптимальность учебной нагрузки (важность)",
    "coord_respect_imp": "Уважительное отношение координатора (важность)",
    "coord_results_imp": "Результативность обращений к координатору (важность)",
    "coord_timely_imp": "Своевременность информирования координатором (важность)",
    "coord_help_imp": "Готовность координатора помочь (важность)",
    "infra_library": "Библиотека",
    "infra_wellbeing": "Сервис Wellbeing",
    "infra_food": "Еда на кампусе",
    "infra_software": "Программное обеспечение в аудиториях",
    "infra_equipment": "Оборудование в аудиториях",
    "infra_classrooms": "Комфорт аудиторий",
    "infra_workshops": "Комфорт мастерских и ресурсных центров",
    "gen_ed_critical_thinking": "Общеобразовательные: критическое мышление",
    "gen_ed_history": "Общеобразовательные: история России",
    "gen_ed_foreign_lang": "Общеобразовательные: иностранный язык",
    "gen_ed_safety": "Общеобразовательные: безопасность жизнедеятельности",
    "gen_ed_statehood": "Общеобразовательные: основы российской государственности",
    "hum_critical_thinking": "Гуманитарные: критическое мышление",
    "hum_history": "Гуманитарные: история России",
    "hum_statehood": "Гуманитарные: основы российской государственности",
    "hum_foreign_lang": "Гуманитарные: иностранный язык",
    "hum_philosophy": "Гуманитарные: философия",
    "hum_communication": "Гуманитарные: теория и практика коммуникации",
    "goal_achieved": "Достижение целей обучения на программе (шкала 1–7)",
    "navigation_ease": "Лёгкость ориентирования в учебном процессе (шкала 1–7)",
    "class_comfort": "Комфорт на занятиях (шкала 1–7)",
    "prev_sem_relevance": "Связь с предыдущим семестром",
    "skill_confidence": "Уверенность в применении навыков",
    "postgrad_masters": "План: магистратура или дополнительное образование",
    "postgrad_same_field": "План: работа в выбранной сфере",
    "postgrad_other_field": "План: работа в другой сфере",
    "postgrad_other": "План: другой вариант",
    "faculty_mean": "Блок «Преподавательская команда» (среднее)",
    "curator_mean": "Блок «Куратор» (среднее)",
    "program_mean": "Блок «Программа» (среднее)",
    "coordinator_mean": "Блок «Координатор» (среднее)",
    "assessment_mean": "Блок «Оценивание» (среднее)",
    "infrastructure_mean": "Блок «Инфраструктура» (среднее)",
}


def label(col: str) -> str:
    return COL_LABELS.get(col, col)


def get_descriptive_metrics(df: pd.DataFrame) -> list[str]:
    """Расширенный список показателей из кодификатора + средние по блокам."""
    preferred_order = [
        "satisf_overall",
        "satisf_teachers",
        "expect_match",
        "fac_support_score",
        "fac_clarity_score",
        "cur_timely_score",
        "cur_help_score",
        "prog_clarity_score",
        "prog_deadlines_score",
        "prog_relevance_score",
        "prog_workload_score",
        "coord_respect_score",
        "coord_results_score",
        "coord_timely_score",
        "coord_help_score",
        "assess_criteria_timely",
        "assess_order_clear",
        "assess_consistent",
        "infra_library",
        "infra_wellbeing",
        "infra_food",
        "infra_software",
        "infra_equipment",
        "infra_classrooms",
        "infra_workshops",
        "gen_ed_critical_thinking",
        "gen_ed_history",
        "gen_ed_foreign_lang",
        "gen_ed_safety",
        "gen_ed_statehood",
        "hum_critical_thinking",
        "hum_history",
        "hum_statehood",
        "hum_foreign_lang",
        "hum_philosophy",
        "hum_communication",
        "goal_achieved",
        "navigation_ease",
        "class_comfort",
        "prev_sem_relevance",
        "skill_confidence",
        "postgrad_masters",
        "postgrad_same_field",
        "postgrad_other_field",
        "postgrad_other",
        "faculty_mean",
        "curator_mean",
        "program_mean",
        "coordinator_mean",
        "assessment_mean",
        "infrastructure_mean",
        "nps",
    ]
    numeric = [
        c
        for c in num_cols(df)
        if c not in TECHNICAL_NUMERIC_EXCLUDE
        and not c.endswith("_comment")
        and c != "date"
    ]
    chosen = [c for c in preferred_order if c in numeric]
    others = sorted([c for c in numeric if c not in chosen], key=label)
    return chosen + others


def render_formula_block(kind: str) -> None:
    if kind == "anova":
        with st.expander("Формулы и обозначения (дисперсионный анализ и критерий Краскела — Уоллиса)", expanded=False):
            st.latex(r"\eta^2=\frac{SS_B}{SS_T}")
            st.caption(
                "Дисперсионный анализ (ANOVA) проверяет, различаются ли средние значения по программам, "
                "а критерий Краскела — Уоллиса — различаются ли распределения рангов (непараметрический метод). "
                "SS_B — межгрупповая сумма квадратов, SS_T — общая сумма квадратов; "
                "η² — доля общей дисперсии, которая объясняется различиями между программами."
            )
        return

    if kind == "spearman":
        with st.expander("Формулы и обозначения (корреляция Спирмена)", expanded=False):
            st.latex(r"\rho_s=r\big(R(X),\,R(Y)\big)")
            st.caption("R — ранги значений, r — коэффициент корреляции Пирсона. Коэффициент рассчитывается по рангам, поэтому лучше подходит для порядковых шкал и распределений, далёких от нормального.")
        return

    if kind == "nps":
        with st.expander("Формулы и обозначения (NPS)", expanded=False):
            st.latex(r"\mathrm{NPS}=\%P-\%D")
            st.caption("%P — доля промоутеров (оценки 9–10), %D — доля критиков (оценки 0–6). Нейтралы (оценки 7–8) учитываются в общем числе ответов, но не входят в формулу.")
        return

    if kind == "csi":
        with st.expander("Формулы и обозначения (CSI)", expanded=False):
            st.latex(r"\mathrm{CSI}_{i,b} = \frac{\bar{S}_{i,b}\cdot\bar{I}_{i,b}}{16}\cdot 100")
            st.latex(r"\bar{S}_{i,b}=\frac{1}{K_b}\sum_{k=1}^{K_b} S_{i,b,k}, \quad \bar{I}_{i,b}=\frac{1}{K_b}\sum_{k=1}^{K_b} I_{i,b,k}")
            st.latex(r"\overline{\mathrm{CSI}}_i = \frac{1}{B_i}\sum_{b=1}^{B_i}\mathrm{CSI}_{i,b}")
            st.markdown("Обозначения индексов:")
            st.markdown("- `i` — респондент.")
            st.markdown("- `b` — блок (`Куратор`, `Преподавательский состав`, `Программа`, `Учебный отдел`).")
            st.markdown("- `k` — отдельный критерий внутри блока.")
            st.markdown("- `S` — оценка удовлетворённости, `I` — оценка важности критерия.")
            st.markdown("- `B_i` — число блоков, на которые ответил респондент; общий CSI — среднее по этим блокам.")
            st.caption(
                "Шкалы удовлетворённости и важности — от 1 до 4, поэтому нормирующий множитель равен 16 (4 × 4). "
                "Если на часть пунктов блока ответов нет, берётся среднее по заполненным пунктам."
            )
        return

    if kind == "teacher_bayes":
        with st.expander("Формулы и обозначения (сглаженное среднее преподавателя)", expanded=False):
            st.latex(r"\hat{\mu}_t=\frac{n_t\cdot\bar{x}_t + m\cdot\mu_0}{n_t+m}")
            st.markdown("Обозначения:")
            st.markdown("- `t` — преподаватель.")
            st.markdown("- `n_t` — число оценок преподавателя.")
            st.markdown("- `\\bar{x}_t` — простое среднее оценок преподавателя.")
            st.markdown("- `\\mu_0` — общее среднее по текущему срезу (с учётом фильтров).")
            st.markdown("- `m` — сила сглаживания: чем больше `m`, тем сильнее оценка смещается к `\\mu_0`.")
            st.caption(
                "Это эмпирическое байесовское сглаживание: оно уменьшает влияние случайных крайних оценок, когда у преподавателя мало ответов."
            )
        return


def add_block_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for block, cols in BLOCKS.items():
        available = [c for c in cols if c in out.columns]
        if available:
            out[f"{block}_mean"] = out[available].mean(axis=1)
    return out


def render_overview(df: pd.DataFrame, cont_slice: pd.DataFrame | None = None, combined_mode: bool = False) -> None:
    st.subheader("Обзор")
    st.info("Что здесь: ключевые показатели по выборке.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ответов", fmt(len(df), 0), help="Число анкет после применения фильтров.")
    c2.metric("Программ", f"{df['program'].nunique() if 'program' in df.columns else 0}", help="Число программ в текущем срезе.")
    c3.metric("Школ", f"{df['school'].nunique() if 'school' in df.columns else 0}", help="Число школ в текущем срезе.")
    c4.metric("Медиана NPS", fmt(df['nps'].median()) if "nps" in df.columns else "н/д", help="Медиана оценок готовности рекомендовать школу (шкала от 0 до 10).")
    if "nps" in df.columns:
        c5, c6 = st.columns(2)
        c5.metric("Минимум NPS", fmt(df['nps'].min()), help="Минимальное значение NPS в текущем срезе.")
        c6.metric("Максимум NPS", fmt(df['nps'].max()), help="Максимальное значение NPS в текущем срезе.")
    render_interp(
        "Как интерпретировать обзор",
        [
            "Если по программе мало ответов, то сравнения по ней менее надёжны.",
            "Если медиана NPS заметно ниже 7, то в срезе преобладают нейтральные и негативные оценки.",
            "Если минимум очень низкий, а максимум высокий, то мнения студентов сильно расходятся.",
        ],
    )

    render_response_rate(
        df,
        cont_slice,
        combined_note=(
            "В режиме «Зима и весна» отклик не рассчитывается: ответы двух волн нельзя соотнести "
            "с одним контингентом, потому что студенты могли отвечать дважды. "
            "Отклик по каждой волне — на вкладке **«Динамика»**."
        )
        if combined_mode
        else None,
    )

    if "program" in df.columns:
        st.caption("График: размер выборки по программам.")
        counts = (
            df.groupby("program", dropna=False)["program_display"]
            .first()
            .to_frame("program_display")
            .join(df["program"].value_counts().rename("n"), how="left")
            .reset_index()
            .rename(columns={"index": "program"})
        )
        counts = counts.sort_values("n", ascending=True)
        fig = px.bar(
            counts,
            x="n",
            y="program_display",
            orientation="h",
            title="Размер выборки по программам",
            color="n",
            color_continuous_scale="Teal",
            labels={"program_display": "Программа", "n": "Число ответов"},
            hover_data={"program": True, "program_display": False, "n": ":.0f"},
        )
        fig.update_layout(height=560, yaxis_title="")
        st.plotly_chart(fig, width='stretch')

    st.markdown("### Навигатор по вкладкам")
    legend_df = pd.DataFrame(
        [
            {"Вкладка": "Динамика", "Что внутри": "Сравнение зимы и весны по сопоставимым показателям, отклик по волнам."},
            {"Вкладка": "Описательные", "Что внутри": "Средние, медианы, минимумы и максимумы, доверительные интервалы, сравнение по программам."},
            {"Вкладка": "Сравнение программ", "Что внутри": "Дисперсионный анализ, критерии Краскела — Уоллиса и Левене, попарные сравнения Данна по выбранному показателю."},
            {"Вкладка": "Корреляции", "Что внутри": "Корреляции Спирмена и самые сильные положительные и отрицательные связи."},
            {"Вкладка": "Матрица приоритетов", "Что внутри": "Сопоставление важности и оценки критериев, зоны первоочередных улучшений."},
            {"Вкладка": "NPS", "Что внутри": "Промоутеры, нейтралы и критики, состав групп и NPS по программам."},
            {"Вкладка": "CSI", "Что внутри": "Индекс удовлетворённости по блокам и общий CSI, формулы и интерпретация."},
            {"Вкладка": "Преподаватели", "Что внутри": "Оценки по школам, программам и курсам, рейтинг отдельных преподавателей со сглаживанием."},
            {"Вкладка": "Комментарии", "Что внутри": "Все ответы на открытые вопросы с фильтрами по разделу, программе, школе, курсу и поиском по слову."},
            {"Вкладка": "Кодификатор", "Что внутри": "Описание переменных, шкал и структуры данных."},
        ]
    )
    show_table(legend_df, width="stretch", hide_index=True)


def render_descriptives(df: pd.DataFrame) -> None:
    st.subheader("Описательная статистика")
    st.info("Что здесь: средние, медианы, разброс и доверительные интервалы (бутстрэп) для выбранных показателей.")

    available_metrics = get_descriptive_metrics(df)
    if not available_metrics:
        st.warning("Ключевые показатели не найдены.")
        return

    default_metrics = [c for c in KEY_METRICS if c in available_metrics]
    selected = st.multiselect(
        "Показатели для анализа",
        available_metrics,
        default=default_metrics if default_metrics else available_metrics[:10],
        format_func=label,
        help="Доступны основные и дополнительные показатели текущего среза.",
    )
    if not selected:
        st.info("Выберите хотя бы один показатель.")
        return
    render_interp(
        "Как интерпретировать описательную статистику",
        [
            "Если среднее и медиана заметно различаются, то распределение асимметрично.",
            "Если 95%-й доверительный интервал (ДИ) узкий, то оценка среднего устойчива; если широкий, то неопределённость выше.",
            "Если минимум и максимум сильно удалены от квартилей, то возможны выбросы или неоднородные подгруппы.",
            "Если доверительные интервалы программ почти не перекрываются, то различия между программами, вероятно, существенны.",
        ],
    )

    rows = []
    for col in selected:
        mean, lo, hi, n = bootstrap_mean_ci(df[col])
        vals = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "Показатель": label(col),
                "n": n,
                "Среднее": mean,
                "Медиана": vals.median(),
                "Минимум": vals.min(),
                "Максимум": vals.max(),
                "Ст. отклонение": vals.std(),
                "Квартиль 1": vals.quantile(0.25),
                "Квартиль 3": vals.quantile(0.75),
                "ДИ 95%, нижняя граница": lo,
                "ДИ 95%, верхняя граница": hi,
            }
        )
    stat_df = pd.DataFrame(rows).sort_values("Среднее", ascending=False)
    stat_df = round_df(stat_df)
    st.caption("Таблица: описательная статистика по выбранным показателям.")
    show_table(stat_df, width='stretch')

    st.caption("График: средние значения с 95%-ми доверительными интервалами (бутстрэп).")
    fig = px.scatter(
        stat_df,
        x="Среднее",
        y="Показатель",
        error_x=stat_df["ДИ 95%, верхняя граница"] - stat_df["Среднее"],
        error_x_minus=stat_df["Среднее"] - stat_df["ДИ 95%, нижняя граница"],
        size="n",
        color="Ст. отклонение",
        color_continuous_scale="RdYlBu_r",
        title="Средние значения и доверительные интервалы",
    )
    fig.update_layout(height=460, yaxis_title="")
    st.plotly_chart(fig, width='stretch')

    metric_for_group = st.selectbox(
        "Показатель по программам",
        selected,
        index=0,
        format_func=label,
        help="Для выбранного показателя приводятся средние по программам с доверительными интервалами.",
    )
    grp_rows = []
    for prog, g in df.groupby("program"):
        mean, lo, hi, n = bootstrap_mean_ci(g[metric_for_group])
        vals = pd.to_numeric(g[metric_for_group], errors="coerce")
        grp_rows.append(
            {
                "program": prog,
                "program_display": g["program_display"].iloc[0] if "program_display" in g.columns else prog,
                "n": n,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "min": vals.min(),
                "max": vals.max(),
            }
        )
    grp = round_df(pd.DataFrame(grp_rows).sort_values("mean"))

    st.caption("Таблица: сводка по программам (с минимумом и максимумом).")
    show_table(
        grp.rename(
            columns={
                "program": "Программа",
                "program_display": "Краткое название",
                "n": "n",
                "mean": "Среднее",
                "ci_low": "ДИ, нижняя граница",
                "ci_high": "ДИ, верхняя граница",
                "min": "Минимум",
                "max": "Максимум",
            }
        ),
        width='stretch',
    )

    st.caption("График: средние по программам с 95%-ми доверительными интервалами (бутстрэп).")
    fig = px.scatter(
        grp,
        x="mean",
        y="program_display",
        error_x=grp["ci_high"] - grp["mean"],
        error_x_minus=grp["mean"] - grp["ci_low"],
        size="n",
        color="mean",
        color_continuous_scale="Viridis",
        title=f"{label(metric_for_group)}: среднее по программам",
        labels={"program_display": "Программа", "mean": "Среднее", "n": "Число ответов"},
        hover_data={"program": True, "program_display": False, "mean": ":.2f", "n": True},
    )
    fig.update_layout(height=620, yaxis_title="")
    st.plotly_chart(fig, width='stretch')

    st.caption("График: распределение значений выбранного показателя по программам.")
    fig = px.violin(
        df,
        y="program_display",
        x=metric_for_group,
        orientation="h",
        box=True,
        points="all",
        title=f"{label(metric_for_group)}: распределение по программам",
        labels={"program_display": "Программа", metric_for_group: label(metric_for_group)},
        hover_data={"program": True, "program_display": False},
    )
    fig.update_layout(height=620, yaxis_title="")
    st.plotly_chart(fig, width='stretch')


def render_program_comparison(df: pd.DataFrame) -> None:
    st.subheader("Сравнение программ")
    st.info(
        "Что здесь: статистическое сравнение программ по выбранному показателю — дисперсионный анализ (ANOVA), "
        "критерии Краскела — Уоллиса и Левене, попарные сравнения по критерию Данна с поправкой Холма."
    )
    render_formula_block("anova")

    if "program" not in df.columns:
        st.warning("Для этой вкладки нужен столбец `program`.")
        return

    candidates = [c for c in get_descriptive_metrics(df) if df[c].dropna().nunique() >= 3]
    if not candidates:
        st.warning("Нет показателей с достаточным разбросом значений для сравнения программ.")
        return

    selected_metric = st.selectbox(
        "Показатель для сравнения программ",
        options=candidates,
        index=candidates.index("nps") if "nps" in candidates else 0,
        format_func=label,
        help="Выберите показатель, по которому нужно сравнить программы.",
    )
    render_interp(
        "Как интерпретировать сравнение программ",
        [
            "Если p-значение дисперсионного анализа меньше 0,05, то средние по программам различаются статистически значимо.",
            "Если p-значение критерия Краскела — Уоллиса меньше 0,05, то различия подтверждаются и непараметрическим методом.",
            "Если p-значение критерия Левене меньше 0,05, то дисперсии в группах неоднородны и в выводах лучше опираться на критерии Краскела — Уоллиса и Данна.",
            "Если η² около 0,01, 0,06 или 0,14 и выше, то эффект обычно считают малым, средним или крупным соответственно.",
            "Если в попарных сравнениях Данна p-значение для пары программ меньше 0,05, то именно эти программы различаются значимо.",
        ],
    )

    min_n = st.slider(
        "Минимальный размер группы для тестов",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        help="Программы, в которых меньше ответов, чем задано, исключаются из статистических тестов.",
    )
    vc = df["program"].value_counts()
    valid_programs = vc[vc >= min_n].index
    keep_cols = ["program", "program_display", selected_metric]
    keep_cols = [c for c in keep_cols if c in df.columns]
    d = df[df["program"].isin(valid_programs)][keep_cols].dropna()
    st.caption(f"В анализе: программ — {d['program'].nunique()}, ответов — {len(d)}.")

    if d["program"].nunique() < 2:
        st.info("Недостаточно программ для проведения тестов.")
        return

    group_vals = [g[selected_metric].values for _, g in d.groupby("program")]
    _, lev_p = stats.levene(*group_vals, center="median")
    anova_model = ols(f"{selected_metric} ~ C(program)", data=d).fit()
    anova_tbl = sm.stats.anova_lm(anova_model, typ=2)
    ss_between = anova_tbl.loc["C(program)", "sum_sq"]
    eta2 = ss_between / anova_tbl["sum_sq"].sum()
    _, kw_p = stats.kruskal(*group_vals)
    dunn = sp.posthoc_dunn(d, val_col=selected_metric, group_col="program", p_adjust="holm")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("p-значение (ANOVA)", fmt_p(anova_tbl.loc['C(program)', 'PR(>F)']), help=f"Проверка различий средних по показателю «{label(selected_metric)}».")
    k2.metric("η²", fmt(eta2), help="Доля дисперсии, объяснённая различиями между программами.")
    k3.metric("p-значение (Краскел — Уоллис)", fmt_p(kw_p), help="Непараметрическая проверка различий между программами.")
    k4.metric("p-значение (Левене)", fmt_p(lev_p), help="Проверка равенства дисперсий в группах.")
    k5.metric("Минимум", fmt(d[selected_metric].min()), help="Минимальное значение показателя в анализируемом срезе.")
    k6.metric("Максимум", fmt(d[selected_metric].max()), help="Максимальное значение показателя в анализируемом срезе.")

    left, right = st.columns(2)
    with left:
        st.caption("Таблица: результаты дисперсионного анализа.")
        show_table(
            anova_tbl.rename(
                columns={"sum_sq": "Сумма квадратов", "df": "Степени свободы", "F": "F", "PR(>F)": "p-значение"},
                index={"C(program)": "Программа", "Residual": "Остаток"},
            ),
            decimals=3,
        )
    with right:
        st.caption("Таблица: попарные сравнения по критерию Данна, p-значения с поправкой Холма.")
        show_table(round_df(dunn), width='stretch')

    st.caption("График: распределение выбранного показателя по программам.")
    fig = px.box(
        d,
        y="program_display",
        x=selected_metric,
        orientation="h",
        points="all",
        color="program_display",
        title=f"{label(selected_metric)} по программам",
        labels={"program_display": "Программа", selected_metric: label(selected_metric)},
        hover_data={"program": True, "program_display": False},
    )
    fig.update_layout(showlegend=False, height=620, yaxis_title="")
    st.plotly_chart(fig, width='stretch')


def render_correlations(df: pd.DataFrame) -> None:
    st.subheader("Корреляции")
    st.info("Что здесь: матрица корреляций Спирмена и самые сильные положительные и отрицательные связи с выбранным показателем.")
    render_formula_block("spearman")

    numeric = num_cols(df)
    default = [c for c in ["nps", "satisf_overall", "expect_match", "program_mean", "coordinator_mean", "infrastructure_mean"] if c in numeric]
    selected = st.multiselect(
        "Показатели для матрицы корреляций",
        numeric,
        default=default if default else numeric[:12],
        format_func=label,
        help="Выберите не менее двух числовых показателей.",
    )
    if len(selected) < 2:
        st.info("Выберите не менее двух показателей.")
        return
    render_interp(
        "Как интерпретировать корреляции",
        [
            "Если коэффициент Спирмена больше 0, то при росте одного показателя другой обычно тоже растёт.",
            "Если коэффициент Спирмена меньше 0, то при росте одного показателя другой обычно снижается.",
            "Если модуль коэффициента около 0,1, 0,3 или 0,5 и выше, то связь обычно считают слабой, умеренной или сильной соответственно.",
            "Даже сильная связь не доказывает причинно-следственную зависимость: учитывайте контекст и возможные скрытые факторы.",
        ],
    )

    corr = df[selected].corr(method="spearman").round(ROUND_DECIMALS)
    corr_display = corr.copy()
    corr_display.index = [label(i) for i in corr_display.index]
    corr_display.columns = [label(i) for i in corr_display.columns]

    st.caption("График: матрица корреляций Спирмена.")
    fig = px.imshow(
        corr_display,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Матрица корреляций Спирмена",
    )
    fig.update_layout(height=680)
    st.plotly_chart(fig, width='stretch')

    target = st.selectbox(
        "Целевой показатель",
        selected,
        index=selected.index("nps") if "nps" in selected else 0,
        format_func=label,
        help="Приводятся коэффициенты корреляции Спирмена остальных показателей с выбранным.",
    )
    with_target = corr[target].drop(target).sort_values(ascending=False).rename("Коэффициент Спирмена").to_frame()
    with_target.index = [label(i) for i in with_target.index]
    st.caption("Таблица: коэффициенты корреляции с целевым показателем.")
    show_table(round_df(with_target), width='stretch')

    top_pos = with_target.head(5).reset_index().rename(columns={"index": "Показатель"})
    top_neg = with_target.tail(5).reset_index().rename(columns={"index": "Показатель"})
    cols = pd.concat([top_pos, top_neg], ignore_index=True).drop_duplicates(subset="Показатель")
    st.caption("График: самые сильные положительные и отрицательные связи.")
    fig = px.bar(
        cols,
        x="Коэффициент Спирмена",
        y="Показатель",
        orientation="h",
        color="Коэффициент Спирмена",
        color_continuous_scale="RdBu",
        title=f"Самые сильные связи с показателем «{label(target)}»",
        labels={"Показатель": ""},
    )
    fig.update_layout(height=420, yaxis_title="")
    st.plotly_chart(fig, width='stretch')


def render_priority_matrix(df: pd.DataFrame) -> None:
    st.subheader("Матрица приоритетов")
    st.info(
        "Что здесь: соотношение важности и оценки по каждому критерию. "
        "Чем больше разрыв между важностью и оценкой, тем выше приоритет улучшений."
    )
    rows = []
    for score_col, imp_col in SCORE_IMP_PAIRS:
        if score_col not in df.columns or imp_col not in df.columns:
            continue
        score = pd.to_numeric(df[score_col], errors="coerce")
        imp = pd.to_numeric(df[imp_col], errors="coerce")
        valid = score.notna() & imp.notna()
        if valid.sum() == 0:
            continue
        rows.append(
            {
                "Критерий": label(score_col).replace(" (оценка)", ""),
                "Средняя оценка": score[valid].mean(),
                "Средняя важность": imp[valid].mean(),
                "Минимум оценки": score[valid].min(),
                "Максимум оценки": score[valid].max(),
                "Минимум важности": imp[valid].min(),
                "Максимум важности": imp[valid].max(),
                "Разрыв (важность − оценка)": imp[valid].mean() - score[valid].mean(),
                "n": int(valid.sum()),
            }
        )

    if not rows:
        st.warning("В текущем срезе нет критериев, по которым есть и оценка, и важность.")
        return
    render_interp(
        "Как интерпретировать матрицу приоритетов",
        [
            "Если важность выше средней, а оценка ниже средней (верхний левый квадрант), то это зона первоочередных улучшений.",
            "Если и важность, и оценка высокие (верхний правый квадрант), то это сильные стороны, которые стоит сохранять.",
            "Если разрыв между важностью и оценкой большой и положительный, то ожидания студентов по этому критерию не оправдываются.",
            "Если разрыв близок к нулю, то ожидания и фактическая оценка совпадают.",
        ],
    )

    m = round_df(pd.DataFrame(rows))
    score_mid = m["Средняя оценка"].mean()
    imp_mid = m["Средняя важность"].mean()

    st.caption("График: матрица «важность — оценка» с делением на квадранты по средним значениям.")
    fig = px.scatter(
        m,
        x="Средняя оценка",
        y="Средняя важность",
        text="Критерий",
        size="n",
        color="Разрыв (важность − оценка)",
        color_continuous_scale="RdYlGn_r",
        hover_data=["Разрыв (важность − оценка)", "n"],
        title="Матрица «важность — оценка»",
    )
    fig.add_vline(score_mid, line_dash="dash", line_color="gray")
    fig.add_hline(imp_mid, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=560)
    st.plotly_chart(fig, width='stretch')

    st.caption("Таблица: приоритеты улучшений (по убыванию разрыва).")
    m = m.sort_values("Разрыв (важность − оценка)", ascending=False)
    show_table(round_df(m), width='stretch')


def render_nps(df: pd.DataFrame) -> None:
    st.subheader("Анализ NPS")
    st.info("Что здесь: структура NPS (промоутеры, нейтралы, критики), состав групп по программам и NPS каждой программы.")
    if "nps" not in df.columns:
        st.warning("Столбец `nps` не найден.")
        return
    render_formula_block("nps")
    render_interp(
        "Как интерпретировать NPS",
        [
            "Если NPS больше 0, то промоутеров больше, чем критиков; если меньше 0 — наоборот.",
            "Если у программы растёт доля критиков, то в первую очередь стоит подробно разобрать отзывы по этой программе.",
            "Если NPS сильно различается между программами, то проблемы и сильные стороны связаны с отдельными программами, а не с университетом в целом.",
            "Если общий NPS высокий, но минимум низкий, то в выборке могут быть отдельные группы недовольных студентов.",
        ],
    )

    d = df.copy()
    d["segment"] = np.where(d["nps"] >= 9, "Промоутеры", np.where(d["nps"] >= 7, "Нейтралы", "Критики"))
    seg = d["segment"].value_counts().rename_axis("segment").reset_index(name="n")
    seg["pct"] = seg["n"] / seg["n"].sum() * 100
    nps_value = float(seg.loc[seg["segment"] == "Промоутеры", "pct"].sum() - seg.loc[seg["segment"] == "Критики", "pct"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("NPS", fmt(nps_value), help="NPS = доля промоутеров − доля критиков (в процентах).")
    c2.metric("Промоутеры, %", fmt(seg.loc[seg['segment']=='Промоутеры', 'pct'].sum()))
    c3.metric("Нейтралы, %", fmt(seg.loc[seg['segment']=='Нейтралы', 'pct'].sum()))
    c4.metric("Критики, %", fmt(seg.loc[seg['segment']=='Критики', 'pct'].sum()))
    c5.metric("Минимум NPS", fmt(d['nps'].min()))
    c6.metric("Максимум NPS", fmt(d['nps'].max()))

    left, right = st.columns(2)
    with left:
        st.caption("График: соотношение групп NPS в целом.")
        fig = px.pie(
            seg,
            names="segment",
            labels={"segment": "Группа", "n": "Число ответов"},
            values="n",
            hole=0.45,
            color="segment",
            color_discrete_map={"Промоутеры": "#2ca02c", "Нейтралы": "#ffbf00", "Критики": "#d62728"},
        )
        fig.update_layout(title="Структура NPS по группам", height=430)
        st.plotly_chart(fig, width='stretch')
    with right:
        if "program" in d.columns:
            st.caption("График: состав групп NPS по программам.")
            by_prog = d.groupby(["program", "program_display", "segment"]).size().rename("n").reset_index()
            tot = d.groupby(["program", "program_display"]).size().rename("total").reset_index()
            by_prog = by_prog.merge(tot, on=["program", "program_display"], how="left")
            by_prog["pct"] = (by_prog["n"] / by_prog["total"] * 100).round(ROUND_DECIMALS)
            fig = px.bar(
                by_prog,
                y="program_display",
                x="pct",
                orientation="h",
                color="segment",
                barmode="stack",
                title="Состав NPS по программам",
                labels={"program_display": "Программа", "pct": "Доля, %", "segment": "Группа"},
                hover_data={"program": True, "program_display": False, "pct": ":.2f"},
            )
            fig.update_layout(height=620, yaxis_title="")
            fig.update_yaxes(tickformat=".2f")
            fig.update_traces(
                hovertemplate="Программа: %{y}<br>Доля: %{x:.1f}%<extra></extra>",
                offsetgroup="0",  # stack the groups instead of placing them side by side
                alignmentgroup="0",
            )
            st.plotly_chart(fig, width='stretch')

    if "program" in d.columns:
        st.caption("График: NPS по программам.")
        nps_by_program = (
            d.groupby(["program", "program_display"])["nps"]
            .agg(
                promoters_pct=lambda s: s.ge(9).mean() * 100,
                detractors_pct=lambda s: s.le(6).mean() * 100,
            )
            .reset_index()
        )
        nps_by_program["nps"] = nps_by_program["promoters_pct"] - nps_by_program["detractors_pct"]
        nps_by_program = nps_by_program[["program", "program_display", "nps"]].round(ROUND_DECIMALS)
        fig = px.bar(
            nps_by_program.sort_values("nps"),
            x="nps",
            y="program_display",
            orientation="h",
            color="nps",
            color_continuous_scale="RdYlGn",
            title="NPS по программам",
            labels={"program_display": "Программа", "nps": "NPS"},
            hover_data={"program": True, "program_display": False, "nps": ":.2f"},
        )
        fig.update_layout(height=620, yaxis_title="")
        st.plotly_chart(fig, width='stretch')


def compute_csi_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSI по принятой методике:
    CSI = (средняя удовлетворённость * средняя важность) / 16 * 100
    где обе шкалы 1-4.
    """
    out = df.copy()
    csi_cols = []
    for block_name, (score_cols, imp_cols) in CSI_BLOCKS.items():
        s_cols = [c for c in score_cols if c in out.columns]
        i_cols = [c for c in imp_cols if c in out.columns]
        if not s_cols or not i_cols:
            continue
        s_mean = out[s_cols].mean(axis=1)
        i_mean = out[i_cols].mean(axis=1)
        csi_col = f"csi_{block_name}"
        out[csi_col] = (s_mean * i_mean / 16.0) * 100.0
        csi_cols.append(csi_col)

    if csi_cols:
        out["csi_Общий"] = out[csi_cols].mean(axis=1)
    return out


def render_csi(df: pd.DataFrame) -> None:
    st.subheader("CSI (индекс удовлетворённости)")
    st.info("Что здесь: CSI по блокам и общий CSI, методика расчёта и интерпретация.")
    render_formula_block("csi")

    render_interp(
        "Как рассчитывается CSI",
        [
            "CSI рассчитывается по блокам «Куратор», «Преподавательский состав», «Программа» и «Учебный отдел».",
            "Общий CSI респондента — среднее из CSI по тем блокам, на которые он ответил.",
            "Сначала CSI рассчитывается для каждого респондента и блока, затем усредняется по выборке.",
        ],
    )
    render_interp(
        "Как интерпретировать CSI",
        [
            "Если CSI близок к 100, то студенты высоко оценивают важные для них критерии.",
            "Если в выбранной школе, на курсе или программе CSI выше, чем в целом, то студенты этой группы довольны больше.",
            "Если блоки сильно различаются, то улучшения стоит начинать с блоков с низким CSI.",
            "Если CSI низкий при высокой важности, то это самые проблемные места.",
        ],
    )

    d = compute_csi_frame(df)
    csi_cols = [c for c in d.columns if c.startswith("csi_")]
    if not csi_cols:
        st.warning("Недостаточно данных для расчёта CSI.")
        return

    pretty_map = {c: c.replace("csi_", "") for c in csi_cols}
    summary_rows = []
    for col in csi_cols:
        vals = pd.to_numeric(d[col], errors="coerce").dropna()
        if vals.empty:
            continue
        summary_rows.append(
            {
                "Блок CSI": pretty_map[col],
                "n": int(vals.shape[0]),
                "Средний CSI": vals.mean(),
                "Медиана CSI": vals.median(),
                "Минимум CSI": vals.min(),
                "Максимум CSI": vals.max(),
                "Ст. отклонение": vals.std(),
            }
        )
    csi_summary = round_df(pd.DataFrame(summary_rows).sort_values("Средний CSI", ascending=False))
    st.caption("Таблица: сводка CSI по блокам.")
    show_table(csi_summary, width="stretch")

    # Прозрачность расчета: показываем средние компоненты формулы по блокам
    comp_rows = []
    for block_name, (score_cols, imp_cols) in CSI_BLOCKS.items():
        s_cols = [c for c in score_cols if c in d.columns]
        i_cols = [c for c in imp_cols if c in d.columns]
        if not s_cols or not i_cols:
            continue
        s_mean_row = d[s_cols].mean(axis=1)
        i_mean_row = d[i_cols].mean(axis=1)
        csi_row = (s_mean_row * i_mean_row / 16.0) * 100.0
        comp_rows.append(
            {
                "Блок": block_name,
                "Средняя удовлетворённость (S̄)": s_mean_row.mean(),
                "Средняя важность (Ī)": i_mean_row.mean(),
                "Средний CSI (по респондентам)": csi_row.mean(),
                "CSI из агрегированных S̄ и Ī": (s_mean_row.mean() * i_mean_row.mean() / 16.0) * 100.0,
            }
        )
    if comp_rows:
        comp_df = round_df(pd.DataFrame(comp_rows))
        st.caption(
            "Таблица: составляющие формулы CSI. "
            "Основной показатель дашборда — «Средний CSI (по респондентам)»."
        )
        show_table(comp_df, width="stretch")

    plot_df = csi_summary.copy()
    fig = px.bar(
        plot_df.sort_values("Средний CSI"),
        x="Средний CSI",
        y="Блок CSI",
        orientation="h",
        color="Средний CSI",
        color_continuous_scale="Tealgrn",
        title="Средний CSI по блокам",
    )
    fig.update_layout(height=420, yaxis_title="")
    fig.update_xaxes(tickformat=".2f")
    st.plotly_chart(fig, width="stretch")

    selected_csi = st.selectbox(
        "Блок CSI для сравнения по программам",
        options=csi_cols,
        format_func=lambda c: pretty_map[c],
        help="Показывает распределение CSI выбранного блока по программам.",
    )
    by_prog = (
        d.groupby(["program", "program_display"], dropna=False)[selected_csi]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "program": "Программа",
                "program_display": "Краткое название",
                "count": "n",
                "mean": "Среднее",
                "median": "Медиана",
                "min": "Минимум",
                "max": "Максимум",
            }
        )
    )
    by_prog = round_df(by_prog.sort_values("Среднее", ascending=False))
    st.caption("Таблица: CSI по программам (выбранный блок).")
    show_table(by_prog, width="stretch")

    fig = px.box(
        d.dropna(subset=[selected_csi]),
        y="program_display",
        x=selected_csi,
        orientation="h",
        points="all",
        title=f"{pretty_map[selected_csi]}: распределение CSI по программам",
        labels={"program_display": "Программа", selected_csi: "CSI"},
        hover_data={"program": True, "program_display": False},
    )
    fig.update_layout(height=620, yaxis_title="")
    fig.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig, width="stretch")


def render_teachers(df_teachers: pd.DataFrame) -> None:
    st.subheader("Преподаватели")
    st.info(
        "Что здесь: оценки преподавателей в разбивке по школам, программам и курсам, а также рейтинг отдельных преподавателей."
    )
    if df_teachers.empty or "rating" not in df_teachers.columns:
        st.warning("Для выбранного среза нет данных о преподавателях.")
        return

    d = df_teachers.dropna(subset=["rating"]).copy()
    if d.empty:
        st.warning("Нет оценок преподавателей в текущем срезе.")
        return

    render_formula_block("teacher_bayes")
    m = st.slider(
        "Сила байесовского сглаживания m",
        min_value=1,
        max_value=50,
        value=8,
        step=1,
        help="Сглаженное среднее рекомендуется, когда у преподавателя мало оценок.",
    )

    global_mean = float(d["rating"].mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Оценок", fmt(len(d), 0))
    c2.metric("Преподавателей", fmt(d['teacher'].nunique(), 0))
    c3.metric("Средняя оценка", fmt(d['rating'].mean()))
    c4.metric("Общее среднее μ0", fmt(global_mean))

    st.markdown("### Оценки по группам")
    dim = st.selectbox(
        "Разбивка",
        ["school", "program", "year"],
        format_func=lambda x: {"school": "Школа", "program": "Программа", "year": "Курс"}[x],
    )

    seg = (
        d.groupby(dim, dropna=False)["rating"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
        .rename(
            columns={
                dim: "Группа",
                "count": "n",
                "mean": "Среднее",
                "median": "Медиана",
                "min": "Минимум",
                "max": "Максимум",
                "std": "Ст. отклонение",
            }
        )
    )
    seg["Сглаженное среднее"] = ((seg["n"] * seg["Среднее"]) + (m * global_mean)) / (seg["n"] + m)
    seg = round_df(seg.sort_values("Сглаженное среднее", ascending=False))
    show_table(seg, width="stretch")

    seg_plot = seg.sort_values("Сглаженное среднее", ascending=True).head(25)
    fig = px.bar(
        seg_plot,
        x="Сглаженное среднее",
        y="Группа",
        orientation="h",
        color="n",
        color_continuous_scale="Blues",
        title="Сглаженная средняя оценка преподавателей по группам",
        labels={"Группа": "", "n": "Число оценок"},
    )
    fig.update_xaxes(tickformat=".2f")
    fig.update_layout(height=560, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Отдельные преподаватели")
    min_n = st.slider("Минимальное число оценок у преподавателя", 1, 50, 5, 1)
    teacher_stats = (
        d.groupby("teacher", dropna=False)["rating"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
        .rename(
            columns={
                "teacher": "Преподаватель",
                "count": "n",
                "mean": "Среднее",
                "median": "Медиана",
                "min": "Минимум",
                "max": "Максимум",
                "std": "Ст. отклонение",
            }
        )
    )
    teacher_stats["Сглаженное среднее"] = ((teacher_stats["n"] * teacher_stats["Среднее"]) + (m * global_mean)) / (
        teacher_stats["n"] + m
    )
    teacher_stats = teacher_stats[teacher_stats["n"] >= min_n]
    if teacher_stats.empty:
        st.info("Нет преподавателей с заданным минимальным числом оценок.")
        return

    teacher_stats = round_df(teacher_stats.sort_values("Сглаженное среднее", ascending=False))
    show_table(teacher_stats, width="stretch")

    top_n_min = 1 if len(teacher_stats) < 5 else 5
    top_n_max = min(40, len(teacher_stats))
    top_n_default = min(20, top_n_max)
    top_n = st.slider("Сколько преподавателей показать на графике", top_n_min, top_n_max, top_n_default)
    top_teachers = teacher_stats.head(top_n).sort_values("Сглаженное среднее")
    fig = px.bar(
        top_teachers,
        x="Сглаженное среднее",
        y="Преподаватель",
        orientation="h",
        color="n",
        color_continuous_scale="Teal",
        title="Преподаватели с наибольшим сглаженным средним",
        labels={"Преподаватель": "", "n": "Число оценок"},
    )
    fig.update_xaxes(tickformat=".2f")
    fig.update_layout(height=640, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    teacher_pick = st.selectbox("Выберите преподавателя для подробного просмотра", teacher_stats["Преподаватель"].tolist())
    one = d[d["teacher"] == teacher_pick].copy()
    if "program_display" not in one.columns:
        one["program_display"] = one["program"]

    by_prog = (
        one.groupby(["program_display", "program"], dropna=False)["rating"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "program_display": "Программа",
                "program": "Полное название",
                "count": "n",
                "mean": "Среднее",
                "median": "Медиана",
                "min": "Минимум",
                "max": "Максимум",
            }
        )
    )
    by_prog = round_df(by_prog.sort_values("Среднее", ascending=False))
    st.caption("Таблица: оценки выбранного преподавателя по программам.")
    show_table(by_prog, width="stretch")


def render_comments(df: pd.DataFrame) -> None:
    _COMMENT_SECTIONS: dict[str, str] = {
        "cur_comment":     "Куратор",
        "prog_comment":    "Программа",
        "coord_comment":   "Учебный отдел",
        "assess_comment":  "Оценивание",
        "fac_comment":     "Преподаватели",
        "seminar_comment": "Лекции и семинары",
        "comment_final":   "Общий комментарий",
    }

    st.subheader("Комментарии")
    st.info(
        "Что здесь: все ответы на открытые вопросы анкеты. "
        "Их можно отфильтровать по разделу, программе, школе, курсу и ключевому слову."
    )
    render_interp(
        "Как читать комментарии",
        [
            "Каждая строка — один ответ одного студента по одному разделу анкеты.",
            "Пустые ответы скрыты автоматически; фильтры по школе, курсу и программе применяются.",
            "Используйте поле поиска, чтобы найти ответы по ключевому слову.",
            "Раздел «Общий комментарий» — последний открытый вопрос анкеты; ответы в нём обычно самые содержательные.",
        ],
    )

    available: dict[str, str] = {col: lbl for col, lbl in _COMMENT_SECTIONS.items() if col in df.columns}
    if not available:
        st.warning("В данных нет столбцов с комментариями.")
        return

    # ── Metric cards: count per section ───────────────────────────────────
    counts: dict[str, int] = {
        lbl: int(df[col].dropna().astype(str).str.strip().ne("").sum())
        for col, lbl in available.items()
    }
    total_comments = sum(counts.values())

    c_total, *section_cols = st.columns(1 + len(available))
    c_total.metric("Всего комментариев", total_comments)
    for col_widget, (lbl, cnt) in zip(section_cols, counts.items()):
        col_widget.metric(lbl, cnt)

    # ── Bar chart ──────────────────────────────────────────────────────────
    if total_comments > 0:
        st.caption("График: число непустых комментариев по разделам анкеты.")
        counts_df = (
            pd.DataFrame(list(counts.items()), columns=["Раздел", "Ответов"])
            .sort_values("Ответов", ascending=True)
        )
        fig = px.bar(
            counts_df,
            x="Ответов",
            y="Раздел",
            orientation="h",
            title="Комментарии по разделам",
            color="Ответов",
            color_continuous_scale="Blues",
            labels={"Ответов": "Число ответов", "Раздел": ""},
        )
        fig.update_layout(height=280, coloraxis_showscale=False, margin=dict(l=0, r=0, t=36, b=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### Просмотр комментариев")

    # ── Controls ───────────────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([2, 3])
    with ctrl1:
        selected_sections = st.multiselect(
            "Раздел анкеты",
            options=list(available.values()),
            default=[lbl for lbl in available.values() if counts[lbl] > 0],
        )
    with ctrl2:
        search_text = st.text_input("Поиск по тексту", placeholder="Ключевое слово…")

    # ── Build long-format table ────────────────────────────────────────────
    prog_col = "program_display" if "program_display" in df.columns else "program"
    meta = [c for c in [prog_col, "school", "year"] if c in df.columns]
    rename_meta = {prog_col: "Программа", "school": "Школа", "year": "Курс"}

    rows = []
    for col, lbl in available.items():
        if lbl not in selected_sections:
            continue
        sub = df[meta + [col]].copy()
        sub = sub[sub[col].notna() & sub[col].astype(str).str.strip().ne("")]
        sub = sub.rename(columns={**rename_meta, col: "Комментарий"})
        sub.insert(0, "Раздел", lbl)
        rows.append(sub)

    if not rows:
        st.info("Нет комментариев по выбранным разделам.")
        return

    comments_df = pd.concat(rows, ignore_index=True)

    if search_text.strip():
        mask = comments_df["Комментарий"].str.contains(search_text.strip(), case=False, na=False)
        comments_df = comments_df[mask]

    st.caption(f"Показано комментариев: **{len(comments_df)}**.")
    show_table(
        comments_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Раздел":      st.column_config.TextColumn("Раздел",      width="small"),
            "Программа":   st.column_config.TextColumn("Программа",   width="medium"),
            "Школа":       st.column_config.TextColumn("Школа",       width="small"),
            "Курс":        st.column_config.TextColumn("Курс",        width="small"),
            "Комментарий": st.column_config.TextColumn("Комментарий", width="large"),
        },
    )


DYNAMICS_METRICS = [
    "nps",
    "satisf_overall",
    "expect_match",
    "assess_criteria_timely",
    "assess_order_clear",
    "assess_consistent",
    "infra_library",
    "infra_wellbeing",
    "infra_food",
    "infra_software",
    "infra_equipment",
    "infra_classrooms",
    "infra_workshops",
    "faculty_mean",
    "curator_mean",
    "program_mean",
    "coordinator_mean",
    "assessment_mean",
    "infrastructure_mean",
    "csi_overall",
]


def _prepare_semester_slice(general_path, filters: FilterState) -> pd.DataFrame:
    raw = load_combined((str(general_path),))
    if raw.empty:
        return raw
    raw = safe_numeric(raw, num_cols(raw))
    sliced = apply_filters(raw, filters)
    prepared = add_program_display(add_block_features(sliced), short_labels=filters.short_program_labels)
    csi = compute_csi_frame(prepared)
    if "csi_Общий" in csi.columns:
        prepared["csi_overall"] = csi["csi_Общий"]
    return prepared


def _metric_mean(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def render_dynamics(filters: FilterState, cont_slice: pd.DataFrame | None) -> None:
    st.subheader("Динамика: зима и весна")
    st.info(
        "Что здесь: сравнение зимней (1-й семестр) и весенней (2-й семестр) анкет по сопоставимым показателям. "
        "Фильтры по программам, курсам и школам применяются; данные обеих волн загружаются "
        "независимо от выбора на боковой панели."
    )
    st.caption(
        "Сравниваются только показатели, измеренные по одинаковой шкале в обеих волнах. "
        "Если в одной из анкет шкала вопроса изменилась, исключите его из списка `DYNAMICS_METRICS`."
    )

    s1 = _prepare_semester_slice(WAVES["winter"]["general"], filters)
    s2 = _prepare_semester_slice(WAVES["spring"]["general"], filters)
    if s1.empty or s2.empty:
        st.warning("Для сравнения нужны данные обеих волн (зимы и весны) в текущем срезе.")
        return

    render_interp(
        "Как интерпретировать динамику",
        [
            "Δ — разность между значением весной и значением зимой.",
            "Если Δ больше нуля, то показатель вырос; если меньше — снизился.",
            "Если Δ небольшая, а ответов мало, то изменение может быть случайным: сверяйтесь с откликом.",
            "NPS здесь — средняя оценка по шкале от 0 до 10, а не классический индекс (доля промоутеров минус доля критиков).",
        ],
    )

    # ── Отклик по волнам ─────────────────────────────────────────────────
    N = float(cont_slice["contingent"].sum()) if cont_slice is not None and not cont_slice.empty else float("nan")
    rr_rows = []
    for name, d in [("Зима", s1), ("Весна", s2)]:
        n = len(d)
        rr = n / N * 100.0 if np.isfinite(N) and N > 0 else float("nan")
        rr_rows.append(
            {
                "Волна": name,
                "Ответов (n)": n,
                "Контингент (N)": int(round(N)) if np.isfinite(N) else None,
                "Отклик, %": rr,
                "Погрешность, ±%": margin_of_error_pct(n, N) if np.isfinite(N) else float("nan"),
            }
        )
    st.caption("Таблица: отклик и достоверность по волнам.")
    show_table(round_df(pd.DataFrame(rr_rows)), width="stretch", hide_index=True)

    # ── Сводная таблица метрик ──────────────────────────────────────────────
    available = [m for m in DYNAMICS_METRICS if (m in s1.columns or m in s2.columns)]
    rows = []
    for m in available:
        v1, v2 = _metric_mean(s1, m), _metric_mean(s2, m)
        rows.append(
            {
                "Показатель": label(m) if m != "csi_overall" else "CSI (общий)",
                "Зима": v1,
                "Весна": v2,
                "Δ (весна−зима)": (v2 - v1) if (np.isfinite(v1) and np.isfinite(v2)) else float("nan"),
            }
        )
    comp = round_df(pd.DataFrame(rows))
    st.caption("Таблица: средние значения сопоставимых показателей зимой и весной.")
    show_table(comp, width="stretch", hide_index=True)

    # ── График дельт ────────────────────────────────────────────────────────
    plot = comp.dropna(subset=["Δ (весна−зима)"]).sort_values("Δ (весна−зима)")
    if not plot.empty:
        fig = px.bar(
            plot,
            x="Δ (весна−зима)",
            y="Показатель",
            orientation="h",
            color="Δ (весна−зима)",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Изменение показателей весной по сравнению с зимой",
        )
        fig.add_vline(0, line_dash="dash", line_color="gray")
        fig.update_layout(height=560, yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    # ── Сравнение по программам для выбранной метрики ────────────────────────
    st.markdown("### По программам")
    metric = st.selectbox(
        "Показатель для сравнения программ по волнам",
        options=available,
        index=available.index("nps") if "nps" in available else 0,
        format_func=lambda m: "CSI (общий)" if m == "csi_overall" else label(m),
    )

    def by_prog(d: pd.DataFrame) -> pd.Series:
        if metric not in d.columns:
            return pd.Series(dtype=float)
        tmp = d.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        return tmp.groupby("program")[metric].mean()

    g1, g2 = by_prog(s1).rename("Зима"), by_prog(s2).rename("Весна")
    prog = pd.concat([g1, g2], axis=1).dropna(how="all")
    if prog.empty:
        st.info("Нет данных по выбранному показателю в разбивке по программам.")
        return
    prog["Δ (весна−зима)"] = prog["Весна"] - prog["Зима"]
    prog = prog.reset_index().rename(columns={"program": "Программа"})

    long = prog.melt(
        id_vars=["Программа"],
        value_vars=["Зима", "Весна"],
        var_name="Волна",
        value_name="Значение",
    ).dropna(subset=["Значение"])
    fig = px.bar(
        long,
        x="Значение",
        y="Программа",
        color="Волна",
        orientation="h",
        barmode="group",
        title=f"{'CSI (общий)' if metric == 'csi_overall' else label(metric)}: зима и весна по программам",
        color_discrete_map={"Зима": MUTED, "Весна": ACCENT},
    )
    fig.update_layout(height=680, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.caption("Таблица: значения по программам и их изменение.")
    show_table(round_df(prog.sort_values("Δ (весна−зима)")), width="stretch", hide_index=True)


def render_codebook(path) -> None:
    st.subheader("Кодификатор")
    st.info("Что здесь: описание всех переменных и шкал SFQ (общее для зимы и весны).")
    codebook = load_codebook(path)
    if not codebook:
        st.warning(f"Файл кодификатора не найден: `{path}`.")
        return
    st.markdown(codebook)


@st.cache_data(show_spinner=False)
def load_combined(paths: tuple[str, ...]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if Path(p).exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined = normalize_unicode_columns(combined)
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    return combined


def render_placeholder(waves: list[str]) -> None:
    names = " и ".join(WAVES[w]["label"].lower() for w in waves)
    st.info(
        f"Данные SFQ ({names}) пока не загружены. После проведения опроса:\n\n"
        + "\n".join(
            f"1. Положите выгрузки по программам (xlsx или csv, по одному файлу на программу) в папку "
            f"`data/raw/{w}/` и запустите `python process_sfq.py {w}`."
            for w in waves
        )
        + "\n\nДашборд автоматически загрузит файл `data/processed/sfq_<волна>.csv`. Вкладки будут те же, "
        "что в SFQ 2025-26: «Обзор», «Динамика», «Описательные», «Сравнение программ», «Корреляции», "
        "«Матрица приоритетов», «NPS», «CSI», «Преподаватели», «Комментарии», «Кодификатор»."
    )


def render_sfq(mode: str) -> None:
    """mode: 'winter', 'spring' or 'both' (зима + весна вместе)."""
    waves = ["winter", "spring"] if mode == "both" else [mode]
    title = "SFQ 2026-27: " + " и ".join(WAVES[w]["label"].lower() for w in waves)
    st.title(title)

    available = [w for w in waves if wave_available(w)]
    if not available:
        render_placeholder(waves)
        return
    if len(available) < len(waves):
        missing = [WAVES[w]["label"] for w in waves if w not in available]
        st.warning(f"Нет данных за период: {', '.join(missing).lower()}. Показаны только загруженные волны.")

    general_paths = tuple(str(WAVES[w]["general"]) for w in available)
    teacher_paths = tuple(str(WAVES[w]["teachers"]) for w in available)
    st.caption("Источник данных: " + " + ".join(f"`{Path(p).name}`" for p in general_paths))

    df = load_combined(general_paths)
    if df.empty:
        st.error(f"Файлы данных пусты: {general_paths}")
        return
    df = safe_numeric(df, num_cols(df))

    df_teachers = load_combined(teacher_paths)
    if not df_teachers.empty and "rating" in df_teachers.columns:
        df_teachers["rating"] = pd.to_numeric(df_teachers["rating"], errors="coerce")
        for col in ["program", "school", "year", "teacher"]:
            if col not in df_teachers.columns:
                df_teachers[col] = np.nan

    filters = build_filters(df, key_prefix=f"sfq_{mode}")
    dff = apply_filters(df, filters)
    if dff.empty:
        st.error("После применения фильтров данных не осталось.")
        return

    cont_slice = filter_contingent(load_contingent(), filters)
    dashboard_df = add_program_display(add_block_features(dff), short_labels=filters.short_program_labels)
    teachers_df = (
        add_program_display(apply_filters(df_teachers, filters), short_labels=filters.short_program_labels)
        if not df_teachers.empty
        else pd.DataFrame()
    )

    combined = len(available) > 1
    tabs = st.tabs(
        ["Обзор", "Динамика", "Описательные", "Сравнение программ", "Корреляции", "Матрица приоритетов",
         "NPS", "CSI", "Преподаватели", "Комментарии", "Кодификатор"]
    )
    with tabs[0]:
        render_overview(dashboard_df, cont_slice, combined_mode=combined)
    with tabs[1]:
        render_dynamics(filters, cont_slice)
    with tabs[2]:
        render_descriptives(dashboard_df)
    with tabs[3]:
        render_program_comparison(dashboard_df)
    with tabs[4]:
        render_correlations(dashboard_df)
    with tabs[5]:
        render_priority_matrix(dashboard_df)
    with tabs[6]:
        render_nps(dashboard_df)
    with tabs[7]:
        render_csi(dashboard_df)
    with tabs[8]:
        render_teachers(teachers_df)
    with tabs[9]:
        render_comments(dashboard_df)
    with tabs[10]:
        render_codebook(WAVES[available[0]]["codebook"])
