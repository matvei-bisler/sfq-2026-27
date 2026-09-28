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
    load_codebook,
    load_contingent,
    margin_of_error_pct,
    normalize_unicode_columns,
    num_cols,
    render_interp,
    render_response_rate,
    round_df,
    safe_numeric,
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
    "nps": "NPS (0-10)",
    "satisf_overall": "Удовлетворенность образовательным процессом",
    "satisf_teachers": "Удовлетворенность преподавателями",
    "expect_match": "Совпадение ожиданий с опытом",
    "assess_criteria_timely": "Критерии оценивания представлены вовремя",
    "assess_order_clear": "Порядок сдачи и оценивания понятен",
    "assess_consistent": "Оценивание соответствует критериям",
    "fac_support_score": "Поддержка преподавательской команды (оценка)",
    "fac_clarity_score": "Понятность объяснений преподавателей (оценка)",
    "cur_timely_score": "Своевременность информации от куратора (оценка)",
    "cur_help_score": "Готовность куратора помочь (оценка)",
    "prog_clarity_score": "Понятность содержания дисциплин (оценка)",
    "prog_deadlines_score": "Реалистичность дедлайнов (оценка)",
    "prog_relevance_score": "Проф. релевантность дисциплин (оценка)",
    "prog_workload_score": "Оптимальность нагрузки (оценка)",
    "coord_respect_score": "Уважительность координатора (оценка)",
    "coord_results_score": "Результативность обращений к координатору (оценка)",
    "coord_timely_score": "Своевременность информации от координатора (оценка)",
    "coord_help_score": "Готовность координатора помочь (оценка)",
    "fac_support_imp": "Поддержка преподавательской команды (важность)",
    "fac_clarity_imp": "Понятность объяснений преподавателей (важность)",
    "cur_timely_imp": "Своевременность информации от куратора (важность)",
    "cur_help_imp": "Готовность куратора помочь (важность)",
    "prog_clarity_imp": "Понятность содержания дисциплин (важность)",
    "prog_deadlines_imp": "Реалистичность дедлайнов (важность)",
    "prog_relevance_imp": "Проф. релевантность дисциплин (важность)",
    "prog_workload_imp": "Оптимальность нагрузки (важность)",
    "coord_respect_imp": "Уважительность координатора (важность)",
    "coord_results_imp": "Результативность обращений к координатору (важность)",
    "coord_timely_imp": "Своевременность информации от координатора (важность)",
    "coord_help_imp": "Готовность координатора помочь (важность)",
    "infra_library": "Библиотека",
    "infra_wellbeing": "Сервис Wellbeing",
    "infra_food": "Еда на кампусе",
    "infra_software": "ПО в аудиториях",
    "infra_equipment": "Оборудование в аудиториях",
    "infra_classrooms": "Комфорт аудиторий",
    "infra_workshops": "Комфорт мастерских/ресурсных центров",
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
    "goal_achieved": "Достижение целей на программе (шкала 1–7)",
    "navigation_ease": "Лёгкость ориентирования в учебном процессе (шкала 1–7)",
    "class_comfort": "Комфорт на занятиях (шкала 1–7)",
    "prev_sem_relevance": "Связь с предыдущим семестром",
    "skill_confidence": "Уверенность в применении навыков",
    "postgrad_masters": "План: магистратура/доп. образование",
    "postgrad_same_field": "План: работа в выбранной сфере",
    "postgrad_other_field": "План: работа в другой сфере",
    "postgrad_other": "План: другой вариант",
    "faculty_mean": "Блок: преподавательская команда (среднее)",
    "curator_mean": "Блок: куратор (среднее)",
    "program_mean": "Блок: программа (среднее)",
    "coordinator_mean": "Блок: координатор (среднее)",
    "assessment_mean": "Блок: оценивание (среднее)",
    "infrastructure_mean": "Блок: инфраструктура (среднее)",
}


def label(col: str) -> str:
    return COL_LABELS.get(col, col)


def get_descriptive_metrics(df: pd.DataFrame) -> list[str]:
    """Расширенный список метрик из codebook + блоковые агрегаты."""
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
        with st.expander("Формулы и обозначения (ANOVA / Kruskal)", expanded=False):
            st.latex(r"\eta^2=\frac{SS_{between}}{SS_{total}}")
            st.caption(
                "ANOVA проверяет различия средних между программами, "
                "а Kruskal-Wallis — различия по рангам (непараметрический подход)."
            )
        return

    if kind == "spearman":
        with st.expander("Формулы и обозначения (Спирмен)", expanded=False):
            st.latex(r"\rho_s=\mathrm{corr}(\mathrm{rank}(X),\mathrm{rank}(Y))")
            st.caption("Метод использует ранги, поэтому устойчивее для порядковых шкал и ненормальных распределений.")
        return

    if kind == "nps":
        with st.expander("Формулы и обозначения (NPS)", expanded=False):
            st.latex(r"\mathrm{NPS}=\%Promoters-\%Detractors")
            st.caption("Границы сегментов: Promoters = 9–10, Passives = 7–8, Detractors = 0–6.")
        return

    if kind == "csi":
        with st.expander("Формулы и обозначения (CSI)", expanded=False):
            st.latex(r"\mathrm{CSI}_{i,b} = \frac{\bar{S}_{i,b}\cdot\bar{I}_{i,b}}{16}\cdot 100")
            st.latex(r"\bar{S}_{i,b}=\frac{1}{K_b}\sum_{k=1}^{K_b} S_{i,b,k}, \quad \bar{I}_{i,b}=\frac{1}{K_b}\sum_{k=1}^{K_b} I_{i,b,k}")
            st.latex(r"\mathrm{CSI}^{\mathrm{overall}}_i = \frac{1}{B_i}\sum_{b=1}^{B_i}\mathrm{CSI}_{i,b}")
            st.markdown("Обозначения индексов:")
            st.markdown("- `i` — респондент.")
            st.markdown("- `b` — блок (`Куратор`, `Преподавательский состав`, `Программа`, `Учебный отдел`).")
            st.markdown("- `k` — отдельный критерий внутри блока.")
            st.caption(
                "Шкалы удовлетворенности и важности — 1..4, поэтому нормирующий коэффициент равен 16 (=4×4). "
                "При пропусках используется среднее по доступным пунктам блока."
            )
        return

    if kind == "teacher_bayes":
        with st.expander("Формулы и обозначения (сглаженное среднее преподавателя)", expanded=False):
            st.latex(r"\hat{\mu}_t=\frac{n_t\cdot\bar{x}_t + m\cdot\mu_0}{n_t+m}")
            st.markdown("Обозначения:")
            st.markdown("- `t` — преподаватель.")
            st.markdown("- `n_t` — число оценок преподавателя.")
            st.markdown("- `\\bar{x}_t` — обычное среднее преподавателя.")
            st.markdown("- `\\mu_0` — глобальное среднее по текущему отфильтрованному срезу.")
            st.markdown("- `m` — сила сглаживания (чем больше `m`, тем сильнее тянет к `\\mu_0`).")
            st.caption(
                "Это эмпирическое байесовское сглаживание, полезное при малом числе оценок у преподавателя."
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
    c1.metric("Ответов", f"{len(df):,}", help="Количество анкет после применения фильтров.")
    c2.metric("Программ", f"{df['program'].nunique() if 'program' in df.columns else 0}", help="Число уникальных программ в текущем срезе.")
    c3.metric("Школ", f"{df['school'].nunique() if 'school' in df.columns else 0}", help="Число уникальных школ в текущем срезе.")
    c4.metric("Медиана NPS", f"{df['nps'].median():.2f}" if "nps" in df.columns else "н/д", help="Центральное значение готовности рекомендовать школу.")
    if "nps" in df.columns:
        c5, c6 = st.columns(2)
        c5.metric("Минимум NPS", f"{df['nps'].min():.2f}", help="Минимальное значение NPS в текущем срезе.")
        c6.metric("Максимум NPS", f"{df['nps'].max():.2f}", help="Максимальное значение NPS в текущем срезе.")
    render_interp(
        "Как интерпретировать обзор",
        [
            "Если число ответов по программе низкое, то сравнения по этой программе менее надежны.",
            "Если медиана NPS существенно ниже 7, то в срезе преобладает нейтральный/негативный опыт.",
            "Если минимум очень низкий, а максимум высокий, то мнения респондентов поляризованы.",
        ],
    )

    render_response_rate(
        df,
        cont_slice,
        combined_note=(
            "В режиме «Зима + Весна» response rate не считается: ответы двух волн "
            "нельзя сопоставлять с одним контингентом (студенты могли отвечать дважды). "
            "Response rate по волнам — на вкладке **«Динамика»**."
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
            labels={"program_display": "Программа", "n": "Количество ответов"},
            hover_data={"program": True, "program_display": False, "n": ":.0f"},
        )
        fig.update_layout(height=560, yaxis_title="")
        st.plotly_chart(fig, width='stretch')

    st.markdown("### Навигатор по вкладкам")
    legend_df = pd.DataFrame(
        [
            {"Вкладка": "Динамика", "Что внутри": "Сравнение зимы и весны по сопоставимым метрикам и response rate."},
            {"Вкладка": "Описательные", "Что внутри": "Средние, медианы, min/max, доверительные интервалы и сравнение по программам."},
            {"Вкладка": "Сравнение программ", "Что внутри": "ANOVA, Kruskal-Wallis, Levene, Dunn post-hoc для выбранной метрики."},
            {"Вкладка": "Корреляции", "Что внутри": "Матрица Спирмена и топ положительных/отрицательных связей."},
            {"Вкладка": "Матрица приоритетов", "Что внутри": "Сопоставление важности и оценки, зоны приоритетных улучшений."},
            {"Вкладка": "NPS", "Что внутри": "Промоутеры/пассивы/критики, состав по программам, NPS по программам."},
            {"Вкладка": "CSI", "Что внутри": "Индекс удовлетворенности по блокам и общий CSI, формулы и интерпретация."},
            {"Вкладка": "Преподаватели", "Что внутри": "Оценки по школам/программам/курсам и по конкретным преподавателям, включая сглаживание."},
            {"Вкладка": "Комментарии", "Что внутри": "Все открытые текстовые ответы: фильтр по разделу, программе, школе, курсу и ключевому слову."},
            {"Вкладка": "Кодбук", "Что внутри": "Полное описание полей, шкал и структуры данных."},
        ]
    )
    st.dataframe(legend_df, width="stretch", hide_index=True)


def render_descriptives(df: pd.DataFrame) -> None:
    st.subheader("Описательная статистика")
    st.info("Что здесь: средние, медианы, разброс и bootstrap доверительные интервалы для выбранных метрик.")

    available_metrics = get_descriptive_metrics(df)
    if not available_metrics:
        st.warning("Ключевые метрики не найдены.")
        return

    default_metrics = [c for c in KEY_METRICS if c in available_metrics]
    selected = st.multiselect(
        "Метрики для анализа",
        available_metrics,
        default=default_metrics if default_metrics else available_metrics[:10],
        format_func=label,
        help="Доступны ключевые и дополнительные метрики из выбранного среза данных.",
    )
    if not selected:
        st.info("Выберите хотя бы одну метрику.")
        return
    render_interp(
        "Как интерпретировать описательные статистики",
        [
            "Если среднее и медиана заметно отличаются, то распределение асимметрично (есть перекос).",
            "Если CI 95% узкий, то оценка среднего более стабильна; если широкий, то неопределенность выше.",
            "Если min/max сильно отличаются от Q25/Q75, то возможны выбросы или неоднородные подгруппы.",
            "Если CI по программам почти не перекрываются, то различия между программами вероятно существенные.",
        ],
    )

    rows = []
    for col in selected:
        mean, lo, hi, n = bootstrap_mean_ci(df[col])
        vals = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "Метрика": label(col),
                "n": n,
                "Среднее": mean,
                "Медиана": vals.median(),
                "Минимум": vals.min(),
                "Максимум": vals.max(),
                "Std": vals.std(),
                "Q25": vals.quantile(0.25),
                "Q75": vals.quantile(0.75),
                "CI 95% нижняя": lo,
                "CI 95% верхняя": hi,
            }
        )
    stat_df = pd.DataFrame(rows).sort_values("Среднее", ascending=False)
    stat_df = round_df(stat_df)
    st.caption("Таблица: описательные статистики по выбранным метрикам.")
    st.dataframe(stat_df, width='stretch')

    st.caption("График: средние значения с bootstrap 95% доверительными интервалами.")
    fig = px.scatter(
        stat_df,
        x="Среднее",
        y="Метрика",
        error_x=stat_df["CI 95% верхняя"] - stat_df["Среднее"],
        error_x_minus=stat_df["Среднее"] - stat_df["CI 95% нижняя"],
        size="n",
        color="Std",
        color_continuous_scale="RdYlBu_r",
        title="Средние значения и доверительные интервалы",
    )
    fig.update_layout(height=460, yaxis_title="")
    st.plotly_chart(fig, width='stretch')

    metric_for_group = st.selectbox(
        "Метрика по программам",
        selected,
        index=0,
        format_func=label,
        help="Для выбранной метрики показываются средние по программам с доверительными интервалами.",
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

    st.caption("Таблица: сводка по программам (с min/max).")
    st.dataframe(
        grp.rename(
            columns={
                "program": "Программа",
                "program_display": "Короткое имя",
                "n": "n",
                "mean": "Среднее",
                "ci_low": "CI нижняя",
                "ci_high": "CI верхняя",
                "min": "Минимум",
                "max": "Максимум",
            }
        ),
        width='stretch',
    )

    st.caption("График: средние по программам с bootstrap 95% CI.")
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
        labels={"program_display": "Программа", "mean": "Среднее"},
        hover_data={"program": True, "program_display": False, "mean": ":.2f", "n": True},
    )
    fig.update_layout(height=620, yaxis_title="")
    st.plotly_chart(fig, width='stretch')

    st.caption("График: распределение значений выбранной метрики по программам.")
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
        "Что здесь: статистическое сравнение программ по выбранной метрике. "
        "Показаны ANOVA, Kruskal-Wallis, Levene и post-hoc Dunn (Holm-коррекция)."
    )
    render_formula_block("anova")

    if "program" not in df.columns:
        st.warning("Для этой вкладки нужен столбец `program`.")
        return

    candidates = [c for c in get_descriptive_metrics(df) if df[c].dropna().nunique() >= 3]
    if not candidates:
        st.warning("Нет метрик с достаточной вариативностью для сравнения программ.")
        return

    selected_metric = st.selectbox(
        "Метрика для сравнения программ",
        options=candidates,
        index=candidates.index("nps") if "nps" in candidates else 0,
        format_func=label,
        help="Выберите метрику, по которой нужно сравнить программы.",
    )
    render_interp(
        "Как интерпретировать сравнение программ",
        [
            "Если ANOVA p-value < 0.05, то средние по программам статистически различаются.",
            "Если Kruskal p-value < 0.05, то различия устойчивы и в непараметрическом тесте.",
            "Если Levene p-value < 0.05, то дисперсии неоднородны; в выводах опирайтесь на Kruskal/Dunn.",
            "Если Eta^2 ~ 0.01/0.06/0.14+, то эффект обычно трактуют как малый/средний/крупный.",
            "Если в Dunn p-value < 0.05 для пары программ, то именно эта пара различается значимо.",
        ],
    )

    min_n = st.slider(
        "Минимальный размер группы для тестов",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        help="Программы с размером меньше порога исключаются из инференциальных тестов.",
    )
    vc = df["program"].value_counts()
    valid_programs = vc[vc >= min_n].index
    keep_cols = ["program", "program_display", selected_metric]
    keep_cols = [c for c in keep_cols if c in df.columns]
    d = df[df["program"].isin(valid_programs)][keep_cols].dropna()
    st.caption(f"В анализе: программ = {d['program'].nunique()}, наблюдений = {len(d)}.")

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
    k1.metric("ANOVA p-value", f"{anova_tbl.loc['C(program)', 'PR(>F)']:.2f}", help=f"Проверка различий средних по метрике: {label(selected_metric)}.")
    k2.metric("Eta^2", f"{eta2:.2f}", help="Доля дисперсии, объясненная фактором программы.")
    k3.metric("Kruskal p-value", f"{kw_p:.2f}", help="Непараметрический тест различий между программами.")
    k4.metric("Levene p-value", f"{lev_p:.2f}", help="Проверка равенства дисперсий между группами.")
    k5.metric("Минимум", f"{d[selected_metric].min():.2f}", help="Минимум выбранной метрики в анализируемом срезе.")
    k6.metric("Максимум", f"{d[selected_metric].max():.2f}", help="Максимум выбранной метрики в анализируемом срезе.")

    left, right = st.columns(2)
    with left:
        st.caption("Таблица: результаты ANOVA.")
        st.dataframe(round_df(anova_tbl), width='stretch')
    with right:
        st.caption("Таблица: Dunn post-hoc, p-value с Holm-коррекцией.")
        st.dataframe(round_df(dunn), width='stretch')

    st.caption("График: распределение выбранной метрики по программам.")
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
    st.info("Что здесь: матрица корреляций Спирмена и наиболее сильные положительные/отрицательные связи с выбранной метрикой.")
    render_formula_block("spearman")

    numeric = num_cols(df)
    default = [c for c in ["nps", "satisf_overall", "expect_match", "program_mean", "coordinator_mean", "infrastructure_mean"] if c in numeric]
    selected = st.multiselect(
        "Числовые столбцы для матрицы Спирмена",
        numeric,
        default=default if default else numeric[:12],
        format_func=label,
        help="Выберите 2 и более числовых показателя.",
    )
    if len(selected) < 2:
        st.info("Выберите минимум 2 столбца.")
        return
    render_interp(
        "Как интерпретировать корреляции",
        [
            "Если коэффициент Спирмена > 0, то при росте одной метрики другая обычно тоже растет.",
            "Если коэффициент Спирмена < 0, то при росте одной метрики другая обычно снижается.",
            "Если |rho| около 0.1/0.3/0.5+, то связь часто трактуют как слабую/умеренную/сильную.",
            "Если связь высокая, это не доказывает причинность; проверяйте контекст и возможные скрытые факторы.",
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
        title="Корреляционная матрица (Spearman)",
    )
    fig.update_layout(height=680)
    st.plotly_chart(fig, width='stretch')

    target = st.selectbox(
        "Целевая метрика",
        selected,
        index=selected.index("nps") if "nps" in selected else 0,
        format_func=label,
        help="Показываются коэффициенты Спирмена остальных признаков с выбранной метрикой.",
    )
    with_target = corr[target].drop(target).sort_values(ascending=False).rename("rho").to_frame()
    with_target.index = [label(i) for i in with_target.index]
    st.caption("Таблица: коэффициенты корреляции с целевой метрикой.")
    st.dataframe(round_df(with_target), width='stretch')

    top_pos = with_target.head(5).reset_index().rename(columns={"index": "Показатель"})
    top_neg = with_target.tail(5).reset_index().rename(columns={"index": "Показатель"})
    cols = pd.concat([top_pos, top_neg], ignore_index=True).drop_duplicates(subset="Показатель")
    st.caption("График: топ положительных и отрицательных связей.")
    fig = px.bar(
        cols,
        x="rho",
        y="Показатель",
        orientation="h",
        color="rho",
        color_continuous_scale="RdBu",
        title=f"Топ связей с метрикой: {label(target)}",
        labels={"rho": "Коэффициент Спирмена"},
    )
    fig.update_layout(height=420, yaxis_title="")
    st.plotly_chart(fig, width='stretch')


def render_priority_matrix(df: pd.DataFrame) -> None:
    st.subheader("Матрица приоритетов")
    st.info(
        "Что здесь: соотношение важности и оценки по критериям. "
        "Чем выше разрыв (важность - оценка), тем выше приоритет для улучшений."
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
                "Разрыв (важность - оценка)": imp[valid].mean() - score[valid].mean(),
                "n": int(valid.sum()),
            }
        )

    if not rows:
        st.warning("В текущем срезе нет доступных пар score/importance.")
        return
    render_interp(
        "Как интерпретировать матрицу приоритетов",
        [
            "Если важность выше средней и оценка ниже средней (верхний левый квадрант), то это зона первоочередного улучшения.",
            "Если важность и оценка обе высокие (верхний правый), то это сильные стороны, которые стоит поддерживать.",
            "Если разрыв (важность - оценка) большой и положительный, то критерий недоудовлетворен.",
            "Если разрыв около нуля, то ожидания и фактическая оценка сбалансированы.",
        ],
    )

    m = round_df(pd.DataFrame(rows))
    score_mid = m["Средняя оценка"].mean()
    imp_mid = m["Средняя важность"].mean()

    st.caption("График: Importance-Performance matrix с разделением на квадранты по средним значениям.")
    fig = px.scatter(
        m,
        x="Средняя оценка",
        y="Средняя важность",
        text="Критерий",
        size="n",
        color="Разрыв (важность - оценка)",
        color_continuous_scale="RdYlGn_r",
        hover_data=["Разрыв (важность - оценка)", "n"],
        title="Матрица важность-оценка",
    )
    fig.add_vline(score_mid, line_dash="dash", line_color="gray")
    fig.add_hline(imp_mid, line_dash="dash", line_color="gray")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=560)
    st.plotly_chart(fig, width='stretch')

    st.caption("Таблица: приоритеты улучшений (по убыванию разрыва).")
    m = m.sort_values("Разрыв (важность - оценка)", ascending=False)
    st.dataframe(round_df(m), width='stretch')


def render_nps(df: pd.DataFrame) -> None:
    st.subheader("NPS-аналитика")
    st.info("Что здесь: структура NPS (промоутеры/пассивы/критики), вклад программ и NPS по программам.")
    if "nps" not in df.columns:
        st.warning("Столбец `nps` не найден.")
        return
    render_formula_block("nps")
    render_interp(
        "Как интерпретировать NPS",
        [
            "Если NPS > 0, то доля промоутеров выше доли критиков; если NPS < 0, ситуация обратная.",
            "Если доля критиков растет у конкретной программы, то стоит приоритизировать качественный разбор этой программы.",
            "Если NPS сильно различается между программами, то проблемы/сильные стороны локальны, а не системны.",
            "Если общий NPS высокий, но есть низкий минимум, то в выборке могут быть отдельные проблемные сегменты.",
        ],
    )

    d = df.copy()
    d["segment"] = np.where(d["nps"] >= 9, "Промоутер", np.where(d["nps"] >= 7, "Пассив", "Критик"))
    seg = d["segment"].value_counts().rename_axis("segment").reset_index(name="n")
    seg["pct"] = seg["n"] / seg["n"].sum() * 100
    nps_value = float(seg.loc[seg["segment"] == "Промоутер", "pct"].sum() - seg.loc[seg["segment"] == "Критик", "pct"].sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("NPS", f"{nps_value:.2f}", help="NPS = %промоутеров - %критиков.")
    c2.metric("% Промоутеров", f"{seg.loc[seg['segment']=='Промоутер', 'pct'].sum():.2f}")
    c3.metric("% Пассивов", f"{seg.loc[seg['segment']=='Пассив', 'pct'].sum():.2f}")
    c4.metric("% Критиков", f"{seg.loc[seg['segment']=='Критик', 'pct'].sum():.2f}")
    c5.metric("Минимум NPS", f"{d['nps'].min():.2f}")
    c6.metric("Максимум NPS", f"{d['nps'].max():.2f}")

    left, right = st.columns(2)
    with left:
        st.caption("График: общий разрез сегментов NPS.")
        fig = px.pie(
            seg,
            names="segment",
            values="n",
            hole=0.45,
            color="segment",
            color_discrete_map={"Промоутер": "#2ca02c", "Пассив": "#ffbf00", "Критик": "#d62728"},
        )
        fig.update_layout(title="Структура NPS по сегментам", height=430)
        st.plotly_chart(fig, width='stretch')
    with right:
        if "program" in d.columns:
            st.caption("График: состав NPS-сегментов по программам.")
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
                labels={"program_display": "Программа", "pct": "%"},
                hover_data={"program": True, "program_display": False, "pct": ":.2f"},
            )
            fig.update_layout(height=620, yaxis_title="")
            fig.update_yaxes(tickformat=".2f")
            fig.update_traces(hovertemplate="Программа: %{y}<br>Доля: %{x:.2f}%<extra></extra>")
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
    CSI = (средняя удовлетворенность * средняя важность) / 16 * 100
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
    st.subheader("CSI (индекс удовлетворенности)")
    st.info("Что здесь: расчет CSI по блокам и общий CSI, методология и интерпретация.")
    render_formula_block("csi")

    render_interp(
        "Как считается CSI",
        [
            "CSI считается по блокам: Куратор, Преподавательский состав, Программа, Учебный отдел.",
            "Общий CSI = среднее из доступных блоковых CSI у респондента.",
            "Сначала считается CSI по каждому блоку и респонденту, затем агрегируется по выборке.",
        ],
    )
    render_interp(
        "Как интерпретировать CSI",
        [
            "Если CSI ближе к 100, то высокий уровень удовлетворенности по важным для студентов критериям.",
            "Если CSI растет после фильтрации по школе/курсу/программе, то в этом сегменте качество опыта выше.",
            "Если высокий разрыв между блоками, то улучшения стоит приоритизировать в блоках с низким CSI.",
            "Если CSI низкий и одновременно важность высокая, это сигнал о наиболее критичных точках сервиса.",
        ],
    )

    d = compute_csi_frame(df)
    csi_cols = [c for c in d.columns if c.startswith("csi_")]
    if not csi_cols:
        st.warning("Недостаточно данных для расчета CSI.")
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
                "Среднее CSI": vals.mean(),
                "Медиана CSI": vals.median(),
                "Минимум CSI": vals.min(),
                "Максимум CSI": vals.max(),
                "Std": vals.std(),
            }
        )
    csi_summary = round_df(pd.DataFrame(summary_rows).sort_values("Среднее CSI", ascending=False))
    st.caption("Таблица: сводка CSI по блокам.")
    st.dataframe(csi_summary, width="stretch")

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
                "Средняя удовлетворенность (S̄)": s_mean_row.mean(),
                "Средняя важность (Ī)": i_mean_row.mean(),
                "Средний CSI (по респондентам)": csi_row.mean(),
                "CSI из агрегированных S̄ и Ī": (s_mean_row.mean() * i_mean_row.mean() / 16.0) * 100.0,
            }
        )
    if comp_rows:
        comp_df = round_df(pd.DataFrame(comp_rows))
        st.caption(
            "Таблица: компоненты формулы CSI. "
            "`Средний CSI (по респондентам)` — основной показатель в дашборде."
        )
        st.dataframe(comp_df, width="stretch")

    plot_df = csi_summary.copy()
    fig = px.bar(
        plot_df.sort_values("Среднее CSI"),
        x="Среднее CSI",
        y="Блок CSI",
        orientation="h",
        color="Среднее CSI",
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
        help="Показывает распределение выбранного CSI по программам.",
    )
    by_prog = (
        d.groupby(["program", "program_display"], dropna=False)[selected_csi]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "program": "Программа",
                "program_display": "Короткое имя",
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
    st.dataframe(by_prog, width="stretch")

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
        "Что здесь: оценки преподавателей в разрезах школы/программы/курса и рейтинг по конкретным преподавателям."
    )
    if df_teachers.empty or "rating" not in df_teachers.columns:
        st.warning("Данные по преподавателям для выбранного среза не найдены или пусты.")
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
        help="Рекомендуется использовать сглаженное среднее при малом числе оценок.",
    )

    global_mean = float(d["rating"].mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Оценок", f"{len(d):,}")
    c2.metric("Преподавателей", f"{d['teacher'].nunique():,}")
    c3.metric("Средняя оценка", f"{d['rating'].mean():.2f}")
    c4.metric("Байесовская база μ0", f"{global_mean:.2f}")

    st.markdown("### Разрез по сегментам")
    dim = st.selectbox(
        "Разрез",
        ["school", "program", "year"],
        format_func=lambda x: {"school": "Школа", "program": "Программа", "year": "Курс"}[x],
    )

    seg = (
        d.groupby(dim, dropna=False)["rating"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
        .rename(
            columns={
                dim: "Сегмент",
                "count": "n",
                "mean": "Среднее",
                "median": "Медиана",
                "min": "Минимум",
                "max": "Максимум",
                "std": "Std",
            }
        )
    )
    seg["Байес. среднее"] = ((seg["n"] * seg["Среднее"]) + (m * global_mean)) / (seg["n"] + m)
    seg = round_df(seg.sort_values("Байес. среднее", ascending=False))
    st.dataframe(seg, width="stretch")

    seg_plot = seg.sort_values("Байес. среднее", ascending=True).head(25)
    fig = px.bar(
        seg_plot,
        x="Байес. среднее",
        y="Сегмент",
        orientation="h",
        color="n",
        color_continuous_scale="Blues",
        title="Сегменты: сглаженное среднее оценки преподавателей",
        labels={"Байес. среднее": "Сглаженное среднее", "Сегмент": ""},
    )
    fig.update_xaxes(tickformat=".2f")
    fig.update_layout(height=560, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Конкретные преподаватели")
    min_n = st.slider("Минимум оценок у преподавателя", 1, 50, 5, 1)
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
                "std": "Std",
            }
        )
    )
    teacher_stats["Байес. среднее"] = ((teacher_stats["n"] * teacher_stats["Среднее"]) + (m * global_mean)) / (
        teacher_stats["n"] + m
    )
    teacher_stats = teacher_stats[teacher_stats["n"] >= min_n]
    if teacher_stats.empty:
        st.info("Нет преподавателей, удовлетворяющих минимальному числу оценок.")
        return

    teacher_stats = round_df(teacher_stats.sort_values("Байес. среднее", ascending=False))
    st.dataframe(teacher_stats, width="stretch")

    top_n_min = 1 if len(teacher_stats) < 5 else 5
    top_n_max = min(40, len(teacher_stats))
    top_n_default = min(20, top_n_max)
    top_n = st.slider("Сколько преподавателей показать в графике", top_n_min, top_n_max, top_n_default)
    top_teachers = teacher_stats.head(top_n).sort_values("Байес. среднее")
    fig = px.bar(
        top_teachers,
        x="Байес. среднее",
        y="Преподаватель",
        orientation="h",
        color="n",
        color_continuous_scale="Teal",
        title="Топ преподавателей по сглаженному среднему",
        labels={"Байес. среднее": "Сглаженное среднее", "Преподаватель": ""},
    )
    fig.update_xaxes(tickformat=".2f")
    fig.update_layout(height=640, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    teacher_pick = st.selectbox("Выберите преподавателя для детализации", teacher_stats["Преподаватель"].tolist())
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
    st.caption("Детализация выбранного преподавателя по программам.")
    st.dataframe(by_prog, width="stretch")


def render_comments(df: pd.DataFrame) -> None:
    _COMMENT_SECTIONS: dict[str, str] = {
        "cur_comment":     "Куратор",
        "prog_comment":    "Программа",
        "coord_comment":   "Учебный отдел",
        "assess_comment":  "Оценивание",
        "fac_comment":     "Преподаватели",
        "seminar_comment": "Семинары/лекции",
        "comment_final":   "Общий комментарий",
    }

    st.subheader("Комментарии")
    st.info(
        "Что здесь: все открытые текстовые ответы из опроса. "
        "Можно фильтровать по разделу опроса, программе, школе, году и ключевому слову."
    )
    render_interp(
        "Как читать комментарии",
        [
            "Каждая строка — один ответ одного студента по одному разделу анкеты.",
            "Пустые ответы отфильтрованы автоматически; глобальные фильтры (школа, курс, программа) применяются.",
            "Используйте поле поиска, чтобы найти ответы по ключевому слову.",
            "Раздел «Общий комментарий» — финальное свободное поле анкеты, наиболее содержательное.",
        ],
    )

    available: dict[str, str] = {col: lbl for col, lbl in _COMMENT_SECTIONS.items() if col in df.columns}
    if not available:
        st.warning("В данных не найдено колонок с комментариями.")
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
        st.caption("График: количество непустых комментариев по разделам опроса.")
        counts_df = (
            pd.DataFrame(list(counts.items()), columns=["Раздел", "Количество"])
            .sort_values("Количество", ascending=True)
        )
        fig = px.bar(
            counts_df,
            x="Количество",
            y="Раздел",
            orientation="h",
            title="Комментарии по разделам",
            color="Количество",
            color_continuous_scale="Blues",
            labels={"Количество": "Кол-во ответов", "Раздел": ""},
        )
        fig.update_layout(height=280, coloraxis_showscale=False, margin=dict(l=0, r=0, t=36, b=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### Просмотр комментариев")

    # ── Controls ───────────────────────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([2, 3])
    with ctrl1:
        selected_sections = st.multiselect(
            "Раздел опроса",
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

    st.caption(f"Показано: **{len(comments_df)}** комментариев.")
    st.dataframe(
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
    st.subheader("Динамика: зима → весна")
    st.info(
        "Что здесь: сравнение зимней (1 семестр) и весенней (2 семестр) волн SFQ по сопоставимым метрикам "
        "(применяются те же фильтры программ/курсов/школ). "
        "Загружаются обе волны независимо от выбора в сайдбаре."
    )
    st.caption(
        "Сравниваются только метрики с единой шкалой в обеих волнах. "
        "Если в анкете одной из волн поменялась шкала вопроса — исключите его из `DYNAMICS_METRICS`."
    )

    s1 = _prepare_semester_slice(WAVES["winter"]["general"], filters)
    s2 = _prepare_semester_slice(WAVES["spring"]["general"], filters)
    if s1.empty or s2.empty:
        st.warning("Для динамики нужны обе волны (зима и весна) с данными в текущем срезе фильтров.")
        return

    render_interp(
        "Как читать динамику",
        [
            "Δ = значение весной минус значение зимой.",
            "Если Δ положительная, метрика выросла; если отрицательная — снизилась.",
            "Малые Δ при небольшом числе ответов могут быть случайными — сверяйтесь с response rate.",
            "NPS здесь — среднее по шкале 0–10 (не классический индекс промоутеров минус критиков).",
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
                "Response rate, %": rr,
                "Погрешность ±%": margin_of_error_pct(n, N) if np.isfinite(N) else float("nan"),
            }
        )
    st.caption("Таблица: отклик и достоверность по волнам.")
    st.dataframe(round_df(pd.DataFrame(rr_rows)), width="stretch", hide_index=True)

    # ── Сводная таблица метрик ──────────────────────────────────────────────
    available = [m for m in DYNAMICS_METRICS if (m in s1.columns or m in s2.columns)]
    rows = []
    for m in available:
        v1, v2 = _metric_mean(s1, m), _metric_mean(s2, m)
        rows.append(
            {
                "Метрика": label(m) if m != "csi_overall" else "CSI (общий)",
                "Зима": v1,
                "Весна": v2,
                "Δ (весна−зима)": (v2 - v1) if (np.isfinite(v1) and np.isfinite(v2)) else float("nan"),
            }
        )
    comp = round_df(pd.DataFrame(rows))
    st.caption("Таблица: средние по сопоставимым метрикам, зима → весна.")
    st.dataframe(comp, width="stretch", hide_index=True)

    # ── График дельт ────────────────────────────────────────────────────────
    plot = comp.dropna(subset=["Δ (весна−зима)"]).sort_values("Δ (весна−зима)")
    if not plot.empty:
        fig = px.bar(
            plot,
            x="Δ (весна−зима)",
            y="Метрика",
            orientation="h",
            color="Δ (весна−зима)",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Изменение метрик: весна минус зима",
        )
        fig.add_vline(0, line_dash="dash", line_color="gray")
        fig.update_layout(height=560, yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

    # ── Сравнение по программам для выбранной метрики ────────────────────────
    st.markdown("### По программам")
    metric = st.selectbox(
        "Метрика для сравнения по программам",
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
        st.info("Нет данных по выбранной метрике в разрезе программ.")
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
        title=f"{'CSI (общий)' if metric == 'csi_overall' else label(metric)}: зима vs весна по программам",
        color_discrete_map={"Зима": MUTED, "Весна": ACCENT},
    )
    fig.update_layout(height=680, yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    st.caption("Таблица: значения по программам и изменение.")
    st.dataframe(round_df(prog.sort_values("Δ (весна−зима)")), width="stretch", hide_index=True)


def render_codebook(path) -> None:
    st.subheader("Кодбук")
    st.info("Что здесь: описание всех колонок и шкал в данных SFQ (одинаково для зимы и весны).")
    codebook = load_codebook(path)
    if not codebook:
        st.warning(f"Файл кодбука не найден: `{path}`.")
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
        f"Данные SFQ ({names}) ещё не загружены. Как только опрос пройдёт:\n\n"
        + "\n".join(
            f"1. Положите выгрузки по программам (xlsx или csv, по файлу на программу) в "
            f"`data/raw/{w}/` и запустите `python process_sfq.py {w}`."
            for w in waves
        )
        + "\n\nДашборд подхватит `data/processed/sfq_<волна>.csv` автоматически — вкладки те же, что в SFQ 2025-26: "
        "Обзор, Динамика, Описательные, Сравнение программ, Корреляции, Матрица приоритетов, NPS, CSI, "
        "Преподаватели, Комментарии, Кодбук."
    )


def render_sfq(mode: str) -> None:
    """mode: 'winter', 'spring' or 'both' (зима + весна вместе)."""
    waves = ["winter", "spring"] if mode == "both" else [mode]
    title = "SFQ 2026-27: " + (" + ".join(WAVES[w]["label"] for w in waves))
    st.title(title)

    available = [w for w in waves if wave_available(w)]
    if not available:
        render_placeholder(waves)
        return
    if len(available) < len(waves):
        missing = [WAVES[w]["label"] for w in waves if w not in available]
        st.warning(f"Нет данных для: {', '.join(missing)}. Показаны только загруженные волны.")

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
         "NPS", "CSI", "Преподаватели", "Комментарии", "Кодбук"]
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
