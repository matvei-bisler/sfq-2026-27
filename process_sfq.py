"""Process winter / spring SFQ surveys (one export per programme) into dashboard CSVs.

Generalised from process_sem2.py of SFQ 2025-26: the same questionnaire is expected
in winter (end of semester 1) and spring (end of semester 2). Raw exports (xlsx or
Testograf csv) go to data/raw/<wave>/, one file per programme; the file name is the
programme name (suffixes like "(2 сем)", "(зима)" are stripped).

Usage:
    python process_sfq.py winter
    python process_sfq.py spring

Outputs:
    data/processed/sfq_<wave>.csv            — one row per respondent, Latin columns
    data/processed/sfq_<wave>_teachers.csv   — long format: one teacher rating per row
"""
from __future__ import annotations

import argparse
import re
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

from survey_utils import METADATA_COLS, PROCESSED_DIR, PROJECT_ROOT, enrich_from_registry, strip_qnum

# Coalescing duplicate (year-branched) survey columns with combine_first emits a
# cosmetic pandas FutureWarning about empty-entry concatenation; output is correct.
warnings.filterwarnings(
    "ignore",
    message="The behavior of array concatenation with empty entries is deprecated",
    category=FutureWarning,
)

WAVES = {"winter": "Зима (1 семестр)", "spring": "Весна (2 семестр)"}

# Export file name (after suffix stripping) -> programme name used everywhere else.
PROGRAM_NAME_OVERRIDES = {
    "Менеджмент в креативных индустриях (Цифровой маркетинг)":
        "Менеджмент в креативных индустриях (сетевая программа)",
}

WAVE_SUFFIX_RE = re.compile(r"\s*\(\s*(?:\d\s*сем|зима|весна)\s*\)\s*$", re.IGNORECASE)


def is_teacher_rating_col(col: str) -> bool:
    s = strip_qnum(col)
    return (
        "оцените работу преподавателей" in s.lower()
        or "оцените работу преподавателей по критериям" in s.lower()
    ) and " - " in s


def is_individual_teacher_col(col: str) -> bool:
    s = unicodedata.normalize("NFKC", strip_qnum(col)).strip()
    low = s.lower()
    if " - " not in s:
        return False
    return (
        low.startswith("оцените преподавателей по дисциплине")
        or low.startswith("оцените преподавателей дисциплины")
        or low.startswith("оцените преподавателей по ")
        or low.startswith("оцените преподавателя по ")
    )


def get_teacher_name(col: str) -> str:
    s = unicodedata.normalize("NFKC", strip_qnum(col))
    return s.rsplit(" - ", 1)[-1].strip()


def is_course_year_col(col: str) -> bool:
    return "выберите курс" in col.lower()


def is_choose_program_col(col: str) -> bool:
    return "выберите программу" in col.lower()


def canonical(col: str) -> str:
    if col in METADATA_COLS:
        return col
    # Mapped questions win over the teacher heuristics: «Оцените преподавателей по
    # общеобразовательным дисциплинам - Критическое мышление» is a gen_ed_* rating,
    # not a teacher called «Критическое мышление» (the sem-1 questionnaire has these).
    if strip_qnum(col) in COL_MAP:
        return strip_qnum(col)
    if is_course_year_col(col):
        return "Курс обучения"
    if is_choose_program_col(col):
        return "Программа (из анкеты)"
    if is_teacher_rating_col(col):
        return f"Оценка преподавателя: {get_teacher_name(col)}"
    if is_individual_teacher_col(col):
        return f"Оценка преподавателя: {get_teacher_name(col)}"
    return strip_qnum(col)


def extract_program_name(filename: str) -> str:
    stem = unicodedata.normalize("NFC", Path(filename).stem)
    name = re.sub(r"\s+", " ", WAVE_SUFFIX_RE.sub("", stem)).strip()
    # "(Цифровой маркетинг 2 сем)" -> "(Цифровой маркетинг)"
    name = re.sub(r"\s*\d\s*сем\s*\)", ")", name)
    return PROGRAM_NAME_OVERRIDES.get(name, name)


def read_export(filepath: Path) -> pd.DataFrame:
    if filepath.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    # Testograf csv: sniff the separator (',' last year, ';' in the autumn export)
    return pd.read_csv(filepath, sep=None, engine="python", encoding="utf-8-sig")


def load_program_file(filepath: Path) -> pd.DataFrame:
    program_name = extract_program_name(filepath.name)
    df = read_export(filepath)

    groups: dict[str, list[str]] = defaultdict(list)
    for col in df.columns:
        groups[canonical(col)].append(col)

    out_series = {}
    for cname, orig_cols in groups.items():
        s = df[orig_cols[0]].copy()
        for c in orig_cols[1:]:
            other = df[c]
            if other.notna().any():
                s = s.combine_first(other)
        out_series[cname] = s

    out = pd.DataFrame(out_series)
    out.insert(0, "Программа", program_name)
    if "Курс обучения" not in out.columns:
        out.insert(1, "Курс обучения", None)
    else:
        out["Курс обучения"] = out["Курс обучения"].astype(object)
    if "Программа (из анкеты)" in out.columns:
        out = out.drop(columns=["Программа (из анкеты)"])
    return out


COL_MAP = {
    "Программа": "program",
    "Курс обучения": "year",
    "Номер ответа": "resp_id",
    "Дата ответа": "date",
    "Время выполнения": "duration_sec",
    "IP - адрес": "ip",
    "Источник распространения": "source",
    "Операционная система": "os",
    "Браузер": "browser",
    "Устройство": "device",
    "Пожалуйста, оцените вашу удовлетворенность образовательным процессом в целом":
        "satisf_overall",
    "Насколько вы в целом удовлетворены работой преподавателей в течение семестра?":
        "satisf_teachers",
    "Насколько ваши ожидания от программы при поступлении совпали с опытом обучения?":
        "expect_match",
    "Пожалуйста, оцените вашу удовлетворенность преподавательским составом на программе - У меня была достаточная поддержка преподавательской команды в профессиональном развитии - Оценка":
        "fac_support_score",
    "Пожалуйста, оцените вашу удовлетворенность преподавательским составом на программе - У меня была достаточная поддержка преподавательской команды в профессиональном развитии - Важность критерия":
        "fac_support_imp",
    "Пожалуйста, оцените вашу удовлетворенность преподавательским составом на программе - Преподавательская команда объясняет сложные вещи понятным языком - Оценка":
        "fac_clarity_score",
    "Пожалуйста, оцените вашу удовлетворенность преподавательским составом на программе - Преподавательская команда объясняет сложные вещи понятным языком - Важность критерия":
        "fac_clarity_imp",
    "Пожалуйста, оцените вашу удовлетворенность куратором на программе - Куратор своевременно сообщает необходимую информацию об образовательном процессе - Оценка":
        "cur_timely_score",
    "Пожалуйста, оцените вашу удовлетворенность куратором на программе - Куратор своевременно сообщает необходимую информацию об образовательном процессе - Важность критерия":
        "cur_timely_imp",
    "Пожалуйста, оцените вашу удовлетворенность куратором на программе - Куратор готов оказать помощь, если у меня появляются вопросы или проблемы - Оценка":
        "cur_help_score",
    "Пожалуйста, оцените вашу удовлетворенность куратором на программе - Куратор готов оказать помощь, если у меня появляются вопросы или проблемы - Важность критерия":
        "cur_help_imp",
    "Здесь вы можете оставить более подробную обратную связь о кураторе":
        "cur_comment",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне было понятно содержание дисциплин, которые я изучал(а) в течение семестра, и их взаимосвязь между собой - Оценка":
        "prog_clarity_score",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне было понятно содержание дисциплин, которые я изучал(а) в течение семестра, и их взаимосвязь между собой - Важность критерия":
        "prog_clarity_imp",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Я мог(-ла) выполнить задания преподавателей в нужные сроки в достаточном объеме - Оценка":
        "prog_deadlines_score",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Я мог(-ла) выполнить задания преподавателей в нужные сроки в достаточном объеме - Важность критерия":
        "prog_deadlines_imp",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне было понятно, как изучаемые дисциплины связаны с моей сферой профессиональной деятельности - Оценка":
        "prog_relevance_score",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне было понятно, как изучаемые дисциплины связаны с моей сферой профессиональной деятельности - Важность критерия":
        "prog_relevance_imp",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне кажется, что за прошедший семестр количество занятий в неделю было оптимальным - Оценка":
        "prog_workload_score",
    "Пожалуйста, оцените вашу удовлетворенность программой по каждому из параметров - Мне кажется, что за прошедший семестр количество занятий в неделю было оптимальным - Важность критерия":
        "prog_workload_imp",
    "Здесь вы можете оставить более подробную обратную связь о программе":
        "prog_comment",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор уважительно взаимодействовал со мной - Оценка":
        "coord_respect_score",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор уважительно взаимодействовал со мной - Важность критерия":
        "coord_respect_imp",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Мои обращения к координатору приводили к ожидаемому результату - Оценка":
        "coord_results_score",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Мои обращения к координатору приводили к ожидаемому результату - Важность критерия":
        "coord_results_imp",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор сообщает необходимую информацию об образовательном процессе своевременно - Оценка":
        "coord_timely_score",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор сообщает необходимую информацию об образовательном процессе своевременно - Важность критерия":
        "coord_timely_imp",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор готов оказать помощь, если возникают вопросы или проблемы - Оценка":
        "coord_help_score",
    "Пожалуйста, оцените вашу удовлетворенность взаимодействием с координатором Учебного отдела программы - Координатор готов оказать помощь, если возникают вопросы или проблемы - Важность критерия":
        "coord_help_imp",
    "Здесь вы можете оставить более подробную обратную связь о взаимодействии с координатором Учебного отдела":
        "coord_comment",
    "Преподаватели своевременно представили критерии оценивания работ":
        "assess_criteria_timely",
    "Преподаватели ясно определили порядок сдачи и оценивания работ":
        "assess_order_clear",
    "Преподаватели проводили оценивание работ в соответствии с заявленными критериями":
        "assess_consistent",
    "Здесь вы можете оставить подробную обратную связь про оценивание на программе":
        "assess_comment",
    "Здесь вы можете оставить более подробную обратную связь о преподавательской команде":
        "fac_comment",
    "Оцените преподавателей по общеобразовательным дисциплинам - Критическое мышление":
        "gen_ed_critical_thinking",
    "Оцените преподавателей по общеобразовательным дисциплинам - История России":
        "gen_ed_history",
    "Оцените преподавателей по общеобразовательным дисциплинам - Иностранный язык":
        "gen_ed_foreign_lang",
    "Оцените преподавателей по общеобразовательным дисциплинам - Безопасность жизнедеятельности":
        "gen_ed_safety",
    "Оцените преподавателей по общеобразовательным дисциплинам - Основы российской государственности":
        "gen_ed_statehood",
    "Оцените преподавателей по гуманитарным дисциплинам: - Критическое мышление":
        "hum_critical_thinking",
    "Оцените преподавателей по гуманитарным дисциплинам: - История России":
        "hum_history",
    "Оцените преподавателей по гуманитарным дисциплинам: - Основы российской государственности":
        "hum_statehood",
    "Оцените преподавателей по гуманитарным дисциплинам: - Иностранный язык":
        "hum_foreign_lang",
    "Оцените преподавателей по гуманитарным дисциплинам: - Философия":
        "hum_philosophy",
    "Оцените преподавателей по гуманитарным дисциплинам: - Теория и практика коммуникации":
        "hum_communication",
    "Оставьте комментарий отдельно о лекционных и семинарских занятиях (укажите преподавателя или семинарский трек). Насколько разнообразными и полезными были семинарские треки:":
        "seminar_comment",
    "Оцените доступность и работу библиотеки": "infra_library",
    "Оцените доступность и качество работы сервиса Wellbeing": "infra_wellbeing",
    "Оцените доступность и качество еды на кампусе": "infra_food",
    "Оцените работоспособность программного обеспечения в аудиториях": "infra_software",
    "Оцените работоспособность оборудования в аудиториях": "infra_equipment",
    "Оцените комфорт пребывания в аудиториях": "infra_classrooms",
    "Оцените комфорт пребывания в мастерских / ресурсных центрах / репетиционных комнатах":
        "infra_workshops",
    "Насколько вероятно, что вы порекомендуете школу друзьям, коллегам или родным, которые планируют обучение":
        "nps",
    "Есть ли у вас дополнительные комментарии или предложения для нас?":
        "comment_final",
    "Насколько дисциплины предыдущего семестра помогли вам в освоении учебного материала текущего семестра?":
        "prev_sem_relevance",
    "Насколько уверенно вы можете применить полученные знания и навыки в реальной профессиональной практике?":
        "skill_confidence",
    "Как вы планируете продолжать свое профессиональное развитие после выпуска? - Продолжение обучения в магистратуре или дополнительном образовании":
        "postgrad_masters",
    "Как вы планируете продолжать свое профессиональное развитие после выпуска? - Работа в выбранной сфере":
        "postgrad_same_field",
    "Как вы планируете продолжать свое профессиональное развитие после выпуска? - Работа в другой сфере":
        "postgrad_other_field",
    "Как вы планируете продолжать свое профессиональное развитие после выпуска? -":
        "postgrad_other",
    # New sem2 questions
    "Вы достигли того, ради чего пришли на эту программу?":
        "goal_achieved",
    "Насколько вам было легко ориентироваться в процессе обучения (расписание, задания, дедлайны)?":
        "navigation_ease",
    "Насколько комфортно вы чувствовали себя на занятиях?":
        "class_comfort",
    # NB: "Оставьте свои контакты..." is intentionally NOT mapped — it is PII
    # (student emails / names) and is dropped, matching the semester-1 pipeline.
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wave", choices=list(WAVES), help="Волна опроса: winter или spring.")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "raw" / args.wave
    files = sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv"))
    if not files:
        print(f"ERROR: no xlsx/csv files in {raw_dir}")
        return

    print("=" * 60)
    print(f"Processing SFQ wave: {WAVES[args.wave]}")
    print("=" * 60)
    dfs = []
    for fp in files:
        df = load_program_file(fp)
        print(f"  loaded: {fp.name} → {extract_program_name(fp.name)} ({len(df)} rows)")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\nCombined: {combined.shape[0]} rows × {combined.shape[1]} columns")

    teacher_cols = sorted(c for c in combined.columns if c.startswith("Оценка преподавателя:"))
    general = combined[[c for c in combined.columns if c not in teacher_cols]]

    # ── general: Latin schema ───────────────────────────────────────────────
    unmapped = [c for c in general.columns if c not in COL_MAP]
    if unmapped:
        print(f"\nWARNING: {len(unmapped)} unmapped columns (dropped). "
              "If a question was reworded, add it to COL_MAP:")
        for c in unmapped:
            print(f"  {c!r}")
    mapped = {c: COL_MAP[c] for c in general.columns if c in COL_MAP}
    df_agg = general.rename(columns=mapped)[list(mapped.values())]
    df_agg = enrich_from_registry(df_agg)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"sfq_{args.wave}.csv"
    df_agg.to_csv(out_path, index=False)
    print(f"\nSaved {out_path.relative_to(PROJECT_ROOT)} ({df_agg.shape[0]} rows × {df_agg.shape[1]} cols)")

    # ── teachers: long format ───────────────────────────────────────────────
    id_cols = [c for c in ["Программа", "Курс обучения", "Номер ответа"] if c in combined.columns]
    df_t = (
        combined[id_cols + teacher_cols]
        .melt(id_vars=id_cols, var_name="teacher", value_name="rating")
        .dropna(subset=["rating"])
        .rename(columns={"Программа": "program", "Курс обучения": "year", "Номер ответа": "resp_id"})
    )
    df_t["teacher"] = df_t["teacher"].str.removeprefix("Оценка преподавателя: ")
    df_t["rating"] = pd.to_numeric(df_t["rating"], errors="coerce")
    df_t = enrich_from_registry(df_t)
    df_t = df_t[["program", "school", "year", "resp_id", "teacher", "rating"]]
    out_t = PROCESSED_DIR / f"sfq_{args.wave}_teachers.csv"
    df_t.to_csv(out_t, index=False)
    print(f"Saved {out_t.relative_to(PROJECT_ROOT)} ({len(df_t)} rows, {df_t['teacher'].nunique()} teachers)")

    print("\nЗаполненность колонок:")
    for c in df_agg.columns:
        print(f"  {c}: {df_agg[c].notna().sum()}/{len(df_agg)}")


if __name__ == "__main__":
    main()
