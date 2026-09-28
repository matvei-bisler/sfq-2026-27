"""Process the autumn start-of-year survey export into the dashboard dataset.

Based on notebooks/clean_data.ipynb, with these changes:
  * columns are mapped by question text instead of fixed number offsets, so the
    script survives renumbering / new course blocks next year;
  * write-in answers ("Впишите свой ответ") go to `<var>_other`, while `<var>`
    keeps only the predefined options (+ "Другое (свой вариант)");
  * personal / restricted data (contact, special-conditions details) is split
    into data/private/ (git-ignored) — the question promises restricted access;
  * test rows are excluded; school/level are added from the programme registry.

Input:  data/raw/autumn/autumn.csv          (Testograf export, ';'-separated)
Output: data/processed/autumn_<year>.csv    (public, used by the dashboard)
        data/processed/autumn_<year>_meta.json
        data/private/autumn_<year>_special_conditions.csv
        docs/codebook_autumn.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from survey_utils import (
    ACADEMIC_YEAR,
    DOCS_DIR,
    PRIVATE_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    clean_text,
    enrich_from_registry,
    normalize_ranges,
    strip_qnum,
)

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "autumn" / "autumn.csv"

WRITE_IN = "Впишите свой ответ"
OTHER_OPTION = "Другое (свой вариант)"
NO_GAMES_OPTION = "Не играю в игры этой категории"

# Question-text prefix (lower-case, number stripped) -> variable name.
# Order matters only for readability; the first matching prefix wins.
QUESTION_MAP: list[tuple[str, str]] = [
    ("на каком курсе", "year"),
    ("на какой программе", "program"),
    ("кто принимал решение", "decision"),
    ("участвовали ли вы в олимпиадах", "olympiads"),
    ("как вы оцениваете свой уровень английского", "english_level"),
    ("что, по вашим ожиданиям, будет самым трудным", "difficulties"),
    ("сколько часов в неделю вы рассчитываете", "hours_planned"),
    ("сколько часов в неделю у вас уходило", "hours_last_year"),
    ("планируете ли вы работать", "work_plans"),
    ("работа связана с профилем", "work_related"),
    ("сколько времени занимает дорога", "travel_time"),
    ("какие у вас ожидания", "expectations"),
    ("что для вас в приоритете", "priorities"),
    ("насколько вы уверены", "confidence"),
    ("думали ли вы в прошлом году о переводе", "thoughts_transfer"),
    ("что на это повлияло", "transfer_influences"),
    ("чего вам не хватает в программе", "missing"),
    ("нуждаетесь ли вы в специальных условиях", "special_conditions"),
    ("какие условия вам нужны", "conditions_desc"),
    ("если вы хотите, чтобы с вами связался", "contact"),
    ("что из этого про вас", "about_you"),
    ("во что вы играете хотя бы раз в неделю? (командные", "games_team"),
    ("во что вы играете хотя бы раз в неделю? (одиночные", "games_solo"),
    ("во что вы играете хотя бы раз в неделю? (мобильные", "games_mobile"),
    ("во что вы играете хотя бы раз в неделю? (интеллектуальные", "games_intellectual"),
    ("было бы вам интересно принять участие в турнире", "tournament_interest"),
    ("в каких внеучебных активностях", "extracurricular"),
]

MULTI_VARS = [
    "difficulties",
    "priorities",
    "thoughts_transfer",
    "missing",
    "about_you",
    "games_team",
    "games_solo",
    "games_mobile",
    "games_intellectual",
]
GAME_VARS = {"games_team", "games_solo", "games_mobile", "games_intellectual"}
SINGLE_VARS = [
    "year",
    "program",
    "decision",
    "olympiads",
    "english_level",
    "hours_planned",
    "hours_last_year",
    "work_plans",
    "work_related",
    "travel_time",
    "tournament_interest",
]
RANGE_VARS = {"hours_planned", "hours_last_year", "work_plans", "travel_time"}
TEXT_VARS = ["expectations", "transfer_influences", "extracurricular"]
PRIVATE_VARS = ["special_conditions", "conditions_desc", "contact"]

# Ordinal answer scales (dashboard keeps this order on axes).
ORDERS: dict[str, list[str]] = {
    "year": ["1 курс", "2 курс", "3 курс", "4 курс", "5 курс"],
    "hours_planned": ["0", "1–5", "6–10", "11–15", "16–20", "21–25", "более 25"],
    "hours_last_year": ["0", "1–5", "6–10", "11–15", "16–20", "21–25", "более 25"],
    "work_plans": [
        "не планирую",
        "до 10 часов в неделю",
        "11–20 часов в неделю",
        "21–30 часов в неделю",
        "более 30 часов в неделю",
        "уже работаю полный день",
        "пока не знаю",
    ],
    "travel_time": ["до 30 минут", "30–60 минут", "60–90 минут", "более 90 минут"],
    "english_level": [
        "Практически не читаю и не говорю",
        "Читаю и говорю с трудом, уходит много времени",
        "Читаю и говорю со словарём, понимаю основное",
        "Читаю и говорю свободно",
        "Затрудняюсь оценить",
    ],
    "olympiads": [
        "Нет",
        "Участвовал(а) без результата",
        "Призёр или победитель школьного либо регионального этапа",
        "Призёр или победитель ВсОШ либо перечневой олимпиады",
    ],
    "special_needs": ["Нет", "Да", "Другое (свободный ответ)", "Предпочли не отвечать"],
    "retention_status": [
        "Не думали",
        "Уже перевелись",
        "Думали в прошлом году",
        "Думают сейчас",
        "Предпочли не отвечать",
    ],
    "hours_change": ["Меньше, чем в прошлом году", "Столько же, сколько в прошлом году", "Больше, чем в прошлом году"],
    "plans_to_work": ["Работают или планируют работать", "Не планируют работать", "Пока не знают"],
    "confidence": ["1", "2", "3", "4", "5"],
}

HOURS_MID = {"0": 0, "1–5": 3, "6–10": 8, "11–15": 13, "16–20": 18, "21–25": 23, "более 25": 28}
TRAVEL_MID = {"до 30 минут": 15, "30–60 минут": 45, "60–90 минут": 75, "более 90 минут": 105}

# "I don't play" in all its spellings: «ни во что», «не во что», «ни чо что»,
# «не играю», «тоже не инграю», «нет», «хз», «-», emoji-only …
NOTHING_RE = re.compile(
    r"^\s*(ни|не|но)\s*(во|в|чо)?\s*(что|чо|ч[её]м)\b"
    r"|не\s*и\w{0,2}граю"
    r"|^\s*(ничего|никак|ни\s*в\s*как).*$"
    r"|^\s*(не+т|no|nope|хз|-+|_+|\.+)\s*[!.]*\s*$"
    r"|^[^\w]*$",
    re.IGNORECASE,
)
TEST_RE = re.compile(r"\bтест\b|ыкцкручеру", re.IGNORECASE)


def match_var(question: str) -> str | None:
    q = question.lower()
    for prefix, var in QUESTION_MAP:
        if q.startswith(prefix):
            return var
    return None


def parse_columns(columns: list[str]) -> tuple[list[dict], dict[str, str], dict[str, list[str]]]:
    """Return per-column info, the question text per var and the option order per multi var."""
    info, qtext, options = [], {}, {}
    unmapped = []
    for col in columns:
        if not re.match(r"^\d+\.", col):
            continue
        body = strip_qnum(col)
        question, option = (body.split(" - ", 1) + [None])[:2]
        var = match_var(question.strip())
        if var is None:
            unmapped.append(col)
            continue
        option = clean_text(option) if option is not None else None
        info.append({"col": col, "var": var, "option": option})
        qtext.setdefault(var, clean_text(question))
        if var in MULTI_VARS and option and option != WRITE_IN:
            options.setdefault(var, [])
            if option not in options[var]:
                options[var].append(option)
    if unmapped:
        print(f"WARNING: {len(unmapped)} unmapped question columns (ignored):")
        for c in unmapped:
            print(f"  {c[:120]}")
    return info, qtext, options


def split_write_in(text: str) -> list[str]:
    parts = re.split(r"[,;\n/]+|\s+и\s+", text)
    return [p.strip(" .-") for p in parts if p and p.strip(" .-")]


def dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def build_rows(df_raw: pd.DataFrame, info: list[dict], options: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for _, raw in df_raw.iterrows():
        rec: dict = {v: np.nan for v in SINGLE_VARS + TEXT_VARS + PRIVATE_VARS + ["confidence"]}
        multi: dict[str, list[str]] = {v: [] for v in MULTI_VARS}
        other: dict[str, list[str]] = {v: [] for v in MULTI_VARS}

        for ci in info:
            value = clean_text(raw[ci["col"]])
            if not isinstance(value, str):
                continue
            var, opt = ci["var"], ci["option"]
            if var in MULTI_VARS:
                if opt and opt != WRITE_IN:
                    multi[var].append(opt)
                    continue
                # write-in answer
                if var in GAME_VARS:
                    known = {o.lower(): o for o in options.get(var, [])}
                    # "Раз в неделю не играю, а так Minecraft" -> check each part separately
                    for item in split_write_in(value) or [value]:
                        if NOTHING_RE.search(item):
                            multi[var].append(NO_GAMES_OPTION)
                        elif item.lower() in known:
                            multi[var].append(known[item.lower()])
                        else:
                            other[var].append(item[0].upper() + item[1:])
                else:
                    multi[var].append(OTHER_OPTION)
                    other[var].append(value)
            elif pd.isna(rec.get(var)):
                rec[var] = value

        for var in MULTI_VARS:
            rec[var] = "; ".join(dedupe(multi[var]))
            rec[f"{var}_other"] = "; ".join(dedupe(other[var]))

        rec["resp_id"] = raw["Номер ответа"]
        rec["date"] = raw["Дата ответа"]
        rec["duration_sec"] = raw["Время выполнения"]
        rows.append(rec)
    return pd.DataFrame(rows)


def classify_special_needs(value) -> str | float:
    if not isinstance(value, str):
        return np.nan
    v = value.strip()
    if v == "Нет" or re.match(r"^пока\s*(что\s*)?нет\b", v, re.IGNORECASE):
        return "Нет"
    if v.startswith("Да"):
        return "Да"
    if v == "Предпочитаю не отвечать":
        return "Предпочли не отвечать"
    return "Другое (свободный ответ)"


def retention_status(value: str) -> str | float:
    if not isinstance(value, str) or not value:
        return np.nan
    items = value.split("; ")
    if "Думаю об этом сейчас" in items:
        return "Думают сейчас"
    if any(i.startswith("Думал(а)") for i in items):
        return "Думали в прошлом году"
    if any(i.startswith("Уже перев") for i in items):
        return "Уже перевелись"
    if "Нет, не думал(а)" in items:
        return "Не думали"
    if "Предпочитаю не отвечать" in items:
        return "Предпочли не отвечать"
    return np.nan


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hours_planned_num"] = out["hours_planned"].map(HOURS_MID)
    out["hours_last_year_num"] = out["hours_last_year"].map(HOURS_MID)
    diff = out["hours_planned_num"] - out["hours_last_year_num"]
    out["hours_change"] = np.select(
        [diff > 0, diff < 0, diff == 0],
        ["Больше, чем в прошлом году", "Меньше, чем в прошлом году", "Столько же, сколько в прошлом году"],
        default=None,
    )
    out["hours_change"] = out["hours_change"].where(diff.notna(), np.nan)
    out["travel_min_num"] = out["travel_time"].map(TRAVEL_MID)
    out["plans_to_work"] = out["work_plans"].map(
        lambda v: np.nan if not isinstance(v, str)
        else "Не планируют работать" if v == "не планирую"
        else "Пока не знают" if v == "пока не знаю"
        else "Работают или планируют работать"
    )
    out["special_needs"] = out["special_conditions"].map(classify_special_needs)
    out["retention_status"] = out["thoughts_transfer"].map(retention_status)
    out["retention_risk"] = out["retention_status"].map(
        {
            "Думают сейчас": 1.0,
            "Думали в прошлом году": 1.0,
            "Не думали": 0.0,
            "Уже перевелись": 0.0,
        }
    )
    return out


def write_codebook(df: pd.DataFrame, meta: dict, path: Path, n_raw: int, n_test: int) -> None:
    lines = [
        f"# Кодификатор осенней анкеты {ACADEMIC_YEAR} (`autumn_{ACADEMIC_YEAR}.csv`)",
        "",
        "Анкета начала учебного года, собранная в Тестографе. Одна строка — один респондент.",
        f"Исходных ответов — **{n_raw}**, исключено тестовых — **{n_test}**, в наборе данных — **{len(df)}**.",
        "",
        "Вопросы анкеты зависят от курса. Первокурсникам задавались вопросы о поступлении "
        "(кто принимал решение, олимпиады, английский язык), студентам 2-го курса и старше — о прошлом "
        "учебном годе (самоподготовка, мысли о переводе или отчислении, чего не хватает в программе).",
        "",
        "## Правила обработки",
        "- Вопросы с несколькими вариантами ответа: выбранные варианты перечислены через `; `, пустое значение — ничего не выбрано.",
        f"- Собственные ответы респондентов (вариант «{WRITE_IN}») вынесены в колонки `<переменная>_other`; "
        f"в самой переменной они учтены как «{OTHER_OPTION}».",
        f"- В вопросах об играх ответы вроде «не играю» и «ни во что» объединены в вариант «{NO_GAMES_OPTION}», "
        "а названия игр из собственных ответов — в колонках `<переменная>_other`.",
        "- Контакты и описания специальных условий **не входят** в публичный набор данных: в анкете обещан "
        "ограниченный доступ к ним (см. `data/private/`).",
        "",
        "## Служебные и производные переменные",
        "",
        "| Переменная | Описание |",
        "|---|---|",
        "| `resp_id` | Номер ответа в выгрузке |",
        "| `date` | Дата и время ответа |",
        "| `duration_sec` | Время заполнения анкеты, с |",
        "| `school`, `level` | Школа и уровень программы (по файлу `data/reference/programs_registry.csv`) |",
        "| `flag_long_time` | Анкету заполняли дольше часа (скорее всего, страницу оставили открытой) |",
        "| `hours_planned_num`, `hours_last_year_num` | Середина интервала часов самоподготовки в неделю: 0, 3, 8, 13, 18, 23, 28 |",
        "| `hours_change` | Планируют ли студенты тратить на самоподготовку больше, столько же или меньше часов, чем в прошлом году (2-й курс и старше) |",
        "| `travel_min_num` | Середина интервала времени в дороге, мин: 15, 45, 75, 105 |",
        "| `plans_to_work` | Работают или планируют работать / не планируют / пока не знают |",
        "| `special_needs` | Нуждаются ли в специальных условиях: «Нет», «Да», «Другое (свободный ответ)», «Предпочли не отвечать» |",
        "| `retention_status` | Итоговая категория по вопросу о переводе, отчислении и академическом отпуске (2-й курс и старше) |",
        "| `retention_risk` | 1 — думали или думают об уходе либо переводе, 0 — не думали; пусто — не ответили или 1-й курс |",
        "",
        "## Вопросы анкеты",
        "",
    ]
    type_names = {
        "single": "один вариант ответа",
        "multi": "несколько вариантов ответа",
        "text": "открытый вопрос",
        "numeric": "шкала от 1 до 5",
    }
    for var, m in meta["variables"].items():
        lines.append(f"### `{var}`")
        lines.append("")
        lines.append(f"**Вопрос:** {m['question']}  ")
        lines.append(f"**Тип:** {type_names.get(m['type'], m['type'])}  ")
        s = df[var] if var in df.columns else pd.Series(dtype=object)
        if m["type"] == "multi":
            s = s[s.fillna("") != ""]
            lines.append(f"**Ответили:** {len(s)}")
            lines.append("")
            counts = s.str.split("; ").explode().value_counts()
            lines += ["| Вариант ответа | Выбрали |", "|---|---|"]
            for opt in m.get("options", []):
                lines.append(f"| {opt} | {int(counts.get(opt, 0))} |")
            if f"{var}_other" in df.columns:
                n_other = int((df[f"{var}_other"].fillna("") != "").sum())
                lines.append("")
                lines.append(f"Собственных ответов (`{var}_other`): {n_other}.")
        elif m["type"] in ("single", "numeric"):
            s = s.dropna()
            lines.append(f"**Ответили:** {len(s)}")
            lines.append("")
            vc = s.astype(str).value_counts()
            order = [o for o in m.get("options", []) if o in vc.index] + [o for o in vc.index if o not in m.get("options", [])]
            lines += ["| Ответ | Число ответов |", "|---|---|"]
            for o in order[:30]:
                lines.append(f"| {str(o).replace('|', '/')} | {int(vc[o])} |")
        else:
            s = s.dropna()
            lines.append(f"**Ответили:** {len(s)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=str(RAW_PATH), help="Путь к выгрузке Тестографа (csv, ';').")
    parser.add_argument("--year", default=ACADEMIC_YEAR, help="Учебный год, например 2026-27.")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    df_raw = pd.read_csv(raw_path, sep=";", encoding="utf-8-sig", dtype=str)
    print(f"Raw: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols  ({raw_path.name})")

    info, qtext, options = parse_columns(list(df_raw.columns))
    df = build_rows(df_raw, info, options)

    for var in RANGE_VARS:
        df[var] = df[var].map(normalize_ranges)
    df["olympiads"] = df["olympiads"].map(clean_text)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["resp_id"] = pd.to_numeric(df["resp_id"], errors="coerce").astype("Int64")
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S", errors="coerce")

    # ── test / junk rows ────────────────────────────────────────────────────
    text_blob = df[TEXT_VARS + ["conditions_desc"]].fillna("").astype(str).agg(" ".join, axis=1)
    is_test = text_blob.str.contains(TEST_RE)
    if is_test.any():
        print(f"Excluded test rows: {int(is_test.sum())}")
        print(df.loc[is_test, ["resp_id", "year", "program", "expectations"]].to_string(index=False))
    n_raw = len(df)
    df = df[~is_test].copy()

    df["flag_long_time"] = df["duration_sec"] > 3600
    df = enrich_from_registry(df)
    df = add_derived(df)

    # Values outside the known scales are worth a look before publishing.
    for var in ["hours_planned", "hours_last_year", "work_plans", "travel_time", "english_level", "olympiads"]:
        unknown = sorted(set(df[var].dropna()) - set(ORDERS[var]))
        if unknown:
            print(f"WARNING: {var}: values outside the known scale: {unknown}")

    # ── private split ───────────────────────────────────────────────────────
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    private = df[["resp_id", "date", "year", "program", "school"] + PRIVATE_VARS]
    # keep only rows with something for the study office to act on
    private = private[
        private["special_conditions"].notna() & (private["special_conditions"] != "Нет")
        | private["conditions_desc"].notna()
        | private["contact"].notna()
    ]
    private_path = PRIVATE_DIR / f"autumn_{args.year}_special_conditions.csv"
    private.to_csv(private_path, index=False, encoding="utf-8-sig")

    public_cols = (
        ["resp_id", "date", "duration_sec", "flag_long_time", "year", "program", "school", "level"]
        + [v for v in SINGLE_VARS if v not in ("year", "program")]
        + ["confidence"]
        + [c for v in MULTI_VARS for c in (v, f"{v}_other")]
        + TEXT_VARS
        + [
            "special_needs",
            "hours_planned_num",
            "hours_last_year_num",
            "hours_change",
            "travel_min_num",
            "plans_to_work",
            "retention_status",
            "retention_risk",
        ]
    )
    public = df[public_cols].sort_values(["year", "program", "resp_id"]).reset_index(drop=True)
    public["date"] = public["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"autumn_{args.year}.csv"
    public.to_csv(out_path, index=False, encoding="utf-8")

    # ── meta (question texts, option orders) ────────────────────────────────
    variables: dict[str, dict] = {}
    for _, var in QUESTION_MAP:
        if var in PRIVATE_VARS or var not in qtext:
            continue
        if var in MULTI_VARS:
            opts = list(options.get(var, []))
            if var in GAME_VARS:
                opts.append(NO_GAMES_OPTION)
            if (public[var].str.contains(re.escape(OTHER_OPTION))).any():
                opts.append(OTHER_OPTION)
            variables[var] = {"question": qtext[var], "type": "multi", "options": opts}
        elif var in TEXT_VARS:
            variables[var] = {"question": qtext[var], "type": "text"}
        elif var == "confidence":
            variables[var] = {"question": qtext[var], "type": "numeric", "options": ORDERS["confidence"]}
        else:
            variables[var] = {"question": qtext[var], "type": "single", "options": ORDERS.get(var, [])}
        courses = sorted(public.loc[public[var].fillna("").astype(str) != "", "year"].dropna().unique())
        variables[var]["courses"] = courses
    variables["special_needs"] = {
        "question": qtext.get("special_conditions", "Нуждаетесь ли вы в специальных условиях организации образовательного процесса?"),
        "type": "single",
        "options": ORDERS["special_needs"],
        "courses": sorted(public["year"].dropna().unique()),
    }
    for derived, question in [
        ("retention_status", "Производная: думали ли студенты о переводе, отчислении или академическом отпуске"),
        ("hours_change", "Производная: самоподготовка в этом году по плану по сравнению с прошлым годом"),
        ("plans_to_work", "Производная: планируют ли студенты работать в этом учебном году"),
    ]:
        variables[derived] = {
            "question": question,
            "type": "single",
            "options": ORDERS.get(derived, sorted(public[derived].dropna().unique())),
            "courses": sorted(public.loc[public[derived].notna(), "year"].dropna().unique()),
        }
    meta = {"academic_year": args.year, "n": len(public), "variables": variables}
    meta_path = PROCESSED_DIR / f"autumn_{args.year}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    write_codebook(public, meta, DOCS_DIR / "codebook_autumn.md", n_raw, int(is_test.sum()))

    print(f"\nSaved {out_path.relative_to(PROJECT_ROOT)} ({public.shape[0]} rows × {public.shape[1]} cols)")
    print(f"Saved {meta_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {private_path.relative_to(PROJECT_ROOT)} ({len(private)} rows, PRIVATE — not committed)")
    print("Saved docs/codebook_autumn.md")
    print("\nОтветов по курсам:")
    print(public["year"].value_counts().sort_index().to_string())
    print("\nОтветов по школам:")
    print(public["school"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
