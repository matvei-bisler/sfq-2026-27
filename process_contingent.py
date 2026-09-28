"""Parse Контингент.xlsx (enrollment) into a clean reference for response-rate analysis.

Input:  data/reference/Контингент.xlsx — columns «Программа» ('Кинопроизводство 2 курс')
        and «Контингент» (number of students), same format as in SFQ 2025-26.
Output: data/reference/contingent.csv with columns program, year, school, contingent.

Programme names are matched against data/reference/programs_registry.csv; rows that
do not match (preparatory programmes, intensives, …) are listed and skipped.
The same snapshot is used for all three waves of the year.
"""
from __future__ import annotations

import re

import pandas as pd

from survey_utils import REFERENCE_DIR, load_registry, normalize_key

CONTINGENT_XLSX = REFERENCE_DIR / "Контингент.xlsx"

# Enrollment-file spelling -> registry programme name.
ALIAS = {
    "foundation art and design/ дизайн": "Foundation Art and Design",
    "цифровой маркетинг": "Менеджмент в креативных индустриях (сетевая программа)",
}


def split_year(name: str) -> tuple[str, str | None]:
    """'Кинопроизводство 2 курс' -> ('Кинопроизводство', '2 курс')."""
    m = re.search(r"(\d)\s*курс\s*$", str(name).strip())
    if m:
        return re.sub(r"\d\s*курс\s*$", "", str(name)).strip(), f"{m.group(1)} курс"
    return str(name).strip(), None


def main() -> None:
    if not CONTINGENT_XLSX.exists():
        print(f"ERROR: {CONTINGENT_XLSX} not found")
        return
    cont = pd.read_excel(CONTINGENT_XLSX).dropna(subset=["Контингент"])
    reg = load_registry()
    by_key = dict(zip(reg["program_key"], reg["Программа"]))
    school = dict(zip(reg["program_key"], reg["Школа"]))

    rows, unmatched = [], []
    for _, r in cont.iterrows():
        prog_raw, year = split_year(r["Программа"])
        target = ALIAS.get(normalize_key(prog_raw)) or by_key.get(normalize_key(prog_raw))
        if target is None:
            unmatched.append((r["Программа"], r["Контингент"]))
            continue
        rows.append(
            {
                "program": target,
                "year": year,
                "school": school.get(normalize_key(target)),
                "contingent": int(r["Контингент"]),
            }
        )

    out = (
        pd.DataFrame(rows)
        .groupby(["program", "year"], dropna=False, as_index=False)
        .agg({"school": "first", "contingent": "sum"})
    )
    out_path = REFERENCE_DIR / "contingent.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} rows, {int(out['contingent'].sum())} students)")
    print(out.groupby("program")["contingent"].sum().to_string())

    missing = sorted(set(reg["Программа"]) - set(out["program"]))
    if missing:
        print("\nПрограммы реестра БЕЗ контингента:")
        for p in missing:
            print(f"  {p}")
    print(f"\nНесопоставленные строки контингента: {len(unmatched)}")
    for name, n in unmatched:
        print(f"  {int(n):>4}  {name}")


if __name__ == "__main__":
    main()
