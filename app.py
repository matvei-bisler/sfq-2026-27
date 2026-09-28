"""SFQ 2026-27 dashboard (Streamlit).

Three survey waves per academic year:
  * Осень — анкета начала учебного года (свой опросник; сравнивается по годам);
  * Зима / Весна — SFQ по итогам семестра (один опросник; сравниваются между собой).
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SFQ 2026-27 — дашборд",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.autumn import render_autumn  # noqa: E402  (set_page_config must run first)
from dashboard.common import PAGE_CSS  # noqa: E402
from dashboard.sfq import WAVES, render_sfq, wave_available  # noqa: E402

st.markdown(PAGE_CSS, unsafe_allow_html=True)

MODES = {
    "autumn": "Осень",
    "winter": "Зима (1-й семестр)",
    "spring": "Весна (2-й семестр)",
    "both": "Зима и весна",
}


def mode_label(mode: str) -> str:
    if mode in WAVES and not wave_available(mode):
        return f"{MODES[mode]} — нет данных"
    if mode == "both" and not all(wave_available(w) for w in WAVES):
        return f"{MODES[mode]} — нет данных"
    return MODES[mode]


def main() -> None:
    st.sidebar.header("Анкета")
    mode = st.sidebar.radio(
        "Выберите данные",
        list(MODES),
        index=0,
        format_func=mode_label,
        help="Осенняя анкета — отдельный опросник, её результаты сравниваются по годам. "
        "Зимняя и весенняя анкеты (SFQ) содержат одни и те же вопросы, их результаты сравниваются между собой.",
    )
    st.sidebar.divider()
    if mode == "autumn":
        render_autumn()
    else:
        render_sfq(mode)


if __name__ == "__main__":
    main()
