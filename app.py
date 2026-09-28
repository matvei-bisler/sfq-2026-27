"""SFQ 2026-27 dashboard (Streamlit).

Three survey waves per academic year:
  * Осень — стартовая анкета (свой опросник; сравнивается год к году);
  * Зима / Весна — SFQ по итогам семестра (один опросник; сравниваются между собой).
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SFQ 2026-27 Дашборд",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.autumn import render_autumn  # noqa: E402  (set_page_config must run first)
from dashboard.common import PAGE_CSS  # noqa: E402
from dashboard.sfq import WAVES, render_sfq, wave_available  # noqa: E402

st.markdown(PAGE_CSS, unsafe_allow_html=True)

MODES = {
    "autumn": "Осень — старт года",
    "winter": "Зима — SFQ, 1 семестр",
    "spring": "Весна — SFQ, 2 семестр",
    "both": "Зима + Весна",
}


def mode_label(mode: str) -> str:
    if mode in WAVES and not wave_available(mode):
        return f"{MODES[mode]} (нет данных)"
    if mode == "both" and not all(wave_available(w) for w in WAVES):
        return f"{MODES[mode]} (нет данных)"
    return MODES[mode]


def main() -> None:
    st.sidebar.header("Опрос")
    mode = st.sidebar.radio(
        "Выберите волну",
        list(MODES),
        index=0,
        format_func=mode_label,
        help="Осенняя анкета — отдельный опросник (сравнение год к году). "
        "Зимняя и весенняя — SFQ с одинаковыми вопросами (сравнение между семестрами).",
    )
    st.sidebar.divider()
    if mode == "autumn":
        render_autumn()
    else:
        render_sfq(mode)


if __name__ == "__main__":
    main()
