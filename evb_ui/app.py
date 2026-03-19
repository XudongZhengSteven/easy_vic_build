from __future__ import annotations

import streamlit as st

from ui_case_steps import render_hydroanalysis_l0_tab, render_init_case_tab
from ui_dpc_step import render_build_dpc_tab
from ui_general_info import render_general_info_tab
from ui_logger import render_logger_tab
from ui_state_profile import init_state, render_sidebar


def main() -> None:
    st.set_page_config(page_title="EVB UI", layout="wide")
    init_state()
    render_sidebar()

    st.title("easy_vic_build UI")
    st.caption(
        "Step 1-2 page edits `general_info.py` and initializes cases; Step 3 runs hydroanalysis level0; Step 4 generates editable `build_dpc.py`."
    )

    tab_general_init, tab_hydro_l0, tab_build_dpc, tab_logger = st.tabs(
        ["Step 1-2 - General Info + Initialize Case", "Step 3 - Hydroanalysis L0", "Step 4 - Build DPC", "Logger"]
    )
    with tab_general_init:
        render_general_info_tab()
        st.divider()
        render_init_case_tab()
    with tab_hydro_l0:
        render_hydroanalysis_l0_tab()
    with tab_build_dpc:
        render_build_dpc_tab()
    with tab_logger:
        render_logger_tab()


if __name__ == "__main__":
    main()
