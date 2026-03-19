from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui_config import DEFAULT_ARTIFACT_ROOT
from ui_helpers import tail_text_lines


def render_logger_tab() -> None:
    st.subheader("Logger")
    st.caption("View runtime console output and case logger files under `VICLog`.")

    st.session_state.setdefault("logs_cases_home", st.session_state.get("active_cases_home", DEFAULT_ARTIFACT_ROOT))
    st.session_state.setdefault("logs_case_name", st.session_state.get("active_case_name", ""))
    st.session_state.setdefault("logger_runtime_tail", 400)
    st.session_state.setdefault("logger_file_tail", 400)

    sync_col, refresh_col, clear_col = st.columns(3)
    with sync_col:
        if st.button("Use active case", use_container_width=True):
            st.session_state["logs_cases_home"] = st.session_state.get("active_cases_home", st.session_state["logs_cases_home"])
            st.session_state["logs_case_name"] = st.session_state.get("active_case_name", st.session_state["logs_case_name"])
            st.rerun()
    with refresh_col:
        if st.button("Refresh logs", use_container_width=True):
            st.rerun()
    with clear_col:
        if st.button("Clear runtime log", use_container_width=True):
            st.session_state["runtime_console_log"] = []
            st.rerun()

    left, right = st.columns(2)
    with left:
        st.text_input("logs cases_home", key="logs_cases_home")
    with right:
        st.text_input("logs case_name", key="logs_case_name")

    runtime_lines: list[str] = st.session_state.get("runtime_console_log", [])
    st.number_input("runtime tail lines", min_value=50, max_value=5000, step=50, key="logger_runtime_tail")
    st.code("\n".join(runtime_lines[-st.session_state["logger_runtime_tail"] :]) or "No runtime logs yet.", language="bash")

    case_dir = Path(str(st.session_state["logs_cases_home"]).strip()) / str(st.session_state["logs_case_name"]).strip()
    vic_log_dir = case_dir / "VICLog"
    st.caption(f"Case dir: `{case_dir}`")
    st.caption(f"VICLog dir: `{vic_log_dir}`")

    if not case_dir.exists():
        st.info("Case directory not found. Run Step 2 (Initialize Case) first or select an existing case.")
        return

    if not vic_log_dir.exists():
        st.info("`VICLog` directory not found for this case.")
        return

    log_files = sorted([path for path in vic_log_dir.iterdir() if path.is_file()], key=lambda path: path.name.lower())
    if not log_files:
        st.info("No log files found in `VICLog`.")
        return

    selected_log = st.selectbox("log file", options=[path.name for path in log_files], key="logger_selected_file")
    st.number_input("log file tail lines", min_value=50, max_value=20000, step=50, key="logger_file_tail")

    selected_path = next(path for path in log_files if path.name == selected_log)
    try:
        content = selected_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        st.error(f"Failed to read log file: {exc}")
        return

    st.caption(f"File: `{selected_path.name}` | Size: {selected_path.stat().st_size} bytes")
    st.code(tail_text_lines(content, st.session_state["logger_file_tail"]) or "(empty)", language="bash")
    st.download_button("Download log file", data=content, file_name=selected_path.name, mime="text/plain")
