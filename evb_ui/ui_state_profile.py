from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from ui_config import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_PRESET,
    GENERAL_INFO_FALLBACKS,
    GENERAL_INFO_STATE_KEYS,
    LEGACY_ARTIFACT_ROOT,
)
from ui_general_info import apply_general_info_defaults


def _general_info_profile_state() -> dict[str, Any]:
    return {key: st.session_state.get(key) for key in GENERAL_INFO_STATE_KEYS}


def save_profile(name: str) -> Path:
    profile_dir = Path(__file__).resolve().parent / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "workflow_preset": st.session_state["workflow_preset"],
        "artifact_root": st.session_state["artifact_root"],
        "general_info_form_state": _general_info_profile_state(),
        "active_cases_home": st.session_state["active_cases_home"],
        "active_case_name": st.session_state["active_case_name"],
        "script_workspace_dir": st.session_state.get("script_workspace_dir", ""),
    }
    profile_path = profile_dir / f"{name.replace(' ', '_')}.json"
    profile_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile_path


def load_profile(name: str) -> Path:
    profile_path = Path(__file__).resolve().parent / "profiles" / f"{name.replace(' ', '_')}.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    form_state = payload.pop("general_info_form_state", None)
    for key, value in payload.items():
        st.session_state[key] = value
    preset = st.session_state.get("workflow_preset", DEFAULT_PRESET)
    apply_general_info_defaults(preset, force=False)
    if isinstance(form_state, dict):
        for key, value in form_state.items():
            if key in GENERAL_INFO_STATE_KEYS:
                st.session_state[key] = value
    return profile_path


def init_state() -> None:
    st.session_state.setdefault("workflow_preset", DEFAULT_PRESET)
    st.session_state.setdefault("artifact_root", DEFAULT_ARTIFACT_ROOT)
    st.session_state.setdefault("gi_final_script_text", "")
    st.session_state.setdefault("ha_l0_final_script_text", "")
    st.session_state.setdefault("ha_l0_cmd", "")
    st.session_state.setdefault("ha_l0_python_cmd", "python")
    st.session_state.setdefault("active_cases_home", DEFAULT_ARTIFACT_ROOT)
    st.session_state.setdefault("active_case_name", "")
    st.session_state.setdefault("script_workspace_dir", "")
    st.session_state.setdefault("runtime_console_log", [])

    if st.session_state["artifact_root"] == LEGACY_ARTIFACT_ROOT:
        st.session_state["artifact_root"] = DEFAULT_ARTIFACT_ROOT
    if st.session_state["active_cases_home"] == LEGACY_ARTIFACT_ROOT:
        st.session_state["active_cases_home"] = DEFAULT_ARTIFACT_ROOT
    if st.session_state["workflow_preset"] not in GENERAL_INFO_FALLBACKS:
        st.session_state["workflow_preset"] = DEFAULT_PRESET
    apply_general_info_defaults(st.session_state["workflow_preset"], force=False)


def render_sidebar() -> None:
    st.sidebar.subheader("Profile")
    profile_name = st.sidebar.text_input("profile_name", value="default")
    left_col, right_col = st.sidebar.columns(2)
    with left_col:
        if st.button("Save profile", use_container_width=True):
            path = save_profile(profile_name)
            st.sidebar.success(f"Saved: {path.name}")
    with right_col:
        if st.button("Load profile", use_container_width=True):
            try:
                path = load_profile(profile_name)
                st.sidebar.success(f"Loaded: {path.name}")
                st.rerun()
            except FileNotFoundError:
                st.sidebar.error("Profile not found.")
            except json.JSONDecodeError:
                st.sidebar.error("Profile JSON invalid.")
