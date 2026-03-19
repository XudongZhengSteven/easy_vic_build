from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from pprint import pformat
from typing import Any

import streamlit as st

from ui_config import DPC_LEVEL_SPECS, REPO_ROOT


def template_general_info_path(preset: str) -> Path:
    return REPO_ROOT / "examples" / preset / "general_info.py"


def default_case_prefix_for_preset(preset: str) -> str:
    return preset.split("_")[0]


def dpc_script_name_for_preset(preset: str) -> str:
    return f"{preset.split('_')[0]}_build_dpc.py"


def hydroanalysis_script_name_for_preset(preset: str) -> str:
    return f"{preset.split('_')[0]}_hydroanalysis.py"


def hydroanalysis_l0_function_for_preset(preset: str) -> str:
    if preset == "HRB_modeling":
        return "hydroanalysis_level0_HRB"
    if preset == "JRB_modeling":
        return "hydroanalysis_level0_JRB"
    return "hydroanalysis_level0"


def dpc_level_specs_for_preset(preset: str) -> list[dict[str, Any]]:
    return DPC_LEVEL_SPECS.get(preset, [])


def strip_evb_dir_section(text: str) -> str:
    lines = text.splitlines()
    stripped: list[str] = []
    skip = False
    for line in lines:
        if line.strip().startswith("# build evb"):
            skip = True
            continue
        if skip:
            continue
        stripped.append(line)
    return "\n".join(stripped).strip() + "\n"


def load_general_info_template(preset: str) -> str:
    template_path = template_general_info_path(preset)
    if not template_path.exists():
        return ""
    return strip_evb_dir_section(template_path.read_text(encoding="utf-8", errors="replace"))


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def station_and_scale_from_form() -> tuple[str, str]:
    station_name = str(st.session_state.get("gi_station_name", "")).strip()
    model_scale = str(st.session_state.get("gi_model_scale", "")).strip()
    return station_name, model_scale


def case_prefix_from_form(preset: str) -> str:
    case_prefix = str(st.session_state.get("gi_case_prefix", "")).strip()
    if case_prefix:
        return case_prefix
    return default_case_prefix_for_preset(preset)


def extract_station_and_scale_from_script(script_text: str) -> tuple[str, str]:
    station_match = re.search(r'^\s*station_name\s*=\s*["\']([^"\']+)["\']', script_text, re.MULTILINE)
    scale_match = re.search(r'^\s*model_scale\s*=\s*["\']([^"\']+)["\']', script_text, re.MULTILINE)
    station_name = station_match.group(1).strip() if station_match else ""
    model_scale = scale_match.group(1).strip() if scale_match else ""
    return station_name, model_scale


def extract_case_prefix_from_script(script_text: str) -> str:
    prefix_match = re.search(r'^\s*case_prefix\s*=\s*["\']([^"\']+)["\']', script_text, re.MULTILINE)
    return prefix_match.group(1).strip() if prefix_match else ""


def station_coord_key(station_name: str, axis: str) -> str:
    station_hash = hashlib.md5(station_name.encode("utf-8")).hexdigest()[:10]
    return f"gi_station_coord_{axis}_{station_hash}"


def station_field_key(station_name: str, field_name: str) -> str:
    station_hash = hashlib.md5(station_name.encode("utf-8")).hexdigest()[:10]
    return f"gi_station_{field_name}_{station_hash}"


def python_repr(value: Any) -> str:
    return pformat(value, width=100, sort_dicts=False)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def case_names(case_prefix: str, station_name: str, model_scale: str) -> tuple[str, str]:
    prefix = case_prefix.strip() or "EVB"
    model_case = f"{prefix}_{station_name}_{model_scale}" if station_name and model_scale else f"{prefix}_<station>_<scale>"
    return model_case, model_case


def workspace_dir(model_case_dir: str) -> Path:
    return Path(model_case_dir) / "scripts"


def tail_text_lines(text: str, line_count: int) -> str:
    lines = text.splitlines()
    if line_count <= 0:
        return ""
    return "\n".join(lines[-line_count:])


def ensure_src_path_for_ui() -> None:
    src_dir = REPO_ROOT / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)


def load_basin_shp_with_shx_repair(Basins: Any, shp_path: str | Path) -> Any:
    shp_path_str = str(shp_path)
    previous_value = os.environ.get("SHAPE_RESTORE_SHX")
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    try:
        return Basins.from_shapefile(shp_path_str)
    finally:
        if previous_value is None:
            os.environ.pop("SHAPE_RESTORE_SHX", None)
        else:
            os.environ["SHAPE_RESTORE_SHX"] = previous_value
