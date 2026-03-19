from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st

from ui_config import GENERAL_INFO_FALLBACKS
from ui_helpers import (
    case_prefix_from_form,
    extract_case_prefix_from_script,
    extract_station_and_scale_from_script,
    is_number,
    json_text,
    python_repr,
    station_field_key,
    template_general_info_path,
)


def form_defaults_for_preset(preset: str) -> dict[str, Any]:
    base = GENERAL_INFO_FALLBACKS[preset]
    return {
        "gi_case_prefix": preset.split("_")[0],
        "gi_enable_nested_basin": bool(base["nest_upstream_map"]),
        "gi_station_names_items": list(base["station_names"]) if base["station_names"] else [""],
        "gi_station_name": base["station_name"],
        "gi_model_scale": base["model_scale"],
        "gi_timestep": base["timestep"],
        "gi_timestep_evaluate": base["timestep_evaluate"],
        "gi_date_start": base["date_period"][0],
        "gi_date_end": base["date_period"][1],
        "gi_warmup_start": base["warmup_date_period"][0],
        "gi_warmup_end": base["warmup_date_period"][1],
        "gi_calibrate_start": base["calibrate_date_period"][0],
        "gi_calibrate_end": base["calibrate_date_period"][1],
        "gi_verify_start": base["verify_date_period"][0],
        "gi_verify_end": base["verify_date_period"][1],
        "gi_reverse_lat": base["reverse_lat"],
        "gi_grid_res_level0": float(base["grid_res_level0"]),
        "gi_scalemap_json": json_text(base["scalemap"]),
        "gi_station_names_json": json_text(base["station_names"]),
        "gi_station_coords_json": json_text(base["station_coords"]),
        "gi_nest_upstream_map_json": json_text(base["nest_upstream_map"]),
        "gi_boundary_xmin": str(base["boundary"][0]) if len(base["boundary"]) > 0 else "",
        "gi_boundary_ymin": str(base["boundary"][1]) if len(base["boundary"]) > 1 else "",
        "gi_boundary_xmax": str(base["boundary"][2]) if len(base["boundary"]) > 2 else "",
        "gi_boundary_ymax": str(base["boundary"][3]) if len(base["boundary"]) > 3 else "",
        "gi_basin_outlets_reference_i_map_json": json_text(base["basin_outlets_reference_i_map"]),
        "gi_stationdata_fname_map_json": json_text(base["stationdata_fname_map"]),
    }


def apply_general_info_defaults(preset: str, force: bool = False) -> None:
    defaults = form_defaults_for_preset(preset)
    for key, value in defaults.items():
        if force or key not in st.session_state:
            st.session_state[key] = value


def _load_json_field(
    field_key: str,
    field_label: str,
    expected_type: type,
    errors: list[str],
    *,
    required: bool = True,
    default_value: Any = None,
) -> Any:
    raw = str(st.session_state.get(field_key, "")).strip()
    if not raw:
        if required:
            errors.append(f"`{field_label}` cannot be empty.")
            return None
        return default_value
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"`{field_label}` JSON parse error: {exc.msg} (line {exc.lineno}, col {exc.colno})")
        return None
    if not isinstance(value, expected_type):
        errors.append(f"`{field_label}` must be `{expected_type.__name__}`.")
        return None
    return value


def collect_general_info_form(preset: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    case_prefix = str(st.session_state.get("gi_case_prefix", "")).strip()
    station_name = str(st.session_state.get("gi_station_name", "")).strip()
    model_scale = str(st.session_state.get("gi_model_scale", "")).strip()
    timestep = str(st.session_state.get("gi_timestep", "")).strip()
    timestep_evaluate = str(st.session_state.get("gi_timestep_evaluate", "")).strip()
    reverse_lat = bool(st.session_state.get("gi_reverse_lat", True))
    grid_res_level0 = float(st.session_state.get("gi_grid_res_level0", 0.00833))

    date_period = [str(st.session_state.get("gi_date_start", "")).strip(), str(st.session_state.get("gi_date_end", "")).strip()]
    warmup_date_period = [
        str(st.session_state.get("gi_warmup_start", "")).strip(),
        str(st.session_state.get("gi_warmup_end", "")).strip(),
    ]
    calibrate_date_period = [
        str(st.session_state.get("gi_calibrate_start", "")).strip(),
        str(st.session_state.get("gi_calibrate_end", "")).strip(),
    ]
    verify_date_period = [
        str(st.session_state.get("gi_verify_start", "")).strip(),
        str(st.session_state.get("gi_verify_end", "")).strip(),
    ]

    if not case_prefix:
        errors.append("`case_prefix` cannot be empty.")
    elif not re.fullmatch(r"[A-Za-z0-9_]+", case_prefix):
        errors.append("`case_prefix` must contain only letters, numbers, or underscore.")
    if not station_name:
        errors.append("`station_name` cannot be empty.")
    if not model_scale:
        errors.append("`model_scale` cannot be empty.")
    if not timestep:
        errors.append("`timestep` cannot be empty.")
    if not timestep_evaluate:
        errors.append("`timestep_evaluate` cannot be empty.")
    if grid_res_level0 <= 0:
        errors.append("`grid_res_level0` must be greater than 0.")
    for label, period in [
        ("date_period", date_period),
        ("warmup_date_period", warmup_date_period),
        ("calibrate_date_period", calibrate_date_period),
        ("verify_date_period", verify_date_period),
    ]:
        if not period[0] or not period[1]:
            errors.append(f"`{label}` requires both start and end.")

    scalemap = _load_json_field("gi_scalemap_json", "scalemap", dict, errors)
    basin_map_raw = _load_json_field(
        "gi_basin_outlets_reference_i_map_json",
        "basin_outlets_reference_i_map",
        dict,
        errors,
        required=False,
        default_value={},
    )
    normalized_basin_map: dict[str, int] = {}
    if isinstance(scalemap, dict):
        invalid_scalemap = [key for key, value in scalemap.items() if not isinstance(key, str) or (value is not None and not is_number(value))]
        if invalid_scalemap:
            errors.append("`scalemap` keys must be strings and values must be number or null.")
        if model_scale and model_scale not in scalemap:
            errors.append(f"`model_scale` ({model_scale}) is not in scalemap keys.")
    if isinstance(basin_map_raw, dict):
        for key, value in basin_map_raw.items():
            if not isinstance(key, str):
                errors.append("`basin_outlets_reference_i_map` keys must be strings.")
                continue
            if isinstance(value, bool):
                errors.append(f"`basin_outlets_reference_i_map[{key}]` must be an integer.")
                continue
            if isinstance(value, int):
                normalized_basin_map[key] = value
                continue
            if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
                normalized_basin_map[key] = int(value.strip())
                continue
            errors.append(f"`basin_outlets_reference_i_map[{key}]` must be an integer.")

    config: dict[str, Any] = {
        "case_prefix": case_prefix,
        "station_name": station_name,
        "model_scale": model_scale,
        "timestep": timestep,
        "timestep_evaluate": timestep_evaluate,
        "date_period": date_period,
        "warmup_date_period": warmup_date_period,
        "calibrate_date_period": calibrate_date_period,
        "verify_date_period": verify_date_period,
        "reverse_lat": reverse_lat,
        "grid_res_level0": grid_res_level0,
        "scalemap": scalemap,
        "basin_outlets_reference_i_map": normalized_basin_map,
    }

    if preset == "HRB_modeling":
        station_names = _load_json_field("gi_station_names_json", "station_names", list, errors)
        station_coords = _load_json_field("gi_station_coords_json", "station_coords", dict, errors)
        nest_upstream_map = _load_json_field(
            "gi_nest_upstream_map_json",
            "nest_upstream_map",
            dict,
            errors,
            required=False,
            default_value={},
        )
        boundary_raw_values = [
            str(st.session_state.get("gi_boundary_xmin", "")).strip(),
            str(st.session_state.get("gi_boundary_ymin", "")).strip(),
            str(st.session_state.get("gi_boundary_xmax", "")).strip(),
            str(st.session_state.get("gi_boundary_ymax", "")).strip(),
        ]
        boundary: list[float] = []
        boundary_labels = ["xmin", "ymin", "xmax", "ymax"]
        for label, raw_value in zip(boundary_labels, boundary_raw_values):
            if not raw_value:
                errors.append(f"`boundary {label}` cannot be empty.")
                continue
            try:
                boundary.append(float(raw_value))
            except ValueError:
                errors.append(f"`boundary {label}` must be numeric.")
        normalized_station_coords: dict[str, list[float]] = {}
        if isinstance(station_names, list):
            if not station_names:
                errors.append("`station_names` cannot be empty.")
            if any(not isinstance(name, str) for name in station_names):
                errors.append("`station_names` must be a list of strings.")
            if any(not str(name).strip() for name in station_names):
                errors.append("`station_names` cannot contain empty values.")
            if len(set(station_names)) != len(station_names):
                errors.append("`station_names` cannot contain duplicates.")
            if station_name and station_name not in station_names:
                errors.append("`station_name` must exist in `station_names`.")
        if isinstance(station_coords, dict):
            for name, coord in station_coords.items():
                if not isinstance(name, str) or not isinstance(coord, list) or len(coord) != 2:
                    errors.append("`station_coords` must be in `{name: [lat, lon]}` format.")
                    break
        if isinstance(station_names, list) and isinstance(station_coords, dict):
            missing_station_coords = [name for name in station_names if name not in station_coords]
            extra_station_coords = [name for name in station_coords if name not in station_names]
            if missing_station_coords:
                errors.append(
                    "`station_coords` is missing stations from `station_names`: "
                    + ", ".join(missing_station_coords)
                )
            if extra_station_coords:
                errors.append(
                    "`station_coords` has extra stations not in `station_names`: "
                    + ", ".join(extra_station_coords)
                )
            for name in station_names:
                coord = station_coords.get(name)
                if not isinstance(coord, list) or len(coord) != 2:
                    errors.append(f"`station_coords[{name}]` must be `[lat, lon]`.")
                    break
                try:
                    lat = float(coord[0])
                    lon = float(coord[1])
                except (TypeError, ValueError):
                    errors.append(f"`station_coords[{name}]` lat/lon must be numeric.")
                    break
                normalized_station_coords[name] = [lat, lon]
        if isinstance(station_names, list):
            missing_basin_outlets = [name for name in station_names if name not in normalized_basin_map]
            extra_basin_outlets = [name for name in normalized_basin_map if name not in station_names]
            if missing_basin_outlets:
                errors.append(
                    "`basin_outlets_reference_i_map` is missing station_names: "
                    + ", ".join(missing_basin_outlets)
                )
            if extra_basin_outlets:
                errors.append(
                    "`basin_outlets_reference_i_map` has extra stations not in station_names: "
                    + ", ".join(extra_basin_outlets)
                )
        if isinstance(nest_upstream_map, dict):
            for name, upstream in nest_upstream_map.items():
                if not isinstance(name, str) or not isinstance(upstream, list) or any(not isinstance(item, str) for item in upstream):
                    errors.append("`nest_upstream_map` must be in `{name: [upstream_names...]}` format.")
                    break
                if isinstance(station_names, list) and name not in station_names:
                    errors.append(f"`nest_upstream_map[{name}]` is not in station_names.")
                    break
                if isinstance(station_names, list):
                    invalid_upstream = [item for item in upstream if item not in station_names]
                    if invalid_upstream:
                        errors.append(
                            f"`nest_upstream_map[{name}]` has stations not in station_names: "
                            + ", ".join(invalid_upstream)
                        )
                        break
                if name in upstream:
                    errors.append(f"`nest_upstream_map[{name}]` cannot contain itself.")
                    break
        if len(boundary) == 4 and not (boundary[0] < boundary[2] and boundary[1] < boundary[3]):
            errors.append("`boundary` must satisfy xmin < xmax and ymin < ymax.")
        config.update(
            {
                "station_names": station_names,
                "station_coords": normalized_station_coords if normalized_station_coords else station_coords,
                "nest_upstream_map": nest_upstream_map,
                "boundary": boundary,
            }
        )
    else:
        stationdata_fname_map = _load_json_field("gi_stationdata_fname_map_json", "stationdata_fname_map", dict, errors)
        if isinstance(stationdata_fname_map, dict):
            invalid_station_data = [key for key, value in stationdata_fname_map.items() if not isinstance(key, str) or not isinstance(value, str)]
            if invalid_station_data:
                errors.append("`stationdata_fname_map` keys and values must be strings.")
        config["stationdata_fname_map"] = stationdata_fname_map

    if errors:
        return None, errors
    return config, []


def render_general_info_script(preset: str, config: dict[str, Any]) -> str:
    lines: list[str] = ["import numpy as np", "import pandas as pd"]
    if preset == "HRB_modeling":
        station_coords = {
            name: (float(coord[0]), float(coord[1]))
            for name, coord in config["station_coords"].items()
        }
        lines.extend(
            [
                "from easy_vic_build import logger",
                "from easy_vic_build.tools.nested_basin_func.nested_basin_func import get_all_upstreams, get_topo_order",
                "",
                "# general info",
                f"scalemap = {python_repr(config['scalemap'])}",
                "",
                "# set stations",
                f"station_name = {config['station_name']!r}",
                f"station_names = {python_repr(config['station_names'])}",
                "",
                f"station_coords = {python_repr(station_coords)}",
                "",
                f"nest_upstream_map = {python_repr(config['nest_upstream_map'])}",
                "",
                "topo_station_order = get_topo_order(station_names, nest_upstream_map)",
                "",
                "station_coords_df = pd.DataFrame({",
                "    \"station_name\": station_names,",
                "    \"lat\": [station_coords[name][0] for name in station_names],",
                "    \"lon\": [station_coords[name][1] for name in station_names],",
                "})",
                "",
                f"boundary = {python_repr(config['boundary'])}",
                "",
                f"basin_outlets_reference_i_map = {python_repr(config['basin_outlets_reference_i_map'])}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "# general info",
                f"scalemap = {python_repr(config['scalemap'])}",
                "",
                f"basin_outlets_reference_i_map = {python_repr(config['basin_outlets_reference_i_map'])}",
                "",
                f"stationdata_fname_map = {python_repr(config['stationdata_fname_map'])}",
                "",
                "# set configuration",
                f"station_name = {config['station_name']!r}",
            ]
        )

    lines.extend(
        [
            f"case_prefix = {config['case_prefix']!r}",
            "",
            f"model_scale = {config['model_scale']!r}",
            f"timestep = {config['timestep']!r}",
            f"timestep_evaluate = {config['timestep_evaluate']!r}",
            "",
            f"date_period = {python_repr(config['date_period'])}",
            f"warmup_date_period = {python_repr(config['warmup_date_period'])}",
            f"calibrate_date_period = {python_repr(config['calibrate_date_period'])}",
            f"verify_date_period = {python_repr(config['verify_date_period'])}",
            "",
            "date = pd.date_range(date_period[0], date_period[1], freq=timestep)",
            "warmup_date = pd.date_range(warmup_date_period[0], warmup_date_period[1], freq=timestep)",
            "calibrate_date = pd.date_range(calibrate_date_period[0], calibrate_date_period[1], freq=timestep)",
            "verify_date = pd.date_range(verify_date_period[0], verify_date_period[1], freq=timestep)",
            "",
            "date_evaluate = pd.date_range(date_period[0], date_period[1], freq=timestep_evaluate)",
            "warmup_date_evaluate = pd.date_range(warmup_date_period[0], warmup_date_period[1], freq=timestep_evaluate)",
            "calibrate_date_evaluate = pd.date_range(calibrate_date_period[0], calibrate_date_period[1], freq=timestep_evaluate)",
            "verify_date_evaluate = pd.date_range(verify_date_period[0], verify_date_period[1], freq=timestep_evaluate)",
            "",
            f"reverse_lat = {config['reverse_lat']!r}",
            "",
            "# set scale level",
            f"grid_res_level0 = {config['grid_res_level0']}",
            "grid_res_level1 = scalemap[model_scale]",
            "grid_res_level2 = scalemap[model_scale]",
            "",
        ]
    )
    return "\n".join(lines)


def render_general_info_tab() -> None:
    st.subheader("Step 1: General Info")
    preset_choices = list(GENERAL_INFO_FALLBACKS.keys())
    selected_preset = st.selectbox(
        "preset",
        options=preset_choices,
        index=preset_choices.index(st.session_state["workflow_preset"]),
    )
    if selected_preset != st.session_state["workflow_preset"]:
        st.session_state["workflow_preset"] = selected_preset
        apply_general_info_defaults(selected_preset, force=True)
        st.session_state["gi_final_script_text"] = ""
        st.rerun()

    preset = st.session_state["workflow_preset"]
    apply_general_info_defaults(preset, force=False)
    st.caption("Fill the form to generate `general_info.py`; originals under `examples/` remain unchanged.")

    template_path = template_general_info_path(preset)
    st.text_input("template path", value=str(template_path), disabled=True)

    reload_col, _ = st.columns([1, 3])
    with reload_col:
        if st.button("Reset form defaults", use_container_width=True):
            apply_general_info_defaults(preset, force=True)
            st.session_state["gi_final_script_text"] = ""
            st.success("Form defaults reloaded.")
            st.rerun()

    basic_left, basic_right = st.columns(2)
    with basic_left:
        st.text_input("case_prefix", key="gi_case_prefix")
        if preset != "HRB_modeling":
            st.text_input("station_name", key="gi_station_name")
        st.text_input("timestep", key="gi_timestep")
        st.text_input("date_period start", key="gi_date_start")
        st.text_input("warmup_date_period start", key="gi_warmup_start")
        st.text_input("calibrate_date_period start", key="gi_calibrate_start")
        st.text_input("verify_date_period start", key="gi_verify_start")
        st.number_input("grid_res_level0", min_value=0.0, key="gi_grid_res_level0", step=0.00001, format="%.6f")
    with basic_right:
        st.text_input("timestep_evaluate", key="gi_timestep_evaluate")
        st.text_input("date_period end", key="gi_date_end")
        st.text_input("warmup_date_period end", key="gi_warmup_end")
        st.text_input("calibrate_date_period end", key="gi_calibrate_end")
        st.text_input("verify_date_period end", key="gi_verify_end")
        st.checkbox("reverse_lat", key="gi_reverse_lat")

    st.text_area("scalemap (JSON)", key="gi_scalemap_json", height=210)
    model_scale_options: list[str] = []
    raw_scalemap = str(st.session_state.get("gi_scalemap_json", "")).strip()
    if raw_scalemap:
        try:
            parsed_scalemap = json.loads(raw_scalemap)
            if isinstance(parsed_scalemap, dict):
                model_scale_options = [str(key) for key in parsed_scalemap.keys()]
            else:
                st.error("`scalemap` must be a JSON object.")
        except json.JSONDecodeError as exc:
            st.error(f"`scalemap` JSON parse error: {exc.msg} (line {exc.lineno}, col {exc.colno})")
    else:
        st.error("`scalemap` cannot be empty.")

    if model_scale_options:
        current_model_scale = str(st.session_state.get("gi_model_scale", "")).strip()
        if current_model_scale not in model_scale_options:
            st.session_state["gi_model_scale"] = model_scale_options[0]
        st.selectbox(
            "model_scale (choose from scalemap keys)",
            options=model_scale_options,
            key="gi_model_scale",
        )
    else:
        st.session_state["gi_model_scale"] = ""
        st.selectbox(
            "model_scale (choose from scalemap keys)",
            options=[""],
            key="gi_model_scale",
            disabled=True,
        )

    if preset == "HRB_modeling":
        existing_station_coords: dict[str, Any] = {}
        raw_station_coords = str(st.session_state.get("gi_station_coords_json", "")).strip()
        if raw_station_coords:
            try:
                parsed_station_coords = json.loads(raw_station_coords)
                if isinstance(parsed_station_coords, dict):
                    existing_station_coords = parsed_station_coords
            except json.JSONDecodeError:
                existing_station_coords = {}

        existing_basin_reference_map: dict[str, Any] = {}
        raw_basin_reference_map = str(st.session_state.get("gi_basin_outlets_reference_i_map_json", "")).strip()
        if raw_basin_reference_map:
            try:
                parsed_basin_reference_map = json.loads(raw_basin_reference_map)
                if isinstance(parsed_basin_reference_map, dict):
                    existing_basin_reference_map = parsed_basin_reference_map
            except json.JSONDecodeError:
                existing_basin_reference_map = {}

        station_name_items = st.session_state.get("gi_station_names_items", [""])
        if not isinstance(station_name_items, list):
            station_name_items = [""]
        if not station_name_items:
            station_name_items = [""]

        st.caption("station_names")
        header_col_name, header_col_lat, header_col_lon, header_col_ref = st.columns([2, 1, 1, 1])
        with header_col_name:
            st.markdown("**station_name**")
        with header_col_lat:
            st.markdown("**lat**")
        with header_col_lon:
            st.markdown("**lon**")
        with header_col_ref:
            st.markdown("**reference_i**")

        edited_station_names: list[str] = []
        synced_station_coords: dict[str, list[str]] = {}
        synced_basin_reference_map: dict[str, str] = {}
        for idx in range(len(station_name_items)):
            station_item_key = f"gi_station_name_item_{idx}"
            lat_item_key = f"gi_station_lat_item_{idx}"
            lon_item_key = f"gi_station_lon_item_{idx}"
            ref_item_key = f"gi_station_ref_item_{idx}"

            if station_item_key not in st.session_state:
                st.session_state[station_item_key] = str(station_name_items[idx])
            station_name_value_for_default = str(st.session_state.get(station_item_key, "")).strip()
            default_coord = existing_station_coords.get(station_name_value_for_default, ["", ""])
            if not isinstance(default_coord, list) or len(default_coord) != 2:
                default_coord = ["", ""]
            default_ref = existing_basin_reference_map.get(station_name_value_for_default, "")

            if lat_item_key not in st.session_state:
                st.session_state[lat_item_key] = str(default_coord[0])
            if lon_item_key not in st.session_state:
                st.session_state[lon_item_key] = str(default_coord[1])
            if ref_item_key not in st.session_state:
                st.session_state[ref_item_key] = str(default_ref)

            station_col, lat_col, lon_col, ref_col = st.columns([2, 1, 1, 1])
            with station_col:
                st.text_input(
                    f"station_name_{idx + 1}",
                    key=station_item_key,
                    label_visibility="collapsed",
                    placeholder="station_name",
                )
            with lat_col:
                st.text_input(
                    f"lat_{idx + 1}",
                    key=lat_item_key,
                    label_visibility="collapsed",
                    placeholder="lat",
                )
            with lon_col:
                st.text_input(
                    f"lon_{idx + 1}",
                    key=lon_item_key,
                    label_visibility="collapsed",
                    placeholder="lon",
                )
            with ref_col:
                st.text_input(
                    f"reference_i_{idx + 1}",
                    key=ref_item_key,
                    label_visibility="collapsed",
                    placeholder="reference_i",
                )

            station_name_value = str(st.session_state.get(station_item_key, "")).strip()
            station_lat_value = str(st.session_state.get(lat_item_key, "")).strip()
            station_lon_value = str(st.session_state.get(lon_item_key, "")).strip()
            station_ref_value = str(st.session_state.get(ref_item_key, "")).strip()

            edited_station_names.append(station_name_value)
            if station_name_value:
                synced_station_coords[station_name_value] = [station_lat_value, station_lon_value]
                synced_basin_reference_map[station_name_value] = station_ref_value

        st.session_state["gi_station_names_items"] = edited_station_names
        st.session_state["gi_station_names_json"] = json.dumps(edited_station_names, ensure_ascii=False, indent=2)
        st.session_state["gi_station_coords_json"] = json.dumps(synced_station_coords, ensure_ascii=False, indent=2)
        st.session_state["gi_basin_outlets_reference_i_map_json"] = json.dumps(
            synced_basin_reference_map,
            ensure_ascii=False,
            indent=2,
        )

        add_col, remove_col = st.columns(2)
        with add_col:
            if st.button("Add station_name", use_container_width=True):
                st.session_state["gi_station_names_items"] = list(edited_station_names) + [""]
                st.rerun()
        with remove_col:
            if st.button("Remove last station_name", use_container_width=True, disabled=len(edited_station_names) <= 1):
                trimmed_station_names = list(edited_station_names[:-1])
                if not trimmed_station_names:
                    trimmed_station_names = [""]
                st.session_state["gi_station_names_items"] = trimmed_station_names
                st.rerun()

        with st.expander("station_coords JSON preview", expanded=False):
            st.code(st.session_state["gi_station_coords_json"], language="json")

        station_name_options: list[str] = [name for name in edited_station_names if name]
        dedup_station_name_options: list[str] = []
        seen_station_names: set[str] = set()
        for station_name_item in station_name_options:
            if station_name_item in seen_station_names:
                continue
            seen_station_names.add(station_name_item)
            dedup_station_name_options.append(station_name_item)
        if len(dedup_station_name_options) < len(station_name_options):
            st.error("`station_names` cannot contain duplicates.")
        station_name_options = dedup_station_name_options

        if station_name_options:
            current_station = str(st.session_state.get("gi_station_name", "")).strip()
            if current_station not in station_name_options:
                st.session_state["gi_station_name"] = station_name_options[0]
            st.selectbox(
                "station_name (main outlet)",
                options=station_name_options,
                key="gi_station_name",
            )
        else:
            st.session_state["gi_station_name"] = ""
            st.selectbox(
                "station_name (main outlet)",
                options=[""],
                key="gi_station_name",
                disabled=True,
            )

        existing_nest_upstream_map: dict[str, Any] = {}
        raw_nest_upstream_map = str(st.session_state.get("gi_nest_upstream_map_json", "")).strip()
        if raw_nest_upstream_map:
            try:
                parsed_nest_upstream_map = json.loads(raw_nest_upstream_map)
                if isinstance(parsed_nest_upstream_map, dict):
                    existing_nest_upstream_map = parsed_nest_upstream_map
            except json.JSONDecodeError:
                existing_nest_upstream_map = {}

        if "gi_enable_nested_basin" not in st.session_state:
            st.session_state["gi_enable_nested_basin"] = bool(existing_nest_upstream_map)
        st.checkbox("Has nested basin", key="gi_enable_nested_basin")
        if st.session_state.get("gi_enable_nested_basin", False):
            synced_nest_upstream_map: dict[str, list[str]] = {}
            for station_name_item in station_name_options:
                upstream_key = station_field_key(station_name_item, "upstreams")
                default_upstream = existing_nest_upstream_map.get(station_name_item, [])
                if not isinstance(default_upstream, list):
                    default_upstream = []
                if upstream_key not in st.session_state:
                    st.session_state[upstream_key] = [
                        upstream_name for upstream_name in default_upstream if upstream_name in station_name_options
                    ]
                upstream_options = [name for name in station_name_options if name != station_name_item]
                st.multiselect(
                    f"{station_name_item} upstream stations",
                    options=upstream_options,
                    key=upstream_key,
                )
                synced_nest_upstream_map[station_name_item] = list(st.session_state.get(upstream_key, []))
            st.session_state["gi_nest_upstream_map_json"] = json.dumps(
                synced_nest_upstream_map,
                ensure_ascii=False,
                indent=2,
            )
        else:
            st.session_state["gi_nest_upstream_map_json"] = json.dumps({}, ensure_ascii=False, indent=2)

        bx1, by1, bx2, by2 = st.columns(4)
        with bx1:
            st.text_input("boundary xmin", key="gi_boundary_xmin")
        with by1:
            st.text_input("boundary ymin", key="gi_boundary_ymin")
        with bx2:
            st.text_input("boundary xmax", key="gi_boundary_xmax")
        with by2:
            st.text_input("boundary ymax", key="gi_boundary_ymax")
    else:
        st.text_area("stationdata_fname_map (JSON object)", key="gi_stationdata_fname_map_json", height=170)
        st.text_area(
            "basin_outlets_reference_i_map (JSON object, optional)",
            key="gi_basin_outlets_reference_i_map_json",
            height=140,
            placeholder="Leave empty if not needed.",
        )

    config, errors = collect_general_info_form(preset)
    generated_script = render_general_info_script(preset, config) if config is not None else ""
    if errors:
        for message in errors:
            st.error(message)

    if not str(st.session_state.get("gi_final_script_text", "")).strip() and generated_script:
        st.session_state["gi_final_script_text"] = generated_script

    sync_col, _ = st.columns([1, 3])
    with sync_col:
        if st.button("Load generated into editor", use_container_width=True, disabled=not bool(generated_script)):
            st.session_state["gi_final_script_text"] = generated_script
            st.rerun()

    st.text_area(
        "final general_info.py (editable, used by Step 2 Initialize Case)",
        key="gi_final_script_text",
        height=360,
    )
    final_script_text = str(st.session_state.get("gi_final_script_text", ""))
    final_station, final_scale = extract_station_and_scale_from_script(final_script_text)
    final_prefix = extract_case_prefix_from_script(final_script_text) or case_prefix_from_form(preset)
    st.caption(
        f"Final script parsed case_prefix: `{final_prefix or 'N/A'}` | station_name: `{final_station or 'N/A'}` | model_scale: `{final_scale or 'N/A'}`"
    )

    if generated_script:
        with st.expander("Generated general_info.py preview", expanded=False):
            st.code(generated_script, language="python")
