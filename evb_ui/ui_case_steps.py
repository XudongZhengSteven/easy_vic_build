from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st

from evb_bridge import EVBUnavailableError, CaseContext, init_case, run_shell_command
from ui_helpers import (
    case_names,
    case_prefix_from_form,
    extract_case_prefix_from_script,
    extract_station_and_scale_from_script,
    hydroanalysis_l0_function_for_preset,
    hydroanalysis_script_name_for_preset,
    python_repr,
    station_and_scale_from_form,
    workspace_dir,
)


def _prepare_case_scripts(general_info_text: str, model_case_dir: str) -> Path:
    target_dir = workspace_dir(model_case_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "general_info.py").write_text(general_info_text, encoding="utf-8")
    return target_dir


def run_command_with_output(command: str, working_dir: str) -> int:
    lines: list[str] = []
    output_box = st.empty()
    runtime_lines = st.session_state.setdefault("runtime_console_log", [])

    def on_output(line: str) -> None:
        lines.append(line)
        runtime_lines.append(line)
        if len(runtime_lines) > 5000:
            del runtime_lines[:-5000]
        output_box.code("\n".join(lines[-400:]) or "Running...", language="bash")

    code = run_shell_command(command=command, working_dir=working_dir, on_output=on_output)
    st.session_state["last_command_output_lines"] = lines[-2000:]
    st.session_state["last_command_exit_code"] = code
    return code


def run_init_case_step() -> bool:
    preset = st.session_state["workflow_preset"]
    general_info_text = str(st.session_state.get("gi_final_script_text", "")).strip()
    if not general_info_text:
        st.error("`final general_info.py` is empty. Complete Step 1 first.")
        return False

    case_prefix = extract_case_prefix_from_script(general_info_text) or case_prefix_from_form(preset)
    station_name, model_scale = extract_station_and_scale_from_script(general_info_text)
    if not station_name or not model_scale:
        st.error("Cannot parse `station_name`/`model_scale` from final general_info.py.")
        return False

    _, model_case = case_names(case_prefix, station_name, model_scale)

    try:
        model_paths = init_case(CaseContext(st.session_state["artifact_root"], model_case))
        workspace_dir_path = _prepare_case_scripts(general_info_text, model_paths["case_dir"])
    except EVBUnavailableError as exc:
        st.error(str(exc))
        return False
    except Exception as exc:
        st.error(f"Initialize failed: {exc}")
        return False

    st.session_state["active_cases_home"] = st.session_state["artifact_root"]
    st.session_state["active_case_name"] = model_case
    st.session_state["script_workspace_dir"] = str(workspace_dir_path)
    st.success("Model case initialized. Only `general_info.py` is written to model case `scripts/`.")
    with st.expander("Initialization outputs", expanded=False):
        st.json(
            {
                "model": model_paths,
                "script_workspace": str(workspace_dir_path),
                "case_name_for_step3": model_case,
            }
        )
    return True


def render_init_case_tab() -> None:
    st.subheader("Step 2: Initialize Case")
    st.caption("Use Step 1 output to initialize only the model case directory, and write only `general_info.py` into model case `scripts/`.")
    st.caption(f"Current preset: `{st.session_state['workflow_preset']}`")

    st.session_state["artifact_root"] = st.text_input(
        "artifact_root (cases_home)",
        value=st.session_state["artifact_root"],
    )

    final_script_text = str(st.session_state.get("gi_final_script_text", ""))
    case_prefix = extract_case_prefix_from_script(final_script_text) or case_prefix_from_form(st.session_state["workflow_preset"])
    station_name, model_scale = extract_station_and_scale_from_script(final_script_text)
    if not station_name or not model_scale:
        station_name, model_scale = station_and_scale_from_form()
    _, model_case = case_names(case_prefix, station_name, model_scale)

    st.caption(f"Case: `{model_case}`")
    st.caption(f"Script workspace: `{st.session_state.get('script_workspace_dir', '') or 'Not initialized'}`")

    if st.button("Initialize Case + Write general_info.py", use_container_width=True):
        run_init_case_step()


def _ha_l0_defaults_for_preset(preset: str) -> dict[str, Any]:
    if preset == "HRB_modeling":
        return {
            "ha_l0_dem_level0_path": r"F:\research\Research\ModelingUncertainty_hanjiang\data\DEM\ASTGTM2_mosaic_clip.tif",
            "ha_l0_flow_direction_pkg": "wbw",
            "ha_l0_stream_acc_threshold": "100000",
            "ha_l0_use_calc_threshold_kwargs": True,
            "ha_l0_calc_method": "drainage_area",
            "ha_l0_calc_drainage_area_km2": "0.01",
            "ha_l0_use_d8_streamnetwork_kwargs": True,
            "ha_l0_d8_snap_dist": "0.001",
            "ha_l0_use_snap_outlet_to_stream_kwargs": True,
            "ha_l0_snap_outlet_dist": "30.0",
            "ha_l0_use_filldem_kwargs": True,
            "ha_l0_fill_add_perturbation": False,
            "ha_l0_fill_depressions_bool": False,
            "ha_l0_fill_max_dist": "500",
            "ha_l0_fill_flat_increment": "0.001",
            "ha_l0_crs_str": "EPSG:4326",
            "ha_l0_esri_pointer": True,
            "ha_l0_outlet_lons_csv": "",
            "ha_l0_outlet_lats_csv": "",
        }
    return {}


def _apply_ha_l0_defaults(preset: str, force: bool = False) -> None:
    defaults = _ha_l0_defaults_for_preset(preset)
    for key, value in defaults.items():
        if force or key not in st.session_state:
            st.session_state[key] = value
    if preset == "HRB_modeling" and "ha_l0_fill_depressions_migrated" not in st.session_state:
        st.session_state["ha_l0_fill_depressions_bool"] = False
        st.session_state["ha_l0_fill_depressions_migrated"] = True


def _parse_optional_number(raw_text: str, label: str, errors: list[str]) -> int | float | None:
    raw = raw_text.strip()
    if not raw:
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+(\.\d+)?", raw):
        return float(raw)
    errors.append(f"`{label}` must be numeric.")
    return None


def _parse_required_csv_float_list(raw_text: str, label: str, errors: list[str]) -> list[float]:
    text = raw_text.strip()
    if not text:
        errors.append(f"`{label}` cannot be empty.")
        return []
    values: list[float] = []
    for token in text.split(","):
        token_str = token.strip()
        if not token_str:
            continue
        try:
            values.append(float(token_str))
        except ValueError:
            errors.append(f"`{label}` contains non-numeric value: {token_str}")
            return []
    if not values:
        errors.append(f"`{label}` cannot be empty.")
    return values


def _parse_stream_acc_threshold(raw_text: str, errors: list[str]) -> int | float | None:
    raw = raw_text.strip()
    if not raw or raw.lower() in {"none", "null"}:
        return None
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+(\.\d+)?", raw):
        return float(raw)
    errors.append("`stream_acc_threshold` must be numeric or `None`.")
    return None


def _script_enables_fill_depressions(script_text: str) -> bool:
    return bool(
        re.search(r"""['"]fill_depressions_bool['"]\s*:\s*True""", script_text)
        or re.search(r"""(?<!\w)fill_depressions_bool\s*=\s*True(?!\w)""", script_text)
    )


def _is_fill_depressions_panic(output_lines: list[str]) -> bool:
    joined = "\n".join(output_lines[-200:]) if output_lines else ""
    return "fill_depressions.rs" in joined and "Error unwrapping 'output'" in joined


def _collect_hrb_hydroanalysis_l0_form() -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    dem_level0_path = str(st.session_state.get("ha_l0_dem_level0_path", "")).strip()
    flow_direction_pkg = str(st.session_state.get("ha_l0_flow_direction_pkg", "")).strip()
    stream_acc_threshold_raw = str(st.session_state.get("ha_l0_stream_acc_threshold", "")).strip()
    crs_str = str(st.session_state.get("ha_l0_crs_str", "")).strip()
    esri_pointer = bool(st.session_state.get("ha_l0_esri_pointer", True))
    outlet_lons_csv = str(st.session_state.get("ha_l0_outlet_lons_csv", "")).strip()
    outlet_lats_csv = str(st.session_state.get("ha_l0_outlet_lats_csv", "")).strip()
    use_calc_threshold_kwargs = bool(st.session_state.get("ha_l0_use_calc_threshold_kwargs", False))
    use_d8_streamnetwork_kwargs = bool(st.session_state.get("ha_l0_use_d8_streamnetwork_kwargs", False))
    use_snap_outlet_to_stream_kwargs = bool(st.session_state.get("ha_l0_use_snap_outlet_to_stream_kwargs", False))
    use_filldem_kwargs = bool(st.session_state.get("ha_l0_use_filldem_kwargs", False))

    if not dem_level0_path:
        errors.append("`dem_level0_path` cannot be empty.")
    if not flow_direction_pkg:
        errors.append("`flow_direction_pkg` cannot be empty.")
    if not crs_str:
        errors.append("`crs_str` cannot be empty.")

    stream_acc_threshold = _parse_stream_acc_threshold(stream_acc_threshold_raw, errors)
    outlet_lons = _parse_required_csv_float_list(outlet_lons_csv, "outlets_with_reference_coords lon list", errors)
    outlet_lats = _parse_required_csv_float_list(outlet_lats_csv, "outlets_with_reference_coords lat list", errors)
    if outlet_lons and outlet_lats and len(outlet_lons) != len(outlet_lats):
        errors.append("`outlets_with_reference_coords` lon/lat length must match.")

    calculate_streamnetwork_threshold_kwargs: dict[str, Any] | None = None
    if use_calc_threshold_kwargs:
        calc_method = str(st.session_state.get("ha_l0_calc_method", "")).strip()
        calc_drainage_area_km2 = _parse_optional_number(
            str(st.session_state.get("ha_l0_calc_drainage_area_km2", "")).strip(),
            "drainage_area_km2",
            errors,
        )
        if not calc_method:
            errors.append("`calculate_streamnetwork_threshold_kwargs.method` cannot be empty.")
        calculate_streamnetwork_threshold_kwargs = {"method": calc_method}
        if calc_drainage_area_km2 is not None:
            calculate_streamnetwork_threshold_kwargs["drainage_area_km2"] = calc_drainage_area_km2

    d8_streamnetwork_kwargs: dict[str, Any] | None = None
    if use_d8_streamnetwork_kwargs:
        d8_snap_dist = _parse_optional_number(
            str(st.session_state.get("ha_l0_d8_snap_dist", "")).strip(),
            "d8_streamnetwork_kwargs.snap_dist",
            errors,
        )
        if d8_snap_dist is None:
            errors.append("`d8_streamnetwork_kwargs.snap_dist` cannot be empty.")
        else:
            d8_streamnetwork_kwargs = {"snap_dist": d8_snap_dist}

    snap_outlet_to_stream_kwargs: dict[str, Any] | None = None
    if use_snap_outlet_to_stream_kwargs:
        snap_outlet_dist = _parse_optional_number(
            str(st.session_state.get("ha_l0_snap_outlet_dist", "")).strip(),
            "snap_outlet_to_stream_kwargs.snap_dist",
            errors,
        )
        if snap_outlet_dist is None:
            errors.append("`snap_outlet_to_stream_kwargs.snap_dist` cannot be empty.")
        else:
            snap_outlet_to_stream_kwargs = {"snap_dist": snap_outlet_dist}

    filldem_kwargs: dict[str, Any] | None = None
    if use_filldem_kwargs:
        filldem_kwargs = {
            "add_perturbation": bool(st.session_state.get("ha_l0_fill_add_perturbation", False)),
            "fill_depressions_bool": bool(st.session_state.get("ha_l0_fill_depressions_bool", False)),
        }
        fill_max_dist = _parse_optional_number(
            str(st.session_state.get("ha_l0_fill_max_dist", "")).strip(),
            "filldem_kwargs.max_dist",
            errors,
        )
        fill_flat_increment = _parse_optional_number(
            str(st.session_state.get("ha_l0_fill_flat_increment", "")).strip(),
            "filldem_kwargs.flat_increment",
            errors,
        )
        if fill_max_dist is not None:
            filldem_kwargs["max_dist"] = fill_max_dist
        if fill_flat_increment is not None:
            filldem_kwargs["flat_increment"] = fill_flat_increment

    if errors:
        return None, errors

    return (
        {
            "dem_level0_path": dem_level0_path,
            "flow_direction_pkg": flow_direction_pkg,
            "stream_acc_threshold": stream_acc_threshold,
            "calculate_streamnetwork_threshold_kwargs": calculate_streamnetwork_threshold_kwargs,
            "d8_streamnetwork_kwargs": d8_streamnetwork_kwargs,
            "snap_outlet_to_stream_kwargs": snap_outlet_to_stream_kwargs,
            "crs_str": crs_str,
            "esri_pointer": esri_pointer,
            "outlets_with_reference_coords": [outlet_lons, outlet_lats],
            "filldem_kwargs": filldem_kwargs,
        },
        [],
    )


def _resolve_uploaded_dem_path(uploaded_file: Any) -> tuple[str | None, str | None]:
    if uploaded_file is None:
        return None, "No DEM file uploaded."

    upload_root = Path(__file__).resolve().parent / "_uploads" / "dem_level0"
    upload_root.mkdir(parents=True, exist_ok=True)

    file_name = Path(uploaded_file.name).name
    suffix = Path(file_name).suffix or ".tif"
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        return None, "Uploaded DEM file is empty."

    digest = hashlib.md5()
    digest.update(file_name.encode("utf-8", errors="ignore"))
    digest.update(file_bytes)
    token = digest.hexdigest()[:16]

    target_dir = upload_root / token
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"dem_level0{suffix}"
    target_path.write_bytes(file_bytes)
    return str(target_path), None


def _load_station_coords_from_general_info() -> dict[str, tuple[float, float]] | None:
    station_names_raw = str(st.session_state.get("gi_station_names_json", "")).strip()
    station_coords_raw = str(st.session_state.get("gi_station_coords_json", "")).strip()
    if not station_names_raw or not station_coords_raw:
        return None
    try:
        station_names = json.loads(station_names_raw)
        station_coords = json.loads(station_coords_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(station_names, list) or not isinstance(station_coords, dict):
        return None

    station_lon_lat: dict[str, tuple[float, float]] = {}
    for station_name in station_names:
        if not isinstance(station_name, str) or station_name not in station_coords:
            return None
        coord = station_coords.get(station_name)
        if not isinstance(coord, list) or len(coord) != 2:
            return None
        try:
            lat = float(coord[0])
            lon = float(coord[1])
        except (TypeError, ValueError):
            return None
        station_lon_lat[station_name] = (lon, lat)
    if not station_lon_lat:
        return None
    return station_lon_lat


def _parse_outlet_csv_for_merge(raw_text: str) -> list[float]:
    raw = raw_text.strip()
    if not raw:
        return []
    values: list[float] = []
    for token in raw.split(","):
        token_str = token.strip()
        if not token_str:
            continue
        values.append(float(token_str))
    return values


def _resolve_case_paths(preset: str) -> tuple[str, Path, Path]:
    final_script_text = str(st.session_state.get("gi_final_script_text", ""))
    case_prefix = extract_case_prefix_from_script(final_script_text) or case_prefix_from_form(preset)
    station_name, model_scale = extract_station_and_scale_from_script(final_script_text)
    if not station_name or not model_scale:
        station_name, model_scale = station_and_scale_from_form()
    _, model_case = case_names(case_prefix, station_name, model_scale)
    artifact_root = Path(str(st.session_state.get("artifact_root", "")).strip())
    case_dir = artifact_root / model_case
    wbw_level0_dir = case_dir / "Hydroanalysis" / "wbw_working_directory_level0"
    return model_case, case_dir, wbw_level0_dir


def _list_shp_files(search_root: Path) -> list[Path]:
    if not search_root.exists():
        return []
    return sorted(
        [path for path in search_root.rglob("*.shp") if path.is_file()],
        key=lambda path: str(path).lower(),
    )


def _list_raster_files(search_root: Path) -> list[Path]:
    if not search_root.exists():
        return []
    rasters: list[Path] = []
    rasters.extend([path for path in search_root.rglob("*.tif") if path.is_file()])
    rasters.extend([path for path in search_root.rglob("*.tiff") if path.is_file()])
    return sorted(rasters, key=lambda path: str(path).lower())


def _plot_shapefile(shp_path: Path) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except Exception as exc:
        return None, None, f"Failed to import geopandas/matplotlib: {exc}"

    try:
        gdf = gpd.read_file(shp_path)
    except Exception as exc:
        return None, None, f"Failed to read shapefile: {exc}"

    fig, axis = plt.subplots(figsize=(4.8, 3.6))
    if gdf.empty:
        axis.set_title(f"{shp_path.name} (empty)")
    else:
        gdf.plot(ax=axis, edgecolor="k", linewidth=0.5, alpha=0.6)
        axis.set_title(shp_path.name)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("lon")
    axis.set_ylabel("lat")
    fig.tight_layout()

    bounds = gdf.total_bounds.tolist() if not gdf.empty else None
    summary: dict[str, Any] = {
        "file": str(shp_path),
        "feature_count": len(gdf),
        "crs": str(gdf.crs) if gdf.crs is not None else None,
        "bounds": bounds,
        "columns": [column for column in gdf.columns if column != "geometry"],
    }
    return fig, summary, None


def _plot_raster(raster_path: Path) -> tuple[Any | None, dict[str, Any] | None, str | None]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
    except Exception as exc:
        return None, None, f"Failed to import raster dependencies: {exc}"

    try:
        with rasterio.open(raster_path, "r") as dataset:
            data = dataset.read(1, masked=True)
            raster_crs = str(dataset.crs) if dataset.crs is not None else None
            raster_bounds = list(dataset.bounds)
            raster_shape = [dataset.height, dataset.width]
            raster_dtype = str(dataset.dtypes[0]) if dataset.dtypes else None
            raster_nodata = dataset.nodata
    except Exception as exc:
        return None, None, f"Failed to read raster file: {exc}"

    fig, axis = plt.subplots(figsize=(5.0, 3.8))
    image = axis.imshow(data, cmap="terrain")
    fig.colorbar(image, ax=axis, shrink=0.8)
    axis.set_title(raster_path.name)
    axis.set_xlabel("col")
    axis.set_ylabel("row")
    fig.tight_layout()

    valid_values = np.asarray(data.compressed()) if hasattr(data, "compressed") else np.asarray(data).ravel()
    if valid_values.size == 0:
        min_value = None
        max_value = None
    else:
        min_value = float(np.nanmin(valid_values))
        max_value = float(np.nanmax(valid_values))

    summary: dict[str, Any] = {
        "file": str(raster_path),
        "shape": raster_shape,
        "dtype": raster_dtype,
        "crs": raster_crs,
        "nodata": raster_nodata,
        "bounds": raster_bounds,
        "min": min_value,
        "max": max_value,
    }
    return fig, summary, None


def _render_hydroanalysis_outputs_panel(preset: str) -> None:
    st.markdown("#### Hydroanalysis outputs visualization")
    model_case, case_dir, wbw_level0_dir = _resolve_case_paths(preset)
    hydroanalysis_dir = case_dir / "Hydroanalysis"
    st.caption(f"Case: `{model_case}`")
    st.caption(f"Hydroanalysis dir: `{hydroanalysis_dir}`")
    st.caption(f"Level0 working dir: `{wbw_level0_dir}`")

    if not hydroanalysis_dir.exists():
        st.info("Hydroanalysis output directory not found. Run Step 3 first.")
        return
    search_root = hydroanalysis_dir

    refresh_col, _ = st.columns([1, 4])
    with refresh_col:
        if st.button("Refresh output files", key="ha_l0_refresh_outputs", use_container_width=True):
            st.rerun()

    shp_files = _list_shp_files(search_root)
    raster_files = _list_raster_files(search_root)
    if not shp_files and not raster_files:
        st.info("No output `.shp/.tif/.tiff` files found yet.")
        return

    vector_tab, raster_tab = st.tabs(["Vector outputs (.shp)", "Raster outputs (.tif/.tiff)"])

    with vector_tab:
        if not shp_files:
            st.info("No `.shp` files found.")
        else:
            shp_options = [str(path) for path in shp_files]
            selected_shp = st.selectbox(
                "select shp output",
                options=shp_options,
                key="ha_l0_selected_output_shp",
            )

            if st.button("Plot selected shp", key="ha_l0_plot_selected_output", use_container_width=True):
                figure, summary, error_message = _plot_shapefile(Path(selected_shp))
                if error_message:
                    st.error(error_message)
                else:
                    st.json(summary)
                    plot_col, _ = st.columns([3, 2])
                    with plot_col:
                        st.pyplot(figure)

    with raster_tab:
        if not raster_files:
            st.info("No `.tif/.tiff` files found.")
        else:
            raster_options = [str(path) for path in raster_files]
            selected_raster = st.selectbox(
                "select raster output",
                options=raster_options,
                key="ha_l0_selected_output_raster",
            )

            if st.button("Plot selected raster", key="ha_l0_plot_selected_raster", use_container_width=True):
                figure, summary, error_message = _plot_raster(Path(selected_raster))
                if error_message:
                    st.error(error_message)
                else:
                    st.json(summary)
                    plot_col, _ = st.columns([3, 2])
                    with plot_col:
                        st.pyplot(figure)


def render_hydroanalysis_l0_hrb_script(config: dict[str, Any]) -> str:
    call_lines = [
        "        evb_dir_hydroanalysis_level0,",
        f"        dem_level0_path={config['dem_level0_path']!r},",
        f"        flow_direction_pkg={config['flow_direction_pkg']!r},",
        f"        stream_acc_threshold={python_repr(config['stream_acc_threshold'])},",
        f"        crs_str={config['crs_str']!r},",
        f"        esri_pointer={config['esri_pointer']!r},",
        f"        outlets_with_reference_coords={python_repr(config['outlets_with_reference_coords'])},",
    ]
    if config["calculate_streamnetwork_threshold_kwargs"] is not None:
        call_lines.append(
            f"        calculate_streamnetwork_threshold_kwargs={python_repr(config['calculate_streamnetwork_threshold_kwargs'])},"
        )
    if config["d8_streamnetwork_kwargs"] is not None:
        call_lines.append(f"        d8_streamnetwork_kwargs={python_repr(config['d8_streamnetwork_kwargs'])},")
    if config["snap_outlet_to_stream_kwargs"] is not None:
        call_lines.append(
            f"        snap_outlet_to_stream_kwargs={python_repr(config['snap_outlet_to_stream_kwargs'])},"
        )
    if config["filldem_kwargs"] is not None:
        call_lines.append(f"        filldem_kwargs={python_repr(config['filldem_kwargs'])},")

    lines = [
        "from pathlib import Path",
        "",
        "from easy_vic_build.Evb_dir_class import Evb_dir",
        "from easy_vic_build.build_hydroanalysis import buildHydroanalysis_level0",
        "from general_info import *",
        "",
        "",
        "def hydroanalysis_level0_HRB(evb_dir_hydroanalysis_level0):",
        "    buildHydroanalysis_level0(",
        *call_lines,
        "    )",
        "",
        "",
        "def _build_case_evb_dir():",
        "    case_scripts_dir = Path(__file__).resolve().parent",
        "    case_dir = case_scripts_dir.parent",
        "    cases_home = case_dir.parent",
        "    case_name = case_dir.name",
        "    evb_dir = Evb_dir(cases_home=str(cases_home))",
        "    evb_dir.builddir(case_name)",
        "    return evb_dir",
        "",
        "",
        "def run_hydroanalysis_level0():",
        "    evb_dir_case = _build_case_evb_dir()",
        "    hydroanalysis_level0_HRB(evb_dir_case)",
        "",
        "",
        "if __name__ == '__main__':",
        "    run_hydroanalysis_level0()",
        "",
    ]
    return "\n".join(lines)


def render_hydroanalysis_l0_wrapper_script(preset: str, function_name: str) -> str:
    script_name = hydroanalysis_script_name_for_preset(preset)
    function_name = function_name.strip() or hydroanalysis_l0_function_for_preset(preset)
    lines = [
        "import importlib.util",
        "import sys",
        "from pathlib import Path",
        "",
        "from general_info import *",
        "from easy_vic_build.Evb_dir_class import Evb_dir",
        "",
        f"PRESET_NAME = {preset!r}",
        f"PRESET_SCRIPT = {script_name!r}",
        f"LEVEL0_FUNCTION = {function_name!r}",
        "",
        "",
        "def _load_preset_module():",
        "    case_scripts_dir = Path(__file__).resolve().parent",
        "    repo_root = case_scripts_dir.parents[2]",
        "    preset_dir = repo_root / 'examples' / PRESET_NAME",
        "    preset_script_path = preset_dir / PRESET_SCRIPT",
        "    if not preset_script_path.exists():",
        "        raise FileNotFoundError(f'Preset hydroanalysis script not found: {preset_script_path}')",
        "",
        "    if str(case_scripts_dir) not in sys.path:",
        "        sys.path.insert(0, str(case_scripts_dir))",
        "    if str(preset_dir) not in sys.path:",
        "        sys.path.append(str(preset_dir))",
        "",
        "    spec = importlib.util.spec_from_file_location('evb_ui_preset_hydroanalysis', preset_script_path)",
        "    if spec is None or spec.loader is None:",
        "        raise RuntimeError(f'Cannot load preset hydroanalysis script: {preset_script_path}')",
        "    module = importlib.util.module_from_spec(spec)",
        "    spec.loader.exec_module(module)",
        "    return module",
        "",
        "",
        "def _build_case_evb_dir():",
        "    case_scripts_dir = Path(__file__).resolve().parent",
        "    case_dir = case_scripts_dir.parent",
        "    cases_home = case_dir.parent",
        "    case_name = case_dir.name",
        "    evb_dir = Evb_dir(cases_home=str(cases_home))",
        "    evb_dir.builddir(case_name)",
        "    return evb_dir",
        "",
        "",
        "def run_hydroanalysis_level0():",
        "    preset = _load_preset_module()",
        "    evb_dir_case = _build_case_evb_dir()",
        "    func = getattr(preset, LEVEL0_FUNCTION, None)",
        "    if func is None:",
        "        raise AttributeError(f'Function not found in preset script: {LEVEL0_FUNCTION}')",
        "    func(evb_dir_case)",
        "",
        "",
        "if __name__ == '__main__':",
        "    run_hydroanalysis_level0()",
        "",
    ]
    return "\n".join(lines)


def render_hydroanalysis_l0_tab() -> None:
    preset = st.session_state["workflow_preset"]
    script_name = hydroanalysis_script_name_for_preset(preset)

    if st.session_state.get("ha_l0_preset_applied") != preset:
        st.session_state["ha_l0_preset_applied"] = preset
        st.session_state["ha_l0_function_name"] = hydroanalysis_l0_function_for_preset(preset)
        st.session_state["ha_l0_final_script_text"] = ""
        st.session_state["ha_l0_cmd"] = ""
        _apply_ha_l0_defaults(preset, force=True)
    _apply_ha_l0_defaults(preset, force=False)

    st.subheader("Step 3: Hydroanalysis Level0")
    st.caption("Generate and run level0 hydroanalysis script based on preset implementation.")
    st.caption(f"Target script name: `{script_name}`")

    generated_script = ""
    if preset == "HRB_modeling":
        st.caption("Reference template: `hydroanalysis_level0_HRB`.")
        pending_outlet_lons = st.session_state.pop("ha_l0_pending_outlet_lons_csv", None)
        pending_outlet_lats = st.session_state.pop("ha_l0_pending_outlet_lats_csv", None)
        pending_outlet_msg = str(st.session_state.pop("ha_l0_pending_outlet_msg", "")).strip()
        if pending_outlet_lons is not None and pending_outlet_lats is not None:
            st.session_state["ha_l0_outlet_lons_csv"] = str(pending_outlet_lons)
            st.session_state["ha_l0_outlet_lats_csv"] = str(pending_outlet_lats)
        if pending_outlet_msg:
            st.success(pending_outlet_msg)

        uploaded_dem = st.file_uploader(
            "Select local DEM file (`.tif` / `.tiff`)",
            type=["tif", "tiff"],
            key="ha_l0_dem_upload",
        )
        if uploaded_dem is not None:
            st.caption(f"Selected local file: `{uploaded_dem.name}`")
            if st.button("Use uploaded DEM as dem_level0_path", key="ha_l0_use_uploaded_dem", use_container_width=True):
                resolved_path, upload_error = _resolve_uploaded_dem_path(uploaded_dem)
                if upload_error:
                    st.error(upload_error)
                else:
                    st.session_state["ha_l0_dem_level0_path"] = resolved_path
                    st.success(f"DEM path set: {resolved_path}")
        st.text_input("dem_level0_path", key="ha_l0_dem_level0_path")
        config_col_left, config_col_right = st.columns(2)
        with config_col_left:
            st.text_input("flow_direction_pkg", key="ha_l0_flow_direction_pkg")
            st.text_input("stream_acc_threshold (numeric or None)", key="ha_l0_stream_acc_threshold")
            st.text_input("crs_str", key="ha_l0_crs_str")
        with config_col_right:
            st.checkbox("esri_pointer", key="ha_l0_esri_pointer")
        outlet_col_left, outlet_col_right = st.columns(2)
        with outlet_col_left:
            st.text_input(
                "outlets_with_reference_coords lon list (comma-separated)",
                key="ha_l0_outlet_lons_csv",
                placeholder="107.023315,107.536583,...",
            )
        with outlet_col_right:
            st.text_input(
                "outlets_with_reference_coords lat list (comma-separated)",
                key="ha_l0_outlet_lats_csv",
                placeholder="33.049,33.218708,...",
            )
        station_coord_map = _load_station_coords_from_general_info()
        if station_coord_map is None:
            st.caption("Step 1 station_coords unavailable.")
        else:
            station_options = list(station_coord_map.keys())
            st.multiselect(
                "Select station coords from Step 1 (can select multiple)",
                options=station_options,
                key="ha_l0_station_coords_selected",
            )
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button(
                    "Append selected into outlets lists",
                    key="ha_l0_append_station_coords",
                    use_container_width=True,
                ):
                    selected_names = list(st.session_state.get("ha_l0_station_coords_selected", []))
                    if not selected_names:
                        st.warning("Select at least one station.")
                    else:
                        try:
                            existing_lons = _parse_outlet_csv_for_merge(str(st.session_state.get("ha_l0_outlet_lons_csv", "")))
                            existing_lats = _parse_outlet_csv_for_merge(str(st.session_state.get("ha_l0_outlet_lats_csv", "")))
                        except ValueError:
                            st.error("Current outlet lon/lat list is invalid. Please fix them first.")
                        else:
                            if len(existing_lons) != len(existing_lats):
                                st.error("Current outlet lon/lat length mismatch. Please fix them first.")
                            else:
                                for station_name in selected_names:
                                    lon, lat = station_coord_map[station_name]
                                    existing_lons.append(lon)
                                    existing_lats.append(lat)
                                st.session_state["ha_l0_pending_outlet_lons_csv"] = ",".join(str(value) for value in existing_lons)
                                st.session_state["ha_l0_pending_outlet_lats_csv"] = ",".join(str(value) for value in existing_lats)
                                st.session_state["ha_l0_pending_outlet_msg"] = "Selected Step 1 station coords appended to outlets lists."
                                st.rerun()
            with action_col2:
                if st.button(
                    "Replace outlets lists with selected",
                    key="ha_l0_replace_station_coords",
                    use_container_width=True,
                ):
                    selected_names = list(st.session_state.get("ha_l0_station_coords_selected", []))
                    if not selected_names:
                        st.warning("Select at least one station.")
                    else:
                        selected_lons: list[float] = []
                        selected_lats: list[float] = []
                        for station_name in selected_names:
                            lon, lat = station_coord_map[station_name]
                            selected_lons.append(lon)
                            selected_lats.append(lat)
                        st.session_state["ha_l0_pending_outlet_lons_csv"] = ",".join(str(value) for value in selected_lons)
                        st.session_state["ha_l0_pending_outlet_lats_csv"] = ",".join(str(value) for value in selected_lats)
                        st.session_state["ha_l0_pending_outlet_msg"] = "Outlets lists replaced by selected Step 1 station coords."
                        st.rerun()

        with st.expander("Optional parameters (fill in)", expanded=True):
            st.checkbox(
                "Enable calculate_streamnetwork_threshold_kwargs",
                key="ha_l0_use_calc_threshold_kwargs",
            )
            if st.session_state.get("ha_l0_use_calc_threshold_kwargs", False):
                calc_left, calc_right = st.columns(2)
                with calc_left:
                    st.text_input("method", key="ha_l0_calc_method")
                with calc_right:
                    st.text_input("drainage_area_km2", key="ha_l0_calc_drainage_area_km2")

            st.checkbox("Enable d8_streamnetwork_kwargs", key="ha_l0_use_d8_streamnetwork_kwargs")
            if st.session_state.get("ha_l0_use_d8_streamnetwork_kwargs", False):
                st.text_input("d8 snap_dist", key="ha_l0_d8_snap_dist")

            st.checkbox("Enable snap_outlet_to_stream_kwargs", key="ha_l0_use_snap_outlet_to_stream_kwargs")
            if st.session_state.get("ha_l0_use_snap_outlet_to_stream_kwargs", False):
                st.text_input("snap_outlet snap_dist", key="ha_l0_snap_outlet_dist")

            st.checkbox("Enable filldem_kwargs", key="ha_l0_use_filldem_kwargs")
            if st.session_state.get("ha_l0_use_filldem_kwargs", False):
                fill_left, fill_mid, fill_right = st.columns(3)
                with fill_left:
                    st.checkbox("add_perturbation", key="ha_l0_fill_add_perturbation")
                    st.checkbox("fill_depressions_bool", key="ha_l0_fill_depressions_bool")
                    if st.session_state.get("ha_l0_fill_depressions_bool", False):
                        st.warning(
                            "Current Whitebox version may panic in `fill_depressions`; "
                            "recommend disabling `fill_depressions_bool` if this happens."
                        )
                with fill_mid:
                    st.text_input("max_dist", key="ha_l0_fill_max_dist")
                with fill_right:
                    st.text_input("flat_increment", key="ha_l0_fill_flat_increment")

        hrb_config, hrb_errors = _collect_hrb_hydroanalysis_l0_form()
        if hrb_errors:
            for message in hrb_errors:
                st.error(message)
        if hrb_config is not None:
            generated_script = render_hydroanalysis_l0_hrb_script(hrb_config)
    else:
        st.text_input("level0 function name", key="ha_l0_function_name")
        generated_script = render_hydroanalysis_l0_wrapper_script(
            preset,
            str(st.session_state.get("ha_l0_function_name", "")).strip(),
        )

    if not str(st.session_state.get("ha_l0_final_script_text", "")).strip():
        st.session_state["ha_l0_final_script_text"] = generated_script

    load_col, _ = st.columns([1, 3])
    with load_col:
        if st.button("Load generated into editor", key="ha_l0_load_generated", use_container_width=True, disabled=not bool(generated_script)):
            st.session_state["ha_l0_final_script_text"] = generated_script
            st.rerun()

    st.text_area("final hydroanalysis.py (editable)", key="ha_l0_final_script_text", height=360)

    workspace_text = str(st.session_state.get("script_workspace_dir", "")).strip()
    workspace_path = Path(workspace_text) if workspace_text else None
    workspace_ready = bool(workspace_path and workspace_path.exists())
    if workspace_ready:
        target_path = workspace_path / script_name
        st.caption(f"Save path: `{target_path}`")
    else:
        target_path = None
        st.warning("Step 2 is required before running Step 3. Initialize case first.")
        if st.button("Initialize case now (run Step 2)", key="ha_l0_init_case_now", use_container_width=True):
            if run_init_case_step():
                st.rerun()

    if not str(st.session_state.get("ha_l0_cmd", "")).strip():
        st.session_state["ha_l0_cmd"] = f"{st.session_state.get('ha_l0_python_cmd', 'python')} {script_name}"
    st.text_input("run command", key="ha_l0_cmd")

    def _write_and_run_hydroanalysis(final_text_to_run: str) -> None:
        if not workspace_ready or target_path is None or workspace_path is None:
            st.error("Run Step 2 first to initialize case and create `scripts/` workspace.")
            return
        final_text = final_text_to_run.strip()
        if not final_text:
            st.error("`final hydroanalysis.py` is empty.")
            return
        if _script_enables_fill_depressions(final_text):
            st.warning(
                "Detected `fill_depressions_bool=True`. If Whitebox panics in `fill_depressions.rs`, "
                "set it to `False` and rerun."
            )
        target_path.write_text(final_text + "\n", encoding="utf-8")
        command = str(st.session_state.get("ha_l0_cmd", "")).strip()
        if not command:
            st.error("Run command is empty.")
            return

        code = run_command_with_output(command, str(workspace_path))
        if code == 0:
            final_script_text = str(st.session_state.get("gi_final_script_text", ""))
            case_prefix = extract_case_prefix_from_script(final_script_text) or case_prefix_from_form(preset)
            station_name, model_scale = extract_station_and_scale_from_script(final_script_text)
            if not station_name or not model_scale:
                station_name, model_scale = station_and_scale_from_form()
            _, model_case = case_names(case_prefix, station_name, model_scale)
            st.session_state["active_cases_home"] = st.session_state["artifact_root"]
            st.session_state["active_case_name"] = model_case
            st.session_state["ha_l0_last_case"] = model_case
            st.success("Hydroanalysis level0 completed.")
        else:
            st.error(f"Hydroanalysis level0 failed: {code}")
            last_output_lines = st.session_state.get("last_command_output_lines", [])
            if isinstance(last_output_lines, list) and _is_fill_depressions_panic(last_output_lines):
                st.error(
                    "Whitebox `fill_depressions` crashed. Edit `filldem_kwargs` in the script and set "
                    "`fill_depressions_bool=False`, then rerun."
                )
                st.info("For deeper debug, run with `set RUST_BACKTRACE=1 && <your command>` on Windows.")

    save_col, run_col = st.columns(2)
    with save_col:
        if st.button("Write hydroanalysis.py to scripts", use_container_width=True, disabled=not workspace_ready):
            final_text = str(st.session_state.get("ha_l0_final_script_text", "")).strip()
            if not final_text:
                st.error("`final hydroanalysis.py` is empty.")
            else:
                assert target_path is not None
                target_path.write_text(final_text + "\n", encoding="utf-8")
                st.success(f"Written: {target_path}")

    with run_col:
        if st.button("Run hydroanalysis", use_container_width=True, disabled=not workspace_ready):
            final_text = str(st.session_state.get("ha_l0_final_script_text", ""))
            _write_and_run_hydroanalysis(final_text)

    _render_hydroanalysis_outputs_panel(preset)
