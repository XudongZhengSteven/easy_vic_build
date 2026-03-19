from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import streamlit as st

from ui_config import DEFAULT_ARTIFACT_ROOT
from ui_general_info import collect_general_info_form
from ui_helpers import (
    case_names,
    case_prefix_from_form,
    dpc_level_specs_for_preset,
    dpc_script_name_for_preset,
    ensure_src_path_for_ui,
    extract_case_prefix_from_script,
    is_number,
    load_basin_shp_with_shx_repair,
    python_repr,
)


def dpc_field_state_key(level_id: str, field_name: str) -> str:
    return f"dpc_field_{level_id}_{field_name}"


def apply_dpc_defaults(preset: str, force: bool = False) -> None:
    preset_changed = st.session_state.get("dpc_preset_applied") != preset
    do_force = force or preset_changed
    st.session_state["dpc_preset_applied"] = preset

    if do_force or "dpc_grid_expand_grids_num" not in st.session_state:
        st.session_state["dpc_grid_expand_grids_num"] = 1
    if do_force or "dpc_grid_plot" not in st.session_state:
        st.session_state["dpc_grid_plot"] = True
    if do_force or "dpc_prepare_basin_shp_path" not in st.session_state:
        st.session_state["dpc_prepare_basin_shp_path"] = ""
    if do_force or "dpc_grid_res_level0_override" not in st.session_state:
        st.session_state["dpc_grid_res_level0_override"] = ""
    if do_force or "dpc_grid_res_level1_override" not in st.session_state:
        st.session_state["dpc_grid_res_level1_override"] = ""
    if do_force or "dpc_grid_res_level2_override" not in st.session_state:
        st.session_state["dpc_grid_res_level2_override"] = ""
    if do_force or "dpc_final_script_text" not in st.session_state:
        st.session_state["dpc_final_script_text"] = ""

    for level_spec in dpc_level_specs_for_preset(preset):
        level_id = str(level_spec["id"])
        enable_key = f"dpc_enable_{level_id}"
        steps_key = f"dpc_steps_{level_id}"
        extra_kwargs_key = f"dpc_extra_kwargs_{level_id}"

        if do_force or enable_key not in st.session_state:
            st.session_state[enable_key] = bool(level_spec.get("default_enabled", False))
        if do_force or steps_key not in st.session_state:
            st.session_state[steps_key] = ""
        if do_force or extra_kwargs_key not in st.session_state:
            st.session_state[extra_kwargs_key] = "{}"

        for field_spec in level_spec.get("fields", []):
            field_name = str(field_spec["name"])
            field_key = dpc_field_state_key(level_id, field_name)
            if do_force or field_key not in st.session_state:
                st.session_state[field_key] = field_spec.get("default")

    if do_force:
        st.session_state["dpc_final_script_text"] = ""


def collect_dpc_prepare_form() -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    expand_grids_num = int(st.session_state.get("dpc_grid_expand_grids_num", 1))
    if expand_grids_num < 0:
        errors.append("`expand_grids_num` must be >= 0.")
    grid_plot = bool(st.session_state.get("dpc_grid_plot", True))
    basin_shp_path = str(st.session_state.get("dpc_prepare_basin_shp_path", "")).strip()

    def parse_optional_float(raw_text: str, label: str) -> float | None:
        raw = str(raw_text).strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            errors.append(f"`{label}` must be numeric if provided.")
            return None
        if value <= 0:
            errors.append(f"`{label}` must be greater than 0 if provided.")
            return None
        return value

    grid_res_level0_override = parse_optional_float(
        st.session_state.get("dpc_grid_res_level0_override", ""),
        "grid_res_level0 override",
    )
    grid_res_level1_override = parse_optional_float(
        st.session_state.get("dpc_grid_res_level1_override", ""),
        "grid_res_level1 override",
    )
    grid_res_level2_override = parse_optional_float(
        st.session_state.get("dpc_grid_res_level2_override", ""),
        "grid_res_level2 override",
    )

    prepare_cfg: dict[str, Any] = {
        "build_basin_and_grid_shp": True,
        "expand_grids_num": expand_grids_num,
        "plot": grid_plot,
        "basin_shp_path": basin_shp_path,
        "grid_res_level0": grid_res_level0_override,
        "grid_res_level1": grid_res_level1_override,
        "grid_res_level2": grid_res_level2_override,
    }
    if errors:
        return None, errors
    return prepare_cfg, []


def collect_build_dpc_form(preset: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []

    prepare_cfg, prepare_errors = collect_dpc_prepare_form()
    errors.extend(prepare_errors)

    levels: dict[str, dict[str, Any]] = {}
    for level_spec in dpc_level_specs_for_preset(preset):
        level_id = str(level_spec["id"])
        level_title = str(level_spec.get("title", level_id))
        enable_key = f"dpc_enable_{level_id}"
        steps_key = f"dpc_steps_{level_id}"
        extra_kwargs_key = f"dpc_extra_kwargs_{level_id}"

        processing_steps = [
            line.strip()
            for line in str(st.session_state.get(steps_key, "")).splitlines()
            if line.strip()
        ]

        raw_extra_kwargs = str(st.session_state.get(extra_kwargs_key, "")).strip()
        if not raw_extra_kwargs:
            extra_kwargs: dict[str, Any] = {}
        else:
            try:
                parsed_extra_kwargs = json.loads(raw_extra_kwargs)
                if not isinstance(parsed_extra_kwargs, dict):
                    errors.append(f"`{level_title}` extra loaddata_kwargs must be a JSON object.")
                    parsed_extra_kwargs = {}
                extra_kwargs = parsed_extra_kwargs
            except json.JSONDecodeError as exc:
                errors.append(
                    f"`{level_title}` extra loaddata_kwargs JSON parse error: {exc.msg} (line {exc.lineno}, col {exc.colno})"
                )
                extra_kwargs = {}

        level_config: dict[str, Any] = {
            "enabled": bool(st.session_state.get(enable_key, False)),
            "processing_steps": processing_steps,
            "extra_loaddata_kwargs": extra_kwargs,
        }

        for field_spec in level_spec.get("fields", []):
            field_name = str(field_spec["name"])
            field_type = str(field_spec.get("type", "text"))
            field_key = dpc_field_state_key(level_id, field_name)
            field_value = st.session_state.get(field_key, field_spec.get("default"))

            if field_type == "text":
                level_config[field_name] = str(field_value).strip()
            elif field_type == "bool":
                level_config[field_name] = bool(field_value)
            else:
                level_config[field_name] = field_value

        levels[level_id] = level_config

    config: dict[str, Any] = {
        "prepare": prepare_cfg if prepare_cfg is not None else {},
        "levels": levels,
    }

    if errors:
        return None, errors
    return config, []


def resolve_prepare_basin_shp(
    preset: str,
    gi_config: dict[str, Any],
    basin_shp_path: str,
    Basins: Any,
) -> tuple[Any, dict[str, Any], str]:
    station_name = str(gi_config["station_name"])

    if basin_shp_path:
        shp_path = Path(basin_shp_path)
        if not shp_path.exists():
            raise FileNotFoundError(f"Custom basin_shp path not found: {shp_path}")

        basin_shp = load_basin_shp_with_shx_repair(Basins, shp_path)
        if preset == "HRB_modeling":
            basin_shps = {name: basin_shp for name in gi_config["station_names"]}
        else:
            basin_shps = {station_name: basin_shp}
        return basin_shp, basin_shps, str(shp_path)

    final_script_text = str(st.session_state.get("gi_final_script_text", ""))
    case_prefix = extract_case_prefix_from_script(final_script_text) or case_prefix_from_form(preset)
    _, model_case = case_names(case_prefix, station_name, str(gi_config["model_scale"]))
    hydroanalysis_dir = Path(str(st.session_state.get("artifact_root", DEFAULT_ARTIFACT_ROOT)).strip()) / model_case / "Hydroanalysis"
    wbw_dir = hydroanalysis_dir / "wbw_working_directory_level0"
    if not wbw_dir.exists():
        raise FileNotFoundError(
            f"Hydroanalysis directory not found: {wbw_dir}. "
            "Run hydroanalysis first or provide custom basin_shp path."
        )

    basin_outlet_map: dict[str, int] = gi_config["basin_outlets_reference_i_map"]
    if preset == "HRB_modeling":
        basin_shps: dict[str, Any] = {}
        for name in gi_config["station_names"]:
            station_id = basin_outlet_map[name]
            shp_path = wbw_dir / f"basin_vector_outlet_with_reference_{station_id}.shp"
            if not shp_path.exists():
                raise FileNotFoundError(f"Station basin_shp not found: {shp_path}")
            basin_shps[name] = load_basin_shp_with_shx_repair(Basins, shp_path)
        return basin_shps[station_name], basin_shps, str(wbw_dir)

    station_id = basin_outlet_map[station_name]
    shp_path = wbw_dir / f"basin_vector_outlet_with_reference_{station_id}.shp"
    if not shp_path.exists():
        raise FileNotFoundError(f"Station basin_shp not found: {shp_path}")
    basin_shp = load_basin_shp_with_shx_repair(Basins, shp_path)
    return basin_shp, {station_name: basin_shp}, str(shp_path)


def resolve_uploaded_basin_shp_path(uploaded_files: list[Any]) -> tuple[str | None, str | None]:
    if not uploaded_files:
        return None, "No files uploaded."

    upload_root = Path(__file__).resolve().parent / "_uploads" / "basin_shp"
    upload_root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.md5()
    for uploaded in uploaded_files:
        digest.update(uploaded.name.encode("utf-8", errors="ignore"))
        digest.update(uploaded.getvalue())
    token = digest.hexdigest()[:16]

    target_dir = upload_root / token
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        for uploaded in uploaded_files:
            file_name = Path(uploaded.name).name
            file_bytes = uploaded.getvalue()
            if file_name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
                    archive.extractall(target_dir)
            else:
                (target_dir / file_name).write_bytes(file_bytes)
    except Exception as exc:
        return None, f"Failed to process uploaded files: {exc}"

    shp_candidates = sorted(target_dir.rglob("*.shp"))
    if not shp_candidates:
        return None, "No `.shp` file found in uploaded files."
    return str(shp_candidates[0]), None


def resolve_prepare_grid_resolutions(
    gi_config: dict[str, Any],
    prepare_cfg: dict[str, Any],
) -> tuple[tuple[float, float, float] | None, str | None]:
    model_scale = str(gi_config["model_scale"])
    scalemap = gi_config["scalemap"]
    if model_scale not in scalemap:
        return None, f"`model_scale` ({model_scale}) not found in `scalemap`."

    default_level0 = float(gi_config["grid_res_level0"])
    default_level1 = scalemap[model_scale]
    default_level2 = scalemap[model_scale]
    if not is_number(default_level1) or not is_number(default_level2):
        return None, "`scalemap[model_scale]` must be numeric for grid building."

    level0 = prepare_cfg.get("grid_res_level0")
    level1 = prepare_cfg.get("grid_res_level1")
    level2 = prepare_cfg.get("grid_res_level2")

    level0_value = float(level0) if level0 is not None else default_level0
    level1_value = float(level1) if level1 is not None else float(default_level1)
    level2_value = float(level2) if level2 is not None else float(default_level2)
    return (level0_value, level1_value, level2_value), None


def run_build_basin_shp_preview(
    basin_shp_path: str,
) -> tuple[dict[str, Any] | None, Any | None, str | None]:
    basin_shp_path = str(basin_shp_path).strip()
    if not basin_shp_path:
        return None, None, "`basin_shp_path` is required for build_basin_shp."

    path_obj = Path(basin_shp_path)
    if not path_obj.exists():
        return None, None, f"`basin_shp_path` not found: {path_obj}"

    try:
        ensure_src_path_for_ui()
        from easy_vic_build.tools.dpc_func.basin_grid_class import Basins
        import matplotlib.pyplot as plt
    except Exception as exc:
        return None, None, f"Failed to import dependencies: {exc}"

    try:
        basin_shp = load_basin_shp_with_shx_repair(Basins, path_obj)
    except Exception as exc:
        message = f"build_basin_shp failed: {exc}"
        if ".shx" in str(exc).lower():
            message += " | Please include `.shp/.shx/.dbf/.prj` together (or upload a full zip)."
        return None, None, message

    fig, axis = plt.subplots(figsize=(6, 5))
    basin_shp.plot(ax=axis, edgecolor="k", alpha=0.45, facecolor="lightskyblue")
    axis.set_title("basin_shp")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("lon")
    axis.set_ylabel("lat")
    fig.tight_layout()

    bounds = basin_shp.total_bounds.tolist() if hasattr(basin_shp, "total_bounds") else None
    summary = {
        "basin_shp_path": str(path_obj),
        "feature_count": len(basin_shp),
        "bounds": bounds,
    }
    return summary, fig, None


def run_build_grid_shp_preview(
    preset: str,
    prepare_cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, Any | None, str | None]:
    gi_config, gi_errors = collect_general_info_form(preset)
    if gi_config is None:
        return None, None, "General Info has errors: " + "; ".join(gi_errors)

    try:
        ensure_src_path_for_ui()
        from easy_vic_build.tools.dpc_func.basin_grid_class import Basins
        from easy_vic_build.tools.dpc_func.basin_grid_func import build_grid_shp
        import matplotlib.pyplot as plt
    except Exception as exc:
        return None, None, f"Failed to import easy_vic_build DPC dependencies: {exc}"

    resolved_grids, resolution_error = resolve_prepare_grid_resolutions(gi_config, prepare_cfg)
    if resolved_grids is None:
        return None, None, resolution_error
    grid_res_level0, grid_res_level1, grid_res_level2 = resolved_grids

    basin_shp_path = str(prepare_cfg.get("basin_shp_path", "")).strip()
    try:
        if basin_shp_path:
            basin_shp = load_basin_shp_with_shx_repair(Basins, basin_shp_path)
            basin_source = basin_shp_path
        else:
            basin_shp, _, basin_source = resolve_prepare_basin_shp(preset, gi_config, "", Basins)

        grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = build_grid_shp(
            basin_shp,
            float(grid_res_level0),
            float(grid_res_level1),
            float(grid_res_level2),
            expand_grids_num=int(prepare_cfg.get("expand_grids_num", 1)),
            plot=False,
        )
    except Exception as exc:
        return None, None, f"build_grid_shp failed: {exc}"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    basin_shp.plot(ax=axes[0], edgecolor="k", alpha=0.35, facecolor="lightskyblue")
    grid_shp_level0.plot(ax=axes[0], alpha=0.6, edgecolor="k", linewidth=0.4)
    axes[0].set_title("Level0")

    basin_shp.plot(ax=axes[1], edgecolor="k", alpha=0.35, facecolor="lightskyblue")
    grid_shp_level1.plot(ax=axes[1], alpha=0.6, edgecolor="k", linewidth=0.4)
    if hasattr(grid_shp_level1, "point_geometry"):
        grid_shp_level1.point_geometry.plot(ax=axes[1], alpha=0.8, color="navy", markersize=1)
    axes[1].set_title("Level1")

    basin_shp.plot(ax=axes[2], edgecolor="k", alpha=0.35, facecolor="lightskyblue")
    grid_shp_level2.plot(ax=axes[2], alpha=0.6, edgecolor="k", linewidth=0.4)
    if hasattr(grid_shp_level2, "point_geometry"):
        grid_shp_level2.point_geometry.plot(ax=axes[2], alpha=0.8, color="navy", markersize=1)
    axes[2].set_title("Level2")

    basin_shp.plot(ax=axes[3], edgecolor="k", alpha=0.35, facecolor="lightskyblue")
    grid_shp_level3.plot(ax=axes[3], alpha=0.6, edgecolor="k", linewidth=0.4)
    axes[3].set_title("Level3")

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("lon")
        axis.set_ylabel("lat")
    fig.tight_layout()

    summary = {
        "basin_source": basin_source,
        "station_name": gi_config["station_name"],
        "grid_counts": {
            "level0": len(grid_shp_level0),
            "level1": len(grid_shp_level1),
            "level2": len(grid_shp_level2),
            "level3": len(grid_shp_level3),
        },
        "grid_resolution": {
            "level0": grid_res_level0,
            "level1": grid_res_level1,
            "level2": grid_res_level2,
        },
        "expand_grids_num": int(prepare_cfg.get("expand_grids_num", 1)),
    }
    return summary, fig, None


def render_build_dpc_script(preset: str, config: dict[str, Any]) -> str:
    script_name = dpc_script_name_for_preset(preset)
    lines: list[str] = [
        "import os",
        "import sys",
        "import importlib.util",
        "from pathlib import Path",
        "",
        "from easy_vic_build.tools.dpc_func.basin_grid_class import Basins",
        "",
        "from general_info import *",
        "",
        f"PRESET_NAME = {preset!r}",
        f"PRESET_SCRIPT = {script_name!r}",
        f"DPC_CONFIG = {python_repr(config)}",
        "",
        "",
        "def _load_preset_module():",
        "    case_scripts_dir = Path(__file__).resolve().parent",
        "    repo_root = case_scripts_dir.parents[2]",
        "    preset_dir = repo_root / 'examples' / PRESET_NAME",
        "    preset_script_path = preset_dir / PRESET_SCRIPT",
        "    if not preset_script_path.exists():",
        "        raise FileNotFoundError(f'Preset script not found: {preset_script_path}')",
        "",
        "    if str(case_scripts_dir) not in sys.path:",
        "        sys.path.insert(0, str(case_scripts_dir))",
        "    if str(preset_dir) not in sys.path:",
        "        sys.path.append(str(preset_dir))",
        "",
        "    spec = importlib.util.spec_from_file_location('evb_ui_preset_build_dpc', preset_script_path)",
        "    if spec is None or spec.loader is None:",
        "        raise RuntimeError(f'Cannot load preset script: {preset_script_path}')",
        "    module = importlib.util.module_from_spec(spec)",
        "    spec.loader.exec_module(module)",
        "    return module",
        "",
        "",
        "preset = _load_preset_module()",
        "",
        "",
        "def _run_pipeline(processor, save_path, loaddata_kwargs, processing_steps=None, clear_before=None):",
        "    if clear_before:",
        "        for item in clear_before:",
        "            try:",
        "                processor.clear_data_from_cache(",
        "                    save_names=item.get('save_names'),",
        "                    step_name=item.get('step_name'),",
        "                )",
        "            except Exception:",
        "                pass",
        "",
        "    if processing_steps:",
        "        processor.loaddata_kwargs = loaddata_kwargs",
        "        for step_name in processing_steps:",
        "            processor._execute_step(step_name, save_path=save_path)",
        "        processor.save_state(save_path)",
        "    else:",
        "        processor.loaddata_pipeline(save_path=save_path, loaddata_kwargs=loaddata_kwargs)",
        "",
        "",
        "def _load_basin_shp(shp_path):",
        "    previous = os.environ.get('SHAPE_RESTORE_SHX')",
        "    os.environ['SHAPE_RESTORE_SHX'] = 'YES'",
        "    try:",
        "        return Basins.from_shapefile(str(shp_path))",
        "    finally:",
        "        if previous is None:",
        "            os.environ.pop('SHAPE_RESTORE_SHX', None)",
        "        else:",
        "            os.environ['SHAPE_RESTORE_SHX'] = previous",
        "",
        "",
    ]

    if preset == "HRB_modeling":
        lines.extend(
            [
                "def build_dpc(evb_dir_hydroanalysis, evb_dir_modeling, date_period, reverse_lat=reverse_lat):",
                "    levels_cfg = DPC_CONFIG['levels']",
                "    prepare_cfg = DPC_CONFIG['prepare']",
                "",
                "    # Step 1: read basin_shp and build grid_shp",
                "    custom_basin_shp_path = str(prepare_cfg.get('basin_shp_path', '')).strip()",
                "    if custom_basin_shp_path:",
                "        basin_shp = _load_basin_shp(custom_basin_shp_path)",
                "        basin_shps = {name: basin_shp for name in station_names}",
                "    else:",
                "        basin_shps = preset.build_basin_shp_JRB(evb_dir_hydroanalysis)",
                "        basin_shp = basin_shps[station_name]",
                "    grid_res_level0_cfg = prepare_cfg.get('grid_res_level0')",
                "    grid_res_level1_cfg = prepare_cfg.get('grid_res_level1')",
                "    grid_res_level2_cfg = prepare_cfg.get('grid_res_level2')",
                "    if grid_res_level0_cfg is None:",
                "        grid_res_level0_cfg = grid_res_level0",
                "    if grid_res_level1_cfg is None:",
                "        grid_res_level1_cfg = grid_res_level1",
                "    if grid_res_level2_cfg is None:",
                "        grid_res_level2_cfg = grid_res_level2",
                "",
                "    grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = preset.build_grid_shp(",
                "        basin_shp,",
                "        grid_res_level0_cfg,",
                "        grid_res_level1_cfg,",
                "        grid_res_level2_cfg,",
                "        expand_grids_num=prepare_cfg['expand_grids_num'],",
                "        plot=prepare_cfg['plot'],",
                "    )",
                "",
                "    level_cfg = levels_cfg['base']",
                "    if level_cfg['enabled']:",
                "        processor = preset.dataProcess_base(",
                "            load_path=os.path.join(evb_dir_modeling.dpcFile_dir, 'dpc_VIC_base.pkl'),",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level1,",
                "            'grid_res': grid_res_level1,",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            os.path.join(evb_dir_modeling.dpcFile_dir, 'dpc_VIC_base.pkl'),",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "",
                "    level_cfg = levels_cfg['level0']",
                "    if level_cfg['enabled']:",
                "        processor = preset.dataProcess_VIC_level0_HRB(",
                "            load_path=evb_dir_modeling._dpc_VIC_level0_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level0,",
                "            'grid_res': grid_res_level0,",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            evb_dir_modeling._dpc_VIC_level0_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "        if level_cfg.get('plot_after', False):",
                "            processor.plot()",
                "",
                "    level_cfg = levels_cfg['level2_cmfd']",
                "    if level_cfg['enabled']:",
                "        save_path = evb_dir_modeling._dpc_VIC_level2_path.replace('.pkl', '_CMFD.pkl')",
                "        processor = preset.dataProcess_VIC_level2_CMFD_HRB(",
                "            load_path=save_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level2,",
                "            'grid_res': grid_res_level2,",
                "            'date_period': date_period,",
                "            'search_method': level_cfg.get('search_method', 'radius_rectangle_reverse'),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            save_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "",
                "    level_cfg = levels_cfg['level1']",
                "    if level_cfg['enabled']:",
                "        processor = preset.dataProcess_VIC_level1_HRB(",
                "            load_path=evb_dir_modeling._dpc_VIC_level1_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level1,",
                "            'grid_res': grid_res_level1,",
                "            'date_period': date_period,",
                "            'evb_dir': evb_dir_modeling,",
                "            'reverse_lat': reverse_lat,",
                "            'search_method_st': level_cfg.get('search_method_st', 'radius_rectangle_reverse'),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            evb_dir_modeling._dpc_VIC_level1_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "        if level_cfg.get('plot_after', False):",
                "            processor.plot()",
                "",
                "    level_cfg = levels_cfg['level3']",
                "    if level_cfg['enabled']:",
                "        processor = preset.dataProcess_VIC_level3_HRB(",
                "            load_path=evb_dir_modeling._dpc_VIC_level3_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'basin_shps': basin_shps,",
                "            'date_period': date_period,",
                "            'station_names': station_names,",
                "            'load_level1': level_cfg.get('load_level1', False),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            evb_dir_modeling._dpc_VIC_level3_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "",
                "    level_cfg = levels_cfg['level3_load_level1']",
                "    if level_cfg['enabled']:",
                "        processor = preset.dataProcess_VIC_level3_HRB(",
                "            load_path=evb_dir_modeling._dpc_VIC_level3_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'basin_shps': basin_shps,",
                "            'date_period': date_period,",
                "            'station_names': station_names,",
                "            'load_level1': level_cfg.get('load_level1', True),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        clear_before = []",
                "        if level_cfg.get('clear_gauge_info', True):",
                "            clear_before.append({'save_names': ['gauge_info'], 'step_name': 'load_gauge_info'})",
                "        _run_pipeline(",
                "            processor,",
                "            evb_dir_modeling._dpc_VIC_level3_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "            clear_before=clear_before,",
                "        )",
            ]
        )
    else:
        lines.extend(
            [
                "def build_dpc(evb_dir_hydroanalysis, evb_dir_modeling, date_period, reverse_lat=reverse_lat):",
                "    levels_cfg = DPC_CONFIG['levels']",
                "    prepare_cfg = DPC_CONFIG['prepare']",
                "",
                "    # Step 1: read basin_shp and build grid_shp",
                "    custom_basin_shp_path = str(prepare_cfg.get('basin_shp_path', '')).strip()",
                "    if custom_basin_shp_path:",
                "        basin_shp = _load_basin_shp(custom_basin_shp_path)",
                "    else:",
                "        basin_shp = preset.build_basin_shp(evb_dir_hydroanalysis)",
                "    grid_res_level0_cfg = prepare_cfg.get('grid_res_level0')",
                "    grid_res_level1_cfg = prepare_cfg.get('grid_res_level1')",
                "    grid_res_level2_cfg = prepare_cfg.get('grid_res_level2')",
                "    if grid_res_level0_cfg is None:",
                "        grid_res_level0_cfg = grid_res_level0",
                "    if grid_res_level1_cfg is None:",
                "        grid_res_level1_cfg = grid_res_level1",
                "    if grid_res_level2_cfg is None:",
                "        grid_res_level2_cfg = grid_res_level2",
                "",
                "    grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = preset.build_grid_shp(",
                "        basin_shp,",
                "        grid_res_level0_cfg,",
                "        grid_res_level1_cfg,",
                "        grid_res_level2_cfg,",
                "        expand_grids_num=prepare_cfg['expand_grids_num'],",
                "        plot=prepare_cfg['plot'],",
                "    )",
                "",
                "    level_cfg = levels_cfg['level0']",
                "    if level_cfg['enabled']:",
                "        processor = preset.dataProcess_VIC_level0_JRB(",
                "            load_path=evb_dir_modeling._dpc_VIC_level0_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level0,",
                "            'grid_res': grid_res_level0,",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            evb_dir_modeling._dpc_VIC_level0_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "        if level_cfg.get('plot_after', False):",
                "            processor.plot()",
                "",
                "    level_cfg = levels_cfg['level2_cmadsv1']",
                "    if level_cfg['enabled']:",
                "        save_path = evb_dir_modeling._dpc_VIC_level2_path.replace('.pkl', '_CMADSV1.pkl')",
                "        processor = preset.dataProcess_VIC_level2_CMADSV1_JRB(",
                "            load_path=save_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level2,",
                "            'grid_res': grid_res_level2,",
                "            'date_period': date_period,",
                "            'reverse_lat': reverse_lat,",
                "            'search_method': level_cfg.get('search_method', 'nearest'),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            save_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "",
                "    level_cfg = levels_cfg['level2_cmfd']",
                "    if level_cfg['enabled']:",
                "        save_path = evb_dir_modeling._dpc_VIC_level2_path.replace('.pkl', '_CMFD.pkl')",
                "        processor = preset.dataProcess_VIC_level2_CMFD_JRB(",
                "            load_path=save_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level2,",
                "            'grid_res': grid_res_level2,",
                "            'date_period': date_period,",
                "            'search_method': level_cfg.get('search_method', 'radius_rectangle'),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            save_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "",
                "    level_cfg = levels_cfg['level2_cdmet']",
                "    if level_cfg['enabled']:",
                "        save_path = evb_dir_modeling._dpc_VIC_level2_path.replace('.pkl', '_CDMet.pkl')",
                "        processor = preset.dataProcess_VIC_level2_CDMet_JRB(",
                "            load_path=save_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level2,",
                "            'grid_res': grid_res_level2,",
                "            'date_period': date_period,",
                "            'search_method': level_cfg.get('search_method', 'radius_rectangle'),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            save_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "",
                "    level_cfg = levels_cfg['level1']",
                "    if level_cfg['enabled']:",
                "        processor = preset.dataProcess_VIC_level1_JRB(",
                "            load_path=evb_dir_modeling._dpc_VIC_level1_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level1,",
                "            'grid_res': grid_res_level1,",
                "            'date_period': date_period,",
                "            'evb_dir': evb_dir_modeling,",
                "            'reverse_lat': reverse_lat,",
                "            'search_method_st': level_cfg.get('search_method_st', 'radius_rectangle'),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            evb_dir_modeling._dpc_VIC_level1_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "        if level_cfg.get('plot_after', False):",
                "            processor.plot()",
                "",
                "    level_cfg = levels_cfg['level3']",
                "    if level_cfg['enabled']:",
                "        processor = preset.dataProcess_VIC_level3_JRB(",
                "            load_path=evb_dir_modeling._dpc_VIC_level3_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'date_period': date_period,",
                "            'station_name': station_name,",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            evb_dir_modeling._dpc_VIC_level3_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
                "",
                "    level_cfg = levels_cfg['level1_gleam']",
                "    if level_cfg['enabled']:",
                "        save_path = os.path.join(evb_dir_modeling.dpcFile_dir, 'dpc_VIC_level1_GLEAM.pkl')",
                "        processor = preset.dataProcess_VIC_level1_GLEAM_JRB(",
                "            load_path=save_path,",
                "            reset_on_load_failure=True,",
                "        )",
                "        kwargs = {",
                "            'basin_shp': basin_shp,",
                "            'grid_shp': grid_shp_level1,",
                "            'grid_res': grid_res_level1,",
                "            'date_period': date_period,",
                "            'search_method': level_cfg.get('search_method', 'radius_rectangle_reverse'),",
                "        }",
                "        kwargs.update(level_cfg.get('extra_loaddata_kwargs', {}))",
                "        _run_pipeline(",
                "            processor,",
                "            save_path,",
                "            kwargs,",
                "            level_cfg.get('processing_steps'),",
                "        )",
            ]
        )

    lines.extend(
        [
            "",
            "",
            "if __name__ == '__main__':",
            "    if 'evb_dir_hydroanalysis' not in globals() or 'evb_dir_modeling' not in globals():",
            "        raise RuntimeError(",
            "            'Please define evb_dir_hydroanalysis and evb_dir_modeling in runtime context before running.'",
            "        )",
            "    build_dpc(evb_dir_hydroanalysis, evb_dir_modeling, date_period, reverse_lat=reverse_lat)",
        ]
    )
    return "\n".join(lines)


def render_build_dpc_tab() -> None:
    preset = st.session_state["workflow_preset"]
    apply_dpc_defaults(preset, force=False)

    st.subheader("Step 4: Build DPC")
    st.caption("Generate editable build_dpc script based on preset structure. Configure each level independently.")

    script_name = dpc_script_name_for_preset(preset)
    st.caption(f"Target script name: `{script_name}`")
    if preset == "HRB_modeling":
        grid_preview = (
            "basin_shps = build_basin_shp_JRB(evb_dir_hydroanalysis)\n"
            "grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = build_grid_shp(\n"
            "    basin_shps[station_name],\n"
            "    grid_res_level0,\n"
            "    grid_res_level1,\n"
            "    grid_res_level2,\n"
            "    expand_grids_num=1,\n"
            "    plot=True,\n"
            ")"
        )
    else:
        grid_preview = (
            "basin_shp = build_basin_shp(evb_dir_hydroanalysis)\n"
            "grid_shp_level0, grid_shp_level1, grid_shp_level2, grid_shp_level3 = build_grid_shp(\n"
            "    basin_shp,\n"
            "    grid_res_level0,\n"
            "    grid_res_level1,\n"
            "    grid_res_level2,\n"
            "    expand_grids_num=1,\n"
            "    plot=True,\n"
            ")"
        )
    with st.expander("Step 1-A: build_basin_shp (standalone)", expanded=True):
        st.caption("Use `basin_shp = Basins.from_shapefile(basin_shp_path)` to load basin shapefile.")
        uploaded_files = st.file_uploader(
            "Drag/drop or choose basin shapefile files (`.zip` or `.shp` + sidecars)",
            type=["zip", "shp", "shx", "dbf", "prj", "cpg"],
            accept_multiple_files=True,
            key="dpc_prepare_basin_uploads",
        )
        if uploaded_files:
            st.caption("Uploaded: " + ", ".join(file_item.name for file_item in uploaded_files))
            if st.button("Use uploaded files as basin_shp path", key="dpc_use_uploaded_basin", use_container_width=True):
                resolved_path, upload_error = resolve_uploaded_basin_shp_path(uploaded_files)
                if upload_error:
                    st.error(upload_error)
                else:
                    st.session_state["dpc_prepare_basin_shp_path"] = resolved_path
                    st.success(f"Resolved basin_shp path: {resolved_path}")

        st.text_input(
            "basin_shp_path",
            key="dpc_prepare_basin_shp_path",
            placeholder="e.g. D:/data/basin_vector_outlet_with_reference_4.shp",
        )
        st.code("basin_shp = Basins.from_shapefile(basin_shp_path)", language="python")

        if st.button("Run build_basin_shp", key="dpc_run_basin_preview", use_container_width=True):
            progress_bar = st.progress(0)
            progress_bar.progress(20)
            summary, fig, error_message = run_build_basin_shp_preview(
                str(st.session_state.get("dpc_prepare_basin_shp_path", "")).strip()
            )
            progress_bar.progress(100)
            if error_message:
                st.error(error_message)
            else:
                st.session_state["dpc_prepare_basin_summary"] = summary
                st.success("build_basin_shp finished.")
                st.json(summary)
                st.pyplot(fig)

    with st.expander("Step 1-B: build_grid_shp (standalone)", expanded=True):
        st.caption("Run `build_grid_shp` independently. If basin path is empty, fallback uses hydroanalysis outputs.")
        st.number_input("expand_grids_num", min_value=0, max_value=50, step=1, key="dpc_grid_expand_grids_num")
        st.checkbox("plot grid_shp while building", key="dpc_grid_plot")
        grid_left, grid_mid, grid_right = st.columns(3)
        with grid_left:
            st.text_input(
                "grid_res_level0 override (optional)",
                key="dpc_grid_res_level0_override",
                placeholder="empty = use general_info",
            )
        with grid_mid:
            st.text_input(
                "grid_res_level1 override (optional)",
                key="dpc_grid_res_level1_override",
                placeholder="empty = use scalemap[model_scale]",
            )
        with grid_right:
            st.text_input(
                "grid_res_level2 override (optional)",
                key="dpc_grid_res_level2_override",
                placeholder="empty = use scalemap[model_scale]",
            )
        st.code(grid_preview, language="python")

        if st.button("Run build_grid_shp", key="dpc_run_grid_preview", use_container_width=True):
            prepare_cfg, prepare_errors = collect_dpc_prepare_form()
            if prepare_cfg is None:
                for err_message in prepare_errors:
                    st.error(err_message)
            else:
                progress_bar = st.progress(0)
                progress_bar.progress(15)
                summary, fig, error_message = run_build_grid_shp_preview(preset, prepare_cfg)
                progress_bar.progress(100)
                if error_message:
                    st.error(error_message)
                else:
                    st.session_state["dpc_prepare_grid_summary"] = summary
                    st.success("build_grid_shp finished.")
                    st.json(summary)
                    st.pyplot(fig)

    for level_spec in dpc_level_specs_for_preset(preset):
        level_id = str(level_spec["id"])
        level_title = str(level_spec.get("title", level_id))
        enable_key = f"dpc_enable_{level_id}"
        steps_key = f"dpc_steps_{level_id}"
        add_step_key = f"dpc_add_step_{level_id}"
        extra_kwargs_key = f"dpc_extra_kwargs_{level_id}"

        with st.expander(level_title, expanded=bool(st.session_state.get(enable_key, False))):
            st.checkbox("enable this level", key=enable_key)

            for field_spec in level_spec.get("fields", []):
                field_name = str(field_spec["name"])
                field_label = str(field_spec.get("label", field_name))
                field_type = str(field_spec.get("type", "text"))
                field_key = dpc_field_state_key(level_id, field_name)
                if field_type == "bool":
                    st.checkbox(field_label, key=field_key)
                else:
                    st.text_input(field_label, key=field_key)

            st.text_area(
                "processing_steps (optional, one step_name per line; empty = run all registered steps)",
                key=steps_key,
                height=100,
            )
            add_col, btn_col = st.columns([4, 1])
            with add_col:
                st.text_input("add processing_step", key=add_step_key, placeholder="e.g. load_dem")
            with btn_col:
                if st.button("Add", key=f"dpc_add_btn_{level_id}", use_container_width=True):
                    new_step = str(st.session_state.get(add_step_key, "")).strip()
                    if new_step:
                        existing_steps = [
                            line.strip()
                            for line in str(st.session_state.get(steps_key, "")).splitlines()
                            if line.strip()
                        ]
                        existing_steps.append(new_step)
                        st.session_state[steps_key] = "\n".join(existing_steps)
                        st.session_state[add_step_key] = ""
                        st.rerun()

            st.text_area(
                "extra loaddata_kwargs (JSON object, optional)",
                key=extra_kwargs_key,
                height=120,
                placeholder='{"custom_key": "custom_value"}',
            )

    config, errors = collect_build_dpc_form(preset)
    generated_script = render_build_dpc_script(preset, config) if config is not None else ""
    if errors:
        for message in errors:
            st.error(message)

    if not str(st.session_state.get("dpc_final_script_text", "")).strip() and generated_script:
        st.session_state["dpc_final_script_text"] = generated_script

    load_col, _ = st.columns([1, 3])
    with load_col:
        if st.button("Load generated into editor", key="dpc_load_generated", use_container_width=True, disabled=not bool(generated_script)):
            st.session_state["dpc_final_script_text"] = generated_script
            st.rerun()

    st.text_area(
        "final build_dpc.py (editable)",
        key="dpc_final_script_text",
        height=420,
    )

    workspace_text = str(st.session_state.get("script_workspace_dir", "")).strip()
    workspace_path = Path(workspace_text) if workspace_text else None
    if workspace_path and workspace_path.exists():
        target_path = workspace_path / script_name
        st.caption(f"Save path: `{target_path}`")
        if st.button("Write build_dpc.py to scripts", use_container_width=True):
            final_script_text = str(st.session_state.get("dpc_final_script_text", "")).strip()
            if not final_script_text:
                st.error("`final build_dpc.py` is empty.")
            else:
                target_path.write_text(final_script_text + "\n", encoding="utf-8")
                st.success(f"Written: {target_path}")
    else:
        st.info("Run Step 2 first to initialize case and create `scripts/` workspace.")
