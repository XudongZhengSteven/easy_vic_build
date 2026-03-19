from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRESET = "HRB_modeling"
DEFAULT_ARTIFACT_ROOT = str(REPO_ROOT / "evb_ui" / "cases")
LEGACY_ARTIFACT_ROOT = str(REPO_ROOT / "examples" / "modeling")

GENERAL_INFO_STATE_KEYS = [
    "gi_final_script_text",
    "gi_case_prefix",
    "gi_enable_nested_basin",
    "gi_station_names_items",
    "gi_station_name",
    "gi_model_scale",
    "gi_timestep",
    "gi_timestep_evaluate",
    "gi_date_start",
    "gi_date_end",
    "gi_warmup_start",
    "gi_warmup_end",
    "gi_calibrate_start",
    "gi_calibrate_end",
    "gi_verify_start",
    "gi_verify_end",
    "gi_reverse_lat",
    "gi_grid_res_level0",
    "gi_scalemap_json",
    "gi_station_names_json",
    "gi_station_coords_json",
    "gi_nest_upstream_map_json",
    "gi_boundary_xmin",
    "gi_boundary_ymin",
    "gi_boundary_xmax",
    "gi_boundary_ymax",
    "gi_basin_outlets_reference_i_map_json",
    "gi_stationdata_fname_map_json",
]

GENERAL_INFO_FALLBACKS = {
    "HRB_modeling": {
        "station_name": "shiquan",
        "model_scale": "6km",
        "timestep": "3h",
        "timestep_evaluate": "D",
        "date_period": ["20030101 00:00:00", "20181231 21:00:00"],
        "warmup_date_period": ["20030101 00:00:00", "20041231 21:00:00"],
        "calibrate_date_period": ["20050101 00:00:00", "20141231 21:00:00"],
        "verify_date_period": ["20150101 00:00:00", "20181231 21:00:00"],
        "reverse_lat": True,
        "grid_res_level0": 0.00833,
        "scalemap": {
            "3km": 0.025,
            "6km": 0.055,
            "8km": 0.072,
            "12km": 0.11,
            "1_32_deg": 0.03125,
            "1_20_deg": 0.05,
            "1_16_deg": 0.0625,
            "1_14_deg": 0.07142857142857142,
            "1_12_deg": 0.08333333333333333,
            "1_10_deg": 0.1,
            "1_8_deg": 0.125,
            "1_6_deg": 0.16666666666666666,
            "1_5_deg": 0.2,
            "1_4_deg": 0.25,
            "1_2_deg": 0.5,
            "1_deg": 1.0,
            "1_grid": None,
        },
        "station_names": ["hanzhong", "yangxian", "youshui", "lianghekou", "shiquan"],
        "station_coords": {
            "hanzhong": [33.049000, 107.023315],
            "yangxian": [33.218708, 107.536583],
            "youshui": [33.267975, 107.766781],
            "lianghekou": [33.26325, 108.06896],
            "shiquan": [33.038635, 108.240737],
        },
        "nest_upstream_map": {
            "hanzhong": [],
            "yangxian": ["hanzhong"],
            "youshui": [],
            "lianghekou": [],
            "shiquan": ["hanzhong", "yangxian", "youshui", "lianghekou"],
        },
        "boundary": [105.6, 32.0, 109.0, 34.8],
        "basin_outlets_reference_i_map": {
            "hanzhong": 0,
            "yangxian": 1,
            "youshui": 2,
            "lianghekou": 3,
            "shiquan": 4,
        },
        "stationdata_fname_map": {},
    },
    "JRB_modeling": {
        "station_name": "Zhangjiashan",
        "model_scale": "1_8_deg",
        "timestep": "3h",
        "timestep_evaluate": "D",
        "date_period": ["20080101 00:00:00", "20181231 21:00:00"],
        "warmup_date_period": ["20080101 00:00:00", "20091231 21:00:00"],
        "calibrate_date_period": ["20100101 00:00:00", "20151231 21:00:00"],
        "verify_date_period": ["20160101 00:00:00", "20181231 21:00:00"],
        "reverse_lat": True,
        "grid_res_level0": 0.00833,
        "scalemap": {
            "3km": 0.025,
            "6km": 0.055,
            "8km": 0.072,
            "12km": 0.11,
            "1_32_deg": 0.03125,
            "1_16_deg": 0.0625,
            "1_8_deg": 0.125,
            "1_6_deg": 0.16666666666666666,
            "1_4_deg": 0.25,
            "1_2_deg": 0.5,
            "1_grid": None,
        },
        "station_names": [],
        "station_coords": {},
        "nest_upstream_map": {},
        "boundary": [],
        "basin_outlets_reference_i_map": {
            "Zhangjiashan": 1,
            "Yangjiapin": 0,
            "Qingyang": 2,
            "2021042103p": 4,
            "2022092592p": 3,
        },
        "stationdata_fname_map": {
            "Zhangjiashan": "stationdata_Zhangjiashan_daily_1960_2020.txt",
            "Yangjiapin": "stationdata_Yangjiapin_daily_1956_2020.txt",
            "Qingyang": "stationdata_Qinyang_daily_1956_2020.txt",
            "2022092592p": "stationdata_2022092592p_Intermittent_2020_2022.txt",
            "2021042103": "stationdata_2021042103p_Intermittent_2016_2024.txt",
        },
    },
}

DPC_LEVEL_SPECS = {
    "HRB_modeling": [
        {
            "id": "base",
            "title": "dpc_base",
            "default_enabled": False,
            "fields": [],
        },
        {
            "id": "level0",
            "title": "dpc_level0",
            "default_enabled": False,
            "fields": [
                {"name": "plot_after", "type": "bool", "label": "plot after pipeline", "default": True},
            ],
        },
        {
            "id": "level2_cmfd",
            "title": "dpc_level2_cmfd",
            "default_enabled": False,
            "fields": [
                {
                    "name": "search_method",
                    "type": "text",
                    "label": "search_method",
                    "default": "radius_rectangle_reverse",
                },
            ],
        },
        {
            "id": "level1",
            "title": "dpc_level1",
            "default_enabled": False,
            "fields": [
                {
                    "name": "search_method_st",
                    "type": "text",
                    "label": "search_method_st",
                    "default": "radius_rectangle_reverse",
                },
                {"name": "plot_after", "type": "bool", "label": "plot after pipeline", "default": True},
            ],
        },
        {
            "id": "level3",
            "title": "dpc_level3",
            "default_enabled": False,
            "fields": [
                {"name": "load_level1", "type": "bool", "label": "load_level1", "default": False},
            ],
        },
        {
            "id": "level3_load_level1",
            "title": "dpc_level3_load_level1",
            "default_enabled": True,
            "fields": [
                {"name": "load_level1", "type": "bool", "label": "load_level1", "default": True},
                {
                    "name": "clear_gauge_info",
                    "type": "bool",
                    "label": "clear gauge_info cache before run",
                    "default": True,
                },
            ],
        },
    ],
    "JRB_modeling": [
        {
            "id": "level0",
            "title": "dpc_level0",
            "default_enabled": False,
            "fields": [
                {"name": "plot_after", "type": "bool", "label": "plot after pipeline", "default": True},
            ],
        },
        {
            "id": "level2_cmadsv1",
            "title": "dpc_level2_cmadsv1",
            "default_enabled": False,
            "fields": [
                {"name": "search_method", "type": "text", "label": "search_method", "default": "nearest"},
            ],
        },
        {
            "id": "level2_cmfd",
            "title": "dpc_level2_cmfd",
            "default_enabled": False,
            "fields": [
                {"name": "search_method", "type": "text", "label": "search_method", "default": "radius_rectangle"},
            ],
        },
        {
            "id": "level2_cdmet",
            "title": "dpc_level2_cdmet",
            "default_enabled": False,
            "fields": [
                {"name": "search_method", "type": "text", "label": "search_method", "default": "radius_rectangle"},
            ],
        },
        {
            "id": "level1",
            "title": "dpc_level1",
            "default_enabled": False,
            "fields": [
                {"name": "search_method_st", "type": "text", "label": "search_method_st", "default": "radius_rectangle"},
                {"name": "plot_after", "type": "bool", "label": "plot after pipeline", "default": True},
            ],
        },
        {
            "id": "level3",
            "title": "dpc_level3",
            "default_enabled": False,
            "fields": [],
        },
        {
            "id": "level1_gleam",
            "title": "dpc_level1_gleam",
            "default_enabled": True,
            "fields": [
                {
                    "name": "search_method",
                    "type": "text",
                    "label": "search_method",
                    "default": "radius_rectangle_reverse",
                },
            ],
        },
    ],
}

