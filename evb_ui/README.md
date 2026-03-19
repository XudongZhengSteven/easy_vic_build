# EVB Workflow Studio

This folder contains a standalone UI for `easy_vic_build`.

The implementation is isolated in `evb_ui/` and does not modify files in `src/easy_vic_build/`.

## Features

1. `Step 1-2 - General Info + Initialize Case`: one page contains the form/editor for `general_info.py` and the case initialization section.
2. Initialization creates only the model case folder under `evb_ui/cases`.
3. Initialization only writes `general_info.py` into `<model_case>/scripts/`.
4. `Step 3 - Hydroanalysis L0`: generate editable preset-based hydroanalysis level0 script, save it to case `scripts/`, run it directly in UI, and preview output `.shp/.tif/.tiff` files.
5. `Step 4 - Build DPC`: generate editable `build_dpc.py` with per-level toggles/settings and optional custom `processing_steps`.
6. In Step 4, `build_basin_shp` and `build_grid_shp` run independently in UI with progress bars and preview plots.
7. Step 4 supports drag/drop or file chooser for basin shapefile inputs (`.zip` or `.shp` + sidecars), and optional grid parameters.
8. `Logger` tab shows runtime console output and reads case log files under `<case>/VICLog`.

Default case root: `evb_ui/cases` (inside repository).

## Quick start

From repository root:

```bash
pip install -r evb_ui/requirements.txt
streamlit run evb_ui/app.py
```

If package dependencies are not installed yet, install this project in editable mode:

```bash
pip install -e .
```

## Notes

- Original `examples/*_modeling` scripts are never modified by the UI.
- Step 3 references preset hydroanalysis structure (`examples/*_modeling/*_hydroanalysis.py`) and runs level0 in case context.
- Step 4 references preset build_dpc structure (`examples/*_modeling/*_build_dpc.py`) and writes editable result into case `scripts/`.

## Module layout

- `app.py`: thin entrypoint + tab routing.
- `ui_state_profile.py`: session defaults and profile save/load sidebar.
- `ui_general_info.py`: Step 1 section (General Info).
- `ui_case_steps.py`: Step 2 section (Initialize Case) + Step 3 (Hydroanalysis L0).
- `ui_dpc_step.py`: Step 4 (Build DPC + basin/grid standalone previews).
- `ui_logger.py`: logger viewer tab.
- `ui_config.py` / `ui_helpers.py`: shared constants and reusable helpers.
