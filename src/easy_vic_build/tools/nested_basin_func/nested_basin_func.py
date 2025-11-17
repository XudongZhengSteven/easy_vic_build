# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from ...bulid_Domain import UTM_proj_map
from ..dpc_func.basin_grid_func import createStand_grids_lat_lon_from_gridshp, createEmptyArray_from_gridshp, assignValue_for_grid_array, gridshp_index_to_grid_array_index
from copy import deepcopy
from tqdm import *

def get_all_upstreams(station, nest_upstream_map):
    visited = set()
    stack = list(nest_upstream_map[station])

    while stack:
        cur = stack.pop()
        if cur not in visited:
            visited.add(cur)
            stack.extend(nest_upstream_map[cur])

    return list(visited)

def get_topo_order(station_names, nest_map):
    visited = set()
    order = []

    def visit(s):
        if s in visited:
            return
        for up in nest_map.get(s, []):
            visit(up)
        visited.add(s)
        order.append(s)

    for s in station_names:
        visit(s)
    return order


def cal_unique_mask_nested_basin(
    station_names,
    grid_shp,
    basin_shps,
    main_basin_shp,
    plot=False,
    reverse_lat=True
):
    """
    example:
        # read basins
        nested_basins_shp = gpd.read_file(os.path.join(evb_dir_hydroanalysis.Hydroanalysis_dir, "wbw_working_directory_level0", "basins_vector_outlets_with_reference.shp"))
        basin_shps = {
            "hanzhong": nested_basins_shp.iloc[0:1, :],
            "yangxian": nested_basins_shp.iloc[1:2, :],
            "lianghekou": nested_basins_shp.iloc[2:3, :],
            "shiquan": nested_basins_shp.iloc[3:4, :],
            "youshui": nested_basins_shp.iloc[4:5, :],
        }
        
        # enforce_unique_masks
        unique_masks_level1 = cal_unique_mask_nested_basin(
            station_names,
            grid_shp_level1,
            basin_shps,
            main_basin_shp=basin_shps["shiquan"],
            plot=True
        )
    
    """
    # Determine the UTM CRS based on the longitude of the basin center
    try:
        lon_cen = main_basin_shp["lon_cen"].values[0]
    except:
        lon_cen = main_basin_shp.centroid.x.values[0]
        
    for k in UTM_proj_map.keys():
        if (
            lon_cen >= UTM_proj_map[k]["lon_min"]
            and lon_cen <= UTM_proj_map[k]["lon_max"]
        ):
            proj_crs = UTM_proj_map[k]["crs_code"]
            
    # Precompute frac for each basin and grid cell
    grid_shp_projection = deepcopy(grid_shp)
    grid_shp_projection = grid_shp_projection.to_crs(proj_crs)
    
    # lon/lat grid map into index to construct array
    stand_grids_lat, stand_grids_lon = createStand_grids_lat_lon_from_gridshp(grid_shp, reverse_lat=reverse_lat)
    rows_index, cols_index = gridshp_index_to_grid_array_index(
        grid_shp, stand_grids_lat, stand_grids_lon
    )
    
    # Initialize arrays for mask, frac, and frac_grid_in_basin
    ny, nx = len(stand_grids_lat), len(stand_grids_lon)
    area_matrix = np.zeros((len(grid_shp_projection), len(station_names)), dtype=float)
    
    for j, s in enumerate(station_names):
        basin_shp = basin_shps[s]
        basin_shp = basin_shp.to_crs(proj_crs)
        inter = gpd.overlay(
            grid_shp_projection.reset_index().rename(columns={'index': 'grid_idx'}),
            basin_shp,
            how='intersection'
        )
        if len(inter) > 0:
            for grid_id, group in inter.groupby('grid_idx'):
                area_matrix[grid_id, j] = group.geometry.area.sum()
    
    no_intersection = area_matrix.sum(axis=1) == 0
    area_matrix[no_intersection, :] = -1
    max_idx = np.argmax(area_matrix, axis=1)
    max_idx[no_intersection] = -1
    
    # assign value
    unique_mask = createEmptyArray_from_gridshp(
        stand_grids_lat, stand_grids_lon, dtype=float, missing_value=np.nan
    )
    
    unique_mask = assignValue_for_grid_array(
        unique_mask,
        np.full((len(grid_shp),), fill_value=max_idx),
        rows_index,
        cols_index,
    )
    
    unique_masks = {s: np.empty((ny, nx), dtype=float) for s in station_names}
    for j, s in enumerate(station_names):
        unique_masks[s][:] = 0
        unique_masks[s][unique_mask == j] = 1
    
    if plot:
        total_plots = len(station_names) + 1
        fig_rows = int(np.ceil(total_plots / 3))
        fig, axes = plt.subplots(fig_rows, 3, figsize=(14, 4 * fig_rows))
        axes = axes.flatten()

        for ax, s in zip(axes, station_names):
            ax.imshow(unique_masks[s], cmap="viridis", interpolation="nearest")
            ax.set_title(f"Mask: {s}")
            ax.axis("off")

        owner_plot = unique_mask.copy().astype(float)
        owner_plot[owner_plot < 0] = np.nan
        ax = axes[len(station_names)]
        cmap = plt.get_cmap("tab20", len(station_names))
        im = ax.imshow(owner_plot, cmap=cmap, interpolation="nearest")
        ax.set_title("Station Ownership Overview")
        ax.axis("off")

        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(station_names)))
        cbar.ax.set_yticklabels(station_names)

        for j in range(len(station_names) + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show(block=True)
        
    return unique_masks

if __name__ == "__main__":
    station_names = ["hanzhong", "yangxian", "youshui", "lianghekou", "shiquan"]
    nest_upstream_map = {
        "hanzhong": [],
        "yangxian": ["hanzhong"],
        "youshui": [],
        "lianghekou": [],
        "shiquan": ["hanzhong", "yangxian", "youshui", "lianghekou"],
    }
    
    topo_station_order = get_topo_order(station_names, nest_upstream_map)