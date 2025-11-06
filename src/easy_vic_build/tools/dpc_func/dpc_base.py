# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Module: dpc_base

This module provides a base class for processing and managing data related to basins and grids in hydrological
and geospatial analyses. The `dataProcess_base` class serves as a template for reading basin and grid data,
aggregating grid data to basins, and visualizing the results. It is designed to be subclassed or extended to
accommodate specific data types or processing steps for particular hydrological modeling needs.

Class:
--------
    - dataProcess_base: A base class that implements a general workflow for processing basin and grid data,
      including reading, aggregating, and visualizing the data. Specific methods for handling data types or
      processing steps should be implemented in subclasses.

Class Methods:
---------------
    - __call__(self, *args: Any, **kwargs: Any): Executes the full data processing pipeline: reading basin
      and grid data, aggregating grid data to basins, and plotting results.
    - read_basin_grid(self): Placeholder method to read basin grid data. To be extended with specific logic.
    - readDataIntoBasins(self): Placeholder method to read data into basins. To be extended with specific logic.
    - readDataIntoGrids(self): Placeholder method to read data into grids. To be extended with specific logic.
    - aggregate_grid_to_basins(self): Placeholder method to aggregate grid data to basins. To be extended with
      specific logic.
    - readBasinAttribute(self): Placeholder method to read basin attributes. To be extended with specific logic.
    - plot(self): Placeholder method to plot the results of the data processing. To be extended with specific logic.

Usage:
------
    1. Instantiate the `dataProcess_base` class with the required basin and grid data:
        - `dp = dataProcess_base(basin_shp, grid_shp, grid_res)`
    2. Call the `__call__` method to trigger the full data processing pipeline:
        - `dp()`
    3. Implement specific logic in subclassed methods to customize data reading, aggregation, and plotting.

Example:
--------
    dp = dataProcess_base(basin_shp, grid_shp, grid_res)
    dp()

Dependencies:
-------------
    - geopandas: For handling and processing spatial data (GeoDataFrame).
    - numpy: For numerical operations and array manipulation.

Author:
-------
    Xudong Zheng
    Email: z786909151@163.com

"""

from abc import ABC, abstractmethod
import geopandas as gpd
import pandas as pd
from typing import Dict, List, Callable, Any, Optional, Union, Set
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from ..decoractors import processing_step
from ... import logger


class dataProcess_base(ABC):
    def __init__(self, load_path: Optional[str] = None, reset_on_load_failure=False, **kwargs):
        self._reset_state()
        
        self.load_path = None
        if load_path is not None:
            self.load_path = load_path
            self.load_state(load_path, reset_on_load_failure, **kwargs)
    
    def _register_decorated_steps(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_step_name"):
                self.register_processing_step(
                    step_name=attr._step_name,
                    save_names=attr._save_names,
                    data_level=attr._data_level,
                    func=attr,
                    dependencies=attr._step_deps
                )

    def register_processing_step(
        self,
        step_name: str,
        save_names: Union[str, List[str]],
        data_level: str,
        func: Callable,
        dependencies: Optional[List[str]] = None
    ):
        self._processing_steps[step_name] = {
            "func": func,
            "deps": dependencies or [],
            "save_names": save_names,
            "data_level": data_level
        }
    
    def loaddata_pipeline(self, save_path=None, loaddata_kwargs: Optional[Dict[str, Dict[str, Any]]] = None):
        self.loaddata_kwargs = loaddata_kwargs or {}
        for step_name in self._processing_steps:
            self._execute_step(step_name, save_path)
    
    def _execute_step(
        self,
        step_name: str,
        save_path: Optional[str] = None,
        visited: Optional[Set[str]] = None
    ):
        if step_name in self._executed_steps:
            return
        
        if visited is None:
            visited = set()
        if step_name in visited:
            raise RuntimeError(f"Circular dependency detected involving step: {step_name}")
        visited.add(step_name)
        
        if step_name not in self._processing_steps:
            raise KeyError(f"Step '{step_name}' not found in registered steps")
        
        step_info = self._processing_steps[step_name]
        save_names = step_info["save_names"]
        save_names_list = [save_names] if isinstance(save_names, str) else save_names
        
        if any(name in self._cache for name in save_names_list):
            logger.info(f"[SKIP] {step_name} (cached)")
            return
        
        for dep in step_info["deps"]:
            if dep not in self._executed_steps:
                self._execute_step(
                    dep,
                    visited=set(visited)
                )
        
        logger.info(f"[RUN ] {step_name}")
        result = step_info["func"]()

        for save_name in save_names_list:
            if save_name not in result:
                raise KeyError(f"Step {step_name} did not produce expected save name: {save_name}")
            
            self._cache[save_name] = {
                "data": result[save_name], 
                "data_level": step_info["data_level"]
            }
            
            if save_path is not None:
                self.save_state(save_path)
            
            logger.info(f"Saved {save_name} from step {step_name} with data level: {step_info['data_level']}")
        
        self._executed_steps.add(step_name)
    
    def merge_basin_data(self) -> gpd.GeoDataFrame:
        if "merged_basin_shp" in self._cache:
            merged_basin_shp = self._cache["merged_basin_shp"]["data"]
        else:
            if "basin_shp" not in self._cache:
                raise KeyError("Missing 'basin_shp' in cache")
            merged_basin_shp = deepcopy(self._cache["basin_shp"]["data"])

        for save_name, entry in self._cache.items():
            if entry["data_level"] != "basin_level" or save_name in ["basin_shp", "merged_basin_shp"]:
                continue
            
            data = entry["data"]

            if isinstance(data, pd.Series):
                data = data.to_frame(name=save_name)
            elif isinstance(data, pd.DataFrame):
                overlapping_cols = set(data.columns) & set(merged_basin_shp.columns)
                if overlapping_cols:
                    raise ValueError(f"Column(s) {overlapping_cols} in '{save_name}' already exist in basin_shp.")
            else:
                raise TypeError(f"Expected DataFrame or Series for '{save_name}', got {type(data)}")

            merged_basin_shp = merged_basin_shp.join(data, how="left")

        self.save_data_to_cache(
            save_name="merged_basin_shp",
            data=merged_basin_shp,
            data_level="basin_level",
            step_name=None,
        )
        
        return merged_basin_shp

    def merge_grid_data(self) -> gpd.GeoDataFrame:
        if "merged_grid_shp" in self._cache:
            merged_grid_shp = self._cache["merged_grid_shp"]["data"]
        else:
            if "grid_shp" not in self._cache:
                raise KeyError("Missing 'grid_shp' in cache")
            merged_grid_shp = deepcopy(self._cache["grid_shp"]["data"])
        
        for save_name, entry in self._cache.items():
            if entry["data_level"] != "grid_level" or save_name in ["grid_shp", "merged_grid_shp"]:
                continue
            
            data = entry["data"]
            
            if isinstance(data, pd.Series):
                data = data.to_frame(name=save_name)
            
            elif isinstance(data, pd.DataFrame):
                cols_to_join = data.columns.difference(merged_grid_shp.columns)
                logger.debug(f"below columns will be added to the merged_grid_shp:\n\nAdded {cols_to_join}\n\n")

            else:
                logger.warning(f"Expected DataFrame for {save_name}, got {type(data)}, will not be added to the merged_grid_shp")
                continue
            
            merged_grid_shp = pd.concat([merged_grid_shp, data[cols_to_join]], axis=1)
            
        self.save_data_to_cache(
            save_name="merged_grid_shp",
            data=merged_grid_shp,
            data_level="grid_level",
            step_name=None,
        )
        
        return merged_grid_shp

    def discard_step_name(self, step_name: str):
        self._executed_steps.discard(step_name)
        
    def get_data_from_cache(
        self,
        save_name: str,
        default: Optional[Any] = None
    ) -> Any:
        """Get processed data from cache"""
        entry = self._cache.get(save_name, default)
        
        if entry is not None:
            return entry["data"], entry.get("data_level", None)
        else:
            return default, None
    
    def list_cache(self) -> List[str]:
        return list(self._cache.keys())
    
    def save_data_to_cache(
        self,
        save_name: str,
        data: Any,
        data_level: str,
        step_name: Optional[str] = None,
    ) -> None:
        """Set processed data in cache
        step_name is not None -> discard this step
        """
        self._cache[save_name] = {"data": data, "data_level": data_level}
        self.discard_step_name(step_name)
        
    def clear_data_from_cache(
        self,
        save_names: Optional[List[str]] = None,
        step_name: Optional[str] = None
    ):
        """Clear cached data for specified keys or all if None,
        if you want to overwrite any data, please use this method to clear old data.
        """
        if save_names is None:
            self._cache.clear()
        else:
            for key in save_names:
                self._cache.pop(key, None)
                self.discard_step_name(step_name)
    
    def save_state(
        self,
        save_path: Optional[str] = None,
    ) -> None:
        """Serialize processor state to file."""
        state = {
            '_cache': self._cache,
            '_executed_steps': self._executed_steps,
            '_processing_steps': self._processing_steps,
        }
        
        if save_path is None and self.load_path is not None:
            save_path = self.load_path
        
        with open(save_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_state(
        self,
        load_path: str,
        reset_on_load_failure: bool = False,
        **kwargs
    ) -> 'dataProcess_base':
        """Load processor state from file."""
        try:
            with open(load_path, "rb") as f:
                state = pickle.load(f)
            
            valid_attrs = {'_cache', '_processing_steps', '_executed_steps'}
            for attr in valid_attrs:
                if attr in state:
                    setattr(self, attr, state[attr])
            
            self._processing_steps.clear()  # Clear existing steps to avoid duplicates
            self._register_decorated_steps()  # register for _processing_steps
        
        except Exception as e:
            if reset_on_load_failure:
                logger.warning(f"Failed to load state from {load_path}: {e}. \nResetting state!", exc_info=False)
                self._reset_state()
            else:
                raise RuntimeError(f"Failed to load state from {load_path}: {e}")
        
        return self
    
    def _reset_state(self) -> None:
        """Reset the processor's internal state to empty."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._processing_steps: Dict[str, Dict[str, Any]] = {}
        self._executed_steps: set = set()
        self._register_decorated_steps()
        
    def aggregate_grid_to_basins(self):
        pass
        
    def plot(
        self,
        fig=None,
        ax=None,
        grid_shp_kwargs=dict(),
        grid_shp_point_kwargs=dict(),
        basin_shp_kwargs=dict(),
    ):
        if fig is None:
            fig, ax = plt.subplots()

        # plot kwargs
        grid_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "linewidth": 0.5}
        grid_shp_kwargs_all.update(grid_shp_kwargs)

        grid_shp_point_kwargs_all = {"alpha": 0.5, "facecolor": "k", "markersize": 1}
        grid_shp_point_kwargs_all.update(grid_shp_point_kwargs)

        basin_shp_kwargs_all = {"edgecolor": "k", "alpha": 0.5, "facecolor": "b"}
        basin_shp_kwargs_all.update(basin_shp_kwargs)

        # plot
        grid_shp, _ = self.get_data_from_cache("grid_shp")
        basin_shp, _ = self.get_data_from_cache("basin_shp")
        
        if grid_shp is not None:
            grid_shp.boundary.plot(ax=ax, **grid_shp_kwargs_all)
            grid_shp["point_geometry"].plot(ax=ax, **grid_shp_point_kwargs_all)
        
        if basin_shp is not None:
            basin_shp.plot(ax=ax, **basin_shp_kwargs_all)

        boundary_x_y = grid_shp.createBoundaryShp()[-1]
        ax.set_xlim(boundary_x_y[0], boundary_x_y[2])
        ax.set_ylim(boundary_x_y[1], boundary_x_y[3])
        
        logger.debug("Generated plot for grid and basin data")
        
        return fig, ax
    
    # general processing step
    @processing_step(
        step_name="load_basin_shp",
        save_names="basin_shp",
        data_level="basin_level",
        deps=None,
    )
    def load_basin_shp(self):
        loaded_basin_shp = deepcopy(self.loaddata_kwargs["basin_shp"])
        
        ret = {"basin_shp": loaded_basin_shp}
        return ret
    
    @processing_step(
        step_name="load_grid_shp",
        save_names=["grid_shp", "grid_res"],
        data_level="grid_level",
        deps=None,
    )
    def load_grid_shp(self):
        loaded_grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        ret = {"grid_shp": loaded_grid_shp, "grid_res": grid_res}
        
        return ret