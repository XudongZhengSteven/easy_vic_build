# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""
Base workflow class for basin/grid data processing.

The class in this module maintains a dependency-aware step graph, executes
registered processing steps, and caches basin/grid-level outputs for reuse.
Subclasses provide concrete loading or aggregation routines through decorated
methods.
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
    """
    Base class for basin/grid data loading pipelines.

    The class manages three internal states:

    - ``_processing_steps``: registered step metadata and dependencies.
    - ``_executed_steps``: names of completed steps.
    - ``_cache``: in-memory data products keyed by ``save_name``.

    Subclasses typically declare loading methods decorated by
    :func:`easy_vic_build.tools.decoractors.processing_step`.
    """

    def __init__(self, load_path: Optional[str] = None, reset_on_load_failure=False, **kwargs):
        """
        Initialize the processing object and optionally restore saved state.

        Parameters
        ----------
        load_path : str, optional
            Path to a serialized processor state (pickle file). If provided, the
            state will be loaded immediately.
        reset_on_load_failure : bool, optional
            If ``True``, reset to a clean state when state loading fails.
            If ``False``, loading errors raise ``RuntimeError``.
        **kwargs : dict
            Extra keyword arguments forwarded to :meth:`load_state`.
        """
        self._reset_state()
        
        self.load_path = None
        if load_path is not None:
            self.load_path = load_path
            self.load_state(load_path, reset_on_load_failure, **kwargs)
    
    def _register_decorated_steps(self):
        """
        Register all bound methods marked by ``@processing_step``.

        The decorator stores metadata on the method object. This helper scans
        instance attributes and converts those metadata into entries in
        ``self._processing_steps``.
        """
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
        """
        Register one processing step in the execution graph.

        Parameters
        ----------
        step_name : str
            Unique step identifier.
        save_names : str or list of str
            Cache key(s) expected to be produced by ``func``.
        data_level : str
            Data scope label, usually ``"basin_level"`` or ``"grid_level"``.
        func : Callable
            Step callable with zero arguments. It must return a ``dict`` keyed
            by ``save_names``.
        dependencies : list of str, optional
            Steps that must be executed before this step.
        """
        self._processing_steps[step_name] = {
            "func": func,
            "deps": dependencies or [],
            "save_names": save_names,
            "data_level": data_level
        }
    
    def loaddata_pipeline(self, save_path=None, loaddata_kwargs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Execute all registered steps in dependency order.

        Parameters
        ----------
        save_path : str, optional
            Path for persisting state after each successful step.
        loaddata_kwargs : dict, optional
            Runtime input dictionary consumed by individual step methods.
        """
        self.loaddata_kwargs = loaddata_kwargs or {}
        for step_name in self._processing_steps:
            self._execute_step(step_name, save_path)
    
    def _execute_step(
        self,
        step_name: str,
        save_path: Optional[str] = None,
        visited: Optional[Set[str]] = None
    ):
        """
        Execute a single step recursively with dependency checks.

        Parameters
        ----------
        step_name : str
            Step to execute.
        save_path : str, optional
            Path used by :meth:`save_state` after caching outputs.
        visited : set of str, optional
            DFS guard set used for cycle detection.

        Raises
        ------
        RuntimeError
            If circular dependencies are detected.
        KeyError
            If step metadata is missing or returned outputs are incomplete.
        """
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
        """
        Merge all cached basin-level outputs into one GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame
            Basin GeoDataFrame containing original geometry plus all joinable
            basin-level products in cache.
        """
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
        """
        Merge all cached grid-level outputs into one GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame
            Grid GeoDataFrame containing original grid fields and appended
            grid-level variables from cache.
        """
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
        """
        Mark one step as not executed.

        Parameters
        ----------
        step_name : str
            Step name to remove from ``_executed_steps``.
        """
        self._executed_steps.discard(step_name)
        
    def get_data_from_cache(
        self,
        save_name: str,
        default: Optional[Any] = None
    ) -> Any:
        """Get cached data and its level by key.

        Parameters
        ----------
        save_name : str
            Cache key to retrieve.
        default : Any, optional
            Value used when the key is not found.

        Returns
        -------
        tuple
            ``(data, data_level)`` if key exists; otherwise ``(default, None)``.
        """
        entry = self._cache.get(save_name, default)
        
        if entry is not None:
            return entry["data"], entry.get("data_level", None)
        else:
            return default, None
    
    def list_cache(self) -> List[str]:
        """
        List available keys in cache.

        Returns
        -------
        list of str
            Current cache key names.
        """
        return list(self._cache.keys())
    
    def save_data_to_cache(
        self,
        save_name: str,
        data: Any,
        data_level: str,
        step_name: Optional[str] = None,
    ) -> None:
        """Save data into cache and optionally reopen its step state.

        Parameters
        ----------
        save_name : str
            Cache key for the data object.
        data : Any
            Data object to cache.
        data_level : str
            Data scope label, usually ``"basin_level"`` or ``"grid_level"``.
        step_name : str, optional
            Step name to discard from ``_executed_steps`` after updating cache.
        """
        self._cache[save_name] = {"data": data, "data_level": data_level}
        self.discard_step_name(step_name)
        
    def clear_data_from_cache(
        self,
        save_names: Optional[List[str]] = None,
        step_name: Optional[str] = None
    ):
        """Clear cached entries by key list or clear all entries.

        Parameters
        ----------
        save_names : list of str, optional
            Keys to remove. If ``None``, all cache entries are removed.
        step_name : str, optional
            Step name to discard from ``_executed_steps`` while clearing keys.
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
        """Serialize processor state to a pickle file.

        Parameters
        ----------
        save_path : str, optional
            Output state path. If omitted, ``self.load_path`` is used when set.
        """
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
        """Load processor state from a pickle file.

        Parameters
        ----------
        load_path : str
            State file path.
        reset_on_load_failure : bool, optional
            Whether to reset to a clean state when loading fails.
        **kwargs : dict
            Reserved for compatibility.

        Returns
        -------
        dataProcess_base
            Current processor instance.

        Raises
        ------
        RuntimeError
            Raised when loading fails and ``reset_on_load_failure`` is ``False``.
        """
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
        """Reset cache, step graph, and executed-step registry."""
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._processing_steps: Dict[str, Dict[str, Any]] = {}
        self._executed_steps: set = set()
        self._register_decorated_steps()
        
    def aggregate_grid_to_basins(self):
        """
        Aggregate grid-level variables to basin-level summaries.

        Notes
        -----
        This method is intentionally left for subclasses.
        """
        pass
        
    def plot(
        self,
        fig=None,
        ax=None,
        grid_shp_kwargs=dict(),
        grid_shp_point_kwargs=dict(),
        basin_shp_kwargs=dict(),
    ):
        """
        Plot cached basin and grid geometry.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Existing figure object. A new one is created when omitted.
        ax : matplotlib.axes.Axes, optional
            Existing axes object. A new one is created when omitted.
        grid_shp_kwargs : dict, optional
            Keyword arguments passed to ``grid_shp.boundary.plot``.
        grid_shp_point_kwargs : dict, optional
            Keyword arguments passed to ``grid_shp["point_geometry"].plot``.
        basin_shp_kwargs : dict, optional
            Keyword arguments passed to ``basin_shp.plot``.

        Returns
        -------
        tuple
            ``(fig, ax)`` with rendered basin/grid layout.
        """
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
        """
        Load basin shapefile-like object from ``loaddata_kwargs``.

        Returns
        -------
        dict
            Dictionary containing key ``"basin_shp"``.
        """
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
        """
        Load grid shapefile-like object and grid resolution from inputs.

        Returns
        -------
        dict
            Dictionary containing keys ``"grid_shp"`` and ``"grid_res"``.
        """
        loaded_grid_shp = deepcopy(self.loaddata_kwargs["grid_shp"])
        grid_res = self.loaddata_kwargs["grid_res"]
        
        ret = {"grid_shp": loaded_grid_shp, "grid_res": grid_res}
        
        return ret
