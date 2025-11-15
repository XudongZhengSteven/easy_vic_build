# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

import numpy as np
from copy import deepcopy
import json


# g params
"""
g_list: global parameters
    [0]             total_depth (g)
    [1, 2]          depth (g1, g2)
    [3, 4]          b_infilt (g1, g2)
    [5, 6, 7]       ksat (g1, g2, g3)
    [8, 9, 10]      phi_s (g1, g2, g3)
    [11, 12, 13]    psis (g1, g2, g3)
    [14, 15, 16]    b_retcurve (g1, g2, g3)
    [17, 18]        expt (g1, g2)
    [19]            fc (g)
    [20]            d4 (g), it can be set as 2
    [21]            d1 (g)
    [22]            d2 (g)
    [23]            d3 (g)
    [24]            dp (g)
    [25, 26]        bubble (g1, g2)
    [27]            quartz (g)
    [28]            bulk_density (g)
    [29, 30, 31]    soil_density (g, g, g), the three g can be set same
    [32]            Wcr_FRACT (g)
    [33]            wp (g)
    [34]            Wpwp_FRACT (g)
    [35]            rough (g), it can be set as 1
    [36]            snow rough (g), it can be set as 1
"""

g_params = {
    "total_depths": {
        "default": [1.0],  # total depth g_params (factor)
        "boundary": [[0.1], [4.0]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "soil_layers_breakpoints": {
        "default": [3, 9],  # soil layer breakpoints, original layers -> modeling layers, note exclusive
        "boundary": [[1, 3], [4, 9]],
        "type": int,
        "optimal": [None, None],
        "free": True,
    },
    
    "b_infilt": {
        "default": [0.0, 1.0],
        "boundary": [[-2.0, 0.8], [1.0, 1.2]],
        "type": float,
        "optimal": [None, None],
        "free": True,
    },
    
    "ksat": {
        "default": [-0.6, 0.0126, -0.0064],  # from Cosby et al. (1984)
        "boundary": [[-0.66, 0.0113, -0.007], [-0.54, 0.0139, -0.0058]],  # +- 10%
        "type": float,
        "optimal": [None, None, None],
        "free": True,
    },
    
    "phi_s": {
        "default": [50.5, -0.142, -0.037],  # from Cosby et al. (1984)
        "boundary": [[45.5, -0.3, -0.1], [55.5, -0.01, -0.01]],
        "type": float,
        "optimal": [None, None, None],
        "free": True,
    },
    
    "psis": {
        "default": [1.54, -0.0095, 0.0063],  # from Cosby et al. (1984)
        "boundary": [[1.0, -0.01, 0.006], [2.0, -0.009, 0.0066]],
        "type": float,
        "optimal": [None, None, None],
        "free": True,
    },
    
    "b_retcurve": {
        "default": [3.1, 0.157, -0.003],  # from Cosby et al. (1984)
        "boundary": [[2.5, 0.1, -0.005], [3.6, 0.2, -0.001]],
        "type": float,
        "optimal": [None, None, None],
        "free": True,
    },
    
    "expt": {
        "default": [3.0, 2.0],  # from Campbell (1974), expt=2b+3
        "boundary": [[2.8, 1.5], [3.2, 2.5]],
        "type": float,
        "optimal": [None, None],
        "free": True,
    },
    
    "fc": {
        "default": [1.0],
        "boundary": [[0.8], [1.2]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "d1": {
        "default": [2.0],
        "boundary": [[1.75], [3.5]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "d2": {
        "default": [2.0],
        "boundary": [[1.75], [3.5]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "d3": {
        "default": [1.0],
        "boundary": [[0.001], [2.0]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "d4": {
        "default": [2.0],  # it can be set as 2
        "boundary": [[1.5], [2.5]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "dp": {
        "default": [1.0],
        "boundary": [[0.9], [1.1]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "bubble": {
        "default": [0.32, 4.3],
        "boundary": [[0.1, 0.0], [0.9, 10.0]],
        "type": float,
        "optimal": [None, None],
        "free": True,
    },
    
    "quartz": {
        "default": [0.8],
        "boundary": [[0.7], [0.9]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "bulk_density": {
        "default": [1.0],
        "boundary": [[0.9], [1.1]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "soil_density": {
        "default": [1.0, 1.0, 1.0],  # the three g can be set same
        "boundary": [[0.9, 0.9, 0.9], [1.1, 1.1, 1.1]],
        "type": float,
        "optimal": [None, None, None],
        "free": True,
    },
    
    "Wcr_FRACT": {
        "default": [1.0],
        "boundary": [[0.8], [1.2]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "wp": {
        "default": [1.0],
        "boundary": [[0.8], [1.2]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "Wpwp_FRACT": {
        "default": [1.0],
        "boundary": [[0.8], [1.2]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "rough": {
        "default": [1.0],  # it can be set as 1
        "boundary": [[0.9], [1.1]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "snow_rough": {
        "default": [1.0],  # it can be set as 1
        "boundary": [[0.9], [1.1]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
}

# g params minimal version
g_params_minimal = deepcopy(g_params)
all_keys = list(g_params.keys())

free_keys = [
    "total_depths",
    "soil_layers_breakpoints",
    "b_infilt",
    "d1",
    "d2",
    "d3",
]

non_free_keys = list(set(all_keys) - set(free_keys))

for key in free_keys:
    g_params_minimal[key]["free"] = True
    
for key in non_free_keys:
    g_params_minimal[key]["free"] = False

# guh params
guh_params = {
    "tp": {
        "default": [1.4],
        "boundary": [[1.0], [24.0]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "mu": {
        "default": [5.0],
        "boundary": [[2.0], [10.0]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "m": {
        "default": [3.0],
        "boundary": [[0.5], [6.0]],
        "type": float,
        "optimal": [None],
        "free": True,
    }
}

# rvic params
rvic_params = {
    "VELOCITY": {
        "default": [1.5],  # velocity in m/s
        "boundary": [[0.01], [3.0]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "DIFFUSION": {
        "default": [800.0],
        "boundary": [[10.0], [4000.0]],
        "type": float,
        "optimal": [None],
        "free": True,
    }
}

rvic_params_spatial = {
    "VELOCITY": {
        "default": [0.2, 0.15, 0.3],
        "boundary": [[0.01, 0.1, 0.2], [0.5, 0.3, 0.4]],
        "type": float,
        "optimal": [None],
        "free": True,
    },
    
    "DIFFUSION": {
        "default": [0.1],
        "boundary": [[0.01], [0.5]],
        "type": float,
        "optimal": [None],
        "free": True,
    }
}

# all params
params = {
    "g_params": g_params,
    "guh_params": guh_params,
    "rvic_params": rvic_params,
}

params_all = {**g_params, **guh_params, **rvic_params}

# default params
default_params = deepcopy(params)
for key in default_params.keys():
    for sub_key in default_params[key].keys():
        default_params[key][sub_key]["optimal"] = default_params[key][sub_key]["default"]
        
# all params minimal version
params_minimal = {
    "g_params": g_params_minimal,
    "guh_params": guh_params,
    "rvic_params": rvic_params,
}

params_all_minimal = {**g_params_minimal, **guh_params, **rvic_params}

# all params minimal version + spatial rvic params
params_minimal_rvic_spatial = {
    "g_params": g_params_minimal,
    "guh_params": guh_params,
    "rvic_params": rvic_params_spatial,
}

params_all_minimal_rvic_spatial = {**g_params_minimal, **guh_params, **rvic_params_spatial}

# ParamManager
class ParamManager:
    def __init__(self, param_dicts: dict):
        """
        Initialize ParamManager with nested parameter dictionaries.

        Parameters
        ----------
        param_dicts : dict
            Nested parameter dictionary, e.g.:
            {
                "rvic_params": {
                    "VELOCITY": {
                        "default": [1.5],
                        "boundary": [0.5, 800.0],
                        "type": float,
                        "optimal": None,
                    }
                },
                ...
            }
        """
        self.param_template = deepcopy(param_dicts)
        self._index_map = self._build_index_map()

    def _build_index_map(self):
        """
        Build an index mapping for parameters.

        Returns
        -------
        list of tuples:
            Each tuple contains (group_name, param_name, dimension, type)
        """
        index_map = []
        for group, param_group in self.param_template.items():
            for param, meta in param_group.items():
                dim = len(meta.get("default", []))
                typ = meta.get("type", float)
                free = meta.get("free", False)
                index_map.append((group, param, dim, typ, free))
        return index_map
    
    def vector_free_mask(self):
        """
        Get a boolean mask indicating which elements of the flattened vector
        correspond to free parameters.

        Returns
        -------
        list of bool
        """
        mask = []
        for _, _, dim, _, free in self._index_map:
            mask.extend([free] * dim)
        return mask
    
    def to_vector(self, field='default', get_free=False):
        """
        Flatten parameters into a single list (vector) from specified field.

        Parameters
        ----------
        field : str
            The key inside parameter dict to extract (e.g. 'default' or 'optimal').

        Returns
        -------
        list:
            Flattened parameter values.
        """
        vec = []
        for group, param, dim, _, free in self._index_map:
            if get_free and not free:
                continue  # skip non-free parameters

            values = self.param_template[group][param].get(field)
            if values is None:
                values = [None] * dim
            vec.extend(values)
            
        return vec
        
    def to_dict(self, vector=None, field="optimal", get_free=False):
        """
        Build and return a full parameter dictionary with values filled from:
        - the internal template (if vector is None), or
        - the provided vector (if vector is given), written to `field`.

        Parameters
        ----------
        vector : list or None
            Flat parameter values to write to the specified field. If None, use existing field values.
        field : str
            The field to populate in the returned structure (e.g. "default" or "optimal").

        Returns
        -------
        dict:
            A new parameter dictionary with updated field values.
        """
        
        new_param = deepcopy(self.param_template)

        if vector is None:
            return new_param  # use stored default structure

        idx = 0
        for group, param, dim, typ, free in self._index_map:
            if get_free and not free:
                new_param[group][param][field] = new_param[group][param]['default']
                continue  # skip non-free parameters, use default

            values = vector[idx:idx+dim]
            idx += dim

            if typ is int:
                values = [int(round(v)) for v in values]
            elif typ is float:
                values = [float(v) for v in values]

            new_param[group][param][field] = values

        return new_param
    
    def format_vector(self, vector, get_free=False):
        
        formatted_vector = deepcopy(vector)
        
        idx = 0
        for _, _, dim, typ, free in self._index_map:
            if get_free and not free:
                continue  # skip non-free parameters, use default

            values = vector[idx:idx+dim]
            idx += dim

            if typ is int:
                values = [int(round(v)) for v in values]
            elif typ is float:
                values = [float(v) for v in values]
            else:
                values = [typ(v) for v in values]
                
            formatted_vector[idx-dim:idx] = values
            
        return formatted_vector
        
    def get_vector_info(self, get_free=False):
        """
        Get combined information of parameters as vectors.

        Returns
        -------
        dict:
            {
                "defaults": list of default values,
                "optimal": list of optimal values,
                "types": list of parameter types,
                "bounds": list of (min, max) tuples,
                "names": list of parameter full names like "group.param"
            }
        """
        defaults = self.to_vector(field='default')
        optimal = self.to_vector(field='optimal')
        types = self.vector_types()
        bounds = self.vector_bounds()
        names = self.vector_names()
    
        if get_free:
            free_mask = self.vector_free_mask()
            defaults = [d for d, f in zip(defaults, free_mask) if f]
            optimal = [o for o, f in zip(optimal, free_mask) if f]
            types = [t for t, f in zip(types, free_mask) if f]
            bounds = [b for b, f in zip(bounds, free_mask) if f]
            names = [n for n, f in zip(names, free_mask) if f]

        return {
            "defaults": defaults,
            "optimal": optimal,
            "types": types,
            "bounds": bounds,
            "names": names,
        }

    def vector_bounds(self, get_free=True):
        """
        Return a flat list of (min, max) tuples for each scalar parameter.

        Each boundary must be specified as a list of two lists:
        e.g., boundary = [[min1, min2, ...], [max1, max2, ...]]
        """
        
        bounds = []
        for group, param, dim, _, free in self._index_map:
            if get_free and not free:
                continue  # skip non-free parameters

            b = self.param_template[group][param].get("boundary")
            if not (isinstance(b, list) and len(b) == 2):
                raise ValueError(f"Boundary for {group}.{param} must be a list of [mins, maxs].")

            b_min, b_max = b
            if not (len(b_min) == len(b_max) == dim):
                raise ValueError(
                    f"Boundary length mismatch in {group}.{param}: "
                    f"expected {dim}, got {len(b_min)} and {len(b_max)}"
                )

            bounds.extend([(minv, maxv) for minv, maxv in zip(b_min, b_max)])
            
        return bounds

    def vector_types(self, get_free=True):
        """
        Get flattened list of parameter types.

        Returns
        -------
        list of types
        """
        types = []
        for _, _, dim, typ, free in self._index_map:
            if get_free and not free:
                continue  # skip non-free parameters

            types.extend([typ] * dim)
        return types

    def vector_names(self, get_free=False):
        """
        Get flattened list of parameter names as "group.param".

        Returns
        -------
        list of str
        """
        names = []
        for group, param, dim, _, free in self._index_map:
            if get_free and not free:
                continue  # skip non-free parameters

            names.extend([f"{group}.{param}"] * dim)
        return names

    def save(self, filepath, param_dict=None):
        """
        Save current parameter structure to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to save JSON file.
        """
        def serialize(d):
            d = deepcopy(d)
            for group in d.values():
                for param in group.values():
                    if "type" in param:
                        param["type"] = param["type"].__name__
            return d

        with open(filepath, "w") as f:
            if param_dict is None:
                json.dump(serialize(self.param_template), f, indent=2)
            else:
                json.dump(serialize(param_dict), f, indent=2)

    @classmethod
    def load(cls, filepath):
        """
        Load parameter structure from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file.

        Returns
        -------
        ParamManager
        """
        def deserialize(d):
            for group in d.values():
                for param in group.values():
                    if "type" in param:
                        if param["type"] == "int":
                            param["type"] = int
                        elif param["type"] == "float":
                            param["type"] = float
            return d

        with open(filepath, "r") as f:
            raw = json.load(f)
        return cls(deserialize(raw))


if __name__ == "__main__":
    # Example usage
    pm = ParamManager(params_minimal)  # params
    bounds = pm.vector_bounds(get_free=False)
    bounds_free = pm.vector_bounds(get_free=True)
    
    vector = pm.to_vector(field='default', get_free=False)
    vector_free = pm.to_vector(field='default', get_free=True)
    
    vector_free_modify = deepcopy(vector_free)
    vector_free_modify[0] = 1.2
    vector_free_modify[1] = 3.2
    
    formatted_vector = pm.format_vector(vector, get_free=False)
    formatted_vector_free = pm.format_vector(vector_free_modify, get_free=True)
    
    restored_params = pm.to_dict(vector, field='default', get_free=False)
    restored_params_get_free = pm.to_dict(vector_free_modify, field='optimal', get_free=True)
    
    print("Flattened vector:", vector)
    print("Restored parameters:", restored_params)
    
    # Save and load example
    pm.save("params.json")
    loaded_pm = ParamManager.load("params.json")
    
    print("Loaded parameters:", loaded_pm.param_template)