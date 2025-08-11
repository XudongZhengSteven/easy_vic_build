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
    [20]            D4 (g), it can be set as 2
    [21]            D1 (g)
    [22]            D2 (g)
    [23]            D3 (g)
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
    },
    
    "soil_layers_breakpoints": {
        "default": [3, 9],  # soil layer breakpoints, original layers -> modeling layers, note exclusive
        "boundary": [[1, 3], [3, 9]],
        "type": int,
        "optimal": [None, None],
    },
    
    "b_infilt": {
        "default": [0.0, 1.0],
        "boundary": [[-2.0, 0.8], [1.0, 1.2]],
        "type": float,
        "optimal": [None, None],
    },
    
    "ksat": {
        "default": [-0.6, 0.0126, -0.0064],  # from Cosby et al. (1984)
        "boundary": [[-0.66, 0.0113, -0.007], [-0.54, 0.0139, -0.0058]],  # +- 10%
        "type": float,
        "optimal": [None, None, None],
    },
    
    "phi_s": {
        "default": [50.5, -0.142, -0.037],  # from Cosby et al. (1984)
        "boundary": [[45.5, -0.3, -0.1], [55.5, -0.01, -0.01]],
        "type": float,
        "optimal": [None, None, None],
    },
    
    "psis": {
        "default": [1.54, -0.0095, 0.0063],  # from Cosby et al. (1984)
        "boundary": [[1.0, -0.01, 0.006], [2.0, -0.009, 0.0066]],
        "type": float,
        "optimal": [None, None, None],
    },
    
    "b_retcurve": {
        "default": [3.1, 0.157, -0.003],  # from Cosby et al. (1984)
        "boundary": [[2.5, 0.1, -0.005], [3.6, 0.2, -0.001]],
        "type": float,
        "optimal": [None, None, None],
    },
    
    "expt": {
        "default": [3.0, 2.0],  # from Campbell (1974), expt=2b+3
        "boundary": [[2.8, 1.5], [3.2, 2.5]],
        "type": float,
        "optimal": [None, None],
    },
    
    "fc": {
        "default": [1.0],
        "boundary": [[0.8], [1.2]],
        "type": float,
        "optimal": [None],
    },
    
    "D4": {
        "default": [2.0],  # it can be set as 2
        "boundary": [[1.5], [2.5]],
        "type": float,
        "optimal": [None],
    },
    
    "D1": {
        "default": [2.0],
        "boundary": [[1.75], [3.5]],
        "type": float,
        "optimal": [None],
    },
    
    "D2": {
        "default": [2.0],
        "boundary": [[1.75], [3.5]],
        "type": float,
        "optimal": [None],
    },
    
    "D3": {
        "default": [1.0],
        "boundary": [[0.001], [2.0]],
        "type": float,
        "optimal": [None],
    },
    
    "dp": {
        "default": [1.0],
        "boundary": [[0.9], [1.1]],
        "type": float,
        "optimal": [None],
    },
    
    "bubble": {
        "default": [0.32, 4.3],
        "boundary": [[0.1, 0.0], [0.9, 10.0]],
        "type": float,
        "optimal": [None, None],
    },
    
    "quartz": {
        "default": [0.8],
        "boundary": [[0.7], [0.9]],
        "type": float,
        "optimal": [None],
    },
    
    "bulk_density": {
        "default": [1.0],
        "boundary": [[0.9], [1.1]],
        "type": float,
        "optimal": [None],
    },
    
    "soil_density": {
        "default": [1.0, 1.0, 1.0],  # the three g can be set same
        "boundary": [[0.9, 0.9, 0.9], [1.1, 1.1, 1.1]],
        "type": float,
        "optimal": [None, None, None],
    },
    
    "Wcr_FRACT": {
        "default": [1.0],
        "boundary": [[0.8], [1.2]],
        "type": float,
        "optimal": [None],
    },
    
    "wp": {
        "default": [1.0],
        "boundary": [[0.8], [1.2]],
        "type": float,
        "optimal": [None],
    },
    
    "Wpwp_FRACT": {
        "default": [1.0],
        "boundary": [[0.8], [1.2]],
        "type": float,
        "optimal": [None],
    },
    
    "rough": {
        "default": [1.0],  # it can be set as 1
        "boundary": [[0.9], [1.1]],
        "type": float,
        "optimal": [None],
    },
    
    "snow_rough": {
        "default": [1.0],  # it can be set as 1
        "boundary": [[0.9], [1.1]],
        "type": float,
        "optimal": [None],
    },
}

# guh params
guh_params = {
    "tp": {
        "default": [1.4],
        "boundary": [[1.0], [24.0]],
        "type": float,
        "optimal": [None],
    },
    
    "mu": {
        "default": [5.0],
        "boundary": [[2.0], [10.0]],
        "type": float,
        "optimal": [None],
    },
    
    "m": {
        "default": [3.0],
        "boundary": [[0.5], [6.0]],
        "type": float,
        "optimal": [None],
    }
}

# rvic params
rvic_params = {
    "VELOCITY": {
        "default": [1.5],  # velocity in m/s
        "boundary": [[0.1], [3.0]],
        "type": float,
        "optimal": [None],
    },
    
    "DIFFUSION": {
        "default": [800.0],
        "boundary": [[10.0], [4000.0]],
        "type": float,
        "optimal": [None],
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
                index_map.append((group, param, dim, typ))
        return index_map

    def to_vector(self, field='default'):
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
        for group, param, dim, _ in self._index_map:
            values = self.param_template[group][param].get(field)
            if values is None:
                vec.extend([None] * dim)
            else:
                vec.extend(values)
        return vec

    def from_vector(self, vector, field='default'):
        """
        Restore nested parameter dict from a flat vector into specified field.

        Parameters
        ----------
        vector : list
            Flat list of parameter values.
        field : str
            The key inside parameter dict to update (e.g. 'default' or 'optimal').

        Returns
        -------
        dict:
            New nested parameter dictionary with updated values.
        """
        new_param = deepcopy(self.param_template)
        idx = 0
        for group, param, dim, typ in self._index_map:
            vals = vector[idx:idx+dim]
            if typ is int:
                vals = [int(round(v)) for v in vals]
            elif typ is float:
                vals = [float(v) for v in vals]
            new_param[group][param][field] = vals
            idx += dim
        return new_param
    
    def to_dict(self, vector=None, field="optimal"):
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
        result = deepcopy(self.param_template)
        
        if vector is None:
            return result  # use stored default structure
        
        idx = 0
        for group, param, dim, typ in self._index_map:
            values = vector[idx:idx+dim]
            if typ is int:
                values = [int(round(v)) for v in values]
            elif typ is float:
                values = [float(v) for v in values]
            result[group][param][field] = values
            idx += dim

        return result
        
    def get_vector_info(self):
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
        return {
            "defaults": self.to_vector(field='default'),
            "optimal": self.to_vector(field='optimal'),
            "types": self.vector_types(),
            "bounds": self.vector_bounds(),
            "names": self.vector_names(),
        }

    def vector_bounds(self):
        """
        Return a flat list of (min, max) tuples for each scalar parameter.

        Each boundary must be specified as a list of two lists:
        e.g., boundary = [[min1, min2, ...], [max1, max2, ...]]
        """
        bounds = []
        for group, param, dim, _ in self._index_map:
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

    def vector_types(self):
        """
        Get flattened list of parameter types.

        Returns
        -------
        list of types
        """
        return [typ for _, _, dim, typ in self._index_map for _ in range(dim)]

    def vector_names(self):
        """
        Get flattened list of parameter names as "group.param".

        Returns
        -------
        list of str
        """
        return [f"{group}.{param}" for group, param, dim, _ in self._index_map for _ in range(dim)]

    def save(self, filepath):
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
            json.dump(serialize(self.param_template), f, indent=2)

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
    pm = ParamManager(params)
    pm.vector_bounds()
    vector = pm.to_vector(field='default')
    restored_params = pm.from_vector(vector, field='default')
    
    print("Flattened vector:", vector)
    print("Restored parameters:", restored_params)
    
    # Save and load example
    pm.save("params.json")
    loaded_pm = ParamManager.load("params.json")
    
    print("Loaded parameters:", loaded_pm.param_template)