# code: utf-8
# author: Xudong Zheng
# email: z786909151@163.com

"""Module ``easy_vic_build.tools.routing_func.river_network``."""

import numpy as np
import networkx as nx
from collections import deque

direction_mapping_num = {
    1: (0, 1),  # (lat/row: 1 south, -1 north, lon/column: 1 east, -1 west)
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
    0: None,
    -1: None,
    255: None,
}

direction_mapping_str = {
    1: "E",
    2: "SE",
    4: "S",
    8: "SW",
    16: "W",
    32: "NW",
    64: "N",
    128: "NE",
    0: None,
    -1: None,
    255: None,
}
    
def cal_threshold(flow_acc):
    """Calculate default river threshold from flow accumulation.

    Parameters
    ----------
    flow_acc : numpy.ndarray
        Flow-accumulation raster.

    Returns
    -------
    float
        80th percentile value of ``flow_acc``.
    """
    threshold = np.percentile(flow_acc, 80)
    return threshold


def get_display_positions(G):
    """Build plotting positions from node matrix coordinates.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes contain ``matrix_pos=(row, col)`` attributes.

    Returns
    -------
    dict
        Mapping ``node -> (x, y)`` where ``x=col`` and ``y=-row`` for display.
    """
    pos = {}
    for node in G.nodes():
        i, j = G.nodes[node]['matrix_pos']
        pos[node] = (j, -i)
    return pos


def create_river_network_graph(flow_direction, flow_acc, threshold=None, mask=None):
    """Create directed river-network graph from D8 flow direction and accumulation.

    Parameters
    ----------
    flow_direction : numpy.ndarray
        2D D8 flow-direction raster using codes in ``direction_mapping_num``.
    flow_acc : numpy.ndarray
        2D flow-accumulation raster with the same shape as ``flow_direction``.
    threshold : float, optional
        River-definition threshold on ``flow_acc``. If ``None``, 80th percentile
        is used via :func:`cal_threshold`.
    mask : numpy.ndarray, optional
        2D mask raster with the same shape as ``flow_direction``.

    Returns
    -------
    tuple
        ``(G, position_to_node, threshold)`` where:

        - ``G`` is a ``networkx.DiGraph`` with node attributes:
          ``matrix_pos``, ``direction_value``, ``flow_acc``, ``node_type``,
          ``is_river`` and optional ``mask``.
        - ``position_to_node`` maps ``(row, col) -> node_name``.
        - ``threshold`` is the effective threshold used.

    Notes
    -----
    Node type rules:

    - ``river``: ``flow_acc >= threshold``
    - ``sink``: direction is invalid (0/-1/255/None)
    - ``hillslope``: otherwise

    Examples
    --------
    >>> flow_direction = np.array([[1, 4], [64, 0]])
    >>> flow_acc = np.array([[10.0, 20.0], [5.0, 1.0]])
    >>> G, pos2node, thr = create_river_network_graph(flow_direction, flow_acc, threshold=8)
    >>> isinstance(G, nx.DiGraph), isinstance(pos2node, dict)
    (True, True)
    """
    # threshold
    if threshold is None:
        threshold = cal_threshold(flow_acc)
        print(f"Calculate threshold: {threshold:.2f}")
    else:
        print(f"Provided threshold: {threshold}")
        
    # create graph (get G and node positions)
    G = nx.DiGraph()
    rows, cols = flow_direction.shape
    
    assert flow_acc.shape == (rows, cols), "flow_acc must have same shape with flow_direction"
    
    if mask is not None:
        assert mask.shape == (rows, cols), "mask must have same shape with flow_direction"
    
    def value_to_direction_code(value):
        if value in [0, -1, 255]:
            return None
        
        if value in direction_mapping_num:
            return value
        
        return None
        
    position_to_node = {}
    
    # create nodes and assign attributes
    for i in range(rows):
        for j in range(cols):
            node_name = f"cell_{i}_{j}"
            direction_value = flow_direction[i, j]
            flow_acc_value = flow_acc[i, j]
            
            if flow_acc_value >= threshold:
                node_type = "river"
            elif direction_value in [0, -1, 255] or direction_value is None:
                node_type = "sink"
            else:
                node_type = "hillslope"
            
            if mask is not None:
                G.add_node(
                    node_name,
                    matrix_pos=(i, j),
                    direction_value=direction_value,
                    flow_acc=flow_acc_value,
                    node_type=node_type,
                    is_river=flow_acc_value>=threshold,
                    mask=mask[i, j],
                )
            else:
                G.add_node(
                    node_name,
                    matrix_pos=(i, j),
                    direction_value=direction_value,
                    flow_acc=flow_acc_value,
                    node_type=node_type,
                    is_river=flow_acc_value>=threshold
                )
            
            position_to_node[(i, j)] = node_name
    
    # create edges
    for i in range(rows):
        for j in range(cols):
            current_node = position_to_node.get((i, j))
            if current_node is None:
                continue
                
            direction_value = flow_direction[i, j]
            direction_code = value_to_direction_code(direction_value)
            
            if direction_code is None:
                continue
            
            if direction_code in direction_mapping_num:
                di, dj = direction_mapping_num[direction_code]
                next_i, next_j = i + di, j + dj
                
                print(f"({i}, {j}) -> ({next_i}, {next_j}): dir: {direction_mapping_str[direction_code]}")
                
                if 0 <= next_i < rows and 0 <= next_j < cols:
                    target_node = position_to_node.get((next_i, next_j))
                    
                    if target_node:
                        from_mask = mask[i, j] if mask is not None else None
                        to_mask = mask[next_i, next_j] if mask is not None else None
                        
                        G.add_edge(current_node, target_node, 
                                  direction_code=direction_code,
                                  from_pos=(i, j),
                                  to_pos=(next_i, next_j),
                                  from_mask=from_mask,
                                  to_mask=to_mask)
    
    return G, position_to_node, threshold


def find_path_to_sink(G, start_node, sinks, visited_edges):
    """Find one downstream river path from a start node to a sink.

    Parameters
    ----------
    G : networkx.DiGraph
        River graph with node attribute ``is_river``.
    start_node : str
        Starting node name.
    sinks : sequence
        Candidate sink nodes.
    visited_edges : set
        Mutable edge set used to reduce duplicate traversals across calls.

    Returns
    -------
    list
        Node sequence of one path. May contain only ``start_node`` when no
        downstream river successor exists.
    """
    path = [start_node]
    current = start_node
    max_steps = 1000
    
    for _ in range(max_steps):
        if current in sinks:
            break
            
        successors = [n for n in G.successors(current) if G.nodes[n].get('is_river', False)]
        
        if not successors:
            break

        unvisited_successors = [n for n in successors if (current, n) not in visited_edges]
        
        if unvisited_successors:
            next_node = unvisited_successors[0]
            visited_edges.add((current, next_node))
        else:
            next_node = successors[0]
        
        path.append(next_node)
        current = next_node
    
    return path

def find_river_paths(G, min_in_degree=None):
    """Extract unique downstream paths across river nodes.

    Parameters
    ----------
    G : networkx.DiGraph
        River graph.
    min_in_degree : int, optional
        Source-node criterion. If ``None``, the minimum in-degree among river
        nodes is used.

    Returns
    -------
    list of list
        Unique river paths, each represented by a node-name sequence.

    Notes
    -----
    If no sink node (out-degree=0) is found, the river node with maximum
    ``flow_acc`` is used as fallback sink.
    """
    river_nodes = [node for node in G.nodes() if G.nodes[node].get("is_river", False)]
    print(f"Find {len(river_nodes)} river nodes")
    
    min_in_degree = min([G.in_degree(node) for node in river_nodes]) if min_in_degree is None else min_in_degree
    sources = [node for node in river_nodes if G.in_degree(node) == min_in_degree]
    print(f"Find {len(sources)} source nodes, min_in_degree: {min_in_degree}")
    
    sinks = [node for node in river_nodes if G.out_degree(node) == 0]
    print(f"Find {len(sinks)} sink nodes")
    
    if not sinks:
        sinks = [max(river_nodes, key=lambda x: G.nodes[x].get('flow_acc', 0))]
        print(f"use the max acc nodes as sinks: {sinks[0]}")
    
    all_paths = []
    visited_edges = set()
    for start_node in river_nodes:
        path_to_sink = find_path_to_sink(G, start_node, sinks, visited_edges)
        if path_to_sink and len(path_to_sink) > 1:
            all_paths.append(path_to_sink)
            
    unique_paths = []
    seen_paths = set()
    
    for path in all_paths:
        path_tuple = tuple(path)
        if path_tuple not in seen_paths:
            seen_paths.add(path_tuple)
            unique_paths.append(path)
    
    return unique_paths
    

def sort_river_paths_by_lengths(river_paths, descending=True):
    """Sort river paths by path length and return ranking metadata.

    Parameters
    ----------
    river_paths : list of list
        Input river paths.
    descending : bool, optional
        If ``True``, longest-first ordering; otherwise shortest-first.

    Returns
    -------
    tuple
        ``(sorted_river_paths, length_info)`` where ``length_info`` is a list
        of dictionaries with rank, length, start/end nodes, and original path.
    """
    paths_with_length = [(path, len(path)) for path in river_paths]
    
    if descending:
        sorted_paths_with_length = sorted(paths_with_length, key=lambda x: x[1], reverse=True)
    else:
        sorted_paths_with_length = sorted(paths_with_length, key=lambda x: x[1])

    sorted_river_paths = [item[0] for item in sorted_paths_with_length]
    
    # save length_info
    length_info = []
    for i, (path, length) in enumerate(sorted_paths_with_length):
        start_node = path[0] if path else None
        end_node = path[-1] if path else None
        length_info.append({
            'rank': i + 1,
            'length': length,
            'start_node': start_node,
            'end_node': end_node,
            'path': path
        })
        
    return sorted_river_paths, length_info


def extract_connected_river_network(G, min_size=10, mask=False):
    """Extract largest weakly connected river subnetwork.

    Parameters
    ----------
    G : networkx.DiGraph
        Full river graph.
    min_size : int, optional
        Minimum component size to keep before selecting largest component.
    mask : bool, optional
        If ``True``, only keep river nodes with node attribute ``mask`` truthy.

    Returns
    -------
    networkx.DiGraph
        Largest qualified weakly connected river subgraph. Returns an empty
        ``DiGraph`` when no component passes ``min_size``.
    """
    if mask:
        river_nodes = [n for n, d in G.nodes(data=True) if d.get("is_river", False) and d.get("mask")]
    else:
        river_nodes = [n for n, d in G.nodes(data=True) if d.get("is_river", False)]
    
    H = G.subgraph(river_nodes).copy()

    components = list(nx.weakly_connected_components(H))

    components = [c for c in components if len(c) >= min_size]

    if len(components) > 0:
        largest_component = max(components, key=len)
        H_main = H.subgraph(largest_component).copy()
    else:
        H_main = nx.DiGraph()
    
    return H_main


def build_node_features(G, matrix_features):
    """Sample raster features to graph nodes by ``matrix_pos``.

    Parameters
    ----------
    G : networkx.Graph
        Graph with node attribute ``matrix_pos=(row, col)``.
    matrix_features : list of numpy.ndarray
        Feature rasters to sample. Supported shapes:

        - 2D: ``(row, col)`` -> scalar sampled per node.
        - 3D: ``(time, row, col)`` -> 1D time series sampled per node.

    Returns
    -------
    list of numpy.ndarray
        Node-aligned features for each input matrix, preserving input order.
    """
    node_features = [[] for _ in range(len(matrix_features))]
    
    for node_name in G.nodes:
        i, j = G.nodes[node_name]["matrix_pos"]
        
        for n in range(len(matrix_features)):
            matrix_feature = matrix_features[n]
            if len(matrix_feature.shape) == 3:
                node_features[n].append(matrix_feature[:, i, j])
            else:
                node_features[n].append(matrix_feature[i, j])
    
    node_features = [np.array(node_feature) for node_feature in node_features]
    
    return node_features


def build_upstream_subgraph_for_node(
    G, target_node, khop=None, cutoff_node=None
):
    """Build upstream subgraph for a target node with optional k-hop limit.

    Parameters
    ----------
    G : networkx.DiGraph
        Directed river graph.
    target_node : str
        Node for which upstream contributors are traced.
    khop : int, optional
        Maximum upstream hop distance. If ``None``, traverse until exhaustion.
    cutoff_node : str, optional
        Upstream node at which traversal stops (node itself is still connected
        to its downstream edge in the returned subgraph).

    Returns
    -------
    tuple
        ``(new_G, new_edges)`` where ``new_G`` is the upstream subgraph and
        ``new_edges`` is the collected upstream edge list.
    """
    new_edges = []
    visited = set()
    queue = deque()
    queue.append((target_node, 0))

    while queue:
        node, hop = queue.popleft()

        if khop is not None and hop >= khop:
            continue

        for source_node in G.predecessors(node):
            new_edges.append((source_node, node))
            
            if source_node == cutoff_node:
                continue  # cutoff
            
            if (source_node, target_node) in visited:
                continue
            
            if source_node not in visited:
                visited.add(source_node)
                queue.append((source_node, hop + 1))

    # create subgraph
    sub_nodes = set([target_node] + [u for u, _ in new_edges])
    new_G = G.subgraph(sub_nodes).copy()
    new_G.add_edges_from(new_edges)

    return new_G, new_edges


if __name__ == "__main__":
    # # read
    # home = "C:\\research\\Routing"
    # data_dir = os.path.join(home, "data")
    # flow_direction_fp = os.path.join(data_dir, "flow_direction_file.nc")
    # domain_fp = os.path.join(data_dir, "domain.nc")
    # with Dataset(flow_direction_fp, "r") as flow_direction_dataset:
    #     flow_direction = flow_direction_dataset.variables["Flow_Direction"][:, :]
    #     flow_acc = flow_direction_dataset.variables["Source_Area"][:, :]
    
    # with Dataset(domain_fp, "r") as domain_dataset:
    #     domain_mask = domain_dataset.variables["mask"][:, :]
    
    # # create graph
    # threshold = 3 # 5
    # river_network_graph, node_positions, threshold = create_river_network_graph(flow_direction, flow_acc, threshold=threshold, mask=domain_mask)  # domain_mask
    
    # # find river
    # river_paths = find_river_paths(river_network_graph)
    # sorted_river_paths, length_info = sort_river_paths_by_lengths(river_paths, descending=True)
    
    # # plot
    # fig, ax = plot_river_network(river_network_graph, mask_by="both", threshold=threshold)  # sorted_river_paths[:2], "both"
    pass
