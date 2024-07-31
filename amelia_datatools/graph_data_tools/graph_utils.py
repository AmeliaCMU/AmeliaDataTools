import matplotlib.colors
import pandas as pd
import networkx 
from typing import Tuple, List
from geographiclib.geodesic import Geodesic
from math import cos, sin, radians


def get_colors() -> List:
    """ 
    Returns color List containing the semantic classes used for plotting. 
    The index of the array corresponds to the categorical numerical value of the class
    """
    file = open("semantic_colors.txt", "r")
    colors = file.read().split(',')
    for i in range(len(colors)):
        colors[i] = matplotlib.colors.to_rgba(colors[i])
    return colors

def get_node_colors(graph: networkx.MultiDiGraph) -> pd.Series:
    """
    Return a series with the corresponding color for each node in the directed graph, used for ploting.
    """
    nc = []
    color_hash = get_colors()
    for key in graph._node.keys():
        nc.append(color_hash[graph._node[key]['node_type']])
    nc = pd.Series(nc,index=graph._node.keys())
    return nc

def get_range_and_bearing(geodesic : Geodesic, reference_point : Tuple, 
                          point: Tuple, to_radians: bool =True) -> Tuple:
    """ 
    Computes the bearing angle and range in meters between the given point and the specified reference.
    ---
    Inputs:
        - geodesic: geode of the the earth.
        - reference_point: tuple containing the point with respect to which calculate bearing and range (lon, lat).
        - point: tuple containing the input point.
        - to_radians: flag that indicates whether the bearing will be reported in radians.
    """
    ref_lon, ref_lat = reference_point
    g = geodesic.Inverse(ref_lat, ref_lon, point[0], point[1])
    bearing = g['azi1']
    range = g['s12']
    range = range/1000
    if to_radians:
        bearing = radians(g['azi1'])
    return (range, bearing)

def calculate_x_y(geodesic: Geodesic, reference_point: Tuple, point: Tuple, to_radians=True):
    """ 
    Computes the local cartesian cordiantes of a point given a specified reference.
    ---
    Inputs:
        - geodesic: geode of the the earth.
        - reference_point: tuple containing the point with respect to which calculate bearing and range (lon, lat).
        - point: tuple containing the input point.
        - to_radians: flag that indicates whether the bearing will be reported in radians.
    """
    r, b = get_range_and_bearing(geodesic, reference_point,point)
    # Apply conversion formula
    x = r * cos(b)
    y = r * sin(b)
    return [x,y]

def _is_endpoint(G, node, strict=True):
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # rule 1
    if node in neighbors:
        # if the node appears in its List of neighbors, it self-loops
        # this is always an endpoint.
        return True

    # rule 2
    elif G.out_degree(node) == 0 or G.in_degree(node) == 0:
        # if node has no incoming edges or no outgoing edges, it is an endpoint
        return True

    # rule 3
    elif not (n == 2 and (d == 2 or d == 4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    # rule 4
    elif not strict:
        # non-strict mode: do its incident edges have different OSM IDs?
        osmids = []

        # add all the edge OSM IDs for incoming edges
        for u in G.predecessors(node):
            for key in G[u][node]:
                osmids.append(G.edges[u, node, key]["osmid"])

        # add all the edge OSM IDs for outgoing edges
        for v in G.successors(node):
            for key in G[node][v]:
                osmids.append(G.edges[node, v, key]["osmid"])

        # if there is more than 1 OSM ID in the List of edge OSM IDs then it is
        # an endpoint, if not, it isn't
        return len(set(osmids)) > 1

    # if none of the preceding rules returned true, then it is not an endpoint
    else:
        return False
    
def _build_path(G, endpoint, endpoint_successor, endpoints, zones_to_avoid):
    """
    Build a path of nodes from one endpoint node to next endpoint node.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    endpoint : int
        the endpoint node from which to start the path
    endpoint_successor : int
        the successor of endpoint through which the path to the next endpoint
        will be built
    endpoints : set
        the set of all nodes in the graph that are endpoints

    Returns
    -------
    path : List
        the first and last items in the resulting path List are endpoint
        nodes, and all other items are interstitial nodes that can be removed
        subsequently
    """
    # start building path from endpoint node through its successor
    path = [endpoint, endpoint_successor]

    # for each successor of the endpoint's successor
    for successor in G.successors(endpoint_successor):
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            while successor not in endpoints and G._node[successor]['node_type'] not in zones_to_avoid:
                # find successors (of current successor) not in path
                successors = [n for n in G.successors(successor) if n not in path]

                # 99%+ of the time there will be only 1 successor: add to path
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)

                # handle relatively rare cases or OSM digitization quirks
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        # we have come to the end of a self-looping edge, so
                        # add first node to end of path to close it and return
                        return path + [endpoint]
                    else:  # pragma: no cover
                        # this can happen due to OSM digitization error where
                        # a one-way street turns into a two-way here, but
                        # duplicate incoming one-way edges are present
                        return path
                else:  # pragma: no cover
                    # if successor has >1 successors, then successor must have
                    # been an endpoint because you can go in 2 new directions.
                    # this should never occur in practice
                    return []

            # if this successor is an endpoint, we've completed the path
            return path

    # if endpoint_successor has no successors not already in the path, return
    # the current path: this is usually due to a digitization quirk on OSM
    return path

def _get_paths_to_simplify(endpoints, G, zones_to_avoid):
    # for each endpoint node, look at each of its successor nodes
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if (successor not in endpoints) and (G._node[successor]['node_type'] not in zones_to_avoid):
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield _build_path(G, endpoint, successor, endpoints, zones_to_avoid)