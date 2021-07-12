""" for use with networkx"""
import networkx as nx

def path_between_nodepair(graph, nodes, fail_if_multiple=True, maxn=10):
    """ find list of edges between nodes
    will be in directed format (but actually undir)
    e.g., [(0,1,0), (1,2,0)]..
    - fail_if_multiple, then makes sure the nodes
    you gave me define a single path
    IN:
    - nodes, list of ints, [0,1]
    OUT:
    - list of edges
    """

    paths = list(nx.all_simple_edge_paths(graph, nodes[0], nodes[1], maxn))
    if fail_if_multiple and len(paths)>1:
        print(paths)
        assert False, "mutlipel paths"
    elif len(paths)==0:
        assert False, "no path found"
        
    path = paths[0]

    return path


def path_through_list_nodes(graph, list_ni):
    """ given ordered nodes, get the corresponding directed edges (but actually undir)
     key is that between each pair should be unambiguous path.
     e..g, list_ni = [0,5,2]
     --> [(0, 5, 0), (5, 2, 0)]
     """
    
    path = []
    for i in range(len(list_ni)-1):
        pair = [list_ni[i], list_ni[i+1]]
        list_of_edges = path_between_nodepair(graph, pair)
        path.extend(list_of_edges)

    return path