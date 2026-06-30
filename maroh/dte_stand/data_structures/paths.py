from dataclasses import dataclass


@dataclass
class GraphPathElement:
    """
    Class to describe an element of path in networkx's MultiDiGraph
        (directed graph where multiple edges connecting two vertices can exist)
    It is used to describe a edge in the graph, so that list of edges makes a path
    from_ -  node id (usually name) of edge's source node
    to_ - node id (usually name) of edge's destination node
    index - edge index to differentiate the edges connecting the same two vertices
        (if there is only one edge connecting a pair of vertices, index is 0 for this edge)
    """
    from_: str
    to_: str
    index: int

