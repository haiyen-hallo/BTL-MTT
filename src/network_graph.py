import networkx as nx
from geopy.distance import geodesic
import math

class NetworkGraph:
    """Compute distances and add capacity for network edges."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def compute_all(self):
        """Compute distances and add capacity to all edges."""
        coords = {node: (data['Latitude'], data['Longitude']) 
                  for node, data in self.graph.nodes(data=True)}
        
        for u, v in self.graph.edges():
            distance = geodesic(coords[u], coords[v]).kilometers
            self.graph[u][v]['distance'] = distance
            self.graph[u][v].setdefault('capacity', math.ceil(distance / 1000) * 100)