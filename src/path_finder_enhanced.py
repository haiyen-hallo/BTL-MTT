import networkx as nx
import pandas as pd
from itertools import islice


class PathFinderEnhanced:
    """K-Shortest Paths Finder for load balancing."""

    def __init__(self, graph: nx.Graph, mst_graph: nx.Graph = None, k_paths: int = 5):
        self.graph = graph
        self.k_paths = k_paths
        self.routes = {}
        
        # Cache only forward edges for undirected graph
        self.edge_dist = {(u, v): data['distance'] for u, v, data in graph.edges(data=True)}
        
        # Cache MST edges (normalized)
        self.mst_edges = {(min(u, v), max(u, v)) for u, v in mst_graph.edges()} if mst_graph else set()

    def find_all_paths(self, demands: pd.DataFrame):
        """Find K alternative paths for each demand."""
        for row in demands.itertuples(index=False):
            d_id, src, tgt, bw = int(row.id), int(row.source), int(row.target), float(row.bandwidth)
            
            try:
                paths = list(islice(nx.shortest_simple_paths(self.graph, src, tgt, weight='distance'), self.k_paths))
                self.routes[d_id] = {
                    'source': src, 'target': tgt, 'bandwidth': bw,
                    'alternative_paths': [self._path_info(p) for p in paths]
                }
            except nx.NetworkXNoPath:
                self.routes[d_id] = {'source': src, 'target': tgt, 'bandwidth': bw, 'alternative_paths': []}

    def _path_info(self, nodes):
        """Compute path info from node sequence."""
        edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
        dist = sum(self._get_dist(u, v) for u, v in edges)
        return {
            'path_nodes': nodes,
            'path_edges': edges,
            'distance': dist,
            'hops': len(edges),
            'uses_only_mst': all((min(u, v), max(u, v)) in self.mst_edges for u, v in edges)
        }

    def _get_dist(self, u, v):
        """Get edge distance (handles both directions)."""
        return self.edge_dist.get((u, v)) or self.edge_dist.get((v, u), 0)

    def get_paths(self, demand_id: int):
        """Get all paths for a demand."""
        return self.routes.get(demand_id, {}).get('alternative_paths', [])