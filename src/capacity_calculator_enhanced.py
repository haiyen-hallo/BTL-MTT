import networkx as nx
import pandas as pd

class CapacityAllocator:
    """Base capacity allocator with efficient edge management."""
    
    def __init__(self, graph: nx.Graph, path_finder, capacity_multiplier: float = 1.0):
        self.graph = graph
        self.path_finder = path_finder
        self.cap_mult = capacity_multiplier
        self.used = {(min(u, v), max(u, v)): 0.0 for u, v in graph.edges()}
        
        # Pre-compute capacities for O(1) lookup
        self.cap = {(min(u, v), max(u, v)): data['capacity'] * capacity_multiplier 
                    for u, v, data in graph.edges(data=True)}
    
    def _norm(self, u, v):
        """Normalize edge to (min, max)"""
        return (min(u, v), max(u, v))
    
    def _has_capacity(self, edges, bw):
        """Check if path has enough capacity"""
        return all(self.used[self._norm(u, v)] + bw <= self.cap[self._norm(u, v)] for u, v in edges)
    
    def _allocate(self, edges, bw):
        """Allocate bandwidth on path"""
        for u, v in edges:
            self.used[self._norm(u, v)] += bw
    
    def _get_paths(self, demand_id):
        """Get paths for demand - decoupled from internal structure"""
        return self.path_finder.get_paths(demand_id)
    
    def _path_max_util(self, edges, bw):
        """Calculate max utilization after allocation"""
        if not self._has_capacity(edges, bw):
            return float('inf')
        return max((self.used[self._norm(u, v)] + bw) / self.cap[self._norm(u, v)] for u, v in edges)


class CapacityCalculatorFCFS(CapacityAllocator):
    """FCFS: First path with capacity"""
    
    def process_demands(self, demands: pd.DataFrame):
        accepted, results = 0, []
        for row in demands.itertuples(index=False):
            d_id, bw = int(row.id), float(row.bandwidth)
            allocated = False
            
            for path in self._get_paths(d_id):
                if self._has_capacity(path['path_edges'], bw):
                    self._allocate(path['path_edges'], bw)
                    results.append((d_id, True, path['path_nodes']))
                    accepted += 1
                    allocated = True
                    break
            
            if not allocated:
                results.append((d_id, False, []))
        
        return accepted, results


class CapacityCalculatorHybrid(CapacityAllocator):
    """Hybrid: Load balancing with multiple strategies"""
    
    def __init__(self, graph, path_finder, capacity_multiplier: float = 1.0):
        super().__init__(graph, path_finder, capacity_multiplier)
    
    def _best_path(self, demand_id, bw):
        """Select path with minimum max utilization"""
        paths = self._get_paths(demand_id)
        if not paths:
            return None
        
        best = min(paths, key=lambda p: self._path_max_util(p['path_edges'], bw), default=None)
        return best if best and self._has_capacity(best['path_edges'], bw) else None
    
    def process_demands(self, demands: pd.DataFrame):
        """Process with multiple sorting strategies"""
        allocated = set()
        
        for ascending in [True, False]:  # Small first, then large first
            for row in demands.sort_values('bandwidth', ascending=ascending).itertuples(index=False):
                d_id, bw = int(row.id), float(row.bandwidth)
                
                if d_id not in allocated:
                    path = self._best_path(d_id, bw)
                    if path:
                        self._allocate(path['path_edges'], bw)
                        allocated.add(d_id)
        
        pending = demands[~demands['id'].isin(allocated)]
        return len(allocated), pending.to_dict('records')