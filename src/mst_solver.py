import networkx as nx
import pandas as pd
import os
import heapq
from typing import Dict, Tuple, List, Optional

class MSTSolver:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.mst = None
        self.backups = {}
        self.parent, self.depth, self.up, self.max_weight = {}, {}, {}, {}
        self._backup_links_cache = None  # ← Cache để tránh tính lại

    def solve(self) -> nx.Graph:
        if not nx.is_connected(self.graph):
            raise ValueError("Graph is not connected")
        self._prim()
        self._build_lifting()
        self._compute_backups()
        total = sum(self.mst[u][v]['distance'] for u, v in self.mst.edges())
        print(f"✓ {self.mst.number_of_edges()} edges, {total:.2f} km, {len(self.backups)} backups")
        return self.mst

    def _prim(self):
        self.mst = nx.Graph()
        self.mst.add_nodes_from(self.graph.nodes(data=True))
        root = min(self.graph.nodes())
        visited, pq = set(), [(0, root, root)]
        while pq:
            d, p, v = heapq.heappop(pq)
            if v in visited:
                continue
            visited.add(v)
            if v != root:
                self.mst.add_edge(p, v, distance=d)
            self.parent[v], self.depth[v] = p, self.depth.get(p, -1) + 1
            for n in self.graph.neighbors(v):
                if n not in visited:
                    heapq.heappush(pq, (self.graph[v][n]['distance'], v, n))

    def _build_lifting(self):
        log_n = max(1, len(self.mst.nodes()).bit_length())
        for v in self.mst.nodes():
            p = self.parent[v]
            w = self.mst[v][p]['distance'] if self.mst.has_edge(v, p) else 0
            self.up[v], self.max_weight[v] = [p] + [None] * log_n, [w] + [0] * log_n
        for i in range(1, log_n + 1):
            for v in self.mst.nodes():
                if self.up[v][i-1]:
                    a = self.up[v][i-1]
                    self.up[v][i] = self.up[a][i-1]
                    self.max_weight[v][i] = max(self.max_weight[v][i-1], self.max_weight[a][i-1])

    def _find_max_on_path(self, u: int, v: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        if self.depth[u] < self.depth[v]:
            u, v = v, u
        max_w, max_e, log_n, diff = 0, None, len(self.up[u]) - 1, self.depth[u] - self.depth[v]
        for i in range(log_n + 1):
            if diff & (1 << i):
                if self.max_weight[u][i] > max_w:
                    max_w, max_e = self.max_weight[u][i], (u, self.up[u][i])
                u = self.up[u][i]
        if u != v:
            for i in range(log_n, -1, -1):
                if self.up[u][i] != self.up[v][i]:
                    for x in [u, v]:
                        if self.max_weight[x][i] > max_w:
                            max_w, max_e = self.max_weight[x][i], (x, self.up[x][i])
                    u, v = self.up[u][i], self.up[v][i]
            for x in [u, v]:
                if self.max_weight[x][0] > max_w:
                    max_w, max_e = self.max_weight[x][0], (x, self.up[x][0])
        return max_w, max_e

    def _compute_backups(self):
        for u, v in self.graph.edges():
            if not self.mst.has_edge(u, v):
                w = self.graph[u][v]['distance']
                mw, me = self._find_max_on_path(u, v)
                if me:
                    key = tuple(sorted(me))
                    if key not in self.backups or w < self.backups[key][1]:
                        self.backups[key] = ((u, v), w)
        
        self._backup_links_cache = [
            {
                'orig_u': orig_u,
                'orig_v': orig_v,
                'backup_u': backup_edge[0],
                'backup_v': backup_edge[1],
                'cost': cost
            }
            for (orig_u, orig_v), (backup_edge, cost) in self.backups.items()
        ]

    def handle_failure(self, node: int) -> Dict:
        if node not in self.mst.nodes():
            return {'error': 'Node not in MST'}
        before = sum(self.mst[u][v]['distance'] for u, v in self.mst.edges())
        self.mst.remove_node(node)
        if node in self.graph.nodes():
            self.graph.remove_node(node)
        applied = []
        for (u, v), (be, w) in list(self.backups.items()):
            if u == node or v == node:
                if be[0] in self.mst.nodes() and be[1] in self.mst.nodes():
                    self.mst.add_edge(*be, distance=w)
                    applied.append((*be, w))
        comps = list(nx.connected_components(self.mst))
        while len(comps) > 1:
            best = None
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    for n1 in comps[i]:
                        for n2 in comps[j]:
                            if self.graph.has_edge(n1, n2):
                                d = self.graph[n1][n2]['distance']
                                if not best or d < best[2]:
                                    best = (n1, n2, d, i, j)
            if not best:
                break
            self.mst.add_edge(best[0], best[1], distance=best[2])
            applied.append((best[0], best[1], best[2]))
            comps[best[3]] = comps[best[3]].union(comps[best[4]])
            comps.pop(best[4])
        after = sum(self.mst[u][v]['distance'] for u, v in self.mst.edges())
        success = len(comps) == 1
        print(f"{'✓' if success else '✗'} Node {node}: {len(applied)} fixes, {before:.2f}→{after:.2f} km")
        return {'success': success, 'applied': applied, 'before': round(before, 2), 'after': round(after, 2)}

    def get_backup_links(self) -> List[Dict]:
        if self._backup_links_cache is None:
            raise RuntimeError("Must call solve() before get_backup_links()")
        return self._backup_links_cache

    def export(self, mst_csv: str = "mst.csv"):
        os.makedirs("results", exist_ok=True)
        pd.DataFrame([{'u': u, 'v': v, 'dist': self.graph[u][v]['distance'], 
                       'in_mst': self.mst.has_edge(u, v)}
                      for u, v in self.graph.edges() if self.graph.has_edge(u, v)]
                     ).to_csv(f"results/{mst_csv}", index=False)
        print("✓ Exported MST")