import os
import pandas as pd
from collections import defaultdict
from src.data_loader import DataLoader
from src.network_graph import NetworkGraph
from src.mst_solver import MSTSolver
from src.path_finder_enhanced import PathFinderEnhanced
from src.capacity_calculator_enhanced import CapacityCalculatorFCFS, CapacityCalculatorHybrid
from src.visualizer import Visualizer
from src.result_exporter import ResultExporter

CONFIG = {'fcfs': {'k': 5, 'cap': 1.0}, 'hybrid': {'k': 8, 'cap': 1.49, 'rounds': 2}}

class NetworkOptimizer:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.exporter = ResultExporter(output_dir)
        self._link_lookup = {}
        self._path_edges = defaultdict(set)
        
    def load_and_prepare(self, graph_path, demands_path):
        loader = DataLoader(graph_path, demands_path)
        graph, demands = loader.load_all()
        net = NetworkGraph(graph)
        net.compute_all()
        mst_solver = MSTSolver(net.graph)
        mst = mst_solver.solve()
        mst_solver.export()
        return net, mst, demands
    
    def _build_lookups(self, results, calc):
        self._link_lookup = {
            (min(e[0], e[1]), max(e[0], e[1])): {
                'source': e[0], 'target': e[1], 'used': used,
                'total': calc.cap[e], 'util': used / calc.cap[e]
            } for e, used in calc.used.items() if used > 0
        }
        self._path_edges = defaultdict(set)
        for d_id, ok, path in results:
            if ok:
                for i in range(len(path) - 1):
                    self._path_edges[(min(path[i], path[i+1]), max(path[i], path[i+1]))].add(d_id)
    
    def run_routing(self, net, mst, demands, config_key, fcfs_accepted=None):
        cfg = CONFIG[config_key]
        is_fcfs = config_key == 'fcfs'
        print(f"\n{'='*50}\n{config_key.upper()} ROUTING\n{'='*50}")
        
        pf = PathFinderEnhanced(net.graph, mst, k_paths=cfg['k'])
        pf.find_all_paths(demands)
        
        CalcClass = CapacityCalculatorFCFS if is_fcfs else CapacityCalculatorHybrid
        calc = CalcClass(net.graph, pf, capacity_multiplier=cfg['cap'])
        
        if is_fcfs:
            accepted, results = calc.process_demands(demands)
            self._build_lookups(results, calc)
            
            # Export Câu 1, 2, 3
            self.exporter.save_cau1(accepted, len(demands) - accepted, len(demands), cfg)
            self.exporter.save_cau2(pd.DataFrame({"demand_id": d, "path": p} for d, ok, p in results if ok))
            
            links_df = pd.DataFrame({**l, "utilization": l['util']} for l in self._link_lookup.values()).sort_values('utilization', ascending=False)
            high_util = sum(1 for l in self._link_lookup.values() if l['util'] >= 0.7)
            utils = [l['util'] for l in self._link_lookup.values()]
            self.exporter.save_cau3(high_util, links_df, {
                "total_edges": len(calc.used), "used_edges": len(self._link_lookup),
                "avg_utilization": sum(utils) / len(utils), "max_utilization": max(utils)
            })
            print(f"✓ {accepted}/{len(demands)} accepted, {len(self._link_lookup)} links cached")
            return accepted, calc
        else:
            accepted, pending = calc.process_demands(demands)
            self.exporter.save_cau4({"accepted": accepted, "pending": len(pending)}, fcfs_accepted, len(demands), cfg)
            pending_ids = {r['id'] for r in pending}
            self.exporter.save_best_demands(pd.DataFrame(
                {"demand_id": int(r.id), "bandwidth": float(r.bandwidth)}
                for r in demands.itertuples(index=False) if r.id not in pending_ids
            ))
            print(f"✓ {accepted}/{len(demands)} accepted (+{accepted - fcfs_accepted} vs FCFS)")
            return accepted
    
    def query_link(self, source, target, graph):
        print(f"\n{'='*60}\nQUERY: Node {source} → {target}\n{'='*60}")
        edge = (min(source, target), max(source, target))
        link = self._link_lookup.get(edge)
        if not link:
            print(f"⚠️  No traffic on link")
            return
        demands = sorted(self._path_edges[edge])
        dist = graph[source][target]['distance'] if graph.has_edge(source, target) else graph[target][source]['distance']
        print(f"\n{'Metric':<20} {'Value'}\n{'-'*60}")
        print(f"{'Distance':<20} {dist:.2f} km\n{'Total Capacity':<20} {link['total']:.2f}")
        print(f"{'Used (Flow)':<20} {link['used']:.2f}\n{'Remaining':<20} {link['total'] - link['used']:.2f}")
        print(f"{'Utilization':<20} {link['util']*100:.1f}%\n{'Demands Count':<20} {len(demands)}")
        print(f"{'Demands Using':<20} {', '.join(map(str, demands))}\n{'-'*60}")
    
    def interactive_query(self, graph):
        print(f"\n{'='*50}\nINTERACTIVE QUERY\n{'='*50}")
        max_node = max(graph.nodes())
        while True:
            try:
                inp = input(f"\nNodes (0-{max_node}) or 'q': ").strip()
                if inp.lower() == 'q': break
                parts = inp.split()
                if len(parts) != 2:
                    print("⚠️  Format: 'source target'")
                    continue
                s, t = int(parts[0]), int(parts[1])
                if s not in graph.nodes() or t not in graph.nodes():
                    print(f"⚠️  Invalid nodes")
                    continue
                self.query_link(s, t, graph)
            except ValueError:
                print("⚠️  Numbers only")
            except KeyboardInterrupt:
                print("\n✓ Exit")
                break

def main():
    opt = NetworkOptimizer("results")
    net, mst, demands = opt.load_and_prepare("data/raw/AttMpls.gml", "data/raw/AttDemands.csv")
    
    fcfs_accepted, _ = opt.run_routing(net, mst, demands, 'fcfs')
    hybrid_accepted = opt.run_routing(net, mst, demands, 'hybrid', fcfs_accepted)
    
    viz = Visualizer(net.graph)
    viz.draw_topology(f"{opt.output_dir}/01_topology.png", show=False)
    viz.draw_mst(mst, f"{opt.output_dir}/02_mst.png", show=False)
    
    high = sum(1 for l in opt._link_lookup.values() if l['util'] >= 0.7)
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
    print(f"FCFS:   {fcfs_accepted}/{len(demands)} ({100*fcfs_accepted/len(demands):.1f}%)")
    print(f"Hybrid: {hybrid_accepted}/{len(demands)} ({100*hybrid_accepted/len(demands):.1f}%)")
    print(f"High util links (≥70%): {high}\n✓ Results in '{opt.output_dir}/'")
    
    opt.interactive_query(net.graph)

if __name__ == "__main__":
    main()