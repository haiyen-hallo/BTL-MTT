import os
import pandas as pd
from src.data_loader import DataLoader
from src.network_graph import NetworkGraph
from src.mst_solver import MSTSolver
from src.path_finder_enhanced import PathFinderEnhanced
from src.capacity_calculator_enhanced import CapacityCalculatorFCFS, CapacityCalculatorHybrid
from src.visualizer import Visualizer
from src.result_exporter import ResultExporter

K_FCFS, CAP_FCFS = 5, 1.0
K_HYBRID, CAP_HYBRID, ROUNDS = 8, 1.49, 2

def query_path_from_data(source, target, df_links, accepted_paths_df, calc):
    """Query path using pre-computed data"""
    print(f"\n{'='*60}")
    print(f"PATH QUERY: Node {source} → Node {target}")
    print(f"{'='*60}")
    
    # Normalize edge
    edge_key = (min(source, target), max(source, target))
    
    # Find link in df_links (already computed in Câu 3)
    link_info = df_links[
        ((df_links['source'] == source) & (df_links['target'] == target)) |
        ((df_links['source'] == target) & (df_links['target'] == source))
    ]
    
    if len(link_info) == 0:
        print(f"⚠️  No traffic on link {source} ↔ {target}")
        return
    
    link = link_info.iloc[0]
    
    # Find demands using this link (from accepted_paths)
    demands_using_link = []
    for _, row in accepted_paths_df.iterrows():
        path = row['path']
        demand_id = row['demand_id']
        
        # Check if this link is in the path
        for i in range(len(path) - 1):
            if (path[i] == source and path[i+1] == target) or \
               (path[i] == target and path[i+1] == source):
                # Get bandwidth from calc (already stored)
                bw = calc.path_finder.routes[demand_id]['bandwidth']
                demands_using_link.append(demand_id)
                break
    
    # Get distance from graph (already computed)
    if calc.graph.has_edge(source, target):
        distance = calc.graph[source][target]['distance']
    else:
        distance = calc.graph[target][source]['distance']
    
    demand_ids = ', '.join(map(str, sorted(demands_using_link)))
    
    # Display results using pre-computed data
    print(f"\n{'Metric':<20} {'Value'}")
    print(f"{'-'*60}")
    print(f"{'Distance':<20} {distance:.2f} km")
    print(f"{'Total Capacity':<20} {link['total_capacity']:.2f}")
    print(f"{'Used (Flow)':<20} {link['used_capacity']:.2f}")
    print(f"{'Remaining':<20} {link['total_capacity'] - link['used_capacity']:.2f}")
    print(f"{'Utilization':<20} {link['utilization']*100:.1f}%")
    print(f"{'Demands Count':<20} {len(demands_using_link)}")
    print(f"{'Demands Using':<20} {demand_ids}")
    print(f"{'-'*60}")


def main():
    os.makedirs("results", exist_ok=True)
    exporter = ResultExporter("results")

    # Load data
    loader = DataLoader("data/raw/AttMpls.gml", "data/raw/AttDemands.csv")
    graph, demands = loader.load_all()

    # Compute distances, capacity & MST
    net = NetworkGraph(graph)
    net.compute_all()
    
    mst_solver = MSTSolver(net.graph)
    mst = mst_solver.solve()
    mst_solver.export()

    # FCFS ROUTING (K=5, Capacity=1.0)
    print("\n" + "="*50)
    print("FCFS ROUTING (Câu 1)")
    print("="*50)
    
    pf_fcfs = PathFinderEnhanced(net.graph, mst, k_paths=K_FCFS)
    pf_fcfs.find_all_paths(demands)
    
    calc_fcfs = CapacityCalculatorFCFS(net.graph, pf_fcfs, capacity_multiplier=CAP_FCFS)
    accepted_fcfs, results_fcfs = calc_fcfs.process_demands(demands)
    
    # Export Câu 1
    exporter.save_cau1(accepted_fcfs, len(demands) - accepted_fcfs, len(demands), 
                       {"k": K_FCFS, "cap": CAP_FCFS})
    print(f"✓ Câu 1: Accepted {accepted_fcfs}/{len(demands)}")
    
    # Export Câu 2
    accepted_paths = [
        {"demand_id": d_id, "path": path} 
        for d_id, ok, path in results_fcfs if ok
    ]
    accepted_paths_df = pd.DataFrame(accepted_paths)
    exporter.save_cau2(accepted_paths_df)
    print(f"✓ Câu 2: Exported {len(accepted_paths)} accepted paths")
    
    # Export Câu 3
    df_links = pd.DataFrame([
        {
            "source": e[0], "target": e[1], 
            "used_capacity": used,
            "total_capacity": calc_fcfs.cap[e],
            "utilization": used / calc_fcfs.cap[e]
        }
        for e, used in calc_fcfs.used.items() if used > 0
    ]).sort_values('utilization', ascending=False)
    
    high_util = len(df_links[df_links['utilization'] >= 0.7])
    exporter.save_cau3(high_util, df_links, {
        "total_edges": len(calc_fcfs.used),
        "used_edges": len(df_links),
        "avg_utilization": df_links['utilization'].mean(),
        "max_utilization": df_links['utilization'].max()
    })
    print(f"✓ Câu 3: Exported {len(df_links)} links, {high_util} high utilization")

    # HYBRID ROUTING (K=8, Capacity=1.49, Rounds=2)
    print("\n" + "="*50)
    print("HYBRID ROUTING (Câu 4)")
    print("="*50)
    
    pf_hybrid = PathFinderEnhanced(net.graph, mst, k_paths=K_HYBRID)
    pf_hybrid.find_all_paths(demands)
    
    calc_hybrid = CapacityCalculatorHybrid(net.graph, pf_hybrid, 
                                          capacity_multiplier=CAP_HYBRID)
    accepted_hybrid, pending = calc_hybrid.process_demands(demands)
    
    # Export Câu 4
    exporter.save_cau4(
        {"accepted": accepted_hybrid, "pending": len(pending)},
        accepted_fcfs, len(demands), 
        {"k": K_HYBRID, "cap": CAP_HYBRID}
    )
    print(f"✓ Câu 4: Accepted {accepted_hybrid}/{len(demands)}")
    
    # Export best demands
    pending_ids = {r['id'] for r in pending}
    exporter.save_best_demands(pd.DataFrame([
        {"demand_id": int(r.id), "bandwidth": float(r.bandwidth)}
        for r in demands.itertuples(index=False) if r.id not in pending_ids
    ]))

    # VISUALIZATION
    viz = Visualizer(net.graph)
    viz.draw_topology("results/01_topology.png", show=False)
    viz.draw_mst(mst, "results/02_mst.png", show=False)

    # SUMMARY
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"\nFCFS (K={K_FCFS}, Cap={CAP_FCFS}):")
    print(f"  ✓ Accepted: {accepted_fcfs}/{len(demands)} ({100*accepted_fcfs/len(demands):.1f}%)")
    print(f"\nHybrid (K={K_HYBRID}, Cap={CAP_HYBRID}, Rounds={ROUNDS}):")
    print(f"  ✓ Accepted: {accepted_hybrid}/{len(demands)} ({100*accepted_hybrid/len(demands):.1f}%)")
    print(f"  ✓ Improvement: +{accepted_hybrid - accepted_fcfs} demands")
    print(f"\nHigh utilization links (≥70%): {high_util}")
    print("\n✓ All results exported to 'results/' folder")
    
    # PATH QUERY FEATURE - SỬ DỤNG DỮ LIỆU ĐÃ CÓ
    print("\n" + "="*50)
    print("PATH QUERY")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nEnter two nodes (format: '5 10') or 'q' to quit: ").strip()
            
            if user_input.lower() == 'q':
                break
            
            parts = user_input.split()
            if len(parts) != 2:
                print("⚠️  Invalid format. Use: 'source target' (e.g., '5 10')")
                continue
            
            source, target = int(parts[0]), int(parts[1])
            
            if source not in graph.nodes() or target not in graph.nodes():
                print(f"⚠️  Invalid nodes. Valid range: 0-{max(graph.nodes())}")
                continue
            
            # SỬ DỤNG DỮ LIỆU ĐÃ TÍNH: df_links, accepted_paths_df, calc_fcfs
            query_path_from_data(source, target, df_links, accepted_paths_df, calc_fcfs)
            
        except ValueError:
            print("⚠️  Invalid input. Use numbers only.")
        except KeyboardInterrupt:
            print("\n\n✓ Exiting...")
            break

if __name__ == "__main__":
    main()