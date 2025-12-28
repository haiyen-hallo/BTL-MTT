import networkx as nx
import matplotlib.pyplot as plt


class Visualizer:
    """Simple network visualization: topology and MST only."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.pos = {n: (d['Longitude'], d['Latitude']) 
                   for n, d in graph.nodes(data=True)}

    def draw_topology(self, save_path: str = None, show: bool = True):
        """Draw network topology."""
        self._draw(
            edges_styles=[{'edgelist': self.graph.edges(), 'edge_color': 'gray', 
                          'width': 1.5, 'style': 'solid'}],
            title="Network Topology",
            save_path=save_path,
            show=show
        )

    def draw_mst(self, mst_graph: nx.Graph, save_path: str = None, show: bool = True):
        """Draw MST with solid lines, non-MST edges with dashed lines."""
        mst_edges = {(min(u, v), max(u, v)) for u, v in mst_graph.edges()}
        non_mst = [(u, v) for u, v in self.graph.edges() 
                   if (min(u, v), max(u, v)) not in mst_edges]
        
        self._draw(
            edges_styles=[
                {'edgelist': non_mst, 'edge_color': 'lightgray', 'width': 1.0, 'style': 'dashed'},
                {'edgelist': mst_graph.edges(), 'edge_color': 'red', 'width': 2.5, 'style': 'solid'}
            ],
            title="Minimum Spanning Tree (MST)",
            save_path=save_path,
            show=show
        )

    def _draw(self, edges_styles: list, title: str, save_path: str = None, show: bool = True):
        """Core drawing method."""
        fig, ax = plt.subplots(figsize=(11, 8))
        
        # Draw edges
        for style in edges_styles:
            nx.draw_networkx_edges(self.graph, self.pos, ax=ax, **style)
        
        # Draw nodes and labels
        nx.draw_networkx_nodes(self.graph, self.pos, node_size=250, 
                              node_color="orange", ax=ax)
        nx.draw_networkx_labels(self.graph, self.pos, font_size=10, 
                               font_color="black", ax=ax)
        
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()