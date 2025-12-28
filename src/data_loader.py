import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Tuple

class DataLoader:
    def __init__(self, gml_path: str, demands_path: str):
        self.gml_path = Path(gml_path)
        self.demands_path = Path(demands_path)
        if not self.gml_path.exists() or not self.demands_path.exists():
            raise FileNotFoundError("File not found")
    
    def load_all(self) -> Tuple[nx.Graph, pd.DataFrame]:
        """Load and validate graph and demands"""
        # Load graph
        graph = nx.read_gml(self.gml_path, label='id')
        if graph.number_of_nodes() == 0 or not nx.is_connected(graph):
            raise ValueError("Invalid graph")
        
        # Load demands WITHOUT header - đọc từ dòng 0
        df = pd.read_csv(self.demands_path, header=None, names=['id', 'source', 'target', 'bandwidth'])
        
        # Validate and return
        df = df.astype({'id': int, 'source': int, 'target': int, 'bandwidth': float})
        
        if (df['bandwidth'] <= 0).any():
            raise ValueError("Bandwidth must be positive")
        
        print(f"✓ Loaded {len(df)} demands (ID: {df['id'].min()}-{df['id'].max()})")
        
        return graph, df