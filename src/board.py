import networkx as nx
import csv
from pathlib import Path

def load_board():
    data_path = Path(__file__).parent.parent / 'data' / 'routes.csv'
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        G = nx.MultiGraph()
        for row in reader:
            city1, city2, distance, color, tunnel, engine = row
            G.add_edge(city1, city2, carriages=int(distance), color=str(color), tunnel=tunnel, engine=int(engine))
        return G
    
if __name__ == "__main__":
    board = load_board()