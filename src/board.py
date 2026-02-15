import networkx as nx
import csv

def load_board():
    with open('data/routes.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        G = nx.MultiGraph()
        for row in reader:
            city1, city2, distance, color, tunnel, engine = row
            G.add_edge(city1, city2, carriages=int(distance), color=str(color), tunnel=tunnel, engine=int(engine))
        print(f"Loaded {G.number_of_edges()} edges and {G.number_of_nodes()} nodes.")
        return G
    
if __name__ == "__main__":
    board = load_board()