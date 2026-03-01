import torch
from torch_geometric.data import Data

COLORS = ['Red', 'Blue', 'Green', 'Yellow', 'Black', 'White', 'Orange', 'Pink', 'false']
CARD_TYPES = ['red_cards', 'blue_cards', 'green_cards', 'yellow_cards',
              'black_cards', 'white_cards', 'orange_cards', 'pink_cards', 'Locomotive']

class StateEncoder:
    def __init__(self, board):
        self.board = board
        self.city_to_idx = {city: i for i, city in enumerate(board.nodes())}
        self.idx_to_city = {i: city for city, i in self.city_to_idx.items()}
        self.num_cities = len(self.city_to_idx)

        self.edge_list = []
        for city1, city2, key, data in board.edges(keys=True, data=True):
            self.edge_list.append((city1, city2, key, data))
        self.num_edges = len(self.edge_list)

    def encode_state(self, game_state, player_idx):
        player = game_state.list_of_players[player_idx]
        opponent_idx = 1 - player_idx
        opponent = game_state.list_of_players[opponent_idx]

        node_features = self._encode_nodes(player, opponent)
        edge_index, edge_features = self._encode_edges(game_state, player)
        private_state = self._encode_private_state(game_state, player, opponent)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            private_state=private_state
        )
        return data

    def _encode_nodes(self, player, opponent):
        features = []

        ticket_sources = set()
        ticket_targets = set()
        for ticket in player.tickets:
            ticket_sources.add(ticket[0])
            ticket_targets.add(ticket[1])

        my_station_cities = set(player.stations_built)
        opp_station_cities = set(opponent.stations_built)

        for city in self.board.nodes():
            node_feat = [
                1.0 if city in ticket_sources else 0.0,
                1.0 if city in ticket_targets else 0.0,
                1.0 if city in my_station_cities else 0.0,
                1.0 if city in opp_station_cities else 0.0,
            ]
            features.append(node_feat)

        return torch.tensor(features, dtype=torch.float)

    def _encode_edges(self, game_state, player):
        edge_index_list = []
        edge_features = []

        player_routes = set()
        for route in player.claimed_routes:
            player_routes.add((route[0], route[1], route[2]))

        opponent_routes = set()
        for p_idx, p in enumerate(game_state.list_of_players):
            if p != player:
                for route in p.claimed_routes:
                    opponent_routes.add((route[0], route[1], route[2]))

        for city1, city2, key, data in self.edge_list:
            idx1 = self.city_to_idx[city1]
            idx2 = self.city_to_idx[city2]

            edge_index_list.append([idx1, idx2])
            edge_index_list.append([idx2, idx1])

            route_id = (city1, city2, key)
            feat = self._encode_single_edge(data, route_id, player_routes, opponent_routes, player)

            edge_features.append(feat)
            edge_features.append(feat)

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return edge_index, edge_attr

    def _encode_single_edge(self, data, route_id, player_routes, opponent_routes, player):
        carriages = data['carriages'] / 6.0

        color = data['color']
        color_onehot = [1.0 if c == color else 0.0 for c in COLORS]

        is_tunnel = 1.0 if data['tunnel'] == 'true' else 0.0
        ferry_req = data['engine'] / 3.0

        claimed_by_me = 1.0 if route_id in player_routes else 0.0
        claimed_by_opp = 1.0 if route_id in opponent_routes else 0.0
        unclaimed = 1.0 if (claimed_by_me == 0 and claimed_by_opp == 0) else 0.0

        can_afford = self._can_afford_route(data, player)

        features = [carriages] + color_onehot + [is_tunnel, ferry_req, claimed_by_me, claimed_by_opp, unclaimed, can_afford]
        return features

    def _can_afford_route(self, data, player):
        length = data['carriages']
        color = data['color']
        ferry_req = data['engine']
        locomotives = player.hand.get('Locomotive', 0)

        if color == 'false':
            for card_type in CARD_TYPES[:-1]:
                count = player.hand.get(card_type, 0)
                if count + locomotives >= length and locomotives >= ferry_req:
                    return 1.0
            return 0.0
        else:
            color_key = color + '_cards'
            count = player.hand.get(color_key, 0)
            if count + locomotives >= length and locomotives >= ferry_req:
                return 1.0
            return 0.0

    def _encode_private_state(self, game_state, player, opponent):
        hand = [player.hand.get(card, 0) / 12.0 for card in CARD_TYPES]

        trains = player.trains / 45.0
        stations = player.stations / 3.0
        opp_trains = opponent.trains / 45.0

        my_points = player.points / 100.0
        opp_points = opponent.points / 100.0

        num_tickets = len(player.tickets) / 10.0

        features = hand + [trains, stations, opp_trains, my_points, opp_points, num_tickets]
        return torch.tensor(features, dtype=torch.float)

    def get_node_feature_dim(self):
        return 4

    def get_edge_feature_dim(self):
        return 1 + len(COLORS) + 6

    def get_private_state_dim(self):
        return len(CARD_TYPES) + 6
