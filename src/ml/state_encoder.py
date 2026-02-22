import torch
import networkx as nx
from torch_geometric.data import Data

COLORS = ['Red', 'Blue', 'Green', 'Yellow', 'Black', 'White', 'Orange', 'Pink', 'false']
CARD_TYPES = ['red_cards', 'blue_cards', 'green_cards', 'yellow_cards',
              'black_cards', 'white_cards', 'orange_cards', 'pink_cards', 'Locomotive']

COLOR_TO_CARD = {
    'Red': 'red_cards', 'Blue': 'blue_cards', 'Green': 'green_cards',
    'Yellow': 'yellow_cards', 'Black': 'black_cards', 'White': 'white_cards',
    'Orange': 'orange_cards', 'Pink': 'pink_cards', 'false': None
}


class StateEncoderV2:
    def __init__(self, board, max_tickets=6, max_paths_per_ticket=3):
        self.board = board
        self.max_tickets = max_tickets
        self.max_paths_per_ticket = max_paths_per_ticket

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
        ticket_features = self._encode_tickets(game_state, player, opponent)
        urgency_features = self._encode_urgency(game_state, player, opponent)

        combined_private = torch.cat([private_state, ticket_features, urgency_features])

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            private_state=combined_private
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

        my_connected = self._get_connected_cities(player)
        opp_connected = self._get_connected_cities(opponent)

        for city in self.board.nodes():
            node_feat = [
                1.0 if city in ticket_sources else 0.0,
                1.0 if city in ticket_targets else 0.0,
                1.0 if city in my_station_cities else 0.0,
                1.0 if city in opp_station_cities else 0.0,
                1.0 if city in my_connected else 0.0,
                1.0 if city in opp_connected else 0.0,
            ]
            features.append(node_feat)

        return torch.tensor(features, dtype=torch.float)

    def _get_connected_cities(self, player):
        connected = set()
        for route in player.claimed_routes:
            connected.add(route[0])
            connected.add(route[1])
        return connected

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

        ticket_cities = set()
        for ticket in player.tickets:
            ticket_cities.add(ticket[0])
            ticket_cities.add(ticket[1])

        for city1, city2, key, data in self.edge_list:
            idx1 = self.city_to_idx[city1]
            idx2 = self.city_to_idx[city2]

            edge_index_list.append([idx1, idx2])
            edge_index_list.append([idx2, idx1])

            route_id = (city1, city2, key)
            feat = self._encode_single_edge(
                data, route_id, player_routes, opponent_routes,
                player, ticket_cities
            )

            edge_features.append(feat)
            edge_features.append(feat)

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return edge_index, edge_attr

    def _encode_single_edge(self, data, route_id, player_routes, opponent_routes, player, ticket_cities):
        carriages = data['carriages'] / 6.0

        color = data['color']
        color_onehot = [1.0 if c == color else 0.0 for c in COLORS]

        is_tunnel = 1.0 if data['tunnel'] == 'true' else 0.0
        ferry_req = data['engine'] / 3.0

        claimed_by_me = 1.0 if route_id in player_routes else 0.0
        claimed_by_opp = 1.0 if route_id in opponent_routes else 0.0
        unclaimed = 1.0 if (claimed_by_me == 0 and claimed_by_opp == 0) else 0.0

        can_afford, affordability_score = self._compute_affordability(data, player)

        city1, city2, _ = route_id
        connects_ticket = 1.0 if (city1 in ticket_cities or city2 in ticket_cities) else 0.0

        points = self._route_points(data['carriages']) / 21.0

        features = (
            [carriages] +
            color_onehot +
            [is_tunnel, ferry_req, claimed_by_me, claimed_by_opp, unclaimed] +
            [can_afford, affordability_score, connects_ticket, points]
        )
        return features

    def _compute_affordability(self, data, player):
        length = data['carriages']
        color = data['color']
        ferry_req = data['engine']
        locomotives = player.hand.get('Locomotive', 0)

        if locomotives < ferry_req:
            return 0.0, 0.0

        if color == 'false':
            best_score = 0.0
            can_afford = False
            for card_type in CARD_TYPES[:-1]:
                count = player.hand.get(card_type, 0)
                total = count + locomotives
                if total >= length:
                    can_afford = True
                    score = min(count, length) / length
                    best_score = max(best_score, score)
            return (1.0 if can_afford else 0.0), best_score
        else:
            color_key = COLOR_TO_CARD.get(color)
            if color_key:
                count = player.hand.get(color_key, 0)
                total = count + locomotives
                if total >= length:
                    score = min(count, length) / length
                    return 1.0, score
            return 0.0, 0.0

    def _route_points(self, length):
        points_table = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
        return points_table.get(length, 0)

    def _encode_private_state(self, game_state, player, opponent):
        hand = [player.hand.get(card, 0) / 12.0 for card in CARD_TYPES]

        total_cards = sum(player.hand.values())
        total_cards_norm = total_cards / 45.0

        trains = player.trains / 45.0
        stations = player.stations / 3.0
        opp_trains = opponent.trains / 45.0
        opp_stations = opponent.stations / 3.0

        my_points = player.points / 100.0
        opp_points = opponent.points / 100.0

        num_tickets = len(player.tickets) / 10.0
        num_claimed = len(player.claimed_routes) / 20.0
        opp_claimed = len(opponent.claimed_routes) / 20.0

        features = (
            hand +
            [total_cards_norm, trains, stations, opp_trains, opp_stations] +
            [my_points, opp_points, num_tickets, num_claimed, opp_claimed]
        )
        return torch.tensor(features, dtype=torch.float)

    def _encode_tickets(self, game_state, player, opponent):
        ticket_features = []

        my_routes = set((r[0], r[1]) for r in player.claimed_routes)
        my_routes.update((r[1], r[0]) for r in player.claimed_routes)

        opp_routes = set((r[0], r[1]) for r in opponent.claimed_routes)
        opp_routes.update((r[1], r[0]) for r in opponent.claimed_routes)

        available_graph = self._build_available_graph(player, my_routes, opp_routes)
        my_network = self._build_player_network(player)

        for i in range(self.max_tickets):
            if i < len(player.tickets):
                ticket = player.tickets[i]
                feat = self._encode_single_ticket(
                    ticket, player, available_graph, my_network, my_routes
                )
            else:
                feat = [0.0] * self._get_ticket_feature_dim()
            ticket_features.extend(feat)

        return torch.tensor(ticket_features, dtype=torch.float)

    def _build_available_graph(self, player, my_routes, opp_routes):
        G = nx.Graph()
        G.add_nodes_from(self.board.nodes())

        for city1, city2, key, data in self.edge_list:
            if (city1, city2) in my_routes or (city2, city1) in my_routes:
                G.add_edge(city1, city2, weight=0, **data)
            elif (city1, city2) not in opp_routes and (city2, city1) not in opp_routes:
                G.add_edge(city1, city2, weight=data['carriages'], **data)

        return G

    def _build_player_network(self, player):
        G = nx.Graph()
        for route in player.claimed_routes:
            G.add_edge(route[0], route[1])
        return G

    def _encode_single_ticket(self, ticket, player, available_graph, my_network, my_routes):
        source, target, points = ticket

        points_norm = points / 21.0

        is_complete = 0.0
        if my_network.has_node(source) and my_network.has_node(target):
            if nx.has_path(my_network, source, target):
                is_complete = 1.0

        if is_complete:
            routes_remaining = 0.0
            cards_needed = 0.0
            cards_have_ratio = 1.0
            best_path_blocked = 0.0
        else:
            try:
                path = nx.shortest_path(available_graph, source, target, weight='weight')
                path_edges = list(zip(path[:-1], path[1:]))

                routes_remaining = 0
                cards_needed = 0

                for c1, c2 in path_edges:
                    if (c1, c2) not in my_routes and (c2, c1) not in my_routes:
                        routes_remaining += 1
                        edge_data = available_graph.get_edge_data(c1, c2)
                        if edge_data and 'carriages' in edge_data:
                            cards_needed += edge_data['carriages']

                routes_remaining = routes_remaining / 8.0
                cards_needed = cards_needed / 30.0

                total_cards = sum(player.hand.values())
                cards_have_ratio = min(total_cards / max(cards_needed * 30, 1), 1.0)
                best_path_blocked = 0.0

            except nx.NetworkXNoPath:
                routes_remaining = 1.0
                cards_needed = 1.0
                cards_have_ratio = 0.0
                best_path_blocked = 1.0

        path_features = self._encode_paths_for_ticket(
            ticket, player, available_graph, my_routes
        )

        base_features = [
            points_norm,
            is_complete,
            routes_remaining,
            cards_needed,
            cards_have_ratio,
            best_path_blocked
        ]

        return base_features + path_features

    def _encode_paths_for_ticket(self, ticket, player, available_graph, my_routes):
        source, target, _ = ticket
        path_features = []

        try:
            paths = list(nx.shortest_simple_paths(available_graph, source, target, weight='weight'))
            paths = paths[:self.max_paths_per_ticket]
        except nx.NetworkXNoPath:
            paths = []

        for i in range(self.max_paths_per_ticket):
            if i < len(paths):
                path = paths[i]
                feat = self._encode_single_path(path, player, available_graph, my_routes)
            else:
                feat = [0.0] * self._get_path_feature_dim()
            path_features.extend(feat)

        return path_features

    def _encode_single_path(self, path, player, available_graph, my_routes):
        path_edges = list(zip(path[:-1], path[1:]))

        total_length = len(path_edges)
        claimed_count = 0
        unclaimed_count = 0
        total_cards_needed = 0
        cards_can_afford = 0
        tunnel_count = 0
        ferry_count = 0

        color_needs = {c: 0 for c in COLORS}

        for c1, c2 in path_edges:
            if (c1, c2) in my_routes or (c2, c1) in my_routes:
                claimed_count += 1
            else:
                unclaimed_count += 1
                edge_data = available_graph.get_edge_data(c1, c2)
                if edge_data:
                    carriages = edge_data.get('carriages', 1)
                    total_cards_needed += carriages

                    color = edge_data.get('color', 'false')
                    color_needs[color] += carriages

                    if edge_data.get('tunnel') == 'true':
                        tunnel_count += 1
                    if edge_data.get('engine', 0) > 0:
                        ferry_count += 1

                    can_afford, _ = self._compute_affordability(edge_data, player)
                    if can_afford:
                        cards_can_afford += 1

        progress = claimed_count / max(total_length, 1)
        completion_ratio = cards_can_afford / max(unclaimed_count, 1)

        total_cards = sum(player.hand.values())
        card_surplus = (total_cards - total_cards_needed) / 20.0

        features = [
            total_length / 8.0,
            progress,
            unclaimed_count / 8.0,
            total_cards_needed / 30.0,
            completion_ratio,
            card_surplus,
            tunnel_count / 4.0,
            ferry_count / 4.0
        ]

        return features

    def _get_path_feature_dim(self):
        return 8

    def _get_ticket_feature_dim(self):
        base = 6
        paths = self.max_paths_per_ticket * self._get_path_feature_dim()
        return base + paths

    def _encode_urgency(self, game_state, player, opponent):
        my_trains = player.trains
        opp_trains = opponent.trains

        end_game_trigger = 1.0 if opp_trains <= 2 else (1.0 - opp_trains / 45.0)
        my_end_game = 1.0 if my_trains <= 2 else 0.0

        total_cards = sum(player.hand.values())
        claimed_routes = len(player.claimed_routes)
        hoard_ratio = total_cards / max(claimed_routes + 1, 1) / 10.0
        hoard_ratio = min(hoard_ratio, 1.0)

        cards_to_trains = total_cards / max(my_trains, 1) / 2.0
        cards_to_trains = min(cards_to_trains, 1.0)

        opp_pace = len(opponent.claimed_routes) / max(len(player.claimed_routes) + 1, 1)
        opp_pace = min(opp_pace / 2.0, 1.0)

        incomplete_tickets = 0
        my_network = self._build_player_network(player)
        for ticket in player.tickets:
            source, target, _ = ticket
            if not (my_network.has_node(source) and my_network.has_node(target) and
                    nx.has_path(my_network, source, target)):
                incomplete_tickets += 1
        incomplete_ratio = incomplete_tickets / max(len(player.tickets), 1)

        urgency_score = end_game_trigger * incomplete_ratio

        features = [
            end_game_trigger,
            my_end_game,
            hoard_ratio,
            cards_to_trains,
            opp_pace,
            incomplete_ratio,
            urgency_score
        ]

        return torch.tensor(features, dtype=torch.float)

    def get_node_feature_dim(self):
        return 6

    def get_edge_feature_dim(self):
        return 1 + len(COLORS) + 5 + 4

    def get_private_state_dim(self):
        base = len(CARD_TYPES) + 10
        tickets = self.max_tickets * self._get_ticket_feature_dim()
        urgency = 7
        return base + tickets + urgency
