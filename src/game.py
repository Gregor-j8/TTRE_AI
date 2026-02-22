import random
import networkx as nx
from src.board import load_board
from src.helpers.game_state import GameState
from src.helpers.action import legal_actions, execute_action

class Game:
    def __init__(self, num_players):
        self.board = load_board()
        self.state = GameState(num_players, self.board)
        self.num_players = num_players
        self.current_player_idx = 0
        self.final_round = False
        self.final_round_starter = None
        self.game_over = False

        self._deal_initial_cards()

    def _deal_initial_cards(self):
        for player in self.state.list_of_players:
            for _ in range(4):
                available = [c for c, count in self.state.draw_pile.items() if count > 0]
                if available:
                    card = random.choice(available)
                    self.state.draw_pile[card] -= 1
                    player.hand[card] += 1

            tickets_to_deal = min(3, len(self.state.ticket_deck))
            player.tickets = self.state.ticket_deck[:tickets_to_deal]
            self.state.ticket_deck = self.state.ticket_deck[tickets_to_deal:]

    def get_current_player(self):
        return self.state.list_of_players[self.current_player_idx]

    def get_legal_actions(self):
        player = self.get_current_player()
        return legal_actions(self.state, player)

    def step(self, action):
        player = self.get_current_player()
        execute_action(action, self.state, player)

        if player.pending_tickets:
            return

        if player.trains <= 2 and not self.final_round:
            self.final_round = True
            self.final_round_starter = self.current_player_idx

        self.current_player_idx = (self.current_player_idx + 1) % self.num_players

        if self.final_round and self.current_player_idx == self.final_round_starter:
            self.game_over = True
            self._final_scoring()

    def _final_scoring(self):
        longest_routes = []

        for idx, player in enumerate(self.state.list_of_players):
            player_graph = nx.Graph()
            for route in player.claimed_routes:
                route_data = self.board.edges[route]
                length = route_data['carriages']
                if player_graph.has_edge(route[0], route[1]):
                    existing = player_graph[route[0]][route[1]].get('weight', 0)
                    player_graph[route[0]][route[1]]['weight'] = existing + length
                else:
                    player_graph.add_edge(route[0], route[1], weight=length)

            longest = self._calculate_longest_path(player_graph)
            longest_routes.append(longest)

            connection_graph = nx.Graph()
            for route in player.claimed_routes:
                connection_graph.add_edge(route[0], route[1])

            for station_city in player.stations_built:
                for other_idx, other_player in enumerate(self.state.list_of_players):
                    if other_idx == idx:
                        continue
                    for route in other_player.claimed_routes:
                        if route[0] == station_city or route[1] == station_city:
                            connection_graph.add_edge(route[0], route[1])
                            break

            for ticket in player.tickets:
                source, target, points = ticket
                if connection_graph.has_node(source) and connection_graph.has_node(target):
                    if nx.has_path(connection_graph, source, target):
                        player.points += points
                    else:
                        player.points -= points
                else:
                    player.points -= points

            player.points += player.stations * 4

        if longest_routes:
            max_length = max(longest_routes)
            if max_length > 0:
                for idx, length in enumerate(longest_routes):
                    if length == max_length:
                        self.state.list_of_players[idx].points += 10

    def _calculate_longest_path(self, graph):
        if not graph.nodes():
            return 0

        longest = 0

        def dfs(node, visited_edges, current_length):
            nonlocal longest
            longest = max(longest, current_length)

            for neighbor in graph.neighbors(node):
                edge = tuple(sorted([node, neighbor]))
                if edge not in visited_edges:
                    weight = graph[node][neighbor].get('weight', 1)
                    visited_edges.add(edge)
                    dfs(neighbor, visited_edges, current_length + weight)
                    visited_edges.remove(edge)

        for start_node in graph.nodes():
            dfs(start_node, set(), 0)

        return longest

    def play_game(self, choose_fns, silent=False):
        turn = 0
        no_action_count = 0
        while not self.game_over:
            actions = self.get_legal_actions()
            if not actions:
                no_action_count += 1
                if no_action_count > self.num_players * 2:
                    if not silent:
                        print("No legal actions available for any player, ending game")
                    break
                self.current_player_idx = (self.current_player_idx + 1) % self.num_players
                continue

            no_action_count = 0
            player = self.get_current_player()
            choose_fn = choose_fns[self.current_player_idx]
            action = choose_fn(self.state, player, actions, self.board)

            if not silent:
                print(f"Turn {turn}: Player {self.current_player_idx} - {action.type}")

            self.step(action)
            turn += 1
            if turn > 1000:
                if not silent:
                    print("Game exceeded 1000 turns, stopping")
                break

        if not silent:
            print(f"Game ended after {turn} turns")
            for i, player in enumerate(self.state.list_of_players):
                print(f"Player {i}: {player.points} points, {player.trains} trains left")

        return [p.points for p in self.state.list_of_players]

    def play_random_game(self):
        from src.players import random_choose
        return self.play_game([random_choose, random_choose])

if __name__ == "__main__":
    game = Game(2)
    game.play_random_game()