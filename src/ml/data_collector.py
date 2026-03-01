import torch
import random
from src.game import Game
from src.ml.state_encoder import StateEncoder
from src.players import ticket_focused_choose


class HeuristicDataCollector:
    def __init__(self):
        self.board = Game(2).board
        self.encoder = StateEncoder(self.board)

        self.action_to_idx = {}
        self.idx_to_action = {}
        self.next_action_idx = 0
        self.num_actions = 1000

    def get_action_idx(self, action):
        key = (action.type, action.source1, action.source2, action.card1, action.card2)
        if key not in self.action_to_idx:
            if self.next_action_idx >= self.num_actions:
                return None
            self.action_to_idx[key] = self.next_action_idx
            self.idx_to_action[self.next_action_idx] = action
            self.next_action_idx += 1
        return self.action_to_idx[key]

    def collect_games(self, num_games=500):
        all_data = []

        for game_num in range(num_games):
            game = Game(2)
            turn = 0

            while not game.game_over and turn < 500:
                actions = game.get_legal_actions()
                if not actions:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                    continue

                player_idx = game.current_player_idx
                player = game.get_current_player()

                data = self.encoder.encode_state(game.state, player_idx)

                chosen_action = ticket_focused_choose(
                    game.state, player, actions, game.board
                )

                action_idx = self.get_action_idx(chosen_action)

                if action_idx is not None:
                    all_data.append({
                        'data': data,
                        'action_idx': action_idx
                    })

                game.step(chosen_action)
                turn += 1

            if (game_num + 1) % 50 == 0:
                print(f"Collected {game_num + 1}/{num_games} games, {len(all_data)} examples")

        return all_data

    def save_data(self, all_data, path):
        torch.save({
            'examples': all_data,
            'action_to_idx': self.action_to_idx,
            'num_examples': len(all_data)
        }, path)
        print(f"Saved {len(all_data)} examples to {path}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=500, help='Number of games to collect')
    parser.add_argument('--output', type=str, default='heuristic_data.pt', help='Output file')
    args = parser.parse_args()

    print("=" * 60)
    print(f"COLLECTING HEURISTIC DATA: {args.games} games")
    print("=" * 60)

    collector = HeuristicDataCollector()

    start = time.time()
    all_data = collector.collect_games(args.games)
    elapsed = time.time() - start

    print(f"\nCollection completed in {elapsed:.1f}s")
    print(f"Total examples: {len(all_data)}")
    print(f"Unique actions seen: {len(collector.action_to_idx)}")
    print(f"Examples per game: {len(all_data) / args.games:.1f}")

    collector.save_data(all_data, args.output)