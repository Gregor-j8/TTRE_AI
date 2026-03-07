import torch
import torch.nn.functional as F
import random
from collections import defaultdict

from src.game import Game
from src.ml.v2.state_encoder import StateEncoderV2
from src.ml.v3.model import TTRModelV3
from src.players import overall_game_choose, blitz_choose
from src.players.ticket_focused import is_ticket_complete


class V3Teacher:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.board = Game(2).board
        self.encoder = StateEncoderV2(self.board)

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        self.model = TTRModelV3(
            node_dim=self.encoder.get_node_feature_dim(),
            edge_dim=self.encoder.get_edge_feature_dim(),
            private_dim=self.encoder.get_private_state_dim(),
            hidden_dim=400,
            num_gnn_layers=5,
            num_actions=1000,
            dropout=0.1
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        self.model.eval()
        print(f"Loaded V3 teacher from {checkpoint_path}")

    def get_action_idx(self, action):
        key = (action.type, action.source1, action.source2, action.card1, action.card2)
        return self.action_to_idx.get(key, None)

    def choose(self, game_state, player, legal_actions, board, player_idx):
        data = self.encoder.encode_state(game_state, player_idx).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(data)

        action_indices = [self.get_action_idx(a) for a in legal_actions]
        valid_pairs = [(i, idx) for i, idx in enumerate(action_indices) if idx is not None]

        if not valid_pairs:
            return random.choice(legal_actions)

        mask = torch.zeros(policy_logits.shape[1], dtype=torch.bool, device=self.device)
        for _, idx in valid_pairs:
            if idx < mask.shape[0]:
                mask[idx] = True

        masked_logits = policy_logits.clone()
        masked_logits[0, ~mask] = float('-inf')

        best_idx = masked_logits[0].argmax().item()

        for action_pos, action_idx in valid_pairs:
            if action_idx == best_idx:
                return legal_actions[action_pos]

        return random.choice(legal_actions)


class HeuristicTeacher:
    def __init__(self, choose_fn, name):
        self.choose_fn = choose_fn
        self.name = name

    def choose(self, game_state, player, legal_actions, board, player_idx):
        return self.choose_fn(game_state, player, legal_actions, board)


class MultiTeacherDataCollector:
    def __init__(self, v3_checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.board = Game(2).board
        self.encoder = StateEncoderV2(self.board)

        self.teachers = {
            'overall_game': HeuristicTeacher(overall_game_choose, 'overall_game'),
            'blitz': HeuristicTeacher(blitz_choose, 'blitz'),
        }

        if v3_checkpoint_path:
            self.teachers['v3'] = V3Teacher(v3_checkpoint_path, self.device)

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

    def count_completed_tickets(self, player):
        return sum(1 for t in player.tickets if is_ticket_complete(player, t))

    def collect_single_game(self, teacher, teacher_name):
        game = Game(2)
        turn = 0
        game_data = []

        while not game.game_over and turn < 500:
            actions = game.get_legal_actions()
            if not actions:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                continue

            player_idx = game.current_player_idx
            player = game.get_current_player()

            data = self.encoder.encode_state(game.state, player_idx)

            chosen_action = teacher.choose(
                game.state, player, actions, game.board, player_idx
            )

            action_idx = self.get_action_idx(chosen_action)

            if action_idx is not None:
                game_data.append({
                    'data': data,
                    'action_idx': action_idx,
                    'teacher': teacher_name
                })

            game.step(chosen_action)
            turn += 1

        tickets_completed_p0 = self.count_completed_tickets(game.state.list_of_players[0])
        tickets_completed_p1 = self.count_completed_tickets(game.state.list_of_players[1])
        total_tickets_completed = tickets_completed_p0 + tickets_completed_p1

        return game_data, total_tickets_completed

    def collect_from_teacher(self, teacher_name, num_games, ticket_weight=2.0):
        teacher = self.teachers[teacher_name]
        all_data = []
        games_with_tickets = 0

        for game_num in range(num_games):
            game_data, tickets_completed = self.collect_single_game(teacher, teacher_name)

            if tickets_completed > 0:
                games_with_tickets += 1
                sample_weight = ticket_weight
            else:
                sample_weight = 1.0

            for sample in game_data:
                sample['sample_weight'] = sample_weight

            all_data.extend(game_data)

            if (game_num + 1) % 100 == 0:
                print(f"  {teacher_name}: {game_num + 1}/{num_games} games, "
                      f"{len(all_data)} samples, {games_with_tickets} w/ tickets")

        return all_data, games_with_tickets

    def collect_all(self, overall_game_games=2000, blitz_games=1000, v3_games=500, ticket_weight=2.0):
        print("=" * 60)
        print(f"V4 DATA COLLECTION")
        print(f"  OverallGame: {overall_game_games} games")
        print(f"  Blitz: {blitz_games} games")
        print(f"  V3: {v3_games} games")
        print(f"  Ticket completion weight: {ticket_weight}x")
        print("=" * 60)

        all_data = []
        stats = {}

        print("\nCollecting from OverallGame...")
        data, tickets = self.collect_from_teacher('overall_game', overall_game_games, ticket_weight)
        all_data.extend(data)
        stats['overall_game'] = {'samples': len(data), 'games_with_tickets': tickets}

        print("\nCollecting from Blitz...")
        data, tickets = self.collect_from_teacher('blitz', blitz_games, ticket_weight)
        all_data.extend(data)
        stats['blitz'] = {'samples': len(data), 'games_with_tickets': tickets}

        if 'v3' in self.teachers:
            print("\nCollecting from V3...")
            data, tickets = self.collect_from_teacher('v3', v3_games, ticket_weight)
            all_data.extend(data)
            stats['v3'] = {'samples': len(data), 'games_with_tickets': tickets}

        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print(f"Total samples: {len(all_data)}")
        print(f"Unique actions: {len(self.action_to_idx)}")
        for teacher, s in stats.items():
            pct = s['games_with_tickets'] / (overall_game_games if teacher == 'overall_game'
                                             else blitz_games if teacher == 'blitz' else v3_games) * 100
            print(f"  {teacher}: {s['samples']} samples, "
                  f"{s['games_with_tickets']} games w/ tickets ({pct:.1f}%)")
        print("=" * 60)

        return all_data, stats

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
    parser.add_argument('--overall-games', type=int, default=2000)
    parser.add_argument('--blitz-games', type=int, default=1000)
    parser.add_argument('--v3-games', type=int, default=500)
    parser.add_argument('--v3-model', type=str,
                        default='model_data/v3_good/model_v3_Final.pt')
    parser.add_argument('--output', type=str, default='v4_training_data.pt')
    parser.add_argument('--ticket-weight', type=float, default=2.0,
                        help='Sample weight multiplier for games with ticket completions')
    args = parser.parse_args()

    collector = MultiTeacherDataCollector(args.v3_model)

    start = time.time()
    all_data, stats = collector.collect_all(
        overall_game_games=args.overall_games,
        blitz_games=args.blitz_games,
        v3_games=args.v3_games,
        ticket_weight=args.ticket_weight
    )
    elapsed = time.time() - start

    print(f"\nCollection completed in {elapsed:.1f}s")
    print(f"Avg samples per game: {len(all_data) / (args.overall_games + args.blitz_games + args.v3_games):.1f}")

    collector.save_data(all_data, args.output)
