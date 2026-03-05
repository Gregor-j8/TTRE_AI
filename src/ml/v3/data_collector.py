import torch
import torch.nn.functional as F
import random
from src.game import Game
from src.ml.v2.state_encoder import StateEncoderV2
from src.ml.state_encoder import StateEncoder
from src.ml.model import TTRModel


class FirstIterationTeacher:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.board = Game(2).board
        self.encoder = StateEncoder(self.board)

        self.model = TTRModel(
            node_dim=self.encoder.get_node_feature_dim(),
            edge_dim=self.encoder.get_edge_feature_dim(),
            private_dim=self.encoder.get_private_state_dim(),
            hidden_dim=256,
            num_actions=1000
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        self.model.eval()
        print(f"Loaded first_iteration teacher from {checkpoint_path}")

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
        probs = F.softmax(masked_logits, dim=1)

        valid_indices = [idx for _, idx in valid_pairs]
        valid_action_positions = [i for i, _ in valid_pairs]

        choice_in_valid = probs[0, valid_indices].argmax().item()
        action_pos = valid_action_positions[choice_in_valid]
        return legal_actions[action_pos]


class FirstIterationDataCollector:
    def __init__(self, teacher_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.board = Game(2).board
        self.encoder = StateEncoderV2(self.board)
        self.teacher = FirstIterationTeacher(teacher_path)

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

    def collect_games(self, num_games=2000):
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

                chosen_action = self.teacher.choose(
                    game.state, player, actions, game.board, player_idx
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
    parser.add_argument('--games', type=int, default=2000, help='Number of games to collect')
    parser.add_argument('--output', type=str, default='first_iter_data_v3.pt', help='Output file')
    parser.add_argument('--teacher', type=str,
                        default='model_data/v1_models/first_iteration/model_final.pt',
                        help='Path to first_iteration model')
    args = parser.parse_args()

    print("=" * 60)
    print(f"COLLECTING FIRST_ITERATION DATA: {args.games} games")
    print("=" * 60)

    collector = FirstIterationDataCollector(args.teacher)

    start = time.time()
    all_data = collector.collect_games(args.games)
    elapsed = time.time() - start

    print(f"\nCollection completed in {elapsed:.1f}s")
    print(f"Total examples: {len(all_data)}")
    print(f"Unique actions seen: {len(collector.action_to_idx)}")
    print(f"Examples per game: {len(all_data) / args.games:.1f}")

    collector.save_data(all_data, args.output)
