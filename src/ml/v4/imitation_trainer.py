import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
import random

from src.game import Game
from src.ml.v2.state_encoder import StateEncoderV2
from src.ml.v4.model import TTRModelV4
from src.ml.state_encoder import StateEncoder
from src.ml.model import TTRModel
from src.players import ticket_focused_choose, random_choose, overall_game_choose, blitz_choose


class FirstIterationOpponent:

    def __init__(self, checkpoint_path, device):
        self.device = device
        self.board = Game(2).board
        self.encoder = StateEncoder(self.board)

        self.model = TTRModel(
            node_dim=self.encoder.get_node_feature_dim(),
            edge_dim=self.encoder.get_edge_feature_dim(),
            private_dim=self.encoder.get_private_state_dim(),
            hidden_dim=256,
            num_actions=1000
        ).to(device)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        self.model.eval()

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


class ImitationTrainerV4:
    def __init__(self, lr=1e-3, label_smoothing=0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.board = Game(2).board
        self.encoder = StateEncoderV2(self.board)

        self.num_actions = 1000
        self.model = TTRModelV4(
            node_dim=self.encoder.get_node_feature_dim(),
            edge_dim=self.encoder.get_edge_feature_dim(),
            private_dim=self.encoder.get_private_state_dim(),
            hidden_dim=704,
            num_gnn_layers=6,
            num_actions=self.num_actions,
            dropout=0.1
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.action_to_idx = {}
        self.label_smoothing = label_smoothing
        self.first_iteration = None

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

    def load_first_iteration(self, path):
        self.first_iteration = FirstIterationOpponent(path, self.device)
        print(f"Loaded first_iteration for evaluation")

    def load_data(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.examples = checkpoint['examples']
        self.action_to_idx = checkpoint['action_to_idx']
        print(f"Loaded {len(self.examples)} examples")
        print(f"Action vocabulary size: {len(self.action_to_idx)}")

        has_weights = any('sample_weight' in ex for ex in self.examples[:10])
        if has_weights:
            avg_weight = sum(ex.get('sample_weight', 1.0) for ex in self.examples) / len(self.examples)
            weighted_count = sum(1 for ex in self.examples if ex.get('sample_weight', 1.0) > 1.0)
            print(f"Sample weights: avg={avg_weight:.2f}, {weighted_count} high-weight samples")
        else:
            print("No sample weights found, using uniform weighting")

        return self.examples

    def train_epoch(self, batch_size=64, use_weights=True):
        self.model.train()
        random.shuffle(self.examples)

        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        for i in range(0, len(self.examples), batch_size):
            batch_examples = self.examples[i:i + batch_size]
            if len(batch_examples) < 4:
                continue

            data_list = [ex['data'] for ex in batch_examples]
            targets = torch.tensor(
                [ex['action_idx'] for ex in batch_examples],
                dtype=torch.long,
                device=self.device
            )

            if use_weights:
                weights = torch.tensor(
                    [ex.get('sample_weight', 1.0) for ex in batch_examples],
                    dtype=torch.float,
                    device=self.device
                )
            else:
                weights = None

            batch = Batch.from_data_list(data_list).to(self.device)

            policy_logits, _ = self.model(batch)

            if self.label_smoothing > 0:
                loss = F.cross_entropy(
                    policy_logits, targets,
                    label_smoothing=self.label_smoothing,
                    weight=None,
                    reduction='none'
                )
            else:
                loss = F.cross_entropy(policy_logits, targets, reduction='none')

            if weights is not None:
                loss = (loss * weights).mean()
            else:
                loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            predictions = policy_logits.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += len(targets)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def get_action_idx(self, action):
        key = (action.type, action.source1, action.source2, action.card1, action.card2)
        return self.action_to_idx.get(key, None)

    def model_choose(self, game_state, player, legal_actions, board, player_idx):
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

    def evaluate_vs_random(self, num_games=100):
        self.model.eval()
        wins = 0

        def model_choose_fn(game_state, player, legal_actions, board):
            player_idx = game_state.list_of_players.index(player)
            return self.model_choose(game_state, player, legal_actions, board, player_idx)

        for i in range(num_games):
            game = Game(2)
            model_player = i % 2

            if model_player == 0:
                scores = game.play_game([model_choose_fn, random_choose], silent=True)
            else:
                scores = game.play_game([random_choose, model_choose_fn], silent=True)

            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def evaluate_vs_heuristic(self, num_games=100):
        self.model.eval()
        wins = 0

        def model_choose_fn(game_state, player, legal_actions, board):
            player_idx = game_state.list_of_players.index(player)
            return self.model_choose(game_state, player, legal_actions, board, player_idx)

        for i in range(num_games):
            game = Game(2)
            model_player = i % 2

            if model_player == 0:
                scores = game.play_game([model_choose_fn, ticket_focused_choose], silent=True)
            else:
                scores = game.play_game([ticket_focused_choose, model_choose_fn], silent=True)

            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def evaluate_vs_overall_game(self, num_games=100):
        self.model.eval()
        wins = 0

        def model_choose_fn(game_state, player, legal_actions, board):
            player_idx = game_state.list_of_players.index(player)
            return self.model_choose(game_state, player, legal_actions, board, player_idx)

        for i in range(num_games):
            game = Game(2)
            model_player = i % 2

            if model_player == 0:
                scores = game.play_game([model_choose_fn, overall_game_choose], silent=True)
            else:
                scores = game.play_game([overall_game_choose, model_choose_fn], silent=True)

            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def evaluate_vs_blitz(self, num_games=100):
        self.model.eval()
        wins = 0

        def model_choose_fn(game_state, player, legal_actions, board):
            player_idx = game_state.list_of_players.index(player)
            return self.model_choose(game_state, player, legal_actions, board, player_idx)

        for i in range(num_games):
            game = Game(2)
            model_player = i % 2

            if model_player == 0:
                scores = game.play_game([model_choose_fn, blitz_choose], silent=True)
            else:
                scores = game.play_game([blitz_choose, model_choose_fn], silent=True)

            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def evaluate_vs_first_iteration(self, num_games=100):
        if not self.first_iteration:
            return None

        self.model.eval()
        wins = 0

        for i in range(num_games):
            game = Game(2)
            model_player = i % 2

            turn = 0
            while not game.game_over and turn < 500:
                actions = game.get_legal_actions()
                if not actions:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                    continue

                player_idx = game.current_player_idx
                player = game.get_current_player()

                if player_idx == model_player:
                    action = self.model_choose(game.state, player, actions, game.board, player_idx)
                else:
                    action = self.first_iteration.choose(game.state, player, actions, game.board, player_idx)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_to_idx': self.action_to_idx,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='v4_training_data.pt', help='Training data file')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (larger for 10M model)')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--output', type=str, default='imitation_model_v4.pt', help='Output model file')
    parser.add_argument('--use-weights', action='store_true', default=True,
                        help='Use sample weights for ticket incentive')
    args = parser.parse_args()

    print("=" * 60)
    print("V4 IMITATION LEARNING TRAINER")
    print("=" * 60)
    print(f"  Model: 10M params (hidden=704, layers=6)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Use sample weights: {args.use_weights}")
    print("=" * 60)

    trainer = ImitationTrainerV4(lr=args.lr, label_smoothing=args.label_smoothing)
    trainer.load_data(args.data)

    print("\n--- Initial Evaluation ---")
    init_random = trainer.evaluate_vs_random(100)
    init_heuristic = trainer.evaluate_vs_heuristic(100)
    init_overall = trainer.evaluate_vs_overall_game(50)
    init_blitz = trainer.evaluate_vs_blitz(50)
    print(f"vs Random: {init_random*100:.1f}%")
    print(f"vs Heuristic: {init_heuristic*100:.1f}%")
    print(f"vs OverallGame: {init_overall*100:.1f}%")
    print(f"vs Blitz: {init_blitz*100:.1f}%")

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    start_time = time.time()
    best_overall = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        loss, accuracy = trainer.train_epoch(
            batch_size=args.batch_size,
            use_weights=args.use_weights
        )

        vs_random = trainer.evaluate_vs_random(50)
        vs_heuristic = trainer.evaluate_vs_heuristic(50)
        vs_overall = trainer.evaluate_vs_overall_game(50)
        vs_blitz = trainer.evaluate_vs_blitz(50)

        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Acc={accuracy*100:.1f}%, "
              f"Random={vs_random*100:.0f}%, Heur={vs_heuristic*100:.0f}%, "
              f"Overall={vs_overall*100:.0f}%, Blitz={vs_blitz*100:.0f}%")

        combined = vs_overall + vs_blitz + vs_heuristic
        if combined > best_overall:
            best_overall = combined
            best_epoch = epoch + 1
            trainer.save(args.output)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best model saved at epoch {best_epoch}")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    trainer.load_model(args.output)
    final_random = trainer.evaluate_vs_random(200)
    final_heuristic = trainer.evaluate_vs_heuristic(200)
    final_overall = trainer.evaluate_vs_overall_game(100)
    final_blitz = trainer.evaluate_vs_blitz(100)

    print(f"vs Random:      {final_random*100:.1f}%")
    print(f"vs Heuristic:   {final_heuristic*100:.1f}%")
    print(f"vs OverallGame: {final_overall*100:.1f}%")
    print(f"vs Blitz:       {final_blitz*100:.1f}%")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Initial -> Final:")
    print(f"  Random:      {init_random*100:.1f}% -> {final_random*100:.1f}%")
    print(f"  Heuristic:   {init_heuristic*100:.1f}% -> {final_heuristic*100:.1f}%")
    print(f"  OverallGame: {init_overall*100:.1f}% -> {final_overall*100:.1f}%")
    print(f"  Blitz:       {init_blitz*100:.1f}% -> {final_blitz*100:.1f}%")
