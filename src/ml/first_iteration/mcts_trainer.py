import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time
import random

from src.game import Game
from .state_encoder import StateEncoder
from .model import TTRModel
from .mcts import MCTS
from src.players import random_choose, ticket_focused_choose


class MCTSTrainer:
    def __init__(self, hidden_dim=256, lr=1e-4, num_simulations=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.board = Game(2).board
        self.encoder = StateEncoder(self.board)

        self.num_actions = 1000
        self.model = TTRModel(
            node_dim=self.encoder.get_node_feature_dim(),
            edge_dim=self.encoder.get_edge_feature_dim(),
            private_dim=self.encoder.get_private_state_dim(),
            hidden_dim=hidden_dim,
            num_actions=self.num_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.num_simulations = num_simulations

        self.action_to_idx = {}
        self.idx_to_action = {}
        self.next_action_idx = 0

    def get_action_idx(self, action):
        key = (action.type, action.source1, action.source2, action.card1, action.card2)
        if key not in self.action_to_idx:
            if self.next_action_idx >= self.num_actions:
                return None
            self.action_to_idx[key] = self.next_action_idx
            self.idx_to_action[self.next_action_idx] = action
            self.next_action_idx += 1
        return self.action_to_idx[key]

    def play_mcts_game(self, temperature=1.0):
        """
        Play a game using MCTS for both players.
        Returns training data: list of (state_data, mcts_policy, player_idx)
        """
        game = Game(2)
        training_data = []

        mcts = MCTS(
            model=self.model,
            encoder=self.encoder,
            action_to_idx=self.action_to_idx,
            c_puct=1.5,
            num_simulations=self.num_simulations
        )

        turn = 0
        while not game.game_over and turn < 500:
            actions = game.get_legal_actions()
            if not actions:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                continue

            player_idx = game.current_player_idx

            # Run MCTS
            best_action, policy = mcts.run(game, player_idx)

            # Store training data
            state_data = self.encoder.encode_state(game.state, player_idx)

            # Convert policy to tensor (action indices -> probabilities)
            # policy is a list of (action, prob) tuples
            policy_target = torch.zeros(self.num_actions)
            for action, prob in policy:
                idx = self.get_action_idx(action)
                if idx is not None:
                    policy_target[idx] = prob

            training_data.append({
                'state': state_data,
                'policy': policy_target,
                'player_idx': player_idx
            })

            # Select action (with temperature for exploration)
            if temperature > 0:
                # Sample from policy (list of (action, prob) tuples)
                actions_list = [a for a, p in policy]
                probs = [p for a, p in policy]
                if temperature != 1.0:
                    probs = [p ** (1/temperature) for p in probs]
                    total = sum(probs)
                    probs = [p / total for p in probs]
                action = random.choices(actions_list, weights=probs)[0]
            else:
                action = best_action

            game.step(action)
            turn += 1

        # Get game result
        scores = [p.points for p in game.state.list_of_players]
        if scores[0] > scores[1]:
            values = [1.0, -1.0]
        elif scores[1] > scores[0]:
            values = [-1.0, 1.0]
        else:
            values = [0.0, 0.0]

        # Assign values to training data
        for data in training_data:
            data['value'] = values[data['player_idx']]

        return training_data, scores

    def train_on_game(self, training_data):
        """Train on data from one MCTS game"""
        if not training_data:
            return 0.0

        total_loss = 0.0
        self.model.train()

        for data in training_data:
            state = data['state'].to(self.device)
            policy_target = data['policy'].to(self.device)
            value_target = torch.tensor([[data['value']]], device=self.device)

            # Forward pass
            policy_logits, value = self.model(state)

            # Policy loss: cross-entropy with MCTS policy
            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(policy_target * log_probs)

            # Value loss: MSE
            value_loss = F.mse_loss(value, value_target)

            # Combined loss
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(training_data)

    def evaluate_vs_random(self, num_games=50):
        """Evaluate model vs random player (no MCTS, just network)"""
        wins = 0
        self.model.eval()

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

                if player_idx == model_player:
                    action = self._model_choose(game, player_idx)
                else:
                    action = random.choice(actions)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def evaluate_vs_heuristic(self, num_games=50):
        """Evaluate model vs heuristic player (no MCTS)"""
        wins = 0
        self.model.eval()

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
                    action = self._model_choose(game, player_idx)
                else:
                    action = ticket_focused_choose(game.state, player, actions, game.board)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def _model_choose(self, game, player_idx):
        """Choose action using network only (no MCTS)"""
        actions = game.get_legal_actions()
        data = self.encoder.encode_state(game.state, player_idx).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(data)

        # Mask illegal actions
        probs = F.softmax(policy_logits, dim=1).squeeze(0)

        best_prob = -1
        best_action = actions[0]

        for action in actions:
            idx = self.get_action_idx(action)
            if idx is not None and idx < probs.shape[0]:
                if probs[idx].item() > best_prob:
                    best_prob = probs[idx].item()
                    best_action = action

        return best_action

    def train(self, num_games=100, eval_every=20):
        """Main training loop"""
        print(f"Starting MCTS training for {num_games} games...")
        print(f"MCTS simulations per move: {self.num_simulations}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        losses = []
        start_time = time.time()

        for game_num in range(num_games):
            game_start = time.time()

            # Play game with MCTS
            training_data, scores = self.play_mcts_game(temperature=1.0)

            # Train on the game
            loss = self.train_on_game(training_data)
            losses.append(loss)

            game_time = time.time() - game_start

            if (game_num + 1) % 5 == 0:
                avg_loss = sum(losses[-5:]) / min(5, len(losses))
                print(f"Game {game_num + 1}: Loss={avg_loss:.4f}, Time={game_time:.1f}s", flush=True)

            if (game_num + 1) % eval_every == 0:
                vs_random = self.evaluate_vs_random(50)
                vs_heuristic = self.evaluate_vs_heuristic(50)
                elapsed = time.time() - start_time
                print(f"  -> vs Random: {vs_random*100:.1f}%, vs Heuristic: {vs_heuristic*100:.1f}% ({elapsed/60:.1f}min)", flush=True)
                self.save(f"mcts_model_game{game_num + 1}.pt")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")

        # Final evaluation
        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        final_random = self.evaluate_vs_random(100)
        final_heuristic = self.evaluate_vs_heuristic(100)
        print(f"vs Random: {final_random*100:.1f}%")
        print(f"vs Heuristic: {final_heuristic*100:.1f}%")

        self.save("mcts_model_final.pt")

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_to_idx': self.action_to_idx,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        self.next_action_idx = len(self.action_to_idx)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('num_games', type=int, nargs='?', default=100)
    parser.add_argument('--simulations', type=int, default=50)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    print("=" * 60)
    print(f"MCTS TRAINING: {args.num_games} games, {args.simulations} sims/move")
    print("=" * 60)

    trainer = MCTSTrainer(lr=args.lr, num_simulations=args.simulations)

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from: {args.resume}")

    trainer.train(num_games=args.num_games, eval_every=20)
