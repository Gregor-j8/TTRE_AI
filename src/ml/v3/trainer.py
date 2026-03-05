import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
import random
import time

from src.game import Game
from src.ml.v2.state_encoder import StateEncoderV2
from src.ml.v3.model import TTRModelV3
from src.ml.state_encoder import StateEncoder
from src.ml.model import TTRModel
from src.players import random_choose, ticket_focused_choose


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
        print(f"Loaded first_iteration opponent from {checkpoint_path}")

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


class TrainerV3:
    def __init__(
        self,
        lr=1e-4,
        batch_size=128,
        gamma=0.99,
        entropy_coef=0.05,
        diverse_opponents=False
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.board = Game(2).board
        self.encoder = StateEncoderV2(self.board)

        self.num_actions = 1000

        self.model = TTRModelV3(
            node_dim=self.encoder.get_node_feature_dim(),
            edge_dim=self.encoder.get_edge_feature_dim(),
            private_dim=self.encoder.get_private_state_dim(),
            hidden_dim=400,
            num_gnn_layers=5,
            num_actions=self.num_actions,
            dropout=0.1
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.diverse_opponents = diverse_opponents

        self.action_to_idx = {}
        self.idx_to_action = {}
        self.next_action_idx = 0

        self.opponent = None

    def load_opponent(self, checkpoint_path):
        self.opponent = FirstIterationOpponent(checkpoint_path, self.device)

    def heuristic_choose(self, game_state, player, legal_actions, board, player_idx):
        return ticket_focused_choose(game_state, player, legal_actions, board)

    def get_action_idx(self, action):
        key = (action.type, action.source1, action.source2, action.card1, action.card2)
        if key not in self.action_to_idx:
            if self.next_action_idx >= self.num_actions:
                return None
            self.action_to_idx[key] = self.next_action_idx
            self.idx_to_action[self.next_action_idx] = action
            self.next_action_idx += 1
        return self.action_to_idx[key]

    def model_choose(self, game_state, player, legal_actions, board, player_idx, explore=True):
        data = self.encoder.encode_state(game_state, player_idx).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(data)

        action_indices = [self.get_action_idx(a) for a in legal_actions]
        valid_pairs = [(i, idx) for i, idx in enumerate(action_indices) if idx is not None]

        if not valid_pairs:
            return random.choice(legal_actions), 0, 1.0 / len(legal_actions)

        mask = torch.zeros(policy_logits.shape[1], dtype=torch.bool, device=self.device)
        for _, idx in valid_pairs:
            if idx < mask.shape[0]:
                mask[idx] = True

        masked_logits = policy_logits.clone()
        masked_logits[0, ~mask] = float('-inf')
        probs = F.softmax(masked_logits, dim=1)

        valid_indices = [idx for _, idx in valid_pairs]
        valid_action_positions = [i for i, _ in valid_pairs]

        if explore:
            valid_probs = probs[0, valid_indices].cpu().numpy()
            valid_probs = valid_probs / valid_probs.sum()
            choice_in_valid = random.choices(range(len(valid_pairs)), weights=valid_probs)[0]
        else:
            choice_in_valid = probs[0, valid_indices].argmax().item()

        action_pos = valid_action_positions[choice_in_valid]
        action_idx = valid_indices[choice_in_valid]
        return legal_actions[action_pos], action_idx, probs[0, action_idx].item()

    def _assign_game_type(self, _game_idx=None):
        if self.diverse_opponents and self.opponent:
            r = random.random()
            if r < 0.33:
                return 'selfplay'
            elif r < 0.67:
                return 'first_iter'
            else:
                return 'heuristic'
        elif self.opponent:
            return 'first_iter' if random.random() < 0.5 else 'selfplay'
        else:
            return 'selfplay'

    def collect_episodes(self, num_episodes):
        all_trajectories = []
        all_rewards = []
        completed = 0

        games = [Game(2) for _ in range(self.batch_size)]
        trajectories = [[[], []] for _ in range(self.batch_size)]
        turns = [0] * self.batch_size

        game_type = ['selfplay'] * self.batch_size
        v3_player = [0] * self.batch_size

        for i in range(self.batch_size):
            game_type[i] = self._assign_game_type(i)
            v3_player[i] = i % 2

        self.model.eval()

        while completed < num_episodes:
            for i, game in enumerate(games):
                if game_type[i] == 'selfplay' or game.game_over or turns[i] >= 500:
                    continue

                player_idx = game.current_player_idx
                if player_idx != v3_player[i]:
                    actions = game.get_legal_actions()
                    if not actions:
                        game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                        continue
                    player = game.get_current_player()

                    if game_type[i] == 'first_iter':
                        action = self.opponent.choose(game.state, player, actions, game.board, player_idx)
                    else:
                        action = self.heuristic_choose(game.state, player, actions, game.board, player_idx)

                    game.step(action)
                    turns[i] += 1

            active_indices = []
            data_list = []
            legal_actions_list = []
            player_indices = []

            for i, game in enumerate(games):
                if game.game_over or turns[i] >= 500:
                    continue

                actions = game.get_legal_actions()
                if not actions:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                    continue

                player_idx = game.current_player_idx

                if game_type[i] != 'selfplay' and player_idx != v3_player[i]:
                    continue

                data = self.encoder.encode_state(game.state, player_idx)

                active_indices.append(i)
                data_list.append(data)
                legal_actions_list.append(actions)
                player_indices.append(player_idx)

            if not data_list:
                for i, game in enumerate(games):
                    if game.game_over or turns[i] >= 500:
                        scores = [p.points for p in game.state.list_of_players]
                        if scores[0] > scores[1]:
                            rewards = [1.0, -1.0]
                        elif scores[1] > scores[0]:
                            rewards = [-1.0, 1.0]
                        else:
                            rewards = [0.0, 0.0]

                        if game_type[i] != 'selfplay':
                            v3_traj = [[],[]]
                            v3_traj[v3_player[i]] = trajectories[i][v3_player[i]]
                            all_trajectories.append(v3_traj)
                            all_rewards.append(rewards)
                        else:
                            all_trajectories.append(trajectories[i])
                            all_rewards.append(rewards)

                        completed += 1

                        if completed >= num_episodes:
                            break

                        games[i] = Game(2)
                        trajectories[i] = [[], []]
                        turns[i] = 0
                        game_type[i] = self._assign_game_type(i)
                        v3_player[i] = completed % 2
                continue

            batch = Batch.from_data_list(data_list).to(self.device)

            with torch.no_grad():
                policy_logits, values = self.model(batch)

            for batch_idx, game_idx in enumerate(active_indices):
                game = games[game_idx]
                actions = legal_actions_list[batch_idx]
                player_idx = player_indices[batch_idx]
                data = data_list[batch_idx]

                logits = policy_logits[batch_idx:batch_idx+1]

                action_indices = [self.get_action_idx(a) for a in actions]
                valid_pairs = [(j, idx) for j, idx in enumerate(action_indices) if idx is not None]

                if not valid_pairs:
                    action = random.choice(actions)
                    action_idx = 0
                    prob = 1.0 / len(actions)
                else:
                    mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=self.device)
                    for _, idx in valid_pairs:
                        if idx < mask.shape[0]:
                            mask[idx] = True

                    masked_logits = logits.clone()
                    masked_logits[0, ~mask] = float('-inf')
                    probs = F.softmax(masked_logits, dim=1)

                    valid_indices = [idx for _, idx in valid_pairs]
                    valid_action_positions = [j for j, _ in valid_pairs]

                    valid_probs = probs[0, valid_indices].cpu().numpy()
                    valid_probs = valid_probs / valid_probs.sum()
                    choice_in_valid = random.choices(range(len(valid_pairs)), weights=valid_probs)[0]

                    action_pos = valid_action_positions[choice_in_valid]
                    action_idx = valid_indices[choice_in_valid]
                    action = actions[action_pos]
                    prob = probs[0, action_idx].item()

                trajectories[game_idx][player_idx].append({
                    'data': data,
                    'action_idx': action_idx,
                    'log_prob': torch.log(torch.tensor(prob + 1e-8))
                })

                game.step(action)
                turns[game_idx] += 1

                if game.game_over or turns[game_idx] >= 500:
                    scores = [p.points for p in game.state.list_of_players]
                    if scores[0] > scores[1]:
                        rewards = [1.0, -1.0]
                    elif scores[1] > scores[0]:
                        rewards = [-1.0, 1.0]
                    else:
                        rewards = [0.0, 0.0]

                    if game_type[game_idx] != 'selfplay':
                        v3_traj = [[], []]
                        v3_traj[v3_player[game_idx]] = trajectories[game_idx][v3_player[game_idx]]
                        all_trajectories.append(v3_traj)
                        all_rewards.append(rewards)
                    else:
                        all_trajectories.append(trajectories[game_idx])
                        all_rewards.append(rewards)

                    completed += 1

                    if completed < num_episodes:
                        games[game_idx] = Game(2)
                        trajectories[game_idx] = [[], []]
                        turns[game_idx] = 0
                        game_type[game_idx] = self._assign_game_type(game_idx)
                        v3_player[game_idx] = completed % 2

        self.model.train()
        return all_trajectories, all_rewards

    def train_on_batch(self, all_trajectories, all_rewards):
        samples = []
        for traj, rewards in zip(all_trajectories, all_rewards):
            for player_idx in range(2):
                player_traj = traj[player_idx]
                reward = rewards[player_idx]

                if not player_traj:
                    continue

                returns = []
                G = reward
                for step in reversed(player_traj):
                    returns.insert(0, G)
                    G = G * self.gamma

                for step, ret in zip(player_traj, returns):
                    samples.append({
                        'data': step['data'],
                        'action_idx': step['action_idx'],
                        'return': ret
                    })

        if not samples:
            return {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

        random.shuffle(samples)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_batches = 0

        mini_batch_size = 32

        for i in range(0, len(samples), mini_batch_size):
            batch_samples = samples[i:i + mini_batch_size]
            if len(batch_samples) < 4:
                continue

            data_list = [s['data'] for s in batch_samples]
            batch = Batch.from_data_list(data_list).to(self.device)

            action_indices = torch.tensor(
                [s['action_idx'] for s in batch_samples],
                device=self.device,
                dtype=torch.long
            )
            returns = torch.tensor(
                [s['return'] for s in batch_samples],
                device=self.device,
                dtype=torch.float32
            )

            policy_logits, values = self.model(batch)
            values = values.squeeze(-1)

            log_probs = F.log_softmax(policy_logits, dim=1)
            probs = F.softmax(policy_logits, dim=1)

            action_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                advantages = returns - values

            policy_loss = -(action_log_probs * advantages).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = -(probs * log_probs).sum(dim=1).mean()

            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_batches += 1

        if num_batches > 0:
            return {
                'loss': total_loss / num_batches,
                'policy_loss': total_policy_loss / num_batches,
                'value_loss': total_value_loss / num_batches,
                'entropy': total_entropy / num_batches
            }
        return {'loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0}

    def evaluate_vs_random(self, num_games=100):
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
                    action, _, _ = self.model_choose(
                        game.state, player, actions, game.board, player_idx, explore=False
                    )
                else:
                    action = random_choose(game.state, player, actions, game.board)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def evaluate_vs_heuristic(self, num_games=100):
        wins = 0
        self.model.eval()

        def model_choose_fn(game_state, player, legal_actions, board):
            player_idx = game_state.list_of_players.index(player)
            action, _, _ = self.model_choose(
                game_state, player, legal_actions, board, player_idx, explore=False
            )
            return action

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

    def evaluate_vs_first_iteration(self, num_games=100):
        if not self.opponent:
            return None

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
                    action, _, _ = self.model_choose(
                        game.state, player, actions, game.board, player_idx, explore=False
                    )
                else:
                    action = self.opponent.choose(game.state, player, actions, game.board, player_idx)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def train(self, num_episodes=20000, eval_every=2000):
        print(f"Starting V3 training for {num_episodes} episodes...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Hyperparameters: lr={self.optimizer.param_groups[0]['lr']}, batch_size={self.batch_size}, gamma={self.gamma}, entropy_coef={self.entropy_coef}")

        print("\n--- Initial Evaluation ---")
        init_random = self.evaluate_vs_random(100)
        init_heuristic = self.evaluate_vs_heuristic(100)
        init_first_iter = self.evaluate_vs_first_iteration(100) if self.opponent else None
        print(f"vs Random: {init_random*100:.1f}%")
        print(f"vs Heuristic: {init_heuristic*100:.1f}%")
        if init_first_iter is not None:
            print(f"vs First Iteration: {init_first_iter*100:.1f}%")

        print("\n" + "=" * 60)
        if self.diverse_opponents and self.opponent:
            print("TRAINING (33% Self-Play + 33% First Iter + 33% Heuristic)")
        elif self.opponent:
            print("TRAINING (50% Self-Play + 50% vs First Iteration)")
        else:
            print("TRAINING (Pure Self-Play)")
        print("=" * 60)

        losses = []
        checkpoints = []
        total_episodes = 0
        batch_num = 0
        start_time = time.time()

        while total_episodes < num_episodes:
            batch_start = time.time()
            all_trajectories, all_rewards = self.collect_episodes(self.batch_size)
            collect_time = time.time() - batch_start

            train_start = time.time()
            stats = self.train_on_batch(all_trajectories, all_rewards)
            train_time = time.time() - train_start

            losses.append(stats['loss'])
            total_episodes += len(all_trajectories)
            batch_num += 1

            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            eps_per_sec = self.batch_size / (collect_time + train_time)
            print(f"Episode {total_episodes}: loss={avg_loss:.4f}, entropy={stats['entropy']:.4f}, {eps_per_sec:.1f} eps/sec", flush=True)

            if total_episodes % eval_every < self.batch_size:
                vs_random = self.evaluate_vs_random(100)
                vs_heuristic = self.evaluate_vs_heuristic(100)
                vs_first_iter = self.evaluate_vs_first_iteration(100) if self.opponent else None
                elapsed = time.time() - start_time
                if vs_first_iter is not None:
                    print(f"  -> vs Heuristic: {vs_heuristic*100:.1f}%, vs First Iter: {vs_first_iter*100:.1f}% ({elapsed/60:.1f}min)", flush=True)
                else:
                    print(f"  -> vs Random: {vs_random*100:.1f}%, vs Heuristic: {vs_heuristic*100:.1f}% ({elapsed/60:.1f}min)", flush=True)
                checkpoints.append({
                    'episode': total_episodes,
                    'vs_random': vs_random,
                    'vs_heuristic': vs_heuristic,
                    'vs_first_iter': vs_first_iter
                })
                self.save(f"model_v3_ep{total_episodes}.pt")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Average speed: {num_episodes/total_time:.1f} episodes/second")

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        final_random = self.evaluate_vs_random(200)
        final_heuristic = self.evaluate_vs_heuristic(200)
        final_first_iter = self.evaluate_vs_first_iteration(200) if self.opponent else None
        print(f"vs Random: {final_random*100:.1f}%")
        print(f"vs Heuristic: {final_heuristic*100:.1f}%")
        if final_first_iter is not None:
            print(f"vs First Iteration: {final_first_iter*100:.1f}%")

        self.save("model_v3_final.pt")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if self.opponent:
            print(f"{'Stage':<25} {'vs Heuristic':<15} {'vs First Iter':<15}")
            print("-" * 55)
            print(f"{'Initial':<25} {init_heuristic*100:.1f}%{'':<10} {init_first_iter*100:.1f}%")
            for cp in checkpoints:
                fi = cp.get('vs_first_iter')
                fi_str = f"{fi*100:.1f}%" if fi is not None else "N/A"
                print(f"{'Episode ' + str(cp['episode']):<25} {cp['vs_heuristic']*100:.1f}%{'':<10} {fi_str}")
            print(f"{'Final':<25} {final_heuristic*100:.1f}%{'':<10} {final_first_iter*100:.1f}%")
        else:
            print(f"{'Stage':<25} {'vs Random':<15} {'vs Heuristic':<15}")
            print("-" * 55)
            print(f"{'Initial':<25} {init_random*100:.1f}%{'':<10} {init_heuristic*100:.1f}%")
            for cp in checkpoints:
                print(f"{'Episode ' + str(cp['episode']):<25} {cp['vs_random']*100:.1f}%{'':<10} {cp['vs_heuristic']*100:.1f}%")
            print(f"{'Final':<25} {final_random*100:.1f}%{'':<10} {final_heuristic*100:.1f}%")

        return losses, checkpoints

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_to_idx': self.action_to_idx,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
            'diverse_opponents': self.diverse_opponents,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        self.next_action_idx = len(self.action_to_idx)
        if 'gamma' in checkpoint:
            self.gamma = checkpoint['gamma']
        if 'entropy_coef' in checkpoint:
            self.entropy_coef = checkpoint['entropy_coef']
        if 'diverse_opponents' in checkpoint:
            self.diverse_opponents = checkpoint['diverse_opponents']
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V3 Training: GCNConv + Pure Self-Play')
    parser.add_argument('num_episodes', type=int, nargs='?', default=20000)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor (default: 0.99)')
    parser.add_argument('--entropy-coef', type=float, default=0.05, help='Entropy coefficient (default: 0.05)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--opponent', type=str, help='Path to first_iteration model for mixed training')
    parser.add_argument('--diverse', action='store_true', help='Use diverse opponents (33%% self-play, 33%% first_iter, 33%% heuristic)')
    parser.add_argument('--eval-every', type=int, default=2000, help='Evaluate every N episodes')
    args = parser.parse_args()

    print("=" * 60)
    if args.diverse and args.opponent:
        print("V3 TRAINING: Diverse (33% Self-Play + 33% First Iter + 33% Heuristic)")
    elif args.opponent:
        print("V3 TRAINING: Mixed (Self-Play + vs First Iteration)")
    else:
        print("V3 TRAINING: GCNConv (Scaled) + Pure Self-Play")
    print("=" * 60)
    print(f"  Episodes: {args.num_episodes}")
    print(f"  LR: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Entropy coef: {args.entropy_coef}")
    if args.opponent:
        print(f"  Opponent: {args.opponent}")
    if args.diverse:
        print(f"  Diverse opponents: enabled")
    print("=" * 60)

    trainer = TrainerV3(
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        diverse_opponents=args.diverse
    )

    if args.opponent:
        trainer.load_opponent(args.opponent)

    if args.resume:
        trainer.load(args.resume)
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.lr
        print(f"Resumed from: {args.resume}")

    trainer.train(
        num_episodes=args.num_episodes,
        eval_every=args.eval_every
    )
