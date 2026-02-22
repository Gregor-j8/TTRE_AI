import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
import random
import time

from src.game import Game
from src.ml.state_encoder import StateEncoder
from src.ml.model import TTRModel
from src.players import random_choose, ticket_focused_choose


class PPOTrainer:
    def __init__(self, hidden_dim=256, lr=1e-4, gamma=0.99, entropy_coef=0.01,
                 epsilon=0.2, ppo_epochs=4, batch_size=64):
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
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

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

    def collect_episodes(self, num_episodes):
        all_trajectories = []
        all_rewards = []

        games = [Game(2) for _ in range(self.batch_size)]
        trajectories = [[[], []] for _ in range(self.batch_size)]
        turns = [0] * self.batch_size
        completed = 0

        while completed < num_episodes:
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

                        all_trajectories.append(trajectories[i])
                        all_rewards.append(rewards)
                        completed += 1

                        if completed >= num_episodes:
                            break

                        games[i] = Game(2)
                        trajectories[i] = [[], []]
                        turns[i] = 0
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
                    log_prob = torch.tensor(-float('inf'))
                else:
                    mask = torch.zeros(logits.shape[1], dtype=torch.bool, device=self.device)
                    for _, idx in valid_pairs:
                        if idx < mask.shape[0]:
                            mask[idx] = True

                    masked_logits = logits.clone()
                    masked_logits[0, ~mask] = float('-inf')
                    log_probs = F.log_softmax(masked_logits, dim=1)
                    probs = F.softmax(masked_logits, dim=1)

                    valid_indices = [idx for _, idx in valid_pairs]
                    valid_action_positions = [j for j, _ in valid_pairs]

                    valid_probs = probs[0, valid_indices].cpu().numpy()
                    valid_probs = valid_probs / valid_probs.sum()
                    choice_in_valid = random.choices(range(len(valid_pairs)), weights=valid_probs)[0]

                    action_pos = valid_action_positions[choice_in_valid]
                    action_idx = valid_indices[choice_in_valid]
                    action = actions[action_pos]
                    log_prob = log_probs[0, action_idx].detach()

                trajectories[game_idx][player_idx].append({
                    'data': data,
                    'action_idx': action_idx,
                    'old_log_prob': log_prob.cpu()
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

                    all_trajectories.append(trajectories[game_idx])
                    all_rewards.append(rewards)
                    completed += 1

                    if completed < num_episodes:
                        games[game_idx] = Game(2)
                        trajectories[game_idx] = [[], []]
                        turns[game_idx] = 0

        return all_trajectories, all_rewards

    def prepare_training_data(self, all_trajectories, all_rewards):
        """Convert trajectories to flat list of training samples with returns computed."""
        samples = []

        for traj, rewards in zip(all_trajectories, all_rewards):
            for player_idx in range(2):
                player_traj = traj[player_idx]
                reward = rewards[player_idx]

                if not player_traj:
                    continue

                returns = []
                G = reward
                for _ in reversed(player_traj):
                    returns.insert(0, G)
                    G = G * self.gamma

                for step, ret in zip(player_traj, returns):
                    samples.append({
                        'data': step['data'],
                        'action_idx': step['action_idx'],
                        'old_log_prob': step['old_log_prob'],
                        'return': ret
                    })

        return samples

    def train_ppo(self, samples):
        """Train using PPO on collected samples for multiple epochs."""
        if not samples:
            return 0.0

        total_loss = 0.0
        num_updates = 0

        for epoch in range(self.ppo_epochs):
            random.shuffle(samples)

            for i in range(0, len(samples), self.batch_size):
                batch_samples = samples[i:i + self.batch_size]
                if len(batch_samples) < 4:
                    continue

                data_list = [s['data'] for s in batch_samples]
                batch = Batch.from_data_list(data_list).to(self.device)

                action_indices = torch.tensor(
                    [s['action_idx'] for s in batch_samples],
                    device=self.device
                )
                old_log_probs = torch.stack(
                    [s['old_log_prob'] for s in batch_samples]
                ).to(self.device)
                returns = torch.tensor(
                    [s['return'] for s in batch_samples],
                    device=self.device,
                    dtype=torch.float32
                )

                policy_logits, values = self.model(batch)
                values = values.squeeze(-1)

                log_probs = F.log_softmax(policy_logits, dim=1)
                probs = F.softmax(policy_logits, dim=1)

                new_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)

                advantages = returns - values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns)

                entropy = -(probs * log_probs).sum(dim=1).mean()

                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return total_loss / num_updates if num_updates > 0 else 0.0

    def model_choose(self, game_state, player, legal_actions, board, player_idx, explore=False):
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

    def evaluate_vs_random(self, num_games=50):
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

        self.model.train()
        return wins / num_games

    def evaluate_vs_heuristic(self, num_games=50):
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
                    action = ticket_focused_choose(game.state, player, actions, game.board)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        self.model.train()
        return wins / num_games

    def train(self, num_episodes=1000, eval_every=500):
        print(f"Starting PPO training for {num_episodes} episodes...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"PPO epochs: {self.ppo_epochs}, epsilon: {self.epsilon}")

        losses = []
        total_episodes = 0
        start_time = time.time()

        while total_episodes < num_episodes:
            batch_start = time.time()

            all_trajectories, all_rewards = self.collect_episodes(self.batch_size)
            collect_time = time.time() - batch_start

            samples = self.prepare_training_data(all_trajectories, all_rewards)

            train_start = time.time()
            loss = self.train_ppo(samples)
            train_time = time.time() - train_start

            losses.append(loss)
            total_episodes += len(all_trajectories)

            if total_episodes % (self.batch_size * 5) < self.batch_size:
                avg_loss = sum(losses[-5:]) / min(5, len(losses))
                eps_per_sec = self.batch_size / (collect_time + train_time)
                print(f"Episode {total_episodes}: Loss={avg_loss:.4f}, {eps_per_sec:.1f} eps/sec", flush=True)

            if total_episodes % eval_every < self.batch_size:
                vs_random = self.evaluate_vs_random(100)
                vs_heuristic = self.evaluate_vs_heuristic(100)
                elapsed = time.time() - start_time
                print(f"  -> vs Random: {vs_random*100:.1f}%, vs Heuristic: {vs_heuristic*100:.1f}% ({elapsed/60:.1f}min)", flush=True)
                self.save(f"ppo_model_ep{total_episodes}.pt")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        final_random = self.evaluate_vs_random(200)
        final_heuristic = self.evaluate_vs_heuristic(200)
        print(f"vs Random: {final_random*100:.1f}%")
        print(f"vs Heuristic: {final_heuristic*100:.1f}%")

        self.save("ppo_model_final.pt")

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
    parser.add_argument('num_episodes', type=int, nargs='?', default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    print("=" * 60)
    print(f"PPO TRAINING: {args.num_episodes} episodes")
    print(f"  batch_size={args.batch_size}, lr={args.lr}")
    print(f"  epsilon={args.epsilon}, ppo_epochs={args.ppo_epochs}")
    print("=" * 60)

    trainer = PPOTrainer(
        lr=args.lr,
        epsilon=args.epsilon,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size
    )

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from: {args.resume}")

    print("\n--- Initial Evaluation ---")
    init_random = trainer.evaluate_vs_random(100)
    init_heuristic = trainer.evaluate_vs_heuristic(100)
    print(f"vs Random: {init_random*100:.1f}%")
    print(f"vs Heuristic: {init_heuristic*100:.1f}%")

    print("\n" + "=" * 60)
    trainer.train(num_episodes=args.num_episodes, eval_every=500)
