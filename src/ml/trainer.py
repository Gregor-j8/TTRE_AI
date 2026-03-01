import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from collections import deque
import random


from src.game import Game
from src.ml.state_encoder import StateEncoder
from src.ml.model import TTRModel
from src.players import random_choose


class ParallelSelfPlay:
    def __init__(self, trainer, batch_size=64):
        self.trainer = trainer
        self.batch_size = batch_size
        self.device = trainer.device
        self.encoder = trainer.encoder
        self.model = trainer.model

    def collect_episodes(self, num_episodes):
        all_trajectories = []
        all_rewards = []
        completed = 0

        games = [Game(2) for _ in range(self.batch_size)]
        trajectories = [[[], []] for _ in range(self.batch_size)]
        turns = [0] * self.batch_size

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

                action_indices = [self.trainer.get_action_idx(a) for a in actions]
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

                    all_trajectories.append(trajectories[game_idx])
                    all_rewards.append(rewards)
                    completed += 1

                    if completed < num_episodes:
                        games[game_idx] = Game(2)
                        trajectories[game_idx] = [[], []]
                        turns[game_idx] = 0

        return all_trajectories, all_rewards
# hidden dim is width of network layers want to try 256 and 384 
class SelfPlayTrainer:
    def __init__(self, hidden_dim=256, lr=3e-4, gamma=0.97, entropy_coef=0.01):
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

    def play_self_play_game(self):
        game = Game(2)

        trajectories = [[], []]

        turn = 0
        while not game.game_over and turn < 500:
            actions = game.get_legal_actions()
            if not actions:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                continue

            player_idx = game.current_player_idx
            player = game.get_current_player()

            data = self.encoder.encode_state(game.state, player_idx)
            action, action_idx, prob = self.model_choose(
                game.state, player, actions, game.board, player_idx, explore=True
            )

            trajectories[player_idx].append({
                'data': data,
                'action_idx': action_idx,
                'log_prob': torch.log(torch.tensor(prob + 1e-8))
            })

            game.step(action)
            turn += 1

        scores = [p.points for p in game.state.list_of_players]

        if scores[0] > scores[1]:
            rewards = [1.0, -1.0]
        elif scores[1] > scores[0]:
            rewards = [-1.0, 1.0]
        else:
            rewards = [0.0, 0.0]

        return trajectories, rewards, scores

    def train_on_trajectories(self, trajectories, rewards):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_steps = 0

        for player_idx in range(2):
            traj = trajectories[player_idx]
            reward = rewards[player_idx]

            if not traj:
                continue

            returns = []
            G = reward
            for step in reversed(traj):
                returns.insert(0, G)
                G = G * self.gamma

            for step, G in zip(traj, returns):
                data = step['data'].to(self.device)
                action_idx = step['action_idx']

                policy_logits, value = self.model(data)

                log_probs = F.log_softmax(policy_logits, dim=1)
                probs = F.softmax(policy_logits, dim=1)

                if action_idx < log_probs.shape[1]:
                    advantage = G - value.item()
                    policy_loss = -log_probs[0, action_idx] * advantage
                    value_loss = F.mse_loss(value, torch.tensor([[G]], device=self.device))
                    entropy = -(probs * log_probs).sum()

                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                    total_entropy += entropy
                    num_steps += 1

        if num_steps > 0:
            loss = (total_policy_loss + 0.5 * total_value_loss - self.entropy_coef * total_entropy) / num_steps

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            return loss.item()
        return 0

    def train_on_batch(self, all_trajectories, all_rewards):
        total_loss = 0
        count = 0
        for traj, rewards in zip(all_trajectories, all_rewards):
            loss = self.train_on_trajectories(traj, rewards)
            total_loss += loss
            count += 1
        return total_loss / count if count > 0 else 0

    def evaluate_vs_random(self, num_games=50):
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

    def train(self, num_episodes=1000, eval_every=100):
        print(f"Starting self-play training for {num_episodes} episodes...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        losses = []
        win_rates = []

        for episode in range(num_episodes):
            trajectories, rewards, scores = self.play_self_play_game()
            loss = self.train_on_trajectories(trajectories, rewards)
            losses.append(loss)

            if (episode + 1) % 10 == 0:
                avg_loss = sum(losses[-10:]) / 10
                print(f"Episode {episode + 1}: Loss = {avg_loss:.4f}")

            if (episode + 1) % eval_every == 0:
                win_rate = self.evaluate_vs_random(50)
                win_rates.append(win_rate)
                print(f"  Eval vs Random: {win_rate*100:.1f}% win rate")

        return losses, win_rates

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

    def evaluate_vs_heuristic(self, num_games=100):
        from src.players import ticket_focused_choose
        wins = 0

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


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('num_episodes', type=int, nargs='?', default=10000)
    parser.add_argument('batch_size', type=int, nargs='?', default=64)
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    args = parser.parse_args()

    num_episodes = args.num_episodes
    batch_size = args.batch_size
    lr = args.lr

    print("=" * 60)
    print(f"PARALLEL TRAINING: {num_episodes:,} episodes, batch_size={batch_size}")
    print("=" * 60)

    trainer = SelfPlayTrainer(lr=lr)

    if args.resume:
        trainer.load(args.resume)
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr
        print(f"Resumed from checkpoint: {args.resume}")

    parallel = ParallelSelfPlay(trainer, batch_size=batch_size)
    print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"Hidden dim: 256, LR: {lr}, Gamma: 0.97, Entropy: 0.01")

    print("\n--- Initial Evaluation ---")
    init_random = trainer.evaluate_vs_random(100)
    init_heuristic = trainer.evaluate_vs_heuristic(100)
    print(f"vs Random: {init_random*100:.1f}%")
    print(f"vs Heuristic: {init_heuristic*100:.1f}%")

    print("\n" + "=" * 60)
    print("TRAINING (Parallel Self-Play)")
    print("=" * 60)

    losses = []
    checkpoints = []
    total_episodes = 0
    batch_num = 0
    episodes_per_batch = batch_size

    start_time = time.time()

    while total_episodes < num_episodes:
        batch_start = time.time()
        all_trajectories, all_rewards = parallel.collect_episodes(episodes_per_batch)
        collect_time = time.time() - batch_start

        train_start = time.time()
        loss = trainer.train_on_batch(all_trajectories, all_rewards)
        train_time = time.time() - train_start

        losses.append(loss)
        total_episodes += len(all_trajectories)
        batch_num += 1

        if batch_num % 10 == 0:
            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            eps_per_sec = episodes_per_batch / (collect_time + train_time)
            print(f"Episode {total_episodes}: Loss={avg_loss:.4f}, {eps_per_sec:.1f} eps/sec", flush=True)

        if total_episodes % 2000 < episodes_per_batch:
            vs_random = trainer.evaluate_vs_random(100)
            vs_heuristic = trainer.evaluate_vs_heuristic(100)
            elapsed = time.time() - start_time
            print(f"  -> vs Random: {vs_random*100:.1f}%, vs Heuristic: {vs_heuristic*100:.1f}% ({elapsed/60:.1f}min)", flush=True)
            checkpoints.append({
                'episode': total_episodes,
                'vs_random': vs_random,
                'vs_heuristic': vs_heuristic
            })
            trainer.save(f"model_ep{total_episodes}.pt")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Average speed: {num_episodes/total_time:.1f} episodes/second")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_random = trainer.evaluate_vs_random(200)
    final_heuristic = trainer.evaluate_vs_heuristic(200)
    print(f"vs Random: {final_random*100:.1f}%")
    print(f"vs Heuristic: {final_heuristic*100:.1f}%")

    trainer.save("model_final.pt")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Stage':<25} {'vs Random':<15} {'vs Heuristic':<15}")
    print("-" * 55)
    print(f"{'Initial':<25} {init_random*100:.1f}%{'':<10} {init_heuristic*100:.1f}%")
    for cp in checkpoints:
        print(f"{'Episode ' + str(cp['episode']):<25} {cp['vs_random']*100:.1f}%{'':<10} {cp['vs_heuristic']*100:.1f}%")
    print(f"{'Final':<25} {final_random*100:.1f}%{'':<10} {final_heuristic*100:.1f}%")
