import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
import random
import time
import copy

from src.game import Game
from src.ml.state_encoder import StateEncoderV2
from src.ml.model import TTRModelV2, TTRModelV2Large
from src.ml.first_iteration.model import TTRModel
from src.ml.first_iteration.state_encoder import StateEncoder
from src.players import random_choose, ticket_focused_choose


class TrainerV2:
    def __init__(
        self,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        entropy_coef=0.02,
        epsilon=0.2,
        ppo_epochs=4,
        batch_size=64,
        use_large_model=False,
        reward_claim_scale=0.05,
        reward_ticket_complete=0.2,
        reward_hoard_penalty=0.01
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.board = Game(2).board
        self.encoder = StateEncoderV2(self.board)

        self.num_actions = 1000

        if use_large_model:
            self.model = TTRModelV2Large(
                node_dim=self.encoder.get_node_feature_dim(),
                edge_dim=self.encoder.get_edge_feature_dim(),
                private_dim=self.encoder.get_private_state_dim(),
                num_actions=self.num_actions
            ).to(self.device)
        else:
            self.model = TTRModelV2(
                node_dim=self.encoder.get_node_feature_dim(),
                edge_dim=self.encoder.get_edge_feature_dim(),
                private_dim=self.encoder.get_private_state_dim(),
                hidden_dim=hidden_dim,
                num_actions=self.num_actions
            ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000, eta_min=1e-5
        )

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.epsilon = epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.reward_claim_scale = reward_claim_scale
        self.reward_ticket_complete = reward_ticket_complete
        self.reward_hoard_penalty = reward_hoard_penalty

        self.action_to_idx = {}
        self.idx_to_action = {}
        self.next_action_idx = 0

        self.benchmark_model = None
        self.benchmark_encoder = None
        self.benchmark_action_to_idx = {}

        self.mixed_opponent_ratio = 0.5

    def load_benchmark_model(self, checkpoint_path):
        print(f"Loading benchmark model from {checkpoint_path}")

        old_encoder = StateEncoder(self.board)
        old_model = TTRModel(
            node_dim=old_encoder.get_node_feature_dim(),
            edge_dim=old_encoder.get_edge_feature_dim(),
            private_dim=old_encoder.get_private_state_dim(),
            hidden_dim=256,
            num_actions=self.num_actions
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        old_model.load_state_dict(checkpoint['model_state_dict'])
        old_model.eval()

        self.benchmark_model = old_model
        self.benchmark_encoder = old_encoder
        self.benchmark_action_to_idx = checkpoint.get('action_to_idx', {})

        print(f"Benchmark model loaded: {sum(p.numel() for p in old_model.parameters()):,} params")

    def collect_imitation_data(self, num_games):
        if self.benchmark_model is None:
            print("No benchmark model loaded, cannot collect imitation data")
            return []

        print(f"Collecting {num_games} games from benchmark model...")
        samples = []

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

                benchmark_action = self.benchmark_choose(
                    game.state, player, actions, game.board, player_idx
                )

                new_data = self.encoder.encode_state(game.state, player_idx)
                action_idx = self.get_action_idx(benchmark_action)

                if action_idx is not None:
                    samples.append({
                        'data': new_data,
                        'action_idx': action_idx
                    })

                game.step(benchmark_action)
                turn += 1

            if (game_num + 1) % 50 == 0:
                print(f"  Collected {game_num + 1}/{num_games} games, {len(samples)} samples")

        print(f"Collected {len(samples)} imitation samples from {num_games} games")
        return samples

    def train_imitation(self, samples, epochs=5):
        if not samples:
            return 0.0

        print(f"Training imitation for {epochs} epochs on {len(samples)} samples...")
        self.model.train()
        total_loss = 0.0
        num_updates = 0

        for epoch in range(epochs):
            random.shuffle(samples)
            epoch_loss = 0.0
            epoch_updates = 0

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

                policy_logits, _ = self.model(batch)
                loss = F.cross_entropy(policy_logits, action_indices)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_updates += 1

            if epoch_updates > 0:
                avg_epoch_loss = epoch_loss / epoch_updates
                print(f"  Epoch {epoch + 1}/{epochs}: Loss={avg_epoch_loss:.4f}")
                total_loss += epoch_loss
                num_updates += epoch_updates

        return total_loss / num_updates if num_updates > 0 else 0.0

    def collect_mixed_episodes(self, num_episodes):
        all_trajectories = []
        all_final_rewards = []

        games = [Game(2) for _ in range(self.batch_size)]
        trajectories = [[[], []] for _ in range(self.batch_size)]
        turns = [0] * self.batch_size
        completed = 0

        use_benchmark = [random.random() < self.mixed_opponent_ratio for _ in range(self.batch_size)]
        new_model_player = [random.randint(0, 1) for _ in range(self.batch_size)]

        while completed < num_episodes:
            active_indices = []
            data_list = []
            legal_actions_list = []
            player_indices = []
            game_copies = []

            for i, game in enumerate(games):
                if game.game_over or turns[i] >= 500:
                    continue

                actions = game.get_legal_actions()
                if not actions:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                    continue

                player_idx = game.current_player_idx

                if use_benchmark[i] and player_idx != new_model_player[i]:
                    player = game.get_current_player()
                    action = self.benchmark_choose(
                        game.state, player, actions, game.board, player_idx
                    )
                    game.step(action)
                    turns[i] += 1
                    continue

                data = self.encoder.encode_state(game.state, player_idx)

                active_indices.append(i)
                data_list.append(data)
                legal_actions_list.append(actions)
                player_indices.append(player_idx)
                game_copies.append(copy.deepcopy(game))

            if not data_list:
                for i, game in enumerate(games):
                    if game.game_over or turns[i] >= 500:
                        scores = [p.points for p in game.state.list_of_players]
                        if scores[0] > scores[1]:
                            final_rewards = [1.0, -1.0]
                        elif scores[1] > scores[0]:
                            final_rewards = [-1.0, 1.0]
                        else:
                            final_rewards = [0.0, 0.0]

                        all_trajectories.append(trajectories[i])
                        all_final_rewards.append(final_rewards)
                        completed += 1

                        if completed >= num_episodes:
                            break

                        games[i] = Game(2)
                        trajectories[i] = [[], []]
                        turns[i] = 0
                        use_benchmark[i] = random.random() < self.mixed_opponent_ratio
                        new_model_player[i] = random.randint(0, 1)
                continue

            batch = Batch.from_data_list(data_list).to(self.device)

            with torch.no_grad():
                policy_logits, values = self.model(batch)

            for batch_idx, game_idx in enumerate(active_indices):
                game = games[game_idx]
                game_before = game_copies[batch_idx]
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
                    valid_probs = valid_probs / (valid_probs.sum() + 1e-8)
                    choice_in_valid = random.choices(range(len(valid_pairs)), weights=valid_probs)[0]

                    action_pos = valid_action_positions[choice_in_valid]
                    action_idx = valid_indices[choice_in_valid]
                    action = actions[action_pos]
                    log_prob = log_probs[0, action_idx].detach()

                game.step(action)
                turns[game_idx] += 1

                intermediate_reward = self.compute_intermediate_reward(
                    game_before, game, player_idx, action
                )

                trajectories[game_idx][player_idx].append({
                    'data': data,
                    'action_idx': action_idx,
                    'old_log_prob': log_prob.cpu(),
                    'intermediate_reward': intermediate_reward
                })

                if game.game_over or turns[game_idx] >= 500:
                    scores = [p.points for p in game.state.list_of_players]
                    if scores[0] > scores[1]:
                        final_rewards = [1.0, -1.0]
                    elif scores[1] > scores[0]:
                        final_rewards = [-1.0, 1.0]
                    else:
                        final_rewards = [0.0, 0.0]

                    all_trajectories.append(trajectories[game_idx])
                    all_final_rewards.append(final_rewards)
                    completed += 1

                    if completed < num_episodes:
                        games[game_idx] = Game(2)
                        trajectories[game_idx] = [[], []]
                        turns[game_idx] = 0
                        use_benchmark[game_idx] = random.random() < self.mixed_opponent_ratio
                        new_model_player[game_idx] = random.randint(0, 1)

        return all_trajectories, all_final_rewards

    def get_action_idx(self, action):
        key = (action.type, action.source1, action.source2, action.card1, action.card2)
        if key not in self.action_to_idx:
            if self.next_action_idx >= self.num_actions:
                return None
            self.action_to_idx[key] = self.next_action_idx
            self.idx_to_action[self.next_action_idx] = action
            self.next_action_idx += 1
        return self.action_to_idx[key]

    def compute_intermediate_reward(self, game_before, game_after, player_idx, action):
        reward = 0.0

        player_before = game_before.state.list_of_players[player_idx]
        player_after = game_after.state.list_of_players[player_idx]

        points_gained = player_after.points - player_before.points
        if points_gained > 0 and action.type == "claim_route":
            reward += points_gained * self.reward_claim_scale

        tickets_complete_before = self._count_complete_tickets(player_before)
        tickets_complete_after = self._count_complete_tickets(player_after)
        new_completions = tickets_complete_after - tickets_complete_before
        if new_completions > 0:
            reward += new_completions * self.reward_ticket_complete

        cards_before = sum(player_before.hand.values())
        claims_before = len(player_before.claimed_routes)

        if cards_before > 15 and claims_before < 3:
            reward -= self.reward_hoard_penalty
        elif cards_before > 20 and claims_before < 5:
            reward -= self.reward_hoard_penalty * 2

        return reward

    def _count_complete_tickets(self, player):
        import networkx as nx
        G = nx.Graph()
        for route in player.claimed_routes:
            G.add_edge(route[0], route[1])

        complete = 0
        for ticket in player.tickets:
            source, target, _ = ticket
            if G.has_node(source) and G.has_node(target):
                if nx.has_path(G, source, target):
                    complete += 1
        return complete

    def collect_episodes(self, num_episodes):
        all_trajectories = []
        all_final_rewards = []

        games = [Game(2) for _ in range(self.batch_size)]
        trajectories = [[[], []] for _ in range(self.batch_size)]
        turns = [0] * self.batch_size
        completed = 0

        while completed < num_episodes:
            active_indices = []
            data_list = []
            legal_actions_list = []
            player_indices = []
            game_copies = []

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
                game_copies.append(copy.deepcopy(game))

            if not data_list:
                for i, game in enumerate(games):
                    if game.game_over or turns[i] >= 500:
                        scores = [p.points for p in game.state.list_of_players]
                        if scores[0] > scores[1]:
                            final_rewards = [1.0, -1.0]
                        elif scores[1] > scores[0]:
                            final_rewards = [-1.0, 1.0]
                        else:
                            final_rewards = [0.0, 0.0]

                        all_trajectories.append(trajectories[i])
                        all_final_rewards.append(final_rewards)
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
                game_before = game_copies[batch_idx]
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
                    valid_probs = valid_probs / (valid_probs.sum() + 1e-8)
                    choice_in_valid = random.choices(range(len(valid_pairs)), weights=valid_probs)[0]

                    action_pos = valid_action_positions[choice_in_valid]
                    action_idx = valid_indices[choice_in_valid]
                    action = actions[action_pos]
                    log_prob = log_probs[0, action_idx].detach()

                game.step(action)
                turns[game_idx] += 1

                intermediate_reward = self.compute_intermediate_reward(
                    game_before, game, player_idx, action
                )

                trajectories[game_idx][player_idx].append({
                    'data': data,
                    'action_idx': action_idx,
                    'old_log_prob': log_prob.cpu(),
                    'intermediate_reward': intermediate_reward
                })

                if game.game_over or turns[game_idx] >= 500:
                    scores = [p.points for p in game.state.list_of_players]
                    if scores[0] > scores[1]:
                        final_rewards = [1.0, -1.0]
                    elif scores[1] > scores[0]:
                        final_rewards = [-1.0, 1.0]
                    else:
                        final_rewards = [0.0, 0.0]

                    all_trajectories.append(trajectories[game_idx])
                    all_final_rewards.append(final_rewards)
                    completed += 1

                    if completed < num_episodes:
                        games[game_idx] = Game(2)
                        trajectories[game_idx] = [[], []]
                        turns[game_idx] = 0

        return all_trajectories, all_final_rewards

    def prepare_training_data(self, all_trajectories, all_final_rewards):
        samples = []

        for traj, final_rewards in zip(all_trajectories, all_final_rewards):
            for player_idx in range(2):
                player_traj = traj[player_idx]
                final_reward = final_rewards[player_idx]

                if not player_traj:
                    continue

                returns = []
                G = final_reward

                for step in reversed(player_traj):
                    G = step['intermediate_reward'] + self.gamma * G
                    returns.insert(0, G)

                for step, ret in zip(player_traj, returns):
                    samples.append({
                        'data': step['data'],
                        'action_idx': step['action_idx'],
                        'old_log_prob': step['old_log_prob'],
                        'return': ret
                    })

        return samples

    def train_ppo(self, samples):
        if not samples:
            return 0.0, 0.0, 0.0

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
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
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_updates += 1

        self.scheduler.step()

        if num_updates > 0:
            return total_loss / num_updates, total_policy_loss / num_updates, total_value_loss / num_updates
        return 0.0, 0.0, 0.0

    def model_choose(self, game_state, player, legal_actions, board, player_idx, explore=False):
        data = self.encoder.encode_state(game_state, player_idx).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(data)

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

        if explore:
            valid_probs = probs[0, valid_indices].cpu().numpy()
            valid_probs = valid_probs / (valid_probs.sum() + 1e-8)
            choice_in_valid = random.choices(range(len(valid_pairs)), weights=valid_probs)[0]
        else:
            choice_in_valid = probs[0, valid_indices].argmax().item()

        action_pos = valid_action_positions[choice_in_valid]
        return legal_actions[action_pos]

    def benchmark_choose(self, game_state, player, legal_actions, board, player_idx):
        if self.benchmark_model is None:
            return random_choose(game_state, player, legal_actions, board)

        data = self.benchmark_encoder.encode_state(game_state, player_idx).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.benchmark_model(data)

        probs = F.softmax(policy_logits, dim=1).squeeze(0)

        best_prob = -1
        best_action = legal_actions[0]

        for action in legal_actions:
            key = (action.type, action.source1, action.source2, action.card1, action.card2)
            idx = self.benchmark_action_to_idx.get(key)
            if idx is not None and idx < probs.shape[0]:
                if probs[idx].item() > best_prob:
                    best_prob = probs[idx].item()
                    best_action = action

        return best_action

    def evaluate_vs_benchmark(self, num_games=100):
        if self.benchmark_model is None:
            print("No benchmark model loaded, skipping benchmark evaluation")
            return 0.5

        wins = 0
        self.model.eval()

        for i in range(num_games):
            game = Game(2)
            new_model_player = i % 2

            turn = 0
            while not game.game_over and turn < 500:
                actions = game.get_legal_actions()
                if not actions:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                    continue

                player_idx = game.current_player_idx
                player = game.get_current_player()

                if player_idx == new_model_player:
                    action = self.model_choose(
                        game.state, player, actions, game.board, player_idx, explore=False
                    )
                else:
                    action = self.benchmark_choose(
                        game.state, player, actions, game.board, player_idx
                    )

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[new_model_player] > scores[1 - new_model_player]:
                wins += 1

        self.model.train()
        return wins / num_games

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
                    action = self.model_choose(
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

    def evaluate_vs_heuristic(self, num_games=100):
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
                    action = self.model_choose(
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

    def train(self, num_episodes=2000, eval_every=500, imitation_games=0, mixed_ratio=0.5):
        print(f"Starting V2 training for {num_episodes} episodes...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Reward shaping: claim={self.reward_claim_scale}, ticket={self.reward_ticket_complete}, hoard_penalty={self.reward_hoard_penalty}")

        self.mixed_opponent_ratio = mixed_ratio

        if imitation_games > 0 and self.benchmark_model is not None:
            print("\n" + "=" * 60)
            print("PHASE 1: IMITATION WARMUP")
            print("=" * 60)
            imitation_samples = self.collect_imitation_data(imitation_games)
            self.train_imitation(imitation_samples, epochs=5)

            print("\n--- Post-Imitation Evaluation ---")
            vs_random = self.evaluate_vs_random(100)
            vs_heuristic = self.evaluate_vs_heuristic(100)
            vs_benchmark = self.evaluate_vs_benchmark(100)
            print(f"vs Random: {vs_random*100:.1f}%")
            print(f"vs Heuristic: {vs_heuristic*100:.1f}%")
            print(f"vs Benchmark: {vs_benchmark*100:.1f}%")
            self.save("model_v2_post_imitation.pt")

        print("\n" + "=" * 60)
        print(f"PHASE 2: MIXED TRAINING (opponent_ratio={mixed_ratio})")
        print("=" * 60)

        losses = []
        total_episodes = 0
        start_time = time.time()

        while total_episodes < num_episodes:
            batch_start = time.time()

            if self.benchmark_model is not None and mixed_ratio > 0:
                all_trajectories, all_rewards = self.collect_mixed_episodes(self.batch_size)
            else:
                all_trajectories, all_rewards = self.collect_episodes(self.batch_size)
            collect_time = time.time() - batch_start

            samples = self.prepare_training_data(all_trajectories, all_rewards)

            train_start = time.time()
            loss, policy_loss, value_loss = self.train_ppo(samples)
            train_time = time.time() - train_start

            losses.append(loss)
            total_episodes += len(all_trajectories)

            if total_episodes % (self.batch_size * 5) < self.batch_size:
                avg_loss = sum(losses[-5:]) / min(5, len(losses))
                eps_per_sec = self.batch_size / (collect_time + train_time)
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Episode {total_episodes}: Loss={avg_loss:.4f} (p={policy_loss:.4f}, v={value_loss:.4f}), LR={lr:.2e}, {eps_per_sec:.1f} eps/sec", flush=True)

            if total_episodes % eval_every < self.batch_size:
                vs_random = self.evaluate_vs_random(100)
                vs_heuristic = self.evaluate_vs_heuristic(100)
                vs_benchmark = self.evaluate_vs_benchmark(100)
                elapsed = time.time() - start_time

                print(f"  -> vs Random: {vs_random*100:.1f}%, vs Heuristic: {vs_heuristic*100:.1f}%, vs Benchmark: {vs_benchmark*100:.1f}% ({elapsed/60:.1f}min)", flush=True)
                self.save(f"model_v2_ep{total_episodes}.pt")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)
        final_random = self.evaluate_vs_random(200)
        final_heuristic = self.evaluate_vs_heuristic(200)
        final_benchmark = self.evaluate_vs_benchmark(200)
        print(f"vs Random: {final_random*100:.1f}%")
        print(f"vs Heuristic: {final_heuristic*100:.1f}%")
        print(f"vs Benchmark (old model): {final_benchmark*100:.1f}%")

        self.save("model_v2_final.pt")

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'action_to_idx': self.action_to_idx,
            'encoder_config': {
                'max_tickets': self.encoder.max_tickets,
                'max_paths_per_ticket': self.encoder.max_paths_per_ticket
            }
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        self.next_action_idx = len(self.action_to_idx)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('num_episodes', type=int, nargs='?', default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--large', action='store_true', help='Use large model variant')
    parser.add_argument('--benchmark', type=str, help='Path to benchmark model checkpoint')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--imitation-games', type=int, default=200, help='Number of games for imitation warmup')
    parser.add_argument('--mixed-ratio', type=float, default=0.5, help='Ratio of games against benchmark (0=self-play, 1=all vs benchmark)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"V2 TRAINING: {args.num_episodes} episodes")
    print(f"  batch_size={args.batch_size}, lr={args.lr}")
    print(f"  large_model={args.large}")
    print(f"  imitation_games={args.imitation_games}, mixed_ratio={args.mixed_ratio}")
    print("=" * 60)

    trainer = TrainerV2(
        lr=args.lr,
        batch_size=args.batch_size,
        use_large_model=args.large
    )

    if args.benchmark:
        trainer.load_benchmark_model(args.benchmark)

    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from: {args.resume}")

    print("\n--- Initial Evaluation ---")
    init_random = trainer.evaluate_vs_random(100)
    init_heuristic = trainer.evaluate_vs_heuristic(100)
    init_benchmark = trainer.evaluate_vs_benchmark(100)
    print(f"vs Random: {init_random*100:.1f}%")
    print(f"vs Heuristic: {init_heuristic*100:.1f}%")
    print(f"vs Benchmark: {init_benchmark*100:.1f}%")

    print("\n" + "=" * 60)
    trainer.train(
        num_episodes=args.num_episodes,
        eval_every=500,
        imitation_games=args.imitation_games,
        mixed_ratio=args.mixed_ratio
    )
