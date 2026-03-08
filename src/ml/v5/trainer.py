import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
import random
import time

from src.game import Game
from src.ml.v5.state_encoder import StateEncoderV5
from src.ml.v5.model import TTRModelV5
from src.players import random_choose, ticket_focused_choose, overall_game_choose, blitz_choose


class TrainerV5:
    def __init__(
        self,
        lr=1e-4,
        batch_size=128,
        gamma=0.99,
        entropy_coef=0.05,
        entropy_decay=0.9995,
        min_entropy=0.01
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.board = Game(2).board
        self.encoder = StateEncoderV5(self.board)

        self.num_actions = 1000

        self.model = TTRModelV5(
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
        self.entropy_decay = entropy_decay
        self.min_entropy = min_entropy

        self.action_to_idx = {}
        self.idx_to_action = {}
        self.next_action_idx = 0

        self.v3_opponent = None
        self.v4_opponent = None

        self.opponent_mix = {
            'selfplay': 0.20,
            'heuristic': 0.20,
            'overall_game': 0.15,
            'blitz': 0.15,
            'v3': 0.15,
            'v4': 0.15
        }

    def load_v3_opponent(self, checkpoint_path):
        from src.ml.v3.model import TTRModelV3
        from src.ml.v2.state_encoder import StateEncoderV2

        v3_encoder = StateEncoderV2(self.board)
        v3_model = TTRModelV3(
            node_dim=v3_encoder.get_node_feature_dim(),
            edge_dim=v3_encoder.get_edge_feature_dim(),
            private_dim=v3_encoder.get_private_state_dim(),
            hidden_dim=400,
            num_gnn_layers=5,
            num_actions=1000,
            dropout=0.1
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        v3_model.load_state_dict(checkpoint['model_state_dict'])
        v3_model.eval()

        self.v3_opponent = {
            'model': v3_model,
            'encoder': v3_encoder,
            'action_to_idx': checkpoint.get('action_to_idx', {})
        }
        print(f"Loaded V3 opponent from {checkpoint_path}")

    def load_v4_opponent(self, checkpoint_path):
        from src.ml.v4.model import TTRModelV4
        from src.ml.v2.state_encoder import StateEncoderV2

        v4_encoder = StateEncoderV2(self.board)
        v4_model = TTRModelV4(
            node_dim=v4_encoder.get_node_feature_dim(),
            edge_dim=v4_encoder.get_edge_feature_dim(),
            private_dim=v4_encoder.get_private_state_dim(),
            hidden_dim=704,
            num_gnn_layers=6,
            num_actions=1000,
            dropout=0.1
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        v4_model.load_state_dict(checkpoint['model_state_dict'])
        v4_model.eval()

        self.v4_opponent = {
            'model': v4_model,
            'encoder': v4_encoder,
            'action_to_idx': checkpoint.get('action_to_idx', {})
        }
        print(f"Loaded V4 opponent from {checkpoint_path}")

    def _assign_game_type(self):
        available_types = {}
        for game_type, prob in self.opponent_mix.items():
            if game_type == 'v3' and not self.v3_opponent:
                continue
            if game_type == 'v4' and not self.v4_opponent:
                continue
            available_types[game_type] = prob

        total = sum(available_types.values())
        if total == 0:
            return 'selfplay'

        r = random.random() * total
        cumulative = 0
        for game_type, prob in available_types.items():
            cumulative += prob
            if r < cumulative:
                return game_type
        return 'selfplay'

    def _get_model_opponent_action(self, opponent, game_state, player_idx, actions):
        data = opponent['encoder'].encode_state(game_state, player_idx).to(self.device)

        with torch.no_grad():
            policy_logits, _ = opponent['model'](data)

        action_to_idx = opponent['action_to_idx']
        action_indices = []
        for a in actions:
            key = (a.type, a.source1, a.source2, a.card1, a.card2)
            action_indices.append(action_to_idx.get(key, None))

        valid_pairs = [(i, idx) for i, idx in enumerate(action_indices) if idx is not None]

        if not valid_pairs:
            return random.choice(actions)

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
        return actions[action_pos]

    def _get_opponent_action(self, game_type, game_state, player, actions, board, player_idx=None):
        if game_type == 'heuristic':
            return ticket_focused_choose(game_state, player, actions, board)
        elif game_type == 'overall_game':
            return overall_game_choose(game_state, player, actions, board)
        elif game_type == 'blitz':
            return blitz_choose(game_state, player, actions, board)
        elif game_type == 'v3' and self.v3_opponent:
            return self._get_model_opponent_action(self.v3_opponent, game_state, player_idx, actions)
        elif game_type == 'v4' and self.v4_opponent:
            return self._get_model_opponent_action(self.v4_opponent, game_state, player_idx, actions)
        else:
            return None

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

    def collect_episodes(self, num_episodes):
        all_trajectories = []
        all_rewards = []
        completed = 0

        games = [Game(2) for _ in range(self.batch_size)]
        trajectories = [[[], []] for _ in range(self.batch_size)]
        turns = [0] * self.batch_size

        self.model.eval()

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

        self.model.train()
        return all_trajectories, all_rewards

    def collect_episodes_mixed(self, num_episodes):
        all_trajectories = []
        all_rewards = []
        completed = 0

        games = [Game(2) for _ in range(self.batch_size)]
        trajectories = [[[], []] for _ in range(self.batch_size)]
        turns = [0] * self.batch_size
        game_types = [self._assign_game_type() for _ in range(self.batch_size)]
        model_players = [i % 2 for i in range(self.batch_size)]

        self.model.eval()

        while completed < num_episodes:
            for i, game in enumerate(games):
                if game_types[i] == 'selfplay' or game.game_over or turns[i] >= 500:
                    continue

                player_idx = game.current_player_idx
                if player_idx != model_players[i]:
                    actions = game.get_legal_actions()
                    if not actions:
                        game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                        continue
                    player = game.get_current_player()
                    action = self._get_opponent_action(
                        game_types[i], game.state, player, actions, game.board, player_idx
                    )
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

                if game_types[i] != 'selfplay' and player_idx != model_players[i]:
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

                        if game_types[i] != 'selfplay':
                            model_traj = [[], []]
                            model_traj[model_players[i]] = trajectories[i][model_players[i]]
                            all_trajectories.append(model_traj)
                        else:
                            all_trajectories.append(trajectories[i])
                        all_rewards.append(rewards)
                        completed += 1

                        if completed >= num_episodes:
                            break

                        games[i] = Game(2)
                        trajectories[i] = [[], []]
                        turns[i] = 0
                        game_types[i] = self._assign_game_type()
                        model_players[i] = completed % 2
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

                    if game_types[game_idx] != 'selfplay':
                        model_traj = [[], []]
                        model_traj[model_players[game_idx]] = trajectories[game_idx][model_players[game_idx]]
                        all_trajectories.append(model_traj)
                    else:
                        all_trajectories.append(trajectories[game_idx])
                    all_rewards.append(rewards)
                    completed += 1

                    if completed < num_episodes:
                        games[game_idx] = Game(2)
                        trajectories[game_idx] = [[], []]
                        turns[game_idx] = 0
                        game_types[game_idx] = self._assign_game_type()
                        model_players[game_idx] = completed % 2

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

        return wins / num_games

    def evaluate_vs_overall_game(self, num_games=50):
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
                    action = overall_game_choose(game.state, player, actions, game.board)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def evaluate_vs_blitz(self, num_games=50):
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
                    action = blitz_choose(game.state, player, actions, game.board)

                game.step(action)
                turn += 1

            scores = [p.points for p in game.state.list_of_players]
            if scores[model_player] > scores[1 - model_player]:
                wins += 1

        return wins / num_games

    def train(self, num_episodes=20000, eval_every=2000):
        print(f"Starting V5 Pure Self-Play training for {num_episodes} episodes...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Hyperparameters: lr={self.optimizer.param_groups[0]['lr']}, "
              f"entropy={self.entropy_coef}, gamma={self.gamma}")

        print("\n--- Initial Evaluation ---")
        init_random = self.evaluate_vs_random(100)
        init_heuristic = self.evaluate_vs_heuristic(100)
        init_overall = self.evaluate_vs_overall_game(50)
        init_blitz = self.evaluate_vs_blitz(50)
        print(f"vs Random: {init_random*100:.1f}%")
        print(f"vs Heuristic: {init_heuristic*100:.1f}%")
        print(f"vs OverallGame: {init_overall*100:.1f}%")
        print(f"vs Blitz: {init_blitz*100:.1f}%")

        print("\n" + "=" * 60)
        print("V5 TRAINING: Pure Self-Play (3.3M params)")
        print("=" * 60)

        losses = []
        checkpoints = []
        total_episodes = 0
        start_time = time.time()
        best_heuristic = init_heuristic

        while total_episodes < num_episodes:
            batch_start = time.time()
            all_trajectories, all_rewards = self.collect_episodes(self.batch_size)
            collect_time = time.time() - batch_start

            train_start = time.time()
            stats = self.train_on_batch(all_trajectories, all_rewards)
            train_time = time.time() - train_start

            losses.append(stats['loss'])
            total_episodes += len(all_trajectories)

            self.entropy_coef = max(self.min_entropy, self.entropy_coef * self.entropy_decay)

            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            eps_per_sec = self.batch_size / (collect_time + train_time)
            print(f"Ep {total_episodes}: loss={avg_loss:.4f}, entropy={stats['entropy']:.4f}, "
                  f"ent_coef={self.entropy_coef:.4f}, {eps_per_sec:.1f} eps/sec", flush=True)

            if total_episodes % eval_every < self.batch_size:
                vs_random = self.evaluate_vs_random(100)
                vs_heuristic = self.evaluate_vs_heuristic(100)
                vs_overall = self.evaluate_vs_overall_game(50)
                vs_blitz = self.evaluate_vs_blitz(50)
                elapsed = time.time() - start_time

                print(f"  -> Rand={vs_random*100:.0f}%, Heur={vs_heuristic*100:.0f}%, "
                      f"Over={vs_overall*100:.0f}%, Blitz={vs_blitz*100:.0f}% ({elapsed/60:.1f}min)")

                checkpoints.append({
                    'episode': total_episodes,
                    'vs_random': vs_random,
                    'vs_heuristic': vs_heuristic,
                    'vs_overall': vs_overall,
                    'vs_blitz': vs_blitz
                })

                if vs_heuristic > best_heuristic:
                    best_heuristic = vs_heuristic
                    self.save("model_v5_best.pt")
                    print(f"  ** New best: {vs_heuristic*100:.1f}% vs Heuristic **")

                self.save(f"model_v5_ep{total_episodes}.pt")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        final_random = self.evaluate_vs_random(200)
        final_heuristic = self.evaluate_vs_heuristic(200)
        final_overall = self.evaluate_vs_overall_game(100)
        final_blitz = self.evaluate_vs_blitz(100)

        print(f"vs Random:      {final_random*100:.1f}%")
        print(f"vs Heuristic:   {final_heuristic*100:.1f}%")
        print(f"vs OverallGame: {final_overall*100:.1f}%")
        print(f"vs Blitz:       {final_blitz*100:.1f}%")

        self.save("model_v5_final.pt")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Episode':<15} {'Heuristic':<12} {'OverallGame':<12} {'Blitz':<10}")
        print("-" * 50)
        print(f"{'Initial':<15} {init_heuristic*100:.1f}%{'':<6} {init_overall*100:.1f}%{'':<6} {init_blitz*100:.1f}%")
        for cp in checkpoints:
            print(f"Ep {cp['episode']:<10} {cp['vs_heuristic']*100:.1f}%{'':<6} "
                  f"{cp['vs_overall']*100:.1f}%{'':<6} {cp['vs_blitz']*100:.1f}%")
        print(f"{'Final':<15} {final_heuristic*100:.1f}%{'':<6} {final_overall*100:.1f}%{'':<6} {final_blitz*100:.1f}%")
        print(f"\nBest vs Heuristic: {best_heuristic*100:.1f}%")

        return losses, checkpoints

    def train_mixed(self, num_episodes=20000, eval_every=2000):
        print(f"Starting V5 Mixed Training for {num_episodes} episodes...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Opponent mix: {self.opponent_mix}")
        print(f"Hyperparameters: lr={self.optimizer.param_groups[0]['lr']}, "
              f"entropy={self.entropy_coef}, gamma={self.gamma}")

        print("\n--- Initial Evaluation ---")
        init_random = self.evaluate_vs_random(100)
        init_heuristic = self.evaluate_vs_heuristic(100)
        init_overall = self.evaluate_vs_overall_game(50)
        init_blitz = self.evaluate_vs_blitz(50)
        print(f"vs Random: {init_random*100:.1f}%")
        print(f"vs Heuristic: {init_heuristic*100:.1f}%")
        print(f"vs OverallGame: {init_overall*100:.1f}%")
        print(f"vs Blitz: {init_blitz*100:.1f}%")

        print("\n" + "=" * 60)
        print("V5 TRAINING: Mixed Opponents (3.3M params)")
        print("=" * 60)

        losses = []
        checkpoints = []
        total_episodes = 0
        start_time = time.time()
        best_heuristic = init_heuristic

        while total_episodes < num_episodes:
            batch_start = time.time()
            all_trajectories, all_rewards = self.collect_episodes_mixed(self.batch_size)
            collect_time = time.time() - batch_start

            train_start = time.time()
            stats = self.train_on_batch(all_trajectories, all_rewards)
            train_time = time.time() - train_start

            losses.append(stats['loss'])
            total_episodes += len(all_trajectories)

            self.entropy_coef = max(self.min_entropy, self.entropy_coef * self.entropy_decay)

            avg_loss = sum(losses[-10:]) / min(10, len(losses))
            eps_per_sec = self.batch_size / (collect_time + train_time)
            print(f"Ep {total_episodes}: loss={avg_loss:.4f}, entropy={stats['entropy']:.4f}, "
                  f"ent_coef={self.entropy_coef:.4f}, {eps_per_sec:.1f} eps/sec", flush=True)

            if total_episodes % eval_every < self.batch_size:
                vs_random = self.evaluate_vs_random(100)
                vs_heuristic = self.evaluate_vs_heuristic(100)
                vs_overall = self.evaluate_vs_overall_game(50)
                vs_blitz = self.evaluate_vs_blitz(50)
                elapsed = time.time() - start_time

                print(f"  -> Rand={vs_random*100:.0f}%, Heur={vs_heuristic*100:.0f}%, "
                      f"Over={vs_overall*100:.0f}%, Blitz={vs_blitz*100:.0f}% ({elapsed/60:.1f}min)")

                checkpoints.append({
                    'episode': total_episodes,
                    'vs_random': vs_random,
                    'vs_heuristic': vs_heuristic,
                    'vs_overall': vs_overall,
                    'vs_blitz': vs_blitz
                })

                if vs_heuristic > best_heuristic:
                    best_heuristic = vs_heuristic
                    self.save("model_v5_mixed_best.pt")
                    print(f"  ** New best: {vs_heuristic*100:.1f}% vs Heuristic **")

                self.save(f"model_v5_mixed_ep{total_episodes}.pt")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")

        print("\n" + "=" * 60)
        print("FINAL EVALUATION")
        print("=" * 60)

        final_random = self.evaluate_vs_random(200)
        final_heuristic = self.evaluate_vs_heuristic(200)
        final_overall = self.evaluate_vs_overall_game(100)
        final_blitz = self.evaluate_vs_blitz(100)

        print(f"vs Random:      {final_random*100:.1f}%")
        print(f"vs Heuristic:   {final_heuristic*100:.1f}%")
        print(f"vs OverallGame: {final_overall*100:.1f}%")
        print(f"vs Blitz:       {final_blitz*100:.1f}%")

        self.save("model_v5_mixed_final.pt")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Episode':<15} {'Heuristic':<12} {'OverallGame':<12} {'Blitz':<10}")
        print("-" * 50)
        print(f"{'Initial':<15} {init_heuristic*100:.1f}%{'':<6} {init_overall*100:.1f}%{'':<6} {init_blitz*100:.1f}%")
        for cp in checkpoints:
            print(f"Ep {cp['episode']:<10} {cp['vs_heuristic']*100:.1f}%{'':<6} "
                  f"{cp['vs_overall']*100:.1f}%{'':<6} {cp['vs_blitz']*100:.1f}%")
        print(f"{'Final':<15} {final_heuristic*100:.1f}%{'':<6} {final_overall*100:.1f}%{'':<6} {final_blitz*100:.1f}%")
        print(f"\nBest vs Heuristic: {best_heuristic*100:.1f}%")

        return losses, checkpoints

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_to_idx': self.action_to_idx,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
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
        print(f"Model loaded from {path}")
