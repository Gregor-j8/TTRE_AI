import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from collections import defaultdict
import random

from src.game import Game
from .state_encoder import StateEncoder
from .model import TTRModel
from src.players import ticket_focused_choose, random_choose


class ModelAnalyzer:
    def __init__(self, checkpoint_path, hidden_dim=256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        self.idx_to_action = {v: k for k, v in self.action_to_idx.items()}
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Action vocabulary: {len(self.action_to_idx)} actions")

    def get_action_idx(self, action):
        key = (action.type, action.source1, action.source2, action.card1, action.card2)
        return self.action_to_idx.get(key, None)

    def format_action(self, action):
        if action.type == "claim_route":
            return f"CLAIM {action.source1} -> {action.source2}"
        elif action.type == "draw_card":
            if action.source1 == "deck":
                return "DRAW from deck"
            else:
                return f"DRAW {action.card1} from face-up"
        elif action.type == "draw_wild_card":
            return "DRAW wild (locomotive)"
        elif action.type == "draw_tickets":
            return "DRAW tickets"
        elif action.type == "keep_tickets":
            return f"KEEP {action.source1} tickets"
        else:
            return f"{action.type}"

    def analyze_decision(self, game_state, player, legal_actions, board, player_idx, top_k=5):
        data = self.encoder.encode_state(game_state, player_idx).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(data)

        action_indices = [self.get_action_idx(a) for a in legal_actions]
        valid_pairs = [(i, idx) for i, idx in enumerate(action_indices) if idx is not None]

        if not valid_pairs:
            return None, []

        mask = torch.zeros(policy_logits.shape[1], dtype=torch.bool, device=self.device)
        for _, idx in valid_pairs:
            if idx < mask.shape[0]:
                mask[idx] = True

        masked_logits = policy_logits.clone()
        masked_logits[0, ~mask] = float('-inf')
        probs = F.softmax(masked_logits, dim=1)

        action_probs = []
        for action_pos, action_idx in valid_pairs:
            prob = probs[0, action_idx].item()
            action_probs.append((legal_actions[action_pos], prob, action_idx))

        action_probs.sort(key=lambda x: x[1], reverse=True)

        return value.item(), action_probs[:top_k]

    def play_analyzed_game(self, vs_heuristic=True, max_turns=200):
        game = Game(2)
        model_player = 0

        print("\n" + "=" * 70)
        print("ANALYZED GAME")
        print(f"Model is Player {model_player + 1}")
        print("=" * 70)

        turn = 0
        decision_stats = defaultdict(int)

        while not game.game_over and turn < max_turns:
            actions = game.get_legal_actions()
            if not actions:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                continue

            player_idx = game.current_player_idx
            player = game.get_current_player()

            if player_idx == model_player:
                value, top_actions = self.analyze_decision(
                    game.state, player, actions, game.board, player_idx
                )

                if top_actions:
                    chosen_action = top_actions[0][0]
                    decision_stats[chosen_action.type] += 1

                    if turn < 30 or turn % 20 == 0:
                        print(f"\n--- Turn {turn + 1} (Model) ---")
                        print(f"Hand: {dict(player.hand)}")
                        print(f"Trains remaining: {player.trains}")
                        print(f"Points: {player.points}")
                        print(f"Value estimate: {value:.3f}")
                        print(f"Top choices:")
                        for action, prob, _ in top_actions[:3]:
                            print(f"  {prob*100:5.1f}% - {self.format_action(action)}")

                    game.step(chosen_action)
                else:
                    game.step(random.choice(actions))
            else:
                if vs_heuristic:
                    action = ticket_focused_choose(game.state, player, actions, game.board)
                else:
                    action = random.choice(actions)
                game.step(action)

            turn += 1

        scores = [p.points for p in game.state.list_of_players]

        print("\n" + "=" * 70)
        print("GAME OVER")
        print("=" * 70)
        print(f"Final scores: Player 1: {scores[0]}, Player 2: {scores[1]}")
        print(f"Model (Player {model_player + 1}) {'WON' if scores[model_player] > scores[1-model_player] else 'LOST'}")

        print(f"\nModel decision breakdown:")
        for action_type, count in sorted(decision_stats.items(), key=lambda x: -x[1]):
            print(f"  {action_type}: {count}")

        return scores[model_player] > scores[1 - model_player]

    def action_distribution_analysis(self, num_games=20):
        print("\n" + "=" * 70)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("=" * 70)

        action_counts = defaultdict(int)
        total_decisions = 0

        for game_num in range(num_games):
            game = Game(2)
            model_player = game_num % 2
            turn = 0

            while not game.game_over and turn < 300:
                actions = game.get_legal_actions()
                if not actions:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                    continue

                player_idx = game.current_player_idx
                player = game.get_current_player()

                if player_idx == model_player:
                    _, top_actions = self.analyze_decision(
                        game.state, player, actions, game.board, player_idx
                    )
                    if top_actions:
                        chosen = top_actions[0][0]
                        action_counts[chosen.type] += 1
                        total_decisions += 1
                        game.step(chosen)
                    else:
                        game.step(random.choice(actions))
                else:
                    game.step(ticket_focused_choose(game.state, player, actions, game.board))

                turn += 1

        print(f"\nAcross {num_games} games ({total_decisions} decisions):")
        for action_type, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            pct = count / total_decisions * 100
            bar = "█" * int(pct / 2)
            print(f"  {action_type:20} {count:5} ({pct:5.1f}%) {bar}")

    def compare_with_heuristic(self, num_games=50):
        print("\n" + "=" * 70)
        print("MODEL vs HEURISTIC COMPARISON")
        print("=" * 70)

        agreements = 0
        disagreements = 0
        disagreement_examples = []

        for game_num in range(num_games):
            game = Game(2)
            turn = 0

            while not game.game_over and turn < 300:
                actions = game.get_legal_actions()
                if not actions:
                    game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                    continue

                player_idx = game.current_player_idx
                player = game.get_current_player()

                _, top_actions = self.analyze_decision(
                    game.state, player, actions, game.board, player_idx
                )

                heuristic_action = ticket_focused_choose(game.state, player, actions, game.board)

                if top_actions:
                    model_action = top_actions[0][0]

                    model_key = (model_action.type, model_action.source1, model_action.source2)
                    heuristic_key = (heuristic_action.type, heuristic_action.source1, heuristic_action.source2)

                    if model_key == heuristic_key:
                        agreements += 1
                    else:
                        disagreements += 1
                        if len(disagreement_examples) < 5:
                            disagreement_examples.append({
                                'model': self.format_action(model_action),
                                'heuristic': self.format_action(heuristic_action),
                                'model_conf': top_actions[0][1]
                            })

                game.step(heuristic_action)
                turn += 1

        total = agreements + disagreements
        agreement_rate = agreements / total * 100 if total > 0 else 0

        print(f"\nAgreement rate: {agreement_rate:.1f}%")
        print(f"Agreements: {agreements}, Disagreements: {disagreements}")

        if disagreement_examples:
            print(f"\nExample disagreements:")
            for ex in disagreement_examples:
                print(f"  Model ({ex['model_conf']*100:.1f}%): {ex['model']}")
                print(f"  Heuristic:          {ex['heuristic']}")
                print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--play', action='store_true', help='Play an analyzed game')
    parser.add_argument('--distribution', action='store_true', help='Analyze action distribution')
    parser.add_argument('--compare', action='store_true', help='Compare with heuristic')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    args = parser.parse_args()

    analyzer = ModelAnalyzer(args.checkpoint)

    if args.all or args.distribution:
        analyzer.action_distribution_analysis(20)

    if args.all or args.compare:
        analyzer.compare_with_heuristic(50)

    if args.all or args.play:
        analyzer.play_analyzed_game(vs_heuristic=True)