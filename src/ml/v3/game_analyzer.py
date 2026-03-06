
import torch
import torch.nn.functional as F
import random
from collections import defaultdict

from src.game import Game
from src.ml.v2.state_encoder import StateEncoderV2
from src.ml.v3.model import TTRModelV3
from src.ml.state_encoder import StateEncoder
from src.ml.model import TTRModel
from src.players import ticket_focused_choose, smart_ticket_choose, greedy_routes_choose, overall_game_choose, blitz_choose
from src.players.ticket_focused import is_ticket_complete


class FirstIterationPlayer:
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

        best_idx = masked_logits[0].argmax().item()

        for action_pos, action_idx in valid_pairs:
            if action_idx == best_idx:
                return legal_actions[action_pos]

        return random.choice(legal_actions)


class GameAnalyzer:
    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.action_to_idx = checkpoint['action_to_idx']
        self.model.eval()
        print(f"Loaded model from {checkpoint_path}")

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

    def analyze_games(self, num_games=100, verbose=False):
        results = {
            'wins': 0,
            'losses': 0,
            'ties': 0,
            'v3_stats': defaultdict(list),
            'heuristic_stats': defaultdict(list),
            'loss_details': []
        }

        for game_num in range(num_games):
            game = Game(2)
            model_player = game_num % 2

            action_counts = [
                {'draw_card': 0, 'claim_route': 0, 'draw_tickets': 0},
                {'draw_card': 0, 'claim_route': 0, 'draw_tickets': 0}
            ]

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
                    action = ticket_focused_choose(game.state, player, actions, game.board)

                if action.type == 'draw_card' or action.type == 'draw_wild_card':
                    action_counts[player_idx]['draw_card'] += 1
                elif action.type == 'claim_route':
                    action_counts[player_idx]['claim_route'] += 1
                elif action.type == 'draw_tickets':
                    action_counts[player_idx]['draw_tickets'] += 1

                game.step(action)
                turn += 1

            v3_player = game.state.list_of_players[model_player]
            heuristic_player = game.state.list_of_players[1 - model_player]

            v3_score = v3_player.points
            heuristic_score = heuristic_player.points

            v3_completed = sum(1 for t in v3_player.tickets if is_ticket_complete(v3_player, t))
            v3_failed = sum(1 for t in v3_player.tickets if not is_ticket_complete(v3_player, t))
            heuristic_completed = sum(1 for t in heuristic_player.tickets if is_ticket_complete(heuristic_player, t))
            heuristic_failed = sum(1 for t in heuristic_player.tickets if not is_ticket_complete(heuristic_player, t))

            v3_routes = len(v3_player.claimed_routes)
            heuristic_routes = len(heuristic_player.claimed_routes)

            v3_trains_left = v3_player.trains
            heuristic_trains_left = heuristic_player.trains

            results['v3_stats']['score'].append(v3_score)
            results['v3_stats']['tickets_completed'].append(v3_completed)
            results['v3_stats']['tickets_failed'].append(v3_failed)
            results['v3_stats']['routes_claimed'].append(v3_routes)
            results['v3_stats']['trains_left'].append(v3_trains_left)
            results['v3_stats']['draw_card'].append(action_counts[model_player]['draw_card'])
            results['v3_stats']['claim_route'].append(action_counts[model_player]['claim_route'])

            results['heuristic_stats']['score'].append(heuristic_score)
            results['heuristic_stats']['tickets_completed'].append(heuristic_completed)
            results['heuristic_stats']['tickets_failed'].append(heuristic_failed)
            results['heuristic_stats']['routes_claimed'].append(heuristic_routes)
            results['heuristic_stats']['trains_left'].append(heuristic_trains_left)
            results['heuristic_stats']['draw_card'].append(action_counts[1 - model_player]['draw_card'])
            results['heuristic_stats']['claim_route'].append(action_counts[1 - model_player]['claim_route'])

            if v3_score > heuristic_score:
                results['wins'] += 1
                outcome = 'WIN'
            elif v3_score < heuristic_score:
                results['losses'] += 1
                outcome = 'LOSS'
                results['loss_details'].append({
                    'game_num': game_num,
                    'v3_score': v3_score,
                    'heuristic_score': heuristic_score,
                    'v3_tickets': f"{v3_completed}/{v3_completed + v3_failed}",
                    'heuristic_tickets': f"{heuristic_completed}/{heuristic_completed + heuristic_failed}",
                    'v3_routes': v3_routes,
                    'heuristic_routes': heuristic_routes,
                    'v3_draws': action_counts[model_player]['draw_card'],
                    'v3_claims': action_counts[model_player]['claim_route'],
                    'heuristic_draws': action_counts[1 - model_player]['draw_card'],
                    'heuristic_claims': action_counts[1 - model_player]['claim_route'],
                })
            else:
                results['ties'] += 1
                outcome = 'TIE'

            if verbose:
                print(f"Game {game_num + 1}: {outcome} | V3: {v3_score} ({v3_completed}/{v3_completed + v3_failed} tickets) | "
                      f"Heuristic: {heuristic_score} ({heuristic_completed}/{heuristic_completed + heuristic_failed} tickets)")

        return results

    def print_analysis(self, results):
        print("\n" + "=" * 70)
        print("GAME ANALYSIS RESULTS")
        print("=" * 70)

        total = results['wins'] + results['losses'] + results['ties']
        print(f"\nOverall: {results['wins']}/{total} wins ({results['wins']/total*100:.1f}%)")
        print(f"  Wins: {results['wins']}, Losses: {results['losses']}, Ties: {results['ties']}")

        print("\n" + "-" * 70)
        print("AVERAGE STATS COMPARISON")
        print("-" * 70)
        print(f"{'Metric':<25} {'V3':>12} {'Heuristic':>12} {'Diff':>10}")
        print("-" * 70)

        for metric in ['score', 'tickets_completed', 'tickets_failed', 'routes_claimed', 'trains_left', 'draw_card', 'claim_route']:
            v3_avg = sum(results['v3_stats'][metric]) / len(results['v3_stats'][metric])
            h_avg = sum(results['heuristic_stats'][metric]) / len(results['heuristic_stats'][metric])
            diff = v3_avg - h_avg
            sign = '+' if diff > 0 else ''
            print(f"{metric:<25} {v3_avg:>12.1f} {h_avg:>12.1f} {sign}{diff:>9.1f}")

        if results['loss_details']:
            print("\n" + "-" * 70)
            print(f"LOSS ANALYSIS ({len(results['loss_details'])} games)")
            print("-" * 70)

            ticket_diff = []
            route_diff = []
            draw_ratio_v3 = []
            draw_ratio_h = []

            for loss in results['loss_details']:
                v3_t = int(loss['v3_tickets'].split('/')[0])
                h_t = int(loss['heuristic_tickets'].split('/')[0])
                ticket_diff.append(h_t - v3_t)
                route_diff.append(loss['heuristic_routes'] - loss['v3_routes'])

                total_v3 = loss['v3_draws'] + loss['v3_claims']
                total_h = loss['heuristic_draws'] + loss['heuristic_claims']
                if total_v3 > 0:
                    draw_ratio_v3.append(loss['v3_draws'] / total_v3)
                if total_h > 0:
                    draw_ratio_h.append(loss['heuristic_draws'] / total_h)

            print(f"\nIn losses, heuristic averaged:")
            print(f"  +{sum(ticket_diff)/len(ticket_diff):.1f} more tickets completed than V3")
            print(f"  +{sum(route_diff)/len(route_diff):.1f} more routes claimed than V3")

            print(f"\nDraw/Claim ratio (draws / total actions):")
            print(f"  V3 in losses: {sum(draw_ratio_v3)/len(draw_ratio_v3)*100:.1f}% draws")
            print(f"  Heuristic in losses: {sum(draw_ratio_h)/len(draw_ratio_h)*100:.1f}% draws")

            score_gaps = [l['heuristic_score'] - l['v3_score'] for l in results['loss_details']]
            print(f"\nScore gap in losses: avg {sum(score_gaps)/len(score_gaps):.1f} points")
            print(f"  Closest loss: {min(score_gaps)} points")
            print(f"  Worst loss: {max(score_gaps)} points")

            print("\n" + "-" * 70)
            print("SAMPLE LOSSES (first 5)")
            print("-" * 70)
            for loss in results['loss_details'][:5]:
                print(f"Game {loss['game_num']}: V3 {loss['v3_score']} vs H {loss['heuristic_score']} | "
                      f"Tickets: V3 {loss['v3_tickets']} vs H {loss['heuristic_tickets']} | "
                      f"Routes: V3 {loss['v3_routes']} vs H {loss['heuristic_routes']}")


def run_match_with_stats(player1_fn, player2_fn, num_games=100):
    wins = 0
    ties = 0
    p1_stats = defaultdict(list)
    p2_stats = defaultdict(list)

    for game_num in range(num_games):
        game = Game(2)
        p1_idx = game_num % 2

        action_counts = [
            {'draw_card': 0, 'claim_route': 0, 'draw_tickets': 0},
            {'draw_card': 0, 'claim_route': 0, 'draw_tickets': 0}
        ]

        turn = 0
        while not game.game_over and turn < 500:
            actions = game.get_legal_actions()
            if not actions:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                continue

            player_idx = game.current_player_idx
            player = game.get_current_player()

            if player_idx == p1_idx:
                action = player1_fn(game.state, player, actions, game.board, player_idx)
            else:
                action = player2_fn(game.state, player, actions, game.board, player_idx)

            if action.type == 'draw_card' or action.type == 'draw_wild_card':
                action_counts[player_idx]['draw_card'] += 1
            elif action.type == 'claim_route':
                action_counts[player_idx]['claim_route'] += 1
            elif action.type == 'draw_tickets':
                action_counts[player_idx]['draw_tickets'] += 1

            game.step(action)
            turn += 1

        p1_player = game.state.list_of_players[p1_idx]
        p2_player = game.state.list_of_players[1 - p1_idx]

        p1_completed = sum(1 for t in p1_player.tickets if is_ticket_complete(p1_player, t))
        p1_failed = sum(1 for t in p1_player.tickets if not is_ticket_complete(p1_player, t))
        p2_completed = sum(1 for t in p2_player.tickets if is_ticket_complete(p2_player, t))
        p2_failed = sum(1 for t in p2_player.tickets if not is_ticket_complete(p2_player, t))

        p1_stats['score'].append(p1_player.points)
        p1_stats['tickets_completed'].append(p1_completed)
        p1_stats['tickets_failed'].append(p1_failed)
        p1_stats['routes_claimed'].append(len(p1_player.claimed_routes))
        p1_stats['trains_left'].append(p1_player.trains)
        p1_stats['draw_card'].append(action_counts[p1_idx]['draw_card'])
        p1_stats['claim_route'].append(action_counts[p1_idx]['claim_route'])

        p2_stats['score'].append(p2_player.points)
        p2_stats['tickets_completed'].append(p2_completed)
        p2_stats['tickets_failed'].append(p2_failed)
        p2_stats['routes_claimed'].append(len(p2_player.claimed_routes))
        p2_stats['trains_left'].append(p2_player.trains)
        p2_stats['draw_card'].append(action_counts[1 - p1_idx]['draw_card'])
        p2_stats['claim_route'].append(action_counts[1 - p1_idx]['claim_route'])

        if p1_player.points > p2_player.points:
            wins += 1
        elif p1_player.points == p2_player.points:
            ties += 1

    return wins, ties, num_games - wins - ties, p1_stats, p2_stats


def run_match(player1_fn, player2_fn, num_games=100):
    wins, ties, losses, _, _ = run_match_with_stats(player1_fn, player2_fn, num_games)
    return wins, ties, losses


def run_round_robin(v3_path, first_iter_path, num_games=100, detailed=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading models...")
    v3_analyzer = GameAnalyzer(v3_path)
    first_iter = FirstIterationPlayer(first_iter_path, device)
    print(f"Loaded first_iteration from {first_iter_path}")

    def v3_choose(game_state, player, actions, board, player_idx):
        return v3_analyzer.model_choose(game_state, player, actions, board, player_idx)

    def first_iter_choose(game_state, player, actions, board, player_idx):
        return first_iter.choose(game_state, player, actions, board, player_idx)

    def heuristic_choose(game_state, player, actions, board, player_idx):
        return ticket_focused_choose(game_state, player, actions, board)

    def smart_choose(game_state, player, actions, board, player_idx):
        return smart_ticket_choose(game_state, player, actions, board)

    def greedy_choose(game_state, player, actions, board, player_idx):
        return greedy_routes_choose(game_state, player, actions, board)

    def overall_choose(game_state, player, actions, board, player_idx):
        return overall_game_choose(game_state, player, actions, board)

    def blitz_choose_fn(game_state, player, actions, board, player_idx):
        return blitz_choose(game_state, player, actions, board)

    players = {
        'V3': v3_choose,
        'First_Iter': first_iter_choose,
        'Heuristic': heuristic_choose,
        'SmartTicket': smart_choose,
        'GreedyRoute': greedy_choose,
        'OverallGame': overall_choose,
        'Blitz': blitz_choose_fn
    }

    print(f"\nRunning round-robin ({num_games} games per matchup)...")
    print("=" * 70)

    results = {}
    all_stats = {}

    for p1_name, p1_fn in players.items():
        for p2_name, p2_fn in players.items():
            if p1_name == p2_name:
                continue

            key = f"{p1_name} vs {p2_name}"
            wins, ties, losses, p1_stats, p2_stats = run_match_with_stats(p1_fn, p2_fn, num_games)
            results[key] = (wins, ties, losses)
            all_stats[key] = (p1_stats, p2_stats)
            win_rate = wins / num_games * 100
            print(f"  {key}: {wins}W-{ties}T-{losses}L ({win_rate:.1f}%)")

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD MATRIX (row beats column)")
    print("=" * 70)
    names = list(players.keys())
    print(f"{'':>12}", end='')
    for name in names:
        print(f"{name:>12}", end='')
    print()

    for p1 in names:
        print(f"{p1:>12}", end='')
        for p2 in names:
            if p1 == p2:
                print(f"{'---':>12}", end='')
            else:
                key = f"{p1} vs {p2}"
                w, t, l = results[key]
                print(f"{w/num_games*100:>11.1f}%", end='')
        print()

    if detailed:
        print("\n" + "=" * 70)
        print("DETAILED STATS BY MATCHUP")
        print("=" * 70)

        metrics = ['score', 'tickets_completed', 'tickets_failed', 'routes_claimed', 'trains_left', 'draw_card', 'claim_route']

        for key, (p1_stats, p2_stats) in all_stats.items():
            p1_name, p2_name = key.split(' vs ')
            print(f"\n--- {key} ---")
            print(f"{'Metric':<20} {p1_name:>12} {p2_name:>12}")
            print("-" * 50)
            for metric in metrics:
                p1_avg = sum(p1_stats[metric]) / len(p1_stats[metric])
                p2_avg = sum(p2_stats[metric]) / len(p2_stats[metric])
                print(f"{metric:<20} {p1_avg:>12.1f} {p2_avg:>12.1f}")

        print("\n" + "=" * 70)
        print("AGGREGATE STATS (averaged across all matchups)")
        print("=" * 70)

        player_aggregate = {name: defaultdict(list) for name in names}

        for key, (p1_stats, p2_stats) in all_stats.items():
            p1_name, p2_name = key.split(' vs ')
            for metric in metrics:
                player_aggregate[p1_name][metric].extend(p1_stats[metric])
                player_aggregate[p2_name][metric].extend(p2_stats[metric])

        print(f"\n{'Metric':<20}", end='')
        for name in names:
            print(f"{name:>12}", end='')
        print()
        print("-" * (20 + 12 * len(names)))

        for metric in metrics:
            print(f"{metric:<20}", end='')
            for name in names:
                avg = sum(player_aggregate[name][metric]) / len(player_aggregate[name][metric])
                print(f"{avg:>12.1f}", end='')
            print()

    return results, all_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze V3 vs Heuristic games')
    parser.add_argument('--model', type=str, default='model_data/v3_good/model_v3_Final.pt',
                        help='Path to V3 model checkpoint')
    parser.add_argument('--first-iter', type=str, default='model_data/v1_models/first_iteration/model_final.pt',
                        help='Path to first_iteration model')
    parser.add_argument('--games', type=int, default=100, help='Number of games per matchup')
    parser.add_argument('--compare', action='store_true', help='Run round-robin comparison')
    parser.add_argument('--detailed', action='store_true', help='Show detailed stats per player')
    parser.add_argument('--verbose', action='store_true', help='Print each game result')
    args = parser.parse_args()

    if args.compare:
        run_round_robin(args.model, args.first_iter, args.games, detailed=args.detailed)
    else:
        analyzer = GameAnalyzer(args.model)
        results = analyzer.analyze_games(args.games, verbose=args.verbose)
        analyzer.print_analysis(results)
