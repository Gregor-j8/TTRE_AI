from src.game import Game
from collections import Counter
import networkx as nx

def run_silent_game():
    game = Game(2)
    turn = 0
    action_counts = Counter()
    tunnel_attempts = 0
    tunnel_failures = 0

    while not game.game_over:
        actions = game.get_legal_actions()
        if not actions:
            game.current_player_idx = (game.current_player_idx + 1) % game.num_players
            continue

        import random
        action = random.choice(actions)
        action_counts[action.type] += 1

        route_data = None
        if action.type == "claim_route":
            route_id = (action.source1, action.source2, int(action.card1))
            route_data = game.state.board.edges[route_id]
            if route_data['tunnel'] == 'true':
                tunnel_attempts += 1

        trains_before = [p.trains for p in game.state.list_of_players]
        game.step(action)
        trains_after = [p.trains for p in game.state.list_of_players]

        if action.type == "claim_route" and route_data and route_data['tunnel'] == 'true':
            if trains_before == trains_after:
                tunnel_failures += 1

        turn += 1
        if turn > 1000:
            break

    results = {
        'turns': turn,
        'scores': [p.points for p in game.state.list_of_players],
        'trains_left': [p.trains for p in game.state.list_of_players],
        'routes_claimed': [],
        'tickets_completed': 0,
        'tickets_failed': 0,
        'action_counts': action_counts,
        'tunnel_attempts': tunnel_attempts,
        'tunnel_failures': tunnel_failures,
        'hand_sizes': [sum(p.hand.values()) for p in game.state.list_of_players],
    }

    for idx, player in enumerate(game.state.list_of_players):
        for route in player.claimed_routes:
            results['routes_claimed'].append(route)

        player_graph = nx.Graph()
        for route in player.claimed_routes:
            player_graph.add_edge(route[0], route[1])

        for ticket in player.tickets:
            source, target, points = ticket
            if player_graph.has_node(source) and player_graph.has_node(target):
                if nx.has_path(player_graph, source, target):
                    results['tickets_completed'] += 1
                else:
                    results['tickets_failed'] += 1
            else:
                results['tickets_failed'] += 1

    return results

def run_validation(num_games=100):
    all_scores = []
    all_turns = []
    all_margins = []
    route_popularity = Counter()
    total_tickets_completed = 0
    total_tickets_failed = 0
    total_action_counts = Counter()
    total_tunnel_attempts = 0
    total_tunnel_failures = 0
    all_hand_sizes = []
    errors = 0

    print(f"Running {num_games} games...")

    for i in range(num_games):
        try:
            results = run_silent_game()
            all_turns.append(results['turns'])
            all_scores.extend(results['scores'])

            scores = results['scores']
            margin = abs(scores[0] - scores[1])
            all_margins.append(margin)

            for route in results['routes_claimed']:
                city_pair = tuple(sorted([route[0], route[1]]))
                route_popularity[city_pair] += 1

            total_tickets_completed += results['tickets_completed']
            total_tickets_failed += results['tickets_failed']
            total_action_counts += results['action_counts']
            total_tunnel_attempts += results['tunnel_attempts']
            total_tunnel_failures += results['tunnel_failures']
            all_hand_sizes.extend(results['hand_sizes'])

        except Exception as e:
            errors += 1
            print(f"Game {i} error: {e}")

    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)

    print(f"\nGames completed: {num_games - errors}/{num_games}")
    print(f"Errors: {errors}")

    print(f"\n--- TURNS ---")
    print(f"Min: {min(all_turns)}, Max: {max(all_turns)}, Mean: {sum(all_turns)/len(all_turns):.1f}")

    print(f"\n--- SCORES ---")
    print(f"Min: {min(all_scores)}, Max: {max(all_scores)}")
    print(f"Mean: {sum(all_scores)/len(all_scores):.1f}")
    import statistics
    print(f"Std: {statistics.stdev(all_scores):.1f}")

    print(f"\n--- WIN MARGINS ---")
    print(f"Min: {min(all_margins)}, Max: {max(all_margins)}, Mean: {sum(all_margins)/len(all_margins):.1f}")

    print(f"\n--- TICKETS ---")
    total_tickets = total_tickets_completed + total_tickets_failed
    completion_rate = total_tickets_completed / total_tickets * 100 if total_tickets > 0 else 0
    print(f"Completed: {total_tickets_completed}, Failed: {total_tickets_failed}")
    print(f"Completion rate: {completion_rate:.1f}%")

    print(f"\n--- TUNNELS ---")
    if total_tunnel_attempts > 0:
        failure_rate = total_tunnel_failures / total_tunnel_attempts * 100
        print(f"Attempts: {total_tunnel_attempts}, Failures: {total_tunnel_failures}")
        print(f"Failure rate: {failure_rate:.1f}%")
    else:
        print("No tunnel attempts")

    print(f"\n--- ACTION DISTRIBUTION ---")
    total_actions = sum(total_action_counts.values())
    for action_type, count in total_action_counts.most_common():
        pct = count / total_actions * 100
        print(f"{action_type}: {count} ({pct:.1f}%)")

    print(f"\n--- FINAL HAND SIZES ---")
    print(f"Mean: {sum(all_hand_sizes)/len(all_hand_sizes):.1f}")

    print(f"\n--- TOP 10 MOST CLAIMED ROUTES ---")
    for route, count in route_popularity.most_common(10):
        print(f"{route[0]} - {route[1]}: {count}")

if __name__ == "__main__":
    run_validation(100)
