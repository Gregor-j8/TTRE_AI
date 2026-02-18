from src.game import Game
from src.players import random_choose, ticket_focused_choose

def run_tournament(num_games=100):
    results = {
        'ticket_vs_random': {'ticket_wins': 0, 'random_wins': 0, 'ties': 0},
        'random_vs_ticket': {'random_wins': 0, 'ticket_wins': 0, 'ties': 0},
    }

    print(f"Running {num_games} games: Ticket-Focused (P0) vs Random (P1)")
    for i in range(num_games):
        try:
            game = Game(2)
            scores = game.play_game([ticket_focused_choose, random_choose], silent=True)

            if scores[0] > scores[1]:
                results['ticket_vs_random']['ticket_wins'] += 1
            elif scores[1] > scores[0]:
                results['ticket_vs_random']['random_wins'] += 1
            else:
                results['ticket_vs_random']['ties'] += 1
        except Exception as e:
            print(f"Game {i} error: {e}")

    print(f"\nRunning {num_games} games: Random (P0) vs Ticket-Focused (P1)")
    for i in range(num_games):
        try:
            game = Game(2)
            scores = game.play_game([random_choose, ticket_focused_choose], silent=True)

            if scores[0] > scores[1]:
                results['random_vs_ticket']['random_wins'] += 1
            elif scores[1] > scores[0]:
                results['random_vs_ticket']['ticket_wins'] += 1
            else:
                results['random_vs_ticket']['ties'] += 1
        except Exception as e:
            print(f"Game {i} error: {e}")

    print("\n" + "=" * 50)
    print("TOURNAMENT RESULTS")
    print("=" * 50)

    t_vs_r = results['ticket_vs_random']
    total1 = t_vs_r['ticket_wins'] + t_vs_r['random_wins'] + t_vs_r['ties']
    print(f"\nTicket-Focused (P0) vs Random (P1): {num_games} games")
    print(f"  Ticket wins: {t_vs_r['ticket_wins']} ({t_vs_r['ticket_wins']/total1*100:.1f}%)")
    print(f"  Random wins: {t_vs_r['random_wins']} ({t_vs_r['random_wins']/total1*100:.1f}%)")
    print(f"  Ties: {t_vs_r['ties']}")

    r_vs_t = results['random_vs_ticket']
    total2 = r_vs_t['ticket_wins'] + r_vs_t['random_wins'] + r_vs_t['ties']
    print(f"\nRandom (P0) vs Ticket-Focused (P1): {num_games} games")
    print(f"  Ticket wins: {r_vs_t['ticket_wins']} ({r_vs_t['ticket_wins']/total2*100:.1f}%)")
    print(f"  Random wins: {r_vs_t['random_wins']} ({r_vs_t['random_wins']/total2*100:.1f}%)")
    print(f"  Ties: {r_vs_t['ties']}")

    total_ticket = t_vs_r['ticket_wins'] + r_vs_t['ticket_wins']
    total_random = t_vs_r['random_wins'] + r_vs_t['random_wins']
    total_games = total1 + total2

    print(f"\n--- OVERALL ---")
    print(f"Ticket-Focused win rate: {total_ticket}/{total_games - t_vs_r['ties'] - r_vs_t['ties']} ({total_ticket/(total_ticket+total_random)*100:.1f}%)")

if __name__ == "__main__":
    run_tournament(1000)
