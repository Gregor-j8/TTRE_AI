import networkx as nx
from .ticket_focused import (
    is_ticket_complete,
    get_path_to_ticket,
    get_needed_segments,
    get_route_color_for_segment
)


ROUTE_POINTS = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}


def get_ticket_value(ticket, board, claimed_routes_global, player):
    source, target, points = ticket

    path = get_path_to_ticket(ticket, board, claimed_routes_global)
    if path is None:
        return -100

    hops = len(path) - 1
    if hops == 0:
        return points * 10

    points_per_hop = points / hops

    player_cities = set()
    for route in player.claimed_routes:
        player_cities.add(route[0])
        player_cities.add(route[1])

    overlap = sum(1 for city in path if city in player_cities)
    overlap_bonus = overlap * 2

    return points_per_hop + overlap_bonus + (points * 0.5)


def get_priority_colors(player, tickets, board, claimed_routes_global):
    needed_colors = {}

    for ticket in tickets:
        if is_ticket_complete(player, ticket):
            continue

        segments = get_needed_segments(player, ticket, board, claimed_routes_global)
        for segment in segments:
            color = get_route_color_for_segment(segment, board, claimed_routes_global)
            if color and color != 'false':
                needed_colors[color] = needed_colors.get(color, 0) + 1

    return needed_colors


def get_route_efficiency(action, board):
    route_id = (action.source1, action.source2, int(action.card1))
    length = board.edges[route_id]['carriages']
    points = ROUTE_POINTS.get(length, 0)
    return points / length if length > 0 else 0


def is_route_on_ticket_path(action, player, tickets, board, claimed_routes_global):
    route_cities = {action.source1, action.source2}

    for ticket in tickets:
        if is_ticket_complete(player, ticket):
            continue
        segments = get_needed_segments(player, ticket, board, claimed_routes_global)
        for seg in segments:
            if set(seg) == route_cities:
                return True
    return False


def extends_network(action, player):
    if not player.claimed_routes:
        return True

    player_cities = set()
    for route in player.claimed_routes:
        player_cities.add(route[0])
        player_cities.add(route[1])

    return action.source1 in player_cities or action.source2 in player_cities


def score_claim_action(action, player, board, claimed_routes_global, tickets, game_phase):
    route_id = (action.source1, action.source2, int(action.card1))
    length = board.edges[route_id]['carriages']
    points = ROUTE_POINTS.get(length, 0)

    score = 0

    efficiency = points / length if length > 0 else 0
    score += efficiency * 10

    if length >= 5:
        score += 20
    elif length >= 4:
        score += 10
    elif length <= 2:
        score -= 5

    if is_route_on_ticket_path(action, player, tickets, board, claimed_routes_global):
        score += 25

    if extends_network(action, player):
        score += 8

    if game_phase == 'early' and length >= 4:
        score += 15

    return score


def get_game_phase(player):
    trains = player.trains
    if trains > 35:
        return 'early'
    elif trains > 20:
        return 'mid'
    else:
        return 'late'


def should_draw_tickets(player, board, claimed_routes_global):
    incomplete = sum(1 for t in player.tickets if not is_ticket_complete(player, t))

    if incomplete >= 2:
        return False

    if player.trains < 15:
        return False

    completed = sum(1 for t in player.tickets if is_ticket_complete(player, t))
    if completed >= 2 and incomplete == 0:
        return True

    return False


def find_best_draw_action(player, tickets, board, claimed_routes_global, legal_actions, game_phase):
    priority_colors = get_priority_colors(player, tickets, board, claimed_routes_global)

    if game_phase == 'early':
        for action in legal_actions:
            if action.type == "draw_card" and action.source1 == "deck" and action.source2 == "deck":
                return action

    if priority_colors:
        best_color = max(priority_colors, key=priority_colors.get)
        for action in legal_actions:
            if action.type == "draw_card" and action.source1 == "face_up":
                if action.card1 == best_color:
                    return action

    for action in legal_actions:
        if action.type == "draw_wild_card":
            return action

    for action in legal_actions:
        if action.type == "draw_card" and action.source1 == "face_up":
            card = action.card1
            if player.hand.get(card, 0) >= 2:
                return action

    for action in legal_actions:
        if action.type == "draw_card" and action.source1 == "deck":
            return action

    return legal_actions[0] if legal_actions else None


def overall_game_choose(game_state, player, legal_actions, board):
    keep_actions = [a for a in legal_actions if a.type == "keep_tickets"]
    if keep_actions:
        def score_keep(action):
            indices = [int(i) for i in action.source1.split(',')]
            tickets = [player.pending_tickets[i] for i in indices]
            total_value = sum(
                get_ticket_value(t, board, game_state.claimed_routes, player)
                for t in tickets
            )
            return total_value

        best = max(keep_actions, key=score_keep)
        return best

    if not player.tickets:
        for action in legal_actions:
            if action.type == "draw_tickets":
                return action

    game_phase = get_game_phase(player)

    claim_actions = [a for a in legal_actions if a.type == "claim_route"]

    if claim_actions:
        scored_claims = []
        for action in claim_actions:
            score = score_claim_action(
                action, player, board, game_state.claimed_routes,
                player.tickets, game_phase
            )
            scored_claims.append((score, action))

        scored_claims.sort(reverse=True, key=lambda x: x[0])
        best_score, best_action = scored_claims[0]

        route_id = (best_action.source1, best_action.source2, int(best_action.card1))
        length = board.edges[route_id]['carriages']

        if best_score > 15 or length >= 4:
            return best_action

        if game_phase == 'late' and player.trains < 10:
            return best_action

        if is_route_on_ticket_path(best_action, player, player.tickets, board, game_state.claimed_routes):
            return best_action

    if should_draw_tickets(player, board, game_state.claimed_routes):
        for action in legal_actions:
            if action.type == "draw_tickets":
                return action

    draw_action = find_best_draw_action(
        player, player.tickets, board, game_state.claimed_routes,
        legal_actions, game_phase
    )
    if draw_action:
        return draw_action

    if claim_actions:
        return max(claim_actions, key=lambda a: get_route_efficiency(a, board))

    return legal_actions[0]
