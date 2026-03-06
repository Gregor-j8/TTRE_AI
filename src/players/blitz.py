import networkx as nx
from .ticket_focused import (
    is_ticket_complete,
    get_path_to_ticket,
    get_needed_segments,
    find_claim_action_for_segment,
    get_route_color_for_segment
)


ROUTE_POINTS = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
POINTS_PER_TRAIN = {1: 1.0, 2: 1.0, 3: 1.33, 4: 1.75, 5: 2.0, 6: 2.5}


def get_ticket_overlap_score(tickets, board, claimed_routes_global):
    all_path_cities = []

    for ticket in tickets:
        path = get_path_to_ticket(ticket, board, claimed_routes_global)
        if path:
            all_path_cities.append(set(path))

    if len(all_path_cities) < 2:
        return 0

    total_overlap = 0
    for i in range(len(all_path_cities)):
        for j in range(i + 1, len(all_path_cities)):
            overlap = len(all_path_cities[i] & all_path_cities[j])
            total_overlap += overlap

    return total_overlap


def score_ticket_for_keeping(ticket, other_tickets, board, claimed_routes_global, player):
    source, target, points = ticket

    path = get_path_to_ticket(ticket, board, claimed_routes_global)
    if path is None:
        return -50

    hops = len(path) - 1

    score = points

    if points >= 15:
        score += 15
    elif points >= 10:
        score += 8

    for other in other_tickets:
        if other == ticket:
            continue
        other_path = get_path_to_ticket(other, board, claimed_routes_global)
        if other_path:
            shared = len(set(path) & set(other_path))
            score += shared * 3

    player_cities = set()
    for route in player.claimed_routes:
        player_cities.add(route[0])
        player_cities.add(route[1])

    already_connected = sum(1 for city in path if city in player_cities)
    score += already_connected * 2

    return score


def is_endpoint_route(action, player, tickets, board, claimed_routes_global):
    route_cities = {action.source1, action.source2}

    for ticket in tickets:
        if is_ticket_complete(player, ticket):
            continue

        path = get_path_to_ticket(ticket, board, claimed_routes_global)
        if not path or len(path) < 2:
            continue

        start_city = path[0]
        end_city = path[-1]

        if start_city in route_cities or end_city in route_cities:
            return True

    return False


def get_route_efficiency(action, board):
    route_id = (action.source1, action.source2, int(action.card1))
    length = board.edges[route_id]['carriages']
    return POINTS_PER_TRAIN.get(length, 1.0)


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


def score_claim_action(action, player, board, claimed_routes_global, tickets):
    route_id = (action.source1, action.source2, int(action.card1))
    length = board.edges[route_id]['carriages']
    base_points = ROUTE_POINTS.get(length, 0)

    score = 0

    efficiency = POINTS_PER_TRAIN.get(length, 1.0)
    score += efficiency * 15

    if length >= 5:
        score += 25
    elif length >= 4:
        score += 15
    elif length <= 2:
        score -= 10

    if is_route_on_ticket_path(action, player, tickets, board, claimed_routes_global):
        score += 30

    if is_endpoint_route(action, player, tickets, board, claimed_routes_global):
        score += 20

    return score


def get_priority_colors(player, tickets, board, claimed_routes_global):
    needed_colors = {}

    for ticket in tickets:
        if is_ticket_complete(player, ticket):
            continue

        segments = get_needed_segments(player, ticket, board, claimed_routes_global)
        for segment in segments:
            color = get_route_color_for_segment(segment, board, claimed_routes_global)
            if color:
                needed_colors[color] = needed_colors.get(color, 0) + 1

    return needed_colors


def find_best_draw(player, tickets, board, claimed_routes_global, legal_actions):
    priority_colors = get_priority_colors(player, tickets, board, claimed_routes_global)

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


def blitz_choose(game_state, player, legal_actions, board):
    keep_actions = [a for a in legal_actions if a.type == "keep_tickets"]
    if keep_actions:
        def score_keep(action):
            indices = [int(i) for i in action.source1.split(',')]
            tickets = [player.pending_tickets[i] for i in indices]

            total = sum(
                score_ticket_for_keeping(t, tickets, board, game_state.claimed_routes, player)
                for t in tickets
            )

            overlap = get_ticket_overlap_score(tickets, board, game_state.claimed_routes)
            total += overlap * 5

            return total

        best = max(keep_actions, key=score_keep)
        return best

    if not player.tickets:
        for action in legal_actions:
            if action.type == "draw_tickets":
                return action

    claim_actions = [a for a in legal_actions if a.type == "claim_route"]

    if claim_actions:
        scored = []
        for action in claim_actions:
            score = score_claim_action(
                action, player, board, game_state.claimed_routes, player.tickets
            )
            scored.append((score, action))

        scored.sort(reverse=True, key=lambda x: x[0])
        best_score, best_action = scored[0]

        route_id = (best_action.source1, best_action.source2, int(best_action.card1))
        length = board.edges[route_id]['carriages']

        if length >= 4:
            return best_action

        if is_route_on_ticket_path(best_action, player, player.tickets, board, game_state.claimed_routes):
            return best_action

        if is_endpoint_route(best_action, player, player.tickets, board, game_state.claimed_routes):
            return best_action

        if player.trains < 15:
            return best_action

    draw_action = find_best_draw(
        player, player.tickets, board, game_state.claimed_routes, legal_actions
    )
    if draw_action:
        return draw_action

    if claim_actions:
        return max(claim_actions, key=lambda a: get_route_efficiency(a, board))

    return legal_actions[0]
