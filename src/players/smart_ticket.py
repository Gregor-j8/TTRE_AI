import networkx as nx
from .ticket_focused import (
    is_ticket_complete,
    get_path_to_ticket,
    get_needed_segments,
    find_claim_action_for_segment,
    get_route_color_for_segment
)


def count_incomplete_tickets(player):
    return sum(1 for t in player.tickets if not is_ticket_complete(player, t))


def get_ticket_difficulty(player, ticket, board, claimed_routes_global):
    if is_ticket_complete(player, ticket):
        return float('inf')

    path = get_path_to_ticket(ticket, board, claimed_routes_global)
    if path is None:
        return float('inf')

    needed = get_needed_segments(player, ticket, board, claimed_routes_global)
    hops_remaining = len(needed)

    source, target, points = ticket
    efficiency = points / max(hops_remaining, 1)

    return hops_remaining - (efficiency * 0.1)


def select_easiest_ticket(player, board, claimed_routes_global):
    if not player.tickets:
        return None

    best_ticket = None
    best_difficulty = float('inf')

    for ticket in player.tickets:
        difficulty = get_ticket_difficulty(player, ticket, board, claimed_routes_global)
        if difficulty < best_difficulty:
            best_difficulty = difficulty
            best_ticket = ticket

    return best_ticket


def find_draw_action_focused(player, needed_segments, board, claimed_routes_global, legal_actions):
    needed_colors = set()
    for segment in needed_segments:
        color = get_route_color_for_segment(segment, board, claimed_routes_global)
        if color:
            needed_colors.add(color)

    for action in legal_actions:
        if action.type == "draw_card" and action.source1 == "face_up":
            card = action.card1
            if card in needed_colors:
                return action

    for action in legal_actions:
        if action.type == "draw_wild_card":
            return action

    for action in legal_actions:
        if action.type == "draw_card" and action.source1 == "deck":
            return action

    return legal_actions[0]


def smart_ticket_choose(game_state, player, legal_actions, board):
    keep_actions = [a for a in legal_actions if a.type == "keep_tickets"]
    if keep_actions:
        def score_keep(action):
            indices = [int(i) for i in action.source1.split(',')]
            tickets = [player.pending_tickets[i] for i in indices]
            total_hops = 0
            for ticket in tickets:
                path = get_path_to_ticket(ticket, board, game_state.claimed_routes)
                if path:
                    total_hops += len(path) - 1
                else:
                    total_hops += 100
            return -total_hops

        if len(keep_actions) > 1:
            min_keep = min(keep_actions, key=lambda a: len(a.source1.split(',')))
            short_keeps = [a for a in keep_actions if len(a.source1.split(',')) == len(min_keep.source1.split(','))]
            best = max(short_keeps, key=score_keep)
        else:
            best = keep_actions[0]
        return best

    if not player.tickets:
        for action in legal_actions:
            if action.type == "draw_tickets":
                return action

    incomplete = count_incomplete_tickets(player)
    if incomplete >= 2:
        draw_ticket_actions = [a for a in legal_actions if a.type == "draw_tickets"]
        legal_actions = [a for a in legal_actions if a.type != "draw_tickets"]
        if not legal_actions:
            return draw_ticket_actions[0] if draw_ticket_actions else None

    target_ticket = select_easiest_ticket(player, board, game_state.claimed_routes)

    if target_ticket is None:
        claim_actions = [a for a in legal_actions if a.type == "claim_route"]
        if claim_actions:
            return claim_actions[0]
        for action in legal_actions:
            if action.type == "draw_card":
                return action
        return legal_actions[0]

    needed_segments = get_needed_segments(player, target_ticket, board, game_state.claimed_routes)

    if not needed_segments:
        target_ticket = select_easiest_ticket(player, board, game_state.claimed_routes)
        if target_ticket:
            needed_segments = get_needed_segments(player, target_ticket, board, game_state.claimed_routes)

    for segment in needed_segments:
        claim_action = find_claim_action_for_segment(segment, legal_actions)
        if claim_action:
            return claim_action

    return find_draw_action_focused(
        player, needed_segments, board, game_state.claimed_routes, legal_actions
    )
