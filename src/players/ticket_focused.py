import networkx as nx

def get_path_to_ticket(ticket, board, claimed_routes_global):
    source, target, points = ticket

    available_board = nx.Graph()
    for city1, city2, key, data in board.edges(keys=True, data=True):
        route_id = (city1, city2, key)
        if route_id not in claimed_routes_global:
            if not available_board.has_edge(city1, city2):
                available_board.add_edge(city1, city2)

    if source not in available_board or target not in available_board:
        return None

    try:
        return nx.shortest_path(available_board, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

def get_ticket_overlap(claimed_routes, ticket, board, claimed_routes_global):
    path = get_path_to_ticket(ticket, board, claimed_routes_global)
    if path is None:
        return -1, float('inf')

    claimed_cities = set()
    for route in claimed_routes:
        claimed_cities.add(route[0])
        claimed_cities.add(route[1])

    overlap = sum(1 for city in path if city in claimed_cities)
    hops = len(path) - 1

    return overlap, hops

def is_ticket_complete(player, ticket):
    player_graph = nx.Graph()
    for route in player.claimed_routes:
        player_graph.add_edge(route[0], route[1])

    source, target, _ = ticket
    if source not in player_graph or target not in player_graph:
        return False
    try:
        return nx.has_path(player_graph, source, target)
    except:
        return False

def select_target_ticket(player, board, claimed_routes_global):
    if not player.tickets:
        return None

    best_ticket = None
    best_overlap = -1
    best_hops = float('inf')

    for ticket in player.tickets:
        if is_ticket_complete(player, ticket):
            continue

        overlap, hops = get_ticket_overlap(
            player.claimed_routes, ticket, board, claimed_routes_global
        )

        if overlap == -1:
            continue

        if overlap > best_overlap or (overlap == best_overlap and hops < best_hops):
            best_ticket = ticket
            best_overlap = overlap
            best_hops = hops

    return best_ticket

def get_needed_segments(player, ticket, board, claimed_routes_global):
    path = get_path_to_ticket(ticket, board, claimed_routes_global)
    if path is None:
        return []

    player_city_pairs = set()
    for route in player.claimed_routes:
        city_pair = tuple(sorted([route[0], route[1]]))
        player_city_pairs.add(city_pair)

    needed = []
    for i in range(len(path) - 1):
        city_pair = tuple(sorted([path[i], path[i + 1]]))
        if city_pair not in player_city_pairs:
            needed.append((path[i], path[i + 1]))

    return needed

def find_claim_action_for_segment(segment, legal_actions):
    city1, city2 = segment
    city_pair = tuple(sorted([city1, city2]))

    for action in legal_actions:
        if action.type == "claim_route":
            action_pair = tuple(sorted([action.source1, action.source2]))
            if action_pair == city_pair:
                return action

    return None

def get_route_color_for_segment(segment, board, claimed_routes_global):
    city1, city2 = segment

    for c1, c2, key, data in board.edges(keys=True, data=True):
        route_id = (c1, c2, key)
        if route_id in claimed_routes_global:
            continue

        if (c1 == city1 and c2 == city2) or (c1 == city2 and c2 == city1):
            color = data['color']
            if color == 'false':
                return None
            return color + '_cards'

    return None

def find_draw_action(player, needed_segments, board, claimed_routes_global, legal_actions, game_state):
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
        if action.type == "draw_card" and action.source1 == "face_up":
            card = action.card1
            if card in player.hand and player.hand[card] > 0:
                return action

    for action in legal_actions:
        if action.type == "draw_wild_card":
            return action

    for action in legal_actions:
        if action.type == "draw_card" and action.source1 == "deck" and action.source2 == "deck":
            return action

    return legal_actions[0]

def ticket_focused_choose(game_state, player, legal_actions, board):
    keep_actions = [a for a in legal_actions if a.type == "keep_tickets"]
    if keep_actions:
        best = max(keep_actions, key=lambda a: len(a.source1.split(',')))
        return best

    if not player.tickets:
        for action in legal_actions:
            if action.type == "draw_tickets":
                return action

    target_ticket = select_target_ticket(player, board, game_state.claimed_routes)

    def get_route_length(action):
        route_id = (action.source1, action.source2, int(action.card1))
        return board.edges[route_id]['carriages']

    if target_ticket is None:
        claim_actions = [a for a in legal_actions if a.type == "claim_route"]
        if claim_actions:
            best = max(claim_actions, key=get_route_length)
            return best

        for action in legal_actions:
            if action.type == "draw_card" and action.source1 == "deck":
                return action
        return legal_actions[0]

    needed_segments = get_needed_segments(player, target_ticket, board, game_state.claimed_routes)

    if not needed_segments:
        claim_actions = [a for a in legal_actions if a.type == "claim_route"]
        if claim_actions:
            best = max(claim_actions, key=get_route_length)
            return best

        for action in legal_actions:
            if action.type == "draw_card" and action.source1 == "deck":
                return action
        return legal_actions[0]

    for segment in needed_segments:
        claim_action = find_claim_action_for_segment(segment, legal_actions)
        if claim_action:
            return claim_action

    claim_actions = [a for a in legal_actions if a.type == "claim_route"]
    long_routes = [a for a in claim_actions if board.edges[(a.source1, a.source2, int(a.card1))]['carriages'] >= 4]
    if long_routes:
        best = max(long_routes, key=lambda a: board.edges[(a.source1, a.source2, int(a.card1))]['carriages'])
        return best

    return find_draw_action(
        player, needed_segments, board, game_state.claimed_routes, legal_actions, game_state
    )
