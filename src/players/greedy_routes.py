def get_route_points(length):
    points_map = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15}
    return points_map.get(length, 0)


def greedy_routes_choose(game_state, player, legal_actions, board):
    keep_actions = [a for a in legal_actions if a.type == "keep_tickets"]
    if keep_actions:
        best = min(keep_actions, key=lambda a: len(a.source1.split(',')))
        return best

    claim_actions = [a for a in legal_actions if a.type == "claim_route"]

    if claim_actions:
        def route_value(action):
            route_id = (action.source1, action.source2, int(action.card1))
            length = board.edges[route_id]['carriages']
            return get_route_points(length)

        best = max(claim_actions, key=route_value)
        return best

    wild_draw = [a for a in legal_actions if a.type == "draw_wild_card"]
    if wild_draw:
        return wild_draw[0]

    face_up_draw = [a for a in legal_actions if a.type == "draw_card" and a.source1 == "face_up"]
    if face_up_draw:
        def card_count(action):
            card = action.card1
            return player.hand.get(card, 0)
        best = max(face_up_draw, key=card_count)
        return best

    deck_draw = [a for a in legal_actions if a.type == "draw_card" and a.source1 == "deck"]
    if deck_draw:
        return deck_draw[0]

    return legal_actions[0]
