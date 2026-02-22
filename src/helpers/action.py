from dataclasses import dataclass

@dataclass
class Action:
    type: str
    source1: str
    card1: str = None
    source2: str = None
    card2: str = None
    color_count: int = 0
    loco_count: int = 0

def _draw_card_action(game_state):
    actions = []
    deck_has_cards = sum(game_state.draw_pile.values()) > 0 or len(game_state.discard_pile) > 0

    if deck_has_cards:
        actions.append(Action(type="draw_card", source1="deck", source2="deck"))
        for card in game_state.face_up_cards:
            if card != "Locomotive":
                actions.append(Action(type="draw_card", source1="deck", source2="face_up", card2=card))
        for card in game_state.face_up_cards:
            if card != "Locomotive":
                actions.append(Action(type="draw_card", source1="face_up", source2="deck", card1=card))
        for i, card1 in enumerate(game_state.face_up_cards):
            for j, card2 in enumerate(game_state.face_up_cards):
                if i != j and card1 != "Locomotive" and card2 != "Locomotive":
                    actions.append(Action(type="draw_card", source1="face_up", source2="face_up", card1=card1, card2=card2))
        for card in game_state.face_up_cards:
            if card == "Locomotive":
                actions.append(Action(type="draw_wild_card", source1="face_up", card1=card))

    return actions

def _claim_route_actions(game_state, player):
    actions = []

    player_city_pairs = set()
    for route in player.claimed_routes:
        city_pair = tuple(sorted([route[0], route[1]]))
        player_city_pairs.add(city_pair)

    for city1, city2, key, data in game_state.board.edges(keys=True, data=True):
        route_id = (city1, city2, key)

        if route_id in game_state.claimed_routes:
            continue

        city_pair = tuple(sorted([city1, city2]))
        if city_pair in player_city_pairs:
            continue

        route_length = data['carriages']
        if player.trains < route_length:
            continue

        route_color = data['color']
        locomotives = player.hand.get('Locomotive', 0)
        ferry_requirement = data['engine']

        if route_color == 'gray' or route_color == 'false':
            for color, count in player.hand.items():
                if color != 'Locomotive':
                    for color_used in range(route_length + 1):
                        loco_used = route_length - color_used
                        if count >= color_used and locomotives >= loco_used and loco_used >= ferry_requirement:
                            actions.append(Action(
                                type="claim_route",
                                source1=city1,
                                source2=city2,
                                card1=str(key),
                                card2=color,
                                color_count=color_used,
                                loco_count=loco_used
                            ))
        else:
            color_key = route_color + '_cards'
            color_cards = player.hand.get(color_key, 0)
            for color_used in range(route_length + 1):
                loco_used = route_length - color_used
                if color_cards >= color_used and locomotives >= loco_used and loco_used >= ferry_requirement:
                    actions.append(Action(
                        type="claim_route",
                        source1=city1,
                        source2=city2,
                        card1=str(key),
                        card2=color_key,
                        color_count=color_used,
                        loco_count=loco_used
                    ))

    return actions

def _draw_tickets_actions(game_state, player):
    actions = []
    if len(game_state.ticket_deck) > 0 and not player.pending_tickets:
        actions.append(Action(type="draw_tickets", source1="ticket_deck"))
    return actions

def _keep_tickets_actions(player):
    actions = []
    if not player.pending_tickets:
        return actions

    n = len(player.pending_tickets)
    for mask in range(1, 2**n):
        indices = [i for i in range(n) if mask & (1 << i)]
        indices_str = ','.join(str(i) for i in indices)
        actions.append(Action(type="keep_tickets", source1=indices_str))
    return actions

def _build_station_actions(game_state, player):
    actions = []

    if player.stations <= 0:
        return actions

    cost = 4 - player.stations
    locomotives = player.hand.get('Locomotive', 0)

    occupied_cities = set()
    for p in game_state.list_of_players:
        for city in p.stations_built:
            occupied_cities.add(city)

    for city in game_state.board.nodes():
        if city in occupied_cities:
            continue
        for color, count in player.hand.items():
            if color != 'Locomotive':
                for color_used in range(cost + 1):
                    loco_used = cost - color_used
                    if count >= color_used and locomotives >= loco_used:
                        actions.append(Action(
                            type="build_station",
                            source1=city,
                            card2=color,
                            color_count=color_used,
                            loco_count=loco_used
                        ))

    return actions

def legal_actions(game_state, player):
    if player.pending_tickets:
        return _keep_tickets_actions(player)

    actions = []
    actions.extend(_draw_card_action(game_state))
    actions.extend(_claim_route_actions(game_state, player))
    actions.extend(_draw_tickets_actions(game_state, player))
    actions.extend(_build_station_actions(game_state, player))
    return actions

def execute_action(action, game_state, player):
    if action.type == "draw_card":
        _execute_draw_card(action, game_state, player)
    elif action.type == "draw_wild_card":
        _execute_draw_wild(action, game_state, player)
    elif action.type == "claim_route":
        _execute_claim_route(action, game_state, player)
    elif action.type == "draw_tickets":
        _execute_draw_tickets(action, game_state, player)
    elif action.type == "keep_tickets":
        _execute_keep_tickets(action, game_state, player)
    elif action.type == "build_station":
        _execute_build_station(action, game_state, player)

def _execute_draw_card(action, game_state, player):
    import random

    if action.source1 == "deck":
        available = [c for c, count in game_state.draw_pile.items() if count > 0]
        if not available:
            game_state.reshuffle_discard()
            available = [c for c, count in game_state.draw_pile.items() if count > 0]
        if available:
            card = random.choice(available)
            game_state.draw_pile[card] -= 1
            player.hand[card] = player.hand.get(card, 0) + 1
    else:
        idx = game_state.face_up_cards.index(action.card1)
        card = game_state.face_up_cards.pop(idx)
        player.hand[card] = player.hand.get(card, 0) + 1
        _refill_face_up(game_state)

    if action.source2 == "deck":
        available = [c for c, count in game_state.draw_pile.items() if count > 0]
        if not available:
            game_state.reshuffle_discard()
            available = [c for c, count in game_state.draw_pile.items() if count > 0]
        if available:
            card = random.choice(available)
            game_state.draw_pile[card] -= 1
            player.hand[card] = player.hand.get(card, 0) + 1
    elif action.source2 == "face_up":
        idx = game_state.face_up_cards.index(action.card2)
        card = game_state.face_up_cards.pop(idx)
        player.hand[card] = player.hand.get(card, 0) + 1
        _refill_face_up(game_state)

def _execute_draw_wild(action, game_state, player):
    idx = game_state.face_up_cards.index(action.card1)
    card = game_state.face_up_cards.pop(idx)
    player.hand[card] = player.hand.get(card, 0) + 1
    _refill_face_up(game_state)

def _execute_claim_route(action, game_state, player):
    import random

    route_id = (action.source1, action.source2, int(action.card1))
    route_data = game_state.board.edges[route_id]
    route_length = route_data['carriages']
    color_key = action.card2
    is_tunnel = route_data['tunnel'] == 'true'

    color_used = action.color_count
    locos_used = action.loco_count

    if is_tunnel:
        revealed = []
        for _ in range(3):
            available = [c for c, count in game_state.draw_pile.items() if count > 0]
            if not available:
                game_state.reshuffle_discard()
                available = [c for c, count in game_state.draw_pile.items() if count > 0]
            if available:
                card = random.choice(available)
                game_state.draw_pile[card] -= 1
                revealed.append(card)

        extra_needed = sum(1 for c in revealed if c == color_key or c == 'Locomotive')
        game_state.discard_pile.extend(revealed)

        extra_color_available = player.hand.get(color_key, 0) - color_used
        extra_loco_available = player.hand.get('Locomotive', 0) - locos_used

        if extra_color_available + extra_loco_available < extra_needed:
            return

        extra_color_used = min(extra_color_available, extra_needed)
        extra_loco_used = extra_needed - extra_color_used

        if color_key in player.hand:
            player.hand[color_key] -= (color_used + extra_color_used)
        player.hand['Locomotive'] -= (locos_used + extra_loco_used)

        game_state.discard_pile.extend([color_key] * (color_used + extra_color_used))
        game_state.discard_pile.extend(['Locomotive'] * (locos_used + extra_loco_used))
    else:
        if color_key in player.hand:
            player.hand[color_key] -= color_used
        player.hand['Locomotive'] -= locos_used

        game_state.discard_pile.extend([color_key] * color_used)
        game_state.discard_pile.extend(['Locomotive'] * locos_used)

    game_state.claimed_routes.add(route_id)
    player.claimed_routes.append(route_id)
    player.trains -= route_length

    points = {1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15, 7: 18, 8: 21}
    player.points += points.get(route_length, 0)

def _execute_draw_tickets(action, game_state, player):
    tickets_to_draw = min(3, len(game_state.ticket_deck))
    drawn = game_state.ticket_deck[:tickets_to_draw]
    game_state.ticket_deck = game_state.ticket_deck[tickets_to_draw:]
    player.pending_tickets = drawn

def _execute_keep_tickets(action, game_state, player):
    indices = [int(i) for i in action.source1.split(',')]
    kept = [player.pending_tickets[i] for i in indices]
    returned = [t for i, t in enumerate(player.pending_tickets) if i not in indices]

    player.tickets.extend(kept)
    game_state.ticket_deck.extend(returned)

    player.pending_tickets = []

def _execute_build_station(action, game_state, player):
    city = action.source1
    color_key = action.card2

    color_used = action.color_count
    locos_used = action.loco_count

    if color_key in player.hand:
        player.hand[color_key] -= color_used
    player.hand['Locomotive'] -= locos_used

    game_state.discard_pile.extend([color_key] * color_used)
    game_state.discard_pile.extend(['Locomotive'] * locos_used)

    player.stations -= 1
    player.stations_built.append(city)

def _refill_face_up(game_state):
    import random
    while len(game_state.face_up_cards) < 5:
        available = [c for c, count in game_state.draw_pile.items() if count > 0]
        if not available:
            if not game_state.reshuffle_discard():
                break
            available = [c for c, count in game_state.draw_pile.items() if count > 0]
            if not available:
                break
        card = random.choice(available)
        game_state.draw_pile[card] -= 1
        game_state.face_up_cards.append(card)

    loco_count = sum(1 for c in game_state.face_up_cards if c == 'Locomotive')
    if loco_count >= 3 and len(game_state.face_up_cards) == 5:
        total_cards = sum(game_state.draw_pile.values())
        if total_cards >= 5 or game_state.discard_pile:
            game_state.discard_pile.extend(game_state.face_up_cards)
            game_state.face_up_cards = []
            _refill_face_up(game_state)