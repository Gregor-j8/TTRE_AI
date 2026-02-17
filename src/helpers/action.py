from dataclasses import dataclass

@dataclass
class Action:
    type: str
    source1: str
    card1: str = None
    source2: str = None
    card2: str = None

def _draw_card_action(game_state):
    actions = []
    deck_has_cards = sum(game_state.draw_pile.values()) > 0
    
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

# def _execute_actions(action, game_state):