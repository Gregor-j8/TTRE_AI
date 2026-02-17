from src.helpers.game_state import GameState
from src.helpers.action import _draw_card_action

game = GameState(2)
print("Face up cards:", game.face_up_cards)
actions = _draw_card_action(game)
print("Number of actions:", len(actions))


if __name__ == "__main__":
    game = GameState(2)
    print("Face up cards:", game.face_up_cards)
    actions = _draw_card_action(game)
    print("Number of actions:", len(actions))