from .player_state import PlayerState
import random

class GameState:
    def __init__(self, num_players):
        self.list_of_players = []
        for i in range(num_players):
            player_state = PlayerState()
            self.list_of_players.append(player_state)

        self.total_cards = {
            "red_cards": 12, 
            "blue_cards": 12,
            "green_cards": 12,
            "yellow_cards": 12, 
            "black_cards": 12, 
            "white_cards": 12, 
            "orange_cards": 12, 
            "pink_cards": 12, 
            "Locomotive": 14
        }
        self.draw_pile = {
            "red_cards": 12, 
            "blue_cards": 12,
            "green_cards": 12,
            "yellow_cards": 12, 
            "black_cards": 12, 
            "white_cards": 12, 
            "orange_cards": 12, 
            "pink_cards": 12, 
            "Locomotive": 14
        }
        self.face_up_cards = []
        self.discard_pile = []
        self.current_player = None
        self.setup_face_up_cards()
    
    def setup_face_up_cards(self):
        for _ in range(5):
            available_cards = [
                card for card in self.draw_pile 
                if self.draw_pile[card] > 0
            ]
            if available_cards:
                random_card = random.choice(available_cards)
                self.face_up_cards.append(random_card)
                self.draw_pile[random_card] -= 1