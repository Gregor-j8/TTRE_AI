class GameState:
    def __init__(self, num_players):
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
        self.list_of_players = []
        self.face_up_cards = {
            "red_cards": 0, 
            "blue_cards": 0, 
            "yellow_cards": 0, 
            "green_cards": 0, 
            "black_cards": 0, 
            "white_cards": 0, 
            "orange_cards": 0, 
            "pink_cards": 0, 
            "Locomotive": 0
        }
        self.discard_pile = []
        self.current_player = None