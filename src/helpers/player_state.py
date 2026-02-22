class PlayerState:
    def __init__(self):
        self.hand = {
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
        self.trains = 45
        self.points = 0
        self.tickets = []
        self.pending_tickets = []
        self.stations = 3
        self.claimed_routes = []
        self.stations_built = []