from .player_state import PlayerState
import random
import csv
import os

class GameState:
    def __init__(self, num_players, board):
        self.board = board
        self.claimed_routes = set()
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
        self.ticket_deck = self._create_ticket_deck()
        self.current_player = None
        self.setup_face_up_cards()

    def _create_ticket_deck(self):
        tickets = []
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'destinations.csv')
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tickets.append((row['Source'], row['Target'], int(row['Points'])))
        random.shuffle(tickets)
        return tickets

    def reshuffle_discard(self):
        if not self.discard_pile:
            return False
        for card in self.discard_pile:
            self.draw_pile[card] = self.draw_pile.get(card, 0) + 1
        self.discard_pile = []
        return True
    
    def setup_face_up_cards(self):
        self._deal_face_up_cards()
        while self._check_locomotive_reset():
            self._deal_face_up_cards()

    def _deal_face_up_cards(self):
        self.discard_pile.extend(self.face_up_cards)
        self.face_up_cards = []
        for _ in range(5):
            available_cards = [
                card for card in self.draw_pile
                if self.draw_pile[card] > 0
            ]
            if not available_cards:
                self.reshuffle_discard()
                available_cards = [
                    card for card in self.draw_pile
                    if self.draw_pile[card] > 0
                ]
            if available_cards:
                random_card = random.choice(available_cards)
                self.face_up_cards.append(random_card)
                self.draw_pile[random_card] -= 1

    def _check_locomotive_reset(self):
        loco_count = sum(1 for c in self.face_up_cards if c == 'Locomotive')
        if loco_count >= 3 and len(self.face_up_cards) == 5:
            total_cards = sum(self.draw_pile.values())
            if total_cards >= 5 or self.discard_pile:
                return True
        return False