import asyncio
from typing import Dict, Optional
from dataclasses import asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.game import Game
from src.players.ticket_focused import ticket_focused_choose
from src.players.random_player import random_choose

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameSession:
    def __init__(self, mode: str = "visualizer"):
        self.game = Game(2)
        self.mode = mode
        self.websocket: Optional[WebSocket] = None
        self.human_player_idx = 0
        self.pending_action = None
        self.action_event = asyncio.Event()

    def get_state_for_frontend(self) -> dict:
        claimed_routes = {}
        for player_idx, player in enumerate(self.game.state.list_of_players):
            for route in player.claimed_routes:
                city1, city2, key = route
                route_id = f"{city1}-{city2}-{key}"
                claimed_routes[route_id] = player_idx

        players = []
        for idx, player in enumerate(self.game.state.list_of_players):
            players.append({
                "hand": dict(player.hand),
                "trains": player.trains,
                "points": player.points,
                "tickets": player.tickets,
                "stations": player.stations,
            })

        legal_actions = []
        if self.mode == "singleplayer" and self.game.current_player_idx == self.human_player_idx:
            actions = self.game.get_legal_actions()
            for i, action in enumerate(actions):
                legal_actions.append({
                    "index": i,
                    "type": action.type,
                    "source1": action.source1,
                    "source2": action.source2,
                    "card1": action.card1,
                    "card2": action.card2,
                })

        return {
            "currentPlayerIdx": self.game.current_player_idx,
            "claimedRoutes": claimed_routes,
            "players": players,
            "faceUpCards": self.game.state.face_up_cards,
            "gameOver": self.game.game_over,
            "finalRound": self.game.final_round,
            "mode": self.mode,
            "legalActions": legal_actions,
        }

sessions: Dict[str, GameSession] = {}

@app.websocket("/ws/game/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await websocket.accept()

    if game_id not in sessions:
        sessions[game_id] = GameSession()

    session = sessions[game_id]
    session.websocket = websocket

    try:
        await websocket.send_json({
            "type": "state",
            "data": session.get_state_for_frontend()
        })

        while True:
            data = await websocket.receive_json()

            if data["type"] == "start_game":
                mode = data.get("mode", "visualizer")
                session.mode = mode
                session.game = Game(2)

                await websocket.send_json({
                    "type": "state",
                    "data": session.get_state_for_frontend()
                })

                if mode == "visualizer":
                    asyncio.create_task(run_visualizer_game(session))
                elif mode == "singleplayer":
                    asyncio.create_task(run_singleplayer_game(session))

            elif data["type"] == "player_action":
                if session.mode == "singleplayer":
                    action_idx = data.get("actionIndex", 0)
                    session.pending_action = action_idx
                    session.action_event.set()

            elif data["type"] == "set_speed":
                pass

    except WebSocketDisconnect:
        if game_id in sessions:
            del sessions[game_id]

async def run_visualizer_game(session: GameSession):
    choose_fns = [ticket_focused_choose, ticket_focused_choose]
    turn = 0

    while not session.game.game_over and session.websocket:
        actions = session.game.get_legal_actions()
        if not actions:
            session.game.current_player_idx = (session.game.current_player_idx + 1) % 2
            continue

        player = session.game.get_current_player()
        choose_fn = choose_fns[session.game.current_player_idx]
        action = choose_fn(session.game.state, player, actions, session.game.board)

        session.game.step(action)
        turn += 1

        try:
            await session.websocket.send_json({
                "type": "state",
                "data": session.get_state_for_frontend()
            })
            await session.websocket.send_json({
                "type": "action",
                "data": {
                    "player": (session.game.current_player_idx - 1) % 2,
                    "action": action.type,
                    "turn": turn
                }
            })
        except:
            break

        await asyncio.sleep(0.5)

        if turn > 500:
            break

    if session.websocket:
        try:
            await session.websocket.send_json({
                "type": "game_over",
                "data": {
                    "scores": [p.points for p in session.game.state.list_of_players]
                }
            })
        except:
            pass

async def run_singleplayer_game(session: GameSession):
    ai_choose = ticket_focused_choose
    turn = 0

    while not session.game.game_over and session.websocket:
        actions = session.game.get_legal_actions()
        if not actions:
            session.game.current_player_idx = (session.game.current_player_idx + 1) % 2
            continue

        current_idx = session.game.current_player_idx

        if current_idx == session.human_player_idx:
            try:
                await session.websocket.send_json({
                    "type": "state",
                    "data": session.get_state_for_frontend()
                })
                await session.websocket.send_json({
                    "type": "your_turn",
                    "data": {"legalActions": len(actions)}
                })
            except:
                break

            session.action_event.clear()
            await session.action_event.wait()

            action_idx = session.pending_action or 0
            if action_idx < len(actions):
                action = actions[action_idx]
            else:
                action = actions[0]
        else:
            player = session.game.get_current_player()
            action = ai_choose(session.game.state, player, actions, session.game.board)
            await asyncio.sleep(0.3)

        session.game.step(action)
        turn += 1

        try:
            await session.websocket.send_json({
                "type": "state",
                "data": session.get_state_for_frontend()
            })
        except:
            break

        if turn > 500:
            break

    if session.websocket:
        try:
            await session.websocket.send_json({
                "type": "game_over",
                "data": {
                    "scores": [p.points for p in session.game.state.list_of_players]
                }
            })
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
