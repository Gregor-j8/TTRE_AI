import asyncio
from typing import Dict, Optional, List, Callable
from dataclasses import asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.game import Game
from src.players import (
    random_choose,
    ticket_focused_choose,
    smart_ticket_choose,
    greedy_routes_choose,
    overall_game_choose,
    blitz_choose,
    v5_best_choose,
    v5_final_choose,
    v5_mixed_choose,
    v4_best_choose,
    v3_final_choose,
    v1_best_choose,
)

AI_PLAYERS = {
    "random": random_choose,
    "greedy": greedy_routes_choose,
    "ticket_focused": ticket_focused_choose,
    "smart_ticket": smart_ticket_choose,
    "overall_game": overall_game_choose,
    "blitz": blitz_choose,
    "v5_best": v5_best_choose,
    "v5_final": v5_final_choose,
    "v5_mixed": v5_mixed_choose,
    "v4_best": v4_best_choose,
    "v3_final": v3_final_choose,
    "v1_best": v1_best_choose,
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameSession:
    def __init__(self, mode: str = "visualizer", player_count: int = 2, ai_types: List[str] = None):
        self.game = Game(player_count)
        self.mode = mode
        self.player_count = player_count
        self.ai_types = ai_types or ["ticket_focused"] * (player_count - 1)
        self.websocket: Optional[WebSocket] = None
        self.human_player_idx = 0
        self.pending_action = None
        self.action_event = asyncio.Event()
        self.speed = 1.0

    def get_ai_choose_fns(self) -> List[Callable]:
        return [AI_PLAYERS.get(ai_type, ticket_focused_choose) for ai_type in self.ai_types]

    def get_state_for_frontend(self) -> dict:
        claimed_routes = {}
        for player_idx, player in enumerate(self.game.state.list_of_players):
            for route in player.claimed_routes:
                city1, city2, key = route
                route_id = f"{city1}-{city2}-{key}"
                claimed_routes[route_id] = player_idx

        players = []
        for idx, player in enumerate(self.game.state.list_of_players):
            tickets = [
                {"source": t[0], "target": t[1], "points": t[2]}
                for t in player.tickets
            ]
            pending_tickets = [
                {"source": t[0], "target": t[1], "points": t[2]}
                for t in player.pending_tickets
            ]
            players.append({
                "hand": dict(player.hand),
                "trains": player.trains,
                "points": player.points,
                "tickets": tickets,
                "pendingTickets": pending_tickets,
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
                    "colorCount": action.color_count,
                    "locoCount": action.loco_count,
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
                player_count = data.get("playerCount", 2)
                ai_types = data.get("aiTypes", ["ticket_focused"] * (player_count - 1))

                session.mode = mode
                session.player_count = player_count
                session.ai_types = ai_types
                session.game = Game(player_count)

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
                session.speed = max(0.1, min(5.0, data.get("speed", 1.0)))

    except WebSocketDisconnect:
        if game_id in sessions:
            del sessions[game_id]

async def run_visualizer_game(session: GameSession):
    choose_fns = session.get_ai_choose_fns()
    turn = 0
    player_count = session.player_count

    while not session.game.game_over and session.websocket:
        actions = session.game.get_legal_actions()
        if not actions:
            session.game.current_player_idx = (session.game.current_player_idx + 1) % player_count
            continue

        player = session.game.get_current_player()
        choose_fn = choose_fns[session.game.current_player_idx % len(choose_fns)]
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
                    "player": (session.game.current_player_idx - 1) % player_count,
                    "action": action.type,
                    "turn": turn
                }
            })
        except Exception:
            break

        await asyncio.sleep(0.5 / session.speed)

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
        except Exception:
            pass

async def run_singleplayer_game(session: GameSession):
    ai_choose_fns = session.get_ai_choose_fns()
    turn = 0
    player_count = session.player_count

    while not session.game.game_over and session.websocket:
        actions = session.game.get_legal_actions()
        if not actions:
            session.game.current_player_idx = (session.game.current_player_idx + 1) % player_count
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
            except Exception:
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
            ai_idx = current_idx - 1 if current_idx > session.human_player_idx else current_idx
            ai_choose = ai_choose_fns[ai_idx % len(ai_choose_fns)]
            action = ai_choose(session.game.state, player, actions, session.game.board)
            await asyncio.sleep(0.3 / session.speed)

        session.game.step(action)
        turn += 1

        try:
            await session.websocket.send_json({
                "type": "state",
                "data": session.get_state_for_frontend()
            })
        except Exception:
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
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
