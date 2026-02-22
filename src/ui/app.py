from nicegui import ui
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Affine2D
import matplotlib.image as mpimg
import numpy as np
import io
import base64
import asyncio
import csv
import json
import os

from src.game import Game
from src.players import ticket_focused_choose, random_choose


CARD_COLORS = {
    'red_cards': '#e74c3c',
    'blue_cards': '#3498db',
    'green_cards': '#2ecc71',
    'yellow_cards': '#f1c40f',
    'black_cards': '#2c3e50',
    'white_cards': '#ecf0f1',
    'orange_cards': '#e67e22',
    'pink_cards': '#ff69b4',
    'Locomotive': '#9b59b6'
}

ROUTE_COLORS = {
    'Red': '#e74c3c',
    'Blue': '#3498db',
    'Green': '#2ecc71',
    'Yellow': '#f1c40f',
    'Black': '#2c3e50',
    'White': '#bdc3c7',
    'Orange': '#e67e22',
    'Pink': '#ff69b4',
    'false': '#95a5a6',
    'gray': '#95a5a6'
}

PLAYER_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']


COORD_BASE_WIDTH = 800
COORD_BASE_HEIGHT = 541


def load_city_coords():
    coords = {}
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'cities.csv')
    with open(data_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords[row['City']] = (float(row['X']), float(row['Y']))
    return coords


def scale_coords(x, y, img_width, img_height):
    scaled_x = x / COORD_BASE_WIDTH * img_width
    scaled_y = y / COORD_BASE_HEIGHT * img_height
    return scaled_x, scaled_y


def load_route_waypoints():
    waypoints = {}
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'route_waypoints.json')
    try:
        with open(data_path, 'r') as f:
            waypoints = json.load(f)
    except Exception as e:
        print(f"Could not load route waypoints: {e}")
    return waypoints


def interpolate_path(waypoints, num_points, padding=0.15):
    if len(waypoints) < 2:
        return waypoints, [0] * len(waypoints)

    segment_lengths = []
    for i in range(len(waypoints) - 1):
        dx = waypoints[i+1][0] - waypoints[i][0]
        dy = waypoints[i+1][1] - waypoints[i][1]
        segment_lengths.append(np.sqrt(dx*dx + dy*dy))

    total_length = sum(segment_lengths)
    if total_length == 0:
        return waypoints, [0] * len(waypoints)

    positions = []
    angles = []

    usable_length = total_length * (1 - 2 * padding)
    start_offset = total_length * padding

    for i in range(num_points):
        t = (i + 0.5) / num_points
        target_dist = start_offset + t * usable_length

        cumulative = 0
        for seg_idx, seg_len in enumerate(segment_lengths):
            if cumulative + seg_len >= target_dist:
                local_t = (target_dist - cumulative) / seg_len if seg_len > 0 else 0
                x = waypoints[seg_idx][0] + local_t * (waypoints[seg_idx+1][0] - waypoints[seg_idx][0])
                y = waypoints[seg_idx][1] + local_t * (waypoints[seg_idx+1][1] - waypoints[seg_idx][1])
                dx = waypoints[seg_idx+1][0] - waypoints[seg_idx][0]
                dy = waypoints[seg_idx+1][1] - waypoints[seg_idx][1]
                angle = np.degrees(np.arctan2(dy, dx))
                positions.append((x, y))
                angles.append(angle)
                break
            cumulative += seg_len

    return positions, angles


class GameUI:
    def __init__(self):
        self.game = None
        self.num_players = 2
        self.game_mode = 'watch'
        self.human_player_idx = 0
        self.ai_speed = 1.0
        self.running = False
        self.waiting_for_human = False
        self.selected_action = None
        self.model_choose_fn = None
        self.city_coords = load_city_coords()
        self.route_waypoints = load_route_waypoints()
        self.map_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'map.png')

    def load_model(self, model_path='model_data/models/first_iteration/model_final.pt'):
        try:
            import torch
            from src.ml.trainer import SelfPlayTrainer
            trainer = SelfPlayTrainer()
            trainer.load(model_path)

            def model_choose(game_state, player, legal_actions, board):
                player_idx = game_state.list_of_players.index(player)
                action, _, _ = trainer.model_choose(game_state, player, legal_actions, board, player_idx)
                return action

            self.model_choose_fn = model_choose
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Could not load model: {e}, using heuristic")
            return False

    def create_board_figure(self):
        if self.game is None:
            return None

        fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        try:
            map_img = mpimg.imread(self.map_path)
            ax.imshow(map_img)
            img_height, img_width = map_img.shape[:2]
        except Exception as e:
            print(f"Could not load map: {e}")
            img_width, img_height = 1200, 800
            ax.set_xlim(0, img_width)
            ax.set_ylim(img_height, 0)

        G = self.game.board

        for u, v, key, data in G.edges(keys=True, data=True):
            route_id = (u, v, key)

            if u not in self.city_coords or v not in self.city_coords:
                continue

            x1, y1 = scale_coords(*self.city_coords[u], img_width, img_height)
            x2, y2 = scale_coords(*self.city_coords[v], img_width, img_height)

            claimed_by = None
            for idx, player in enumerate(self.game.state.list_of_players):
                if route_id in player.claimed_routes:
                    claimed_by = idx
                    break

            offset = key * 6
            x1_off, y1_off = x1 + offset, y1 + offset
            x2_off, y2_off = x2 + offset, y2 + offset

            if claimed_by is not None:
                player_color = PLAYER_COLORS[claimed_by]
                num_cars = data.get('carriages', 3)
                car_width = 18
                car_height = 8

                route_color = data.get('color', 'false')
                waypoint_key = f"{u}-{v}-{route_color}"
                waypoint_key_rev = f"{v}-{u}-{route_color}"

                waypoints = self.route_waypoints.get(waypoint_key)
                reversed_waypoints = False
                if not waypoints:
                    waypoints = self.route_waypoints.get(waypoint_key_rev)
                    reversed_waypoints = True

                if waypoints:
                    if reversed_waypoints:
                        waypoints = waypoints[::-1]
                    scaled_waypoints = [scale_coords(wp[0], wp[1], img_width, img_height) for wp in waypoints]
                    positions, angles = interpolate_path(scaled_waypoints, num_cars)

                    for i, ((cx, cy), angle) in enumerate(zip(positions, angles)):
                        rect = FancyBboxPatch(
                            (-car_width/2, -car_height/2),
                            car_width, car_height,
                            boxstyle="round,pad=0.02,rounding_size=2",
                            facecolor=player_color,
                            edgecolor='black',
                            linewidth=1,
                            zorder=8
                        )
                        transform = Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData
                        rect.set_transform(transform)
                        ax.add_patch(rect)
                else:
                    angle = np.degrees(np.arctan2(y2_off - y1_off, x2_off - x1_off))
                    for i in range(num_cars):
                        t = (i + 0.5) / num_cars
                        cx = x1_off + t * (x2_off - x1_off)
                        cy = y1_off + t * (y2_off - y1_off)

                        rect = FancyBboxPatch(
                            (-car_width/2, -car_height/2),
                            car_width, car_height,
                            boxstyle="round,pad=0.02,rounding_size=2",
                            facecolor=player_color,
                            edgecolor='black',
                            linewidth=1,
                            zorder=8
                        )
                        transform = Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData
                        rect.set_transform(transform)
                        ax.add_patch(rect)

        ax.axis('off')
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return f'data:image/png;base64,{img_base64}'

    def get_choose_fn(self, player_idx):
        if self.game_mode == 'watch':
            if self.model_choose_fn:
                return self.model_choose_fn
            return ticket_focused_choose
        else:
            if player_idx == self.human_player_idx:
                return None
            if self.model_choose_fn:
                return self.model_choose_fn
            return ticket_focused_choose


game_ui = GameUI()


def create_ui():
    state = {'start_game': None, 'stop_game': None}

    with ui.column().classes('w-full p-2'):
        with ui.row().classes('w-full items-center gap-4 mb-2'):
            ui.label('Ticket to Ride Europe').classes('text-2xl font-bold')
            status_label = ui.label('No game running').classes('text-gray-500')
            turn_label = ui.label('')
            ui.html('<div style="flex-grow:1"></div>')
            scores_container = ui.row().classes('gap-2')

        with ui.card().classes('w-full p-1 mb-2'):
            board_image = ui.image().classes('w-full')

        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('p-3'):
                ui.label('Setup').classes('font-bold mb-1')
                with ui.row().classes('gap-2 items-center'):
                    num_players_select = ui.select([2, 3, 4, 5], value=2, label='Players').classes('w-20')
                    mode_select = ui.select({'watch': 'Watch AI', 'play': 'Play vs AI'}, value='watch', label='Mode').classes('w-28')
                    human_select = ui.select({0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4', 4: 'P5'}, value=0, label='Play as').classes('w-16')
                    ui.label('Speed:')
                    speed_slider = ui.slider(min=0.1, max=2.0, value=0.5, step=0.1).classes('w-24')
                    ui.button('Start', on_click=lambda: asyncio.create_task(state['start_game']())).props('color=primary')
                    ui.button('Stop', on_click=lambda: state['stop_game']()).props('color=red')

            with ui.card().classes('p-3'):
                ui.label('Hand').classes('font-bold mb-1')
                hand_container = ui.row().classes('gap-1 flex-wrap')

            with ui.card().classes('p-3'):
                ui.label('Face Up').classes('font-bold mb-1')
                faceup_container = ui.row().classes('gap-1 flex-wrap')

            tickets_expansion = ui.expansion('Destination Tickets', icon='confirmation_number')
            with tickets_expansion:
                tickets_container = ui.column().classes('max-h-40 overflow-y-auto gap-1')

            log_expansion = ui.expansion('Log', icon='list')
            with log_expansion:
                log_container = ui.column().classes('max-h-32 overflow-y-auto')

        with ui.card().classes('w-full p-3 mt-2') as action_card:
            ui.label('Your Turn').classes('font-bold mb-1')
            action_container = ui.row().classes('gap-2 flex-wrap')
        action_card.visible = False

    def update_board():
        img = game_ui.create_board_figure()
        if img:
            board_image.set_source(img)

    def update_scores():
        scores_container.clear()
        if game_ui.game:
            with scores_container:
                for idx, player in enumerate(game_ui.game.state.list_of_players):
                    color = PLAYER_COLORS[idx]
                    with ui.card().classes('p-2').style(f'border-left: 4px solid {color}'):
                        ui.label(f'P{idx+1}').classes('font-bold')
                        ui.label(f'{player.points} pts')
                        ui.label(f'{player.trains} trains').classes('text-sm text-gray-500')

    def update_hand():
        hand_container.clear()
        if game_ui.game:
            current_idx = game_ui.game.current_player_idx
            player = game_ui.game.state.list_of_players[current_idx]
            with hand_container:
                for card, count in player.hand.items():
                    if count > 0:
                        color = CARD_COLORS.get(card, '#666')
                        text_color = 'white' if card not in ['white_cards', 'yellow_cards'] else 'black'
                        card_name = card.replace('_cards', '').title()[:3]
                        ui.html(f'''
                            <div style="background:{color};color:{text_color};padding:4px 8px;
                                        border-radius:4px;font-weight:bold;text-align:center;font-size:12px">
                                {card_name}<br>{count}
                            </div>
                        ''')

    def update_faceup():
        faceup_container.clear()
        if game_ui.game:
            with faceup_container:
                for card in game_ui.game.state.face_up_cards:
                    color = CARD_COLORS.get(card, '#666')
                    text_color = 'white' if card not in ['white_cards', 'yellow_cards'] else 'black'
                    card_name = card.replace('_cards', '').title()[:3]
                    ui.html(f'''
                        <div style="background:{color};color:{text_color};padding:4px 8px;
                                    border-radius:4px;font-weight:bold;text-align:center;font-size:12px">
                            {card_name}
                        </div>
                    ''')

    def update_tickets():
        tickets_container.clear()
        if game_ui.game:
            current_idx = game_ui.game.current_player_idx
            player = game_ui.game.state.list_of_players[current_idx]
            with tickets_container:
                if not player.tickets:
                    ui.label('No tickets').classes('text-gray-500 text-sm')
                else:
                    for source, target, points in player.tickets:
                        with ui.row().classes('items-center gap-2'):
                            ui.html(f'''
                                <div style="background:#34495e;color:white;padding:4px 10px;
                                            border-radius:4px;font-size:12px;display:flex;align-items:center;gap:6px">
                                    <span style="font-weight:bold">{source}</span>
                                    <span>→</span>
                                    <span style="font-weight:bold">{target}</span>
                                    <span style="background:#e67e22;padding:2px 6px;border-radius:3px;margin-left:4px">
                                        {points} pts
                                    </span>
                                </div>
                            ''')

    def add_log(message):
        with log_container:
            ui.label(message).classes('text-sm')

    def show_actions(actions):
        action_card.visible = True
        action_container.clear()

        grouped = {'draw_card': [], 'claim_route': [], 'draw_tickets': [], 'build_station': [], 'draw_wild_card': [], 'keep_tickets': []}
        for a in actions:
            if a.type in grouped:
                grouped[a.type].append(a)

        with action_container:
            if grouped['keep_tickets']:
                player = game_ui.game.get_current_player()
                ui.label('Choose tickets to keep (must keep at least 1):').classes('font-bold mb-2')
                with ui.column().classes('gap-2'):
                    for a in grouped['keep_tickets']:
                        indices = [int(i) for i in a.source1.split(',')]
                        ticket_names = []
                        for i in indices:
                            t = player.pending_tickets[i]
                            ticket_names.append(f"{t[0]}→{t[1]} ({t[2]}pts)")
                        label = ', '.join(ticket_names)
                        ui.button(f'Keep: {label}',
                                 on_click=lambda act=a: select_action(act)).props('size=sm')
            else:
                with ui.row().classes('gap-2 flex-wrap'):
                    if grouped['draw_card']:
                        ui.button('Draw from Deck', on_click=lambda: select_action(grouped['draw_card'][0])).props('color=primary')

                    if grouped['draw_wild_card']:
                        ui.button('Draw Locomotive', on_click=lambda: select_action(grouped['draw_wild_card'][0])).props('color=purple')

                    if grouped['draw_tickets']:
                        ui.button('Draw Tickets', on_click=lambda: select_action(grouped['draw_tickets'][0])).props('color=orange')

                if grouped['claim_route']:
                    ui.label('Claim Route:').classes('font-bold mt-2')
                    with ui.row().classes('gap-2 flex-wrap'):
                        seen = set()
                        for a in grouped['claim_route']:
                            key = (a.source1, a.source2)
                            if key not in seen:
                                seen.add(key)
                                ui.button(f'{a.source1} → {a.source2}',
                                         on_click=lambda act=a: select_action(act)).props('size=sm')
                            if len(seen) >= 12:
                                break

    def hide_actions():
        action_card.visible = False
        action_container.clear()

    def select_action(action):
        game_ui.selected_action = action
        game_ui.waiting_for_human = False
        hide_actions()

    async def game_loop():
        while game_ui.running and not game_ui.game.game_over:
            actions = game_ui.game.get_legal_actions()
            if not actions:
                game_ui.game.current_player_idx = (game_ui.game.current_player_idx + 1) % game_ui.game.num_players
                continue

            current_idx = game_ui.game.current_player_idx
            player = game_ui.game.get_current_player()

            turn_label.set_text(f'Turn: Player {current_idx + 1}')
            update_hand()
            update_faceup()
            update_tickets()

            choose_fn = game_ui.get_choose_fn(current_idx)

            if choose_fn is None:
                game_ui.waiting_for_human = True
                game_ui.selected_action = None
                show_actions(actions)

                while game_ui.waiting_for_human and game_ui.running:
                    await asyncio.sleep(0.1)

                if not game_ui.running:
                    break

                action = game_ui.selected_action
            else:
                await asyncio.sleep(game_ui.ai_speed)
                action = choose_fn(game_ui.game.state, player, actions, game_ui.game.board)

            if action:
                add_log(f'P{current_idx + 1}: {action.type}')
                game_ui.game.step(action)
                update_board()
                update_scores()

        if game_ui.game and game_ui.game.game_over:
            status_label.set_text('Game Over!')
            scores = [p.points for p in game_ui.game.state.list_of_players]
            winner = scores.index(max(scores))
            add_log(f'Game Over! P{winner + 1} wins with {max(scores)} pts!')

    async def start_game_fn():
        game_ui.num_players = num_players_select.value
        game_ui.game_mode = mode_select.value
        game_ui.human_player_idx = human_select.value
        game_ui.ai_speed = speed_slider.value
        game_ui.running = True

        game_ui.game = Game(game_ui.num_players)
        game_ui.load_model()

        status_label.set_text('Game running...')
        log_container.clear()
        add_log('Game started!')

        update_board()
        update_scores()
        update_faceup()
        update_tickets()

        asyncio.create_task(game_loop())

    def stop_game_fn():
        game_ui.running = False
        game_ui.waiting_for_human = False
        status_label.set_text('Game stopped')
        hide_actions()

    state['start_game'] = start_game_fn
    state['stop_game'] = stop_game_fn


if __name__ in {"__main__", "__mp_main__"}:
    create_ui()
    ui.run(title='Ticket to Ride Europe', port=8080, reload=False)