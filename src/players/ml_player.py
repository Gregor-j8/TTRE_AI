import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Callable

MODEL_CACHE = {}


def load_v1_model(model_path: str, board):
    from src.ml.first_iteration.model import TTRModel
    from src.ml.first_iteration.state_encoder import StateEncoder

    encoder = StateEncoder(board)
    model = TTRModel(
        node_dim=encoder.get_node_feature_dim(),
        edge_dim=encoder.get_edge_feature_dim(),
        private_dim=encoder.get_private_state_dim(),
        hidden_dim=64,
        num_actions=100,
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, encoder


def load_v3_model(model_path: str, board):
    from src.ml.v3.model import TTRModelV3
    from src.ml.v5.state_encoder import StateEncoderV5

    encoder = StateEncoderV5(board)
    model = TTRModelV3(
        node_dim=encoder.get_node_feature_dim(),
        edge_dim=encoder.get_edge_feature_dim(),
        private_dim=encoder.get_private_state_dim(),
        hidden_dim=400,
        num_gnn_layers=5,
        num_actions=1000,
        dropout=0.0
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, encoder


def load_v4_model(model_path: str, board):
    from src.ml.v4.model import TTRModelV4
    from src.ml.v5.state_encoder import StateEncoderV5

    encoder = StateEncoderV5(board)
    model = TTRModelV4(
        node_dim=encoder.get_node_feature_dim(),
        edge_dim=encoder.get_edge_feature_dim(),
        private_dim=encoder.get_private_state_dim(),
        hidden_dim=704,
        num_gnn_layers=6,
        num_actions=1000,
        dropout=0.0
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, encoder


def load_v5_model(model_path: str, board):
    from src.ml.v5.model import TTRModelV5
    from src.ml.v5.state_encoder import StateEncoderV5

    encoder = StateEncoderV5(board)
    model = TTRModelV5(
        node_dim=encoder.get_node_feature_dim(),
        edge_dim=encoder.get_edge_feature_dim(),
        private_dim=encoder.get_private_state_dim(),
        hidden_dim=400,
        num_gnn_layers=5,
        num_actions=1000,
        dropout=0.0
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, encoder


def create_ml_choose(model_path: str, loader_fn) -> Callable:
    def ml_choose(game_state, player, legal_actions, board):
        if model_path not in MODEL_CACHE:
            MODEL_CACHE[model_path] = loader_fn(model_path, board)

        model, encoder = MODEL_CACHE[model_path]

        player_idx = game_state.list_of_players.index(player)
        data = encoder.encode_state(game_state, player_idx)

        with torch.no_grad():
            policy_logits, _ = model(data)

        action_probs = F.softmax(policy_logits.squeeze(), dim=-1)
        max_idx = min(len(action_probs), len(legal_actions))

        if max_idx == 0:
            return legal_actions[0]

        legal_probs = action_probs[:max_idx]
        best_legal_idx = legal_probs.argmax().item()

        if best_legal_idx < len(legal_actions):
            return legal_actions[best_legal_idx]
        return legal_actions[0]

    return ml_choose


MODEL_DIR = Path(__file__).parent.parent.parent / "model_data"

v5_best_choose = create_ml_choose(str(MODEL_DIR / "v5" / "model_v5_best.pt"), load_v5_model)
v5_final_choose = create_ml_choose(str(MODEL_DIR / "v5" / "model_v5_final.pt"), load_v5_model)
v5_mixed_choose = create_ml_choose(str(MODEL_DIR / "v5" / "model_v5_mixed_best.pt"), load_v5_model)
v4_best_choose = create_ml_choose(str(MODEL_DIR / "v4" / "model_v4_BEST_64pct_heuristic.pt"), load_v4_model)
v3_final_choose = create_ml_choose(str(MODEL_DIR / "v3_good" / "model_v3_Final.pt"), load_v3_model)
v1_best_choose = create_ml_choose(str(MODEL_DIR / "v1_models" / "first_iteration" / "model_100pct_heuristic_6016.pt"), load_v1_model)
