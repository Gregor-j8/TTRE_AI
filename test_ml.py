from src.game import Game
from src.ml.state_encoder import StateEncoder
from src.ml.model import TTRModel

def test_encoder_and_model():
    game = Game(2)

    encoder = StateEncoder(game.board)
    print(f"Number of cities: {encoder.num_cities}")
    print(f"Number of edges: {encoder.num_edges}")
    print(f"Node feature dim: {encoder.get_node_feature_dim()}")
    print(f"Edge feature dim: {encoder.get_edge_feature_dim()}")
    print(f"Private state dim: {encoder.get_private_state_dim()}")

    data = encoder.encode_state(game.state, player_idx=0)
    print(f"\nEncoded state:")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge index shape: {data.edge_index.shape}")
    print(f"  Edge features shape: {data.edge_attr.shape}")
    print(f"  Private state shape: {data.private_state.shape}")

    model = TTRModel(
        node_dim=encoder.get_node_feature_dim(),
        edge_dim=encoder.get_edge_feature_dim(),
        private_dim=encoder.get_private_state_dim(),
        hidden_dim=112,
        num_actions=100
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    policy_logits, value = model(data)
    print(f"\nModel output:")
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value estimate: {value.item():.4f}")

    print("\nTest passed!")

if __name__ == "__main__":
    test_encoder_and_model()
