import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool


class TTRModelV2(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        private_dim,
        hidden_dim=256,
        num_heads=4,
        num_gnn_layers=4,
        num_actions=1000,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            self.gnn_layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                    concat=True
                )
            )
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))

        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        combined_dim = hidden_dim * 3

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        private_state = data.private_state

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            x_new = conv(x, edge_index, edge_attr)
            x = norm(x + x_new)
            x = F.relu(x)

        if hasattr(data, 'batch') and data.batch is not None:
            graph_mean = global_mean_pool(x, data.batch)
            graph_max = global_max_pool(x, data.batch)
            num_graphs = data.batch.max().item() + 1
            private_state = private_state.view(num_graphs, -1)
        else:
            graph_mean = x.mean(dim=0, keepdim=True)
            graph_max = x.max(dim=0, keepdim=True)[0]
            if private_state.dim() == 1:
                private_state = private_state.unsqueeze(0)

        private_embedding = self.private_encoder(private_state)

        combined = torch.cat([graph_mean, graph_max, private_embedding], dim=1)

        fused = self.fusion(combined)

        policy_logits = self.policy_head(fused)
        value = self.value_head(fused)

        return policy_logits, value

    def get_action_probs(self, data, legal_action_mask):
        policy_logits, value = self.forward(data)

        masked_logits = policy_logits.clone()
        masked_logits[~legal_action_mask] = float('-inf')

        action_probs = F.softmax(masked_logits, dim=1)

        return action_probs, value


class TTRModelV2Large(TTRModelV2):
    def __init__(self, node_dim, edge_dim, private_dim, num_actions=1000):
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            private_dim=private_dim,
            hidden_dim=512,
            num_heads=8,
            num_gnn_layers=6,
            num_actions=num_actions,
            dropout=0.1
        )
