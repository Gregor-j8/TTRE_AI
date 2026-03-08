import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class TTRModelV5(nn.Module):
    def __init__(
        self,
        node_dim,
        edge_dim,
        private_dim,
        hidden_dim=400,
        num_gnn_layers=5,
        num_actions=1000,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()

        for _ in range(num_gnn_layers):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
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
        private_state = data.private_state

        x = self.node_encoder(x)

        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            x_new = conv(x, edge_index)
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
