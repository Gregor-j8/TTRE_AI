import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class TTRModel(nn.Module):
    def __init__(self, node_dim, edge_dim, private_dim, hidden_dim=64, num_actions=100):
        super().__init__()

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.private_encoder = nn.Sequential(
            nn.Linear(private_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        combined_dim = hidden_dim + hidden_dim

        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        private_state = data.private_state

        x = self.node_encoder(x)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        if hasattr(data, 'batch') and data.batch is not None:
            graph_embedding = global_mean_pool(x, data.batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)

        if private_state.dim() == 1:
            private_state = private_state.unsqueeze(0)
        private_embedding = self.private_encoder(private_state)

        combined = torch.cat([graph_embedding, private_embedding], dim=1)

        policy_logits = self.policy_head(combined)
        value = self.value_head(combined)

        return policy_logits, value

    def get_action_probs(self, data, legal_action_mask):
        policy_logits, value = self.forward(data)

        masked_logits = policy_logits.clone()
        masked_logits[~legal_action_mask] = float('-inf')

        action_probs = F.softmax(masked_logits, dim=1)

        return action_probs, value
