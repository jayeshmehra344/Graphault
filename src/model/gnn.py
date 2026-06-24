import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation


class CodeRiskGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        if pooling == "attention":
            # conv3 produces hidden_dim embeddings; gate + classifier handle graph-level logit
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.attn_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
            self.classifier = nn.Linear(hidden_dim, 1)
        else:
            self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        if self.pooling == "attention":
            x = self.attn_pool(x, batch)   # [n_graphs, hidden_dim]
            return self.classifier(x)       # [n_graphs, 1]
        if batch is not None:
            return global_mean_pool(x, batch)  # [n_graphs, 1]
        return x  # [total_nodes, 1] — legacy path, no batch provided


if __name__ == "__main__":
    model = CodeRiskGNN(input_dim=4, hidden_dim=64, output_dim=1)
    print(model)
    model_attn = CodeRiskGNN(input_dim=4, hidden_dim=64, output_dim=1, pooling="attention")
    print(model_attn)
