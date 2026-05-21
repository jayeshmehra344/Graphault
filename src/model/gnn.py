import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv

class CodeRiskGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CodeRiskGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
        
def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)
    
    
if __name__ == "__main__":
    model = CodeRiskGNN(input_dim=4, hidden_dim=64, output_dim=1)
    print(model)