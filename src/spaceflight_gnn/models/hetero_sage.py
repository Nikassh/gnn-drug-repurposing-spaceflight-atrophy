import torch
import torch.nn as nn
import torch.nn.functional as F

class HeteroGraphSAGEPredictor(nn.Module):
    """
    Heterogeneous GraphSAGE Model for Drug-Gene Link Prediction.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.gene_lin = nn.Linear(in_channels, hidden_channels)
        self.drug_lin = nn.Linear(in_channels, hidden_channels)
        self.out_lin = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, gene_x: torch.Tensor, drug_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_gene = F.relu(self.gene_lin(gene_x))
        h_drug = F.relu(self.drug_lin(drug_x))
        
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h_gene[src], h_drug[dst]], dim=-1)
        score = self.out_lin(edge_feat)
        return torch.sigmoid(score)
