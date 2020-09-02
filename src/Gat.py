import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from neural_network import get_evaluation
import scipy.sparse as sp


def test_gat(dataset_graph, x_feat, ys, train_mask, test_mask):

    class Net(torch.nn.Module):
        def __init__(self, edge_index, graph_features):
            super(Net, self).__init__()
            self.edge_index = edge_index
            self.graph_features = graph_features
            self.conv1 = GATConv(graph_features.shape[1], 8, heads=8, dropout=0.6)
            self.conv2 = GATConv(8 * 8, 1, heads=8, concat=False,
                                 dropout=0.6)
            self.lin = torch.nn.Linear(1, 1)

        def forward(self):
            x = F.dropout(self.graph_features, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, self.edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, self.edge_index)
            return F.relu(x)

    return get_evaluation(dataset_graph, sp.csr_matrix(ys).todense().T,
                          x_feat, Net, train_mask, test_mask)
