import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from neural_network import get_evaluation
import scipy.sparse as sp


def test_gin(dataset_graph, x_feat, ys, train_mask, test_mask):
    class Net(torch.nn.Module):
        def __init__(self, edge_index, graph_features):
            super(Net, self).__init__()
            self.edge_index = edge_index
            self.graph_features = graph_features

            num_features = x_feat.shape[1]
            dim = 32

            nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            self.conv1 = GINConv(nn1)
            self.bn1 = torch.nn.BatchNorm1d(dim)

            nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            self.conv2 = GINConv(nn2)
            self.bn2 = torch.nn.BatchNorm1d(dim)

            nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            self.conv3 = GINConv(nn3)
            self.bn3 = torch.nn.BatchNorm1d(dim)

            nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            self.conv4 = GINConv(nn4)
            self.bn4 = torch.nn.BatchNorm1d(dim)

            nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            self.conv5 = GINConv(nn5)
            self.bn5 = torch.nn.BatchNorm1d(dim)

            self.fc1 = Linear(dim, dim)
            self.fc2 = Linear(dim, 1)

        def forward(self):
            x = self.graph_features
            x = F.relu(self.conv1(x, self.edge_index))
            x = self.bn1(x)
            x = F.relu(self.conv2(x, self.edge_index))
            x = self.bn2(x)
            x = F.relu(self.conv3(x, self.edge_index))
            x = self.bn3(x)
            x = F.relu(self.conv4(x, self.edge_index))
            x = self.bn4(x)
            x = F.relu(self.conv5(x, self.edge_index))
            x = self.bn5(x)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.fc2(x)
            return F.relu(x)

    return get_evaluation(dataset_graph, sp.csr_matrix(ys).todense().T,
                          x_feat, Net, train_mask, test_mask)
