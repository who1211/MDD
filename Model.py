import torch
import torch.nn.functional as func
from Variable_chevconv import chef_conv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):


    def __init__(self,
                 num_features,
                 num_classes,
                 k_order,
                 dropout=.3):
        super(GCN, self).__init__()

        self.p = dropout

        self.conv1 = chef_conv(int(num_features), 128, K=k_order)
        self.conv2 = chef_conv(128, 64, K=k_order)
        self.conv3 = chef_conv(64, 32, K=k_order)

        self.lin1 = torch.nn.Linear(32, int(num_classes))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        x = func.relu(self.conv1(x, edge_index, edge_attr))
        x = func.dropout(x, p=self.p, training=self.training)
        x = func.relu(self.conv2(x, edge_index, edge_attr))
        x = func.dropout(x, p=self.p, training=self.training)
        x = torch.relu(self.conv3(x, edge_index, edge_attr))

        x = global_mean_pool(x, batch)
        x = self.lin1(x)

        return x




