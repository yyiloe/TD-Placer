import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import numpy as np
import random

class GATEncoder(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers=2, heads=2):
        super().__init__()
        self.gat_layers = nn.ModuleList()

        self.gat_layers.append(GATConv(in_feats, hidden_feats, heads=heads, concat=True))

        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_feats * heads, hidden_feats, heads=heads, concat=True))

        self.gat_layers.append(GATConv(hidden_feats * heads, hidden_feats, heads=1, concat=True))

    def forward(self, x, edge_index, batch):
        x = x.clone()
        x[:, 0] = x[:, 0] / 100.0
        x[:, 1] = x[:, 1] / 400.0
        x[:, -1] = x[:, -1] / 100.0

        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)

        return x


class MLP(nn.Module):
    def __init__(self, hidden_feats, net_feats, out_feats, depth=2):
        super(MLP, self).__init__()
        self.fc_hg_transform = nn.Linear(hidden_feats, net_feats)
        self.alpha = nn.Parameter(torch.tensor(0.05))
        self.beta = nn.Parameter(torch.tensor(0.05))
        input_dim = net_feats + 3 + 7
        #input_dim =  7
        layers = []
        layers.append(nn.Linear(input_dim, hidden_feats))
        layers.append(nn.ReLU())

        for _ in range(depth): 
            layers.append(nn.Linear(hidden_feats, hidden_feats))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_feats, input_dim))
        layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.norm_hg = nn.LayerNorm(net_feats)

        self.out = nn.Sequential(
            nn.Linear(input_dim, hidden_feats),
            nn.ReLU(),
            nn.Linear(hidden_feats, out_feats)
        )

        self.bn1 = nn.BatchNorm1d(input_dim)

    def forward(self, hg, nets_f, nodes_f):
        hg_transformed = self.fc_hg_transform(hg)
        hg_transformed = self.norm_hg(hg_transformed)
        hg_transformed = F.relu(hg_transformed)
        hg_transformed = self.alpha * hg_transformed   # self.alpha = nn.Parameter(torch.tensor(0.1))
        nets_f = self.beta * nets_f

        all_features = torch.cat((hg_transformed, nets_f, nodes_f), dim=1)
        #all_features = torch.cat((hg_transformed, nodes_f), dim=1)
        #all_features = nodes_f
        all_features = self.bn1(all_features)  

        out = self.mlp(all_features)  

        return self.out(out)
    

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_attn=None):
        super().__init__()
        if d_attn is None:
            d_attn = d_model

        self.query_proj = nn.Linear(d_model, d_attn)
        self.key_proj = nn.Linear(d_model, d_attn)
        self.value_proj = nn.Linear(d_model, d_attn)
        self.out_proj = nn.Linear(d_attn, d_model)

    def forward(self, query, key, value):
        """
        Args:
            query: Tensor, shape [B, d_model]
            key: Tensor, shape [N, d_model]
            value: Tensor, shape [N, d_model]

        Returns:
            out: Tensor, shape [B, d_model]
        """
        Q = self.query_proj(query).unsqueeze(1)  # [B, 1, d_attn]

        K = self.key_proj(key).unsqueeze(0)      # [1, N, d_attn]
        V = self.value_proj(value).unsqueeze(0)  # [1, N, d_attn]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # [B, 1, N]
        attn = F.softmax(scores, dim=-1)                                     # [B, 1, N]

        context = torch.matmul(attn, V)  # [B, 1, d_attn]
        context = context.squeeze(1)     # [B, d_attn]

        out = self.out_proj(context)     # [B, d_model]

        return out

class NetLogicMLP(nn.Module):
    def __init__(self, hidden_feats, net_feats, out_feats, depth=2):
        super(NetLogicMLP, self).__init__()
        #self.crossAttention = CrossAttention(hidden_feats)
        self.mlp = MLP(hidden_feats, net_feats, out_feats, depth=depth)
        mlp_layers = []
        for _ in range(depth):
            mlp_layers.append(nn.Linear(hidden_feats, hidden_feats))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(hidden_feats, out_feats))
        self.mlp_logic = nn.Sequential(*mlp_layers)

    def forward(self, h, h_in, nets_f, nodes_f):
        #hg_out = global_mean_pool(h, batch)
        out = self.mlp(h, nets_f, nodes_f)


        #hg_in = global_mean_pool(h_in, batch_in)
        #hg = hg_in + self.crossAttention(hg_in, h, h)
        hg = h + h_in
        out_logic = self.mlp_logic(hg)
        return torch.cat([out, out_logic], dim=1)
    

class NetLogicDelayPredict(nn.Module):
    def __init__(self, in_feats, hidden_feats, net_feats, out_feats, depth=2):
        super(NetLogicDelayPredict, self).__init__()
        self.netLogicMLP = NetLogicMLP(hidden_feats, net_feats, out_feats, depth)
        self.gcn = GATEncoder(in_feats, hidden_feats, depth, heads=2)
    def forward(self, x, edge_index, batch, nets_f, in_x, batch_in, in_edge_index, nodes_f):
        h = self.gcn(x, edge_index, batch)
        h_in = self.gcn(in_x, in_edge_index, batch_in)
        return self.netLogicMLP(h, h_in, nets_f, nodes_f)



def set_seed(seed=42):
    random.seed(seed)                      
    np.random.seed(seed)                  
    torch.manual_seed(seed)                
    torch.cuda.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed)       
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False     

set_seed(42)


in_feats = 2
hidden_feats = 3
net_feats = 4
out_feats = 1
batch_size = 2

nodes1 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
edge_index1 = [[0, 1, 2], [1, 2, 0]]
data1 = Data(torch.tensor(nodes1, dtype=torch.float),
             torch.tensor(edge_index1, dtype=torch.long))

nodes2 = [[0.5, 0.5], [0.2, 0.8]]
edge_index2 = [[0], [1]]
data2 = Data(torch.tensor(nodes2, dtype=torch.float),
             torch.tensor(edge_index2, dtype=torch.long))

batch = Batch.from_data_list([data1, data2])



nodes1_in = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
edge_index1_in = [[0, 1, 2], [1, 2, 0]]
data1_in = Data(torch.tensor(nodes1, dtype=torch.float),
             torch.tensor(edge_index1, dtype=torch.long))

nodes2_in = [[0.5, 0.5], [0.2, 0.8]]
edge_index2_in = [[0], [1]]
data2_in = Data(torch.tensor(nodes2, dtype=torch.float),
             torch.tensor(edge_index2, dtype=torch.long))

batch_in = Batch.from_data_list([data1_in, data2_in])


nets_f = torch.randn(batch_size, 3)
nodes_f = torch.randn(batch_size, 7)

model = NetLogicDelayPredict(in_feats=in_feats,
                             hidden_feats=hidden_feats,
                             net_feats=net_feats,
                             out_feats=out_feats)
model.eval()

parameters = model.parameters()
    
total_num = [p.numel() for p in model.parameters()]
trainable_num = [p.numel() for p in model.parameters() if p.requires_grad]

print("Total parameters: {}".format(sum(total_num)))
print("Trainable parameters: {}".format(sum(trainable_num)))
whetherIn = torch.tensor([1, 0])
print(batch.x)
print(batch.edge_index)
print(batch.batch)

out= model(batch.x, batch.edge_index, batch.batch, nets_f, batch_in.x, batch_in.batch, batch_in.edge_index, nodes_f)
print(out) 
