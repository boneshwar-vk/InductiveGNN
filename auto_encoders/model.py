
"""
    1) Feature AE
    2) GAE
    3) Cluster AE
"""

import torch
from torch import nn, optim
from torch.nn import functional as F

from auto_encoders.gae.layers import GraphConvolution, GAT, GCN

# from torch_geometric.data import NeighborSampler
# from torch_geometric.nn import SAGEConv
import dgl
from dgl.nn import SAGEConv

class Feature_AE(nn.Module):
    ''' 
    Autoencoder for dimensional reduction
    Args:
        x: Tensor, mini-batch
        dim: int, feature dimension
    Return:
        self.decode(z): reconstructed input 
        z: feature encoding
    '''
    def __init__(self, dim):
        super(Feature_AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.relu(self.fc4(h3))

        # h3 = torch.sigmoid(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return z, self.decode(z)

class Graph_AE(nn.Module):
    def __init__(self, dim, embedding_size, 
                            gat_dropout=0,
                            multi_heads=2,
                            layers = 2,
                            hid_embed=16,agg_cls='mean',use_layer='GCNConv'):
        super(Graph_AE, self).__init__()

        # Layers defined  (dim = 128, hid_embed = 16, embedding size = 16)
        if use_layer == 'GAT':
            self.gat = GAT(num_of_layers = layers, 
                            num_heads_per_layer = [multi_heads, multi_heads], 
                            num_features_per_layer = [dim, hid_embed, embedding_size],
                            dropout=gat_dropout)
            
        elif use_layer == 'GCNConv':
            self.gcn = GCN(input_dim=dim, embedding_dim=embedding_size, feature_pre=True,hidden_dim=64,layer_num=layers)

        # elif use_layer == 'GATConv':
        #     self.gat_conv = GAT_(input_dim=dim, feature_pre=True,feature_dim=32,hidden_layers=[hid_embed*4]*(layers+1), output_dim=embedding_size*2,layer_num=layers)   

        # elif use_layer == 'SAGEConv':
        #     self.sage_conv = SAGE_(input_dim=dim, feature_pre=True,feature_dim=32,hidden_layers=[hid_embed*4]*(layers+1), output_dim=embedding_size*2,layer_num=layers)  

        elif use_layer == 'GCN':
            self.gc1 = GraphConvolution(dim, 32, 0, act=F.relu)
            self.gc2 = GraphConvolution(32, embedding_size*2, 0, act=lambda x: x)
            self.gc3 = GraphConvolution(embedding_size*2, embedding_size*4, 0, act=lambda x: x)
            self.gc4 = GraphConvolution(embedding_size*4, embedding_size*2, 0, act=lambda x: x)
        
        elif use_layer == 'SAGE':
            self.sage1 = SAGEConv(dim, hid_embed,agg_cls)
            self.sage2 = SAGEConv(hid_embed, embedding_size*2,agg_cls)
            # self.sage3 = SAGEConv(hid_embed*2, embedding_size*2,agg_cls)

        elif use_layer == 'PGNN':
            pass
        
        else:
            raise Exception("Invalid layer for GAE")

        self.mu_layer = nn.Linear(embedding_size*2, embedding_size)
        self.logvar_layer = nn.Linear(embedding_size*2, embedding_size)

        self.decode = InnerProductDecoder(0, act=lambda x: x)

    def encode_gat(self, in_nodes_features, edge_index):
        return self.gat((in_nodes_features, edge_index))
    
    def encode_gcn_conv(self, in_nodes_features, edge_index):
        return self.gcn(in_nodes_features, edge_index)[0], self.gcn(in_nodes_features, edge_index)[1]
    
    # def encode_gat_conv(self, in_nodes_features, edge_index):
    #     return self.gat_conv(in_nodes_features, edge_index)
    
    def encode_sage_conv(self, in_nodes_features, edge_index):
        return self.sage_conv(in_nodes_features, edge_index)

    def encode_gae(self, x, adj):
        x = self.gc1(x, adj)
        F.relu(torch.relu(x))
        x = self.gc2(x, adj)
        F.relu(torch.relu(x))
        x = self.gc3(x,adj)
        F.relu(torch.relu(x))
        x = self.gc4(x,adj)
        F.relu(torch.relu(x))
        return self.mu_layer(x), self.logvar_layer(x)

    def encode_sage(self, x, adj):
        x = self.sage1(x, adj)
        x = F.relu(torch.relu(x))
        x = self.sage2(x, adj)
        x = F.relu(torch.relu(x))
        # x = self.sage3(x,adj)
        # x = F.relu(torch.relu(x))
        return self.mu_layer(x), self.logvar_layer(x)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, in_nodes_features, edge_index, encode=False, use_GAT=True, use_layer = 'GCN'):
        # [0] just extracts the node_features part of the data (index 1 contains the edge_index)
        gae_info= None
        
        if use_layer == 'GAT':
            out_nodes_features = self.encode_gat(in_nodes_features, edge_index)[0]

        elif use_layer == 'GCNConv':
            out_nodes_features = self.encode_gcn_conv(in_nodes_features, edge_index)[0]

        elif use_layer == 'SAGE':
            gae_info = self.encode_sage(in_nodes_features, edge_index)
            out_nodes_features = self.reparameterize(*gae_info)

        elif use_layer == 'SAGEConv':
            out_nodes_features = self.encode_sage_conv(in_nodes_features, edge_index)[0]

        # elif use_layer == 'GATConv':
        #     out_nodes_features = self.encode_gat_conv(in_nodes_features, edge_index)[0]
                       
        else:
            gae_info = self.encode_gae(in_nodes_features, edge_index)
            out_nodes_features = self.reparameterize(*gae_info)
        
        recon_graph = self.decode(out_nodes_features)
        return out_nodes_features, gae_info, recon_graph

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Cluster_AE(Feature_AE):
    def __init__(self, dim):
        super().__init__(dim)
