
"""
    1) Feature AE
    2) GAE
    3) Cluster AE
"""

import torch
from torch import nn, optim
from torch.nn import functional as F

from auto_encoders.gae.layers import GraphConvolution, GAT, GCN, PGNN_layer, PGNN

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
                            hid_embed=16,agg_cls='mean',use_layer='GCN'):
        super(Graph_AE, self).__init__()

        # Layers defined  (dim = 128, hid_embed = 16, embedding size = 16)
        if use_layer == 'GAT':
            self.gat = GAT(num_of_layers = layers, 
                            num_heads_per_layer = [multi_heads, multi_heads], 
                            num_features_per_layer = [dim, hid_embed, embedding_size],
                            dropout=gat_dropout)
            
        elif use_layer == 'GCNConv':
            self.gcn = GCN(input_dim=dim, embedding_dim=embedding_size, feature_pre=True,hidden_dim=64,layer_num=layers)


        elif use_layer == 'GCN':
            self.gc1 = GraphConvolution(dim, 32, 0, act=F.relu)
            self.gc2 = GraphConvolution(32, embedding_size*2, 0, act=lambda x: x)
            self.gc3 = GraphConvolution(embedding_size*2, embedding_size*4, 0, act=lambda x: x)
            self.gc4 = GraphConvolution(embedding_size*4, embedding_size*2, 0, act=lambda x: x)
        
        else:
            raise Exception("Invalid layer for Non Inductive GAE")

        self.mu_layer = nn.Linear(embedding_size*2, embedding_size)
        self.logvar_layer = nn.Linear(embedding_size*2, embedding_size)

        self.decode = InnerProductDecoder(0, act=lambda x: x)

    def encode_gat(self, in_nodes_features, edge_index):
        return self.gat((in_nodes_features, edge_index))
    
    def encode_gcn_conv(self, in_nodes_features, edge_index):
        return self.gcn(in_nodes_features, edge_index)[0], self.gcn(in_nodes_features, edge_index)[1]

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
    

        elif 'GCN':
            gae_info = self.encode_gae(in_nodes_features, edge_index)
            out_nodes_features = self.reparameterize(*gae_info)
            
        else:
            raise Exception("Invalid layer for Graph AE")
        
        recon_graph = self.decode(out_nodes_features)
        return out_nodes_features, gae_info, recon_graph
    
class InductiveGraph_AE(nn.Module):

    def __init__(self, input_dim,
                 hid_embed=16,
                 embedding_size=16,
                 agg_cls='mean',
                 use_layer='SAGE',
                 dropout=0,
                 layers = 2,):
        super(InductiveGraph_AE, self).__init__()
        self.embedding_size = embedding_size

        # Layers defined  (dim = 128, hid_embed = 16, embedding size = 16)
        if use_layer == 'SAGE': 
            if layers == 2:
                self.sage1 = SAGEConv(input_dim, hid_embed,agg_cls)
                self.sage2 = SAGEConv(hid_embed, embedding_size*2,agg_cls)
                # self.sage3 = SAGEConv(hid_embed*2, embedding_size*2,agg_cls)
                self.dropout = nn.Dropout(0.4)
            # self.sage3 = SAGEConv(hid_embed*2, embedding_size*2,agg_cls)
        
        elif use_layer == 'PGNN':
            self.dropout = nn.Dropout(0.4)
            self.linear_pre = nn.Linear(input_dim, hid_embed*2)
            self.pgnn1 = PGNN_layer(hid_embed*2, embedding_size*4)
            self.pgnn2 = PGNN_layer(embedding_size*4, embedding_size*2)
            
        else:
            raise Exception("Invalid layer for Inductive GAE")
        

        
        self.mu_layer = nn.Linear(embedding_size*2, embedding_size)
        self.logvar_layer = nn.Linear(embedding_size*2, embedding_size)

        self.decode = InnerProductDecoder(0, act=lambda x: x)


    def encode_sage(self, graph, x):
        h = self.sage1(graph, x)
        h = F.relu(h)
        h = self.sage2(graph,h)
        h = F.relu(h)
        return self.mu_layer(h), self.logvar_layer(h)
    
    def encode_pgnn(self,data,concatenate=False):
        x = data["graph"].ndata["feat"]
        x_old = x
        graph = data["graph"]
        x = self.linear_pre(x)
        x_position, x = self.pgnn1(
            graph, x, data["anchor_eid"], data["dists_max"])
        x = self.dropout(x)
        x_position, x = self.pgnn2(
            graph, x, data["anchor_eid"], data["dists_max"])
        if concatenate:
            x_new = torch.cat((x_old,x_position),dim=-1)
            final_layer = nn.Linear(x_new.shape[1],self.embedding_size*2)
            x_final = final_layer(x_new)
            x_final = F.relu(x_final)
                # x_final = F.normalize(x_final, p=2, dim=-1)
            return self.mu_layer(x_final), self.logvar_layer(x_final)

        else:
            mu_layer = nn.Linear(x_position.shape[1], self.embedding_size)
            log_var_layer = nn.Linear(x_position.shape[1], self.embedding_size)
            # x_final = F.normalize(x_final, p=2, dim=-1)
            return mu_layer(F.relu(x_position)), log_var_layer(F.relu(x_position))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, blocks, in_features=None, encode=False, use_GAT=True, use_Graphlayer = 'SAGE'):
        if use_Graphlayer == 'SAGE':
            gae_info = self.encode_sage(blocks, in_features)
            out_nodes_features = self.reparameterize(*gae_info)

        elif use_Graphlayer == 'PGNN':
            gae_info = self.encode_pgnn(blocks)
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
