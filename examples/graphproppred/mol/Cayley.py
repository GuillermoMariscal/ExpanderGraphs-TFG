from primePy import primes
import networkx as nx
import numpy as np
import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

def compute_n(n_vertices):
    argmin = -1
    count = 2
    while(argmin < n_vertices):
        prod = 1
        primes_list = primes.between(2,count+1)
        primes_list.append(2)
        for p in primes_list:
            if count % p == 0:
                prod = prod * (1 - 1/(p**2))
        argmin = count**3 * prod
        count += 1
    return count

def cayley_graph(n):
    index = 0
    mat1 = np.array([[1, 1], [0, 1]])
    mat2 = np.array([[1, 0], [1, 1]])
    id = np.array([[1, 0], [0, 1]])
    Sn = [mat1, mat2]
    visited_nodes = []
    added_nodes = []
    nodes_to_visit = [id]
    G = nx.Graph()
    G.add_node(index,matrix=id)
    added_nodes.append(id)

    while nodes_to_visit:
        curr_node = nodes_to_visit.pop(0)
        visited_nodes.append(curr_node)
        for mat in Sn:
            new_node = np.dot(curr_node, mat) % n
            is_new_node = True
            for added_node in added_nodes:
                if np.array_equal(new_node, added_node):
                    is_new_node = False
            if is_new_node:
                index += 1
                G.add_node(index, matrix=new_node)
                added_nodes.append(new_node)
            is_visited = False
            for visited_node in visited_nodes:
                if np.array_equal(new_node, visited_node):
                    is_visited = True
            if not is_visited:
                G.add_edge(tuple(curr_node.flatten()), tuple(new_node.flatten()))
                is_in_nodes_to_visit = False
                for node_to_visit in nodes_to_visit:
                    if np.array_equal(new_node, node_to_visit):
                        is_in_nodes_to_visit = True
                if not is_in_nodes_to_visit:
                    nodes_to_visit.append(new_node)
    return G

def get_ad_matrix(G):
    return nx.adjacency_matrix(G)

#Classe equivalent a la definida per la GIN al conv.py però aplicant la matriu d'adjacència del Cayley Graph
class CayleyConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(CayleyConv, self).__init__(aggr="add")
        self.n_value = compute_n(emb_dim)

        G = cayley_graph(self.n_value)
        adj_matrix = get_ad_matrix(G)
        adj_tensor = torch.tensor(adj_matrix.toarray(), dtype=torch.int)
        self.edge_index = adj_tensor.nonzero(as_tuple=False).t()

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):

        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        edge_attr = edge_attr[:x_j.shape[0]]
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out