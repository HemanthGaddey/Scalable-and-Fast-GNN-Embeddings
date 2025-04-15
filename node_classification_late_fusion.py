import torch
import torch.nn.functional as F
from torch_geometric.datasets import Flickr, Coauthor, CoraFull, Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import copy
from tqdm import tqdm
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#-------------------SEED------------------
SEED=548
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import sklearn
sklearn.utils.check_random_state(SEED)
#-----------------------------------------
def edge_index_to_adj(edge_index, num_nodes):
    values = torch.ones(edge_index.shape[1])
    adj_matrix = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    return adj_matrix.to_dense()

dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]  # The dataset contains a single graph
print(data)
adj_matrix = edge_index_to_adj(data.edge_index, len(data.x))

features = torch.tensor(data.x)
# finding feature similarity across all the nodes via dot product
similarities = (features@features.T)

# making the diagnol elements to zero as the max similarity is obtained with a same
similarities = similarities * (torch.eye(len(similarities)) == 0).long()

maxi=similarities.max()
print(f"maximum cross similarity:{maxi}")

# normalizing similarty to lie in range [0,1]
similarities=torch.div(similarities, maxi)

alpha = 0.25
print(f"alpha:{alpha}")
max_similarities = (similarities > alpha).long()

attr_edges = torch.nonzero(max_similarities, as_tuple=False).T
# print(attr_edges)
#---------------------mapping attr edges to virtual indices-----------------
connected_nodes = torch.unique(attr_edges)
map_attr_edges={idx.item():i for i, idx in enumerate(connected_nodes)}
rev_map_attr_edges={j:i for i,j in map_attr_edges.items()}
mapped_attr_edges = torch.tensor([map_attr_edges[idx.item()] for idx in attr_edges.flatten()], device=attr_edges.device).reshape(attr_edges.shape)

# print(mapped_attr_edges)

#--------------------------------------------------------------------------

# Find all nodes in the graph
all_nodes = torch.arange(len(data.x))

# Find disconnected nodes (nodes not present in attr_edges)
disconnected_nodes = torch.tensor([node for node in all_nodes if node not in connected_nodes])

print(f"Connected nodes ({len(connected_nodes)}))")
print(f"Disconnected nodes ({len(disconnected_nodes)})")

print(f"Total edges in structure network created: {data.edge_index.shape[1]}")
print(f"Total edges in attribute network created: {attr_edges.shape[1]}")

#--------------------------------------------------------LOUVAINNE-----------------------------------------------------------

save_edges = lambda args: pd.DataFrame(args[0].T).to_csv(args[1],index=None, header=None, sep=' ', mode='a')
def get_embeddings(
    edges, 
    edges_name="edgelist.txt", 
    k=128,  # Embedding dimension
    a=0.01,  # damping parameter
    partition=1,
    output=False
):
    '''
    "partition": the partition algorithm to use, default is 1.
    0: random bisection
    1: Louvain partition
    2: Louvain first-level partition
    3: Label Propagation partition
    '''
    open(edges_name,'w').close()
    save_edges((edges, edges_name))

    if not os.path.exists('./hierarchy.txt'):
        os.mknod('./hierarchy.txt')
    else:
        open('hierarchy.txt','w').close()
    if not os.path.exists('./vectors.txt'):
        os.mknod('./vectors.txt')
    else:
        open('vectors.txt','w').close()
    
    # do hierarchical clustering using Louvain algorithm
    endstr = '> /dev/null 2>&1'
    if(output):
        endstr = ''
    os.system(f'./LouvainNE/recpart ./{edges_name} ./hierarchy.txt {partition} {endstr}')
    
    # obtain node embedding of each node at every hierarchy
    os.system(f'./LouvainNE/hi2vec {k} {a} ./hierarchy.txt ./vectors.txt {endstr}')
    
    # Path to your output node embeddings text file
    file_path = 'vectors.txt'

    data_ = np.loadtxt(file_path)
    data_tensor = torch.from_numpy(data_)
    # The first column contains node IDs
    node_ids = data_tensor[:, 0].to(torch.int)
    # The remaining columns contain embeddings
    embeddings = data_tensor[:, 1:]
    
    return node_ids, embeddings

#------------------------------------------------------NODE CLASSIFICATION-----------------------------------------
def evaluate_node_classification(aa_node_ids, aa_embeddings, data):
    """
    Evaluates multi-class node classification using Logistic Regression.

    Args:
        aa_node_ids: List of node IDs.
        aa_embeddings: Corresponding node embeddings.
        data: PyTorch Geometric data object containing labels and masks.

    Returns:
        Micro-F1 and Macro-F1 scores.
    """

    # Map node IDs to indices in embeddings
    node_to_idx = {node_id.item(): idx for idx, node_id in enumerate(aa_node_ids)}
    # Reorder embeddings to match the label ordering
    ordered_embeddings = np.zeros((len(data.y), aa_embeddings.shape[1]))
    for i in range(len(data.y)):
        ordered_embeddings[i] = aa_embeddings[node_to_idx[i]]

    # Convert masks to NumPy arrays
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    # Prepare train and test data
    X_train, X_test = ordered_embeddings[train_mask], ordered_embeddings[test_mask]
    y_train, y_test = data.y[train_mask].numpy(), data.y[test_mask].numpy()

    # Train Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Compute Micro-F1 and Macro-F1 scores
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }

def evaluate_node_classification_latefusion(aa_node_ids_a, aa_embeddings_a, aa_node_ids_b, aa_embeddings_b, data, agg_type):
    """
    Evaluates multi-class node classification using Logistic Regression.

    Args:
        aa_node_ids: List of node IDs.
        aa_embeddings: Corresponding node embeddings.
        data: PyTorch Geometric data object containing labels and masks.

    Returns:
        Micro-F1 and Macro-F1 scores.
    """

    # Map node IDs to indices in embeddings
    node_to_idx_a = {node_id.item(): idx for idx, node_id in enumerate(aa_node_ids_a)}
    node_to_idx_b = {node_id.item(): idx for idx, node_id in enumerate(aa_node_ids_b)}

    # Reorder embeddings to match the label ordering
    ordered_embeddings_a = np.zeros((len(data.y), aa_embeddings_a.shape[1]))
    ordered_embeddings_b = np.zeros((len(data.y), aa_embeddings_b.shape[1]))

    for i in range(len(data.y)):
        ordered_embeddings_a[i] = aa_embeddings_a[node_to_idx_a[i]]
        ordered_embeddings_b[i] = aa_embeddings_b[node_to_idx_b[i]]

    ordered_embeddings = np.zeros((len(data.y), aa_embeddings_a.shape[1]))
    if agg_type == 'sum':
        ordered_embeddings = ordered_embeddings_a+ordered_embeddings_b
    elif agg_type == 'concat':
        if isinstance(ordered_embeddings_a, np.ndarray):
            ordered_embeddings_a = torch.tensor(ordered_embeddings_a, dtype=torch.float32)

        if isinstance(ordered_embeddings_b, np.ndarray):
            ordered_embeddings_b = torch.tensor(ordered_embeddings_b, dtype=torch.float32)
        ordered_embeddings = torch.cat([ordered_embeddings_a, ordered_embeddings_b],dim=1)
    elif agg_type == 'avg':
        ordered_embeddings = (ordered_embeddings_a+ordered_embeddings_b)/2
    else: 
        raise ValueError('Invalid agg_type, choose from sum/concat/avg')
    # Convert masks to NumPy arrays
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    # Prepare train and test data
    X_train, X_test = ordered_embeddings[train_mask], ordered_embeddings[test_mask]
    y_train, y_test = data.y[train_mask].numpy(), data.y[test_mask].numpy()

    # Train Logistic Regression classifier
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Compute Micro-F1 and Macro-F1 scores
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }

#--------------------------------------EVALUATION-----------------------------------
def append(idxs, emb, data):
    l=data.x.shape[0]
    ids=[i for i in range(data.x.shape[0])]
    final_ids=copy.deepcopy(idxs)
    final_ids=final_ids.tolist()
    final_emb = copy.deepcopy(emb)
    final_emb=final_emb.tolist()
    for i in ids:
        if i in final_ids:
            continue
        else:
            final_ids.append(i)
            final_emb.append(torch.zeros(len(final_emb[0])))
    final_ids=torch.tensor(final_ids)
    final_emb=torch.tensor(final_emb)
    return final_ids, final_emb


print('On  Structural Graph:')
struc_edges = data.edge_index
a = []
b = []
for i in tqdm(range(128)):
    aa_node_ids, aa_embeddings = get_embeddings(struc_edges, "struc_edgelist_late_fusion.txt", k=512, partition=1)
    result = evaluate_node_classification(aa_node_ids, aa_embeddings, data)
    a.append(result['micro_f1'])
    b.append(result['macro_f1'])

print(f"Micro-F1: {np.mean(a):.4f} ± {np.std(a):.4f}")
print(f"Macro-F1: {np.mean(b):.4f} ± {np.std(b):.4f}")

print('Late Fusion (Sum):')
a = []
b = []
for i in tqdm(range(128)):
    aa_node_ids_struc, aa_embeddings_struc = get_embeddings(struc_edges, "struc_edgelist_late_fusion.txt", k=512, partition=1)
    aa_node_ids_attr, aa_embeddings_attr = get_embeddings(mapped_attr_edges, "attr_edgelist_late_fusion.txt", k=512, partition=1)
    # reverse mapping the indices from virtual index to actual index
    org_aa_node_ids_attr = torch.tensor([rev_map_attr_edges[idx.item()] for idx in aa_node_ids_attr])
    aa_node_ids_attr, aa_embeddings_attr = append(org_aa_node_ids_attr, aa_embeddings_attr, data)
    agg_type='sum' # sum/concat/avg
    result = evaluate_node_classification_latefusion(aa_node_ids_struc, aa_embeddings_struc, aa_node_ids_attr, aa_embeddings_attr, data, agg_type)
    a.append(result['micro_f1'])
    b.append(result['macro_f1'])

print(f"Micro-F1: {np.mean(a):.4f} ± {np.std(a):.4f}")
print(f"Macro-F1: {np.mean(b):.4f} ± {np.std(b):.4f}")