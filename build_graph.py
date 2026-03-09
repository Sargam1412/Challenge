import pandas as pd
import torch
from torch_geometric.data import Data

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
# Why: read the raw CSV files into memory

train = pd.read_csv("data/train_compressed.csv.gz")
test = pd.read_csv("data/test.csv")
labels = pd.read_csv("data/labels.csv")
edges = pd.read_csv("data/edges.csv")


# -------------------------------------------------
# 2. EXTRACT NODE IDS
# -------------------------------------------------
# Why: node identifiers are stored in the column "Unnamed: 0"

train_ids = train["Unnamed: 0"].values
test_ids = test["Unnamed: 0"].values


# -------------------------------------------------
# 3. REMOVE NON-FEATURE COLUMNS
# -------------------------------------------------
# Why: neural networks should only receive gene expression features

train_features = train.drop(columns=["Unnamed: 0", "is_perturbed"])
test_features = test.drop(columns=["Unnamed: 0"])


# -------------------------------------------------
# 4. MERGE TRAIN + TEST FEATURES
# -------------------------------------------------
# Why: the graph contains ALL nodes (4700)

features = pd.concat([train_features, test_features], axis=0)


# -------------------------------------------------
# 5. BUILD NODE ID → INDEX MAPPING
# -------------------------------------------------
# Why: GNN libraries require indices 0..N-1

node_ids = pd.concat([train["Unnamed: 0"], test["Unnamed: 0"]])
node_id_map = {nid: i for i, nid in enumerate(node_ids)}


# -------------------------------------------------
# 6. CONVERT FEATURES TO TENSOR
# -------------------------------------------------

x = torch.tensor(features.values, dtype=torch.float)


# -------------------------------------------------
# 7. BUILD EDGE INDEX
# -------------------------------------------------
# Why: edges must use the internal indices

edge_src = edges["source"].map(node_id_map)
edge_dst = edges["target"].map(node_id_map)

edge_index = torch.tensor(
    [edge_src.values, edge_dst.values],
    dtype=torch.long
)


# -------------------------------------------------
# 8. BUILD LABEL VECTOR
# -------------------------------------------------
# Why: only training nodes have labels

num_nodes = len(node_ids)

y = torch.full((num_nodes,), -1, dtype=torch.long)

for _, row in labels.iterrows():
    idx = node_id_map[row["Unnamed: 0"]]
    y[idx] = row["cell_type"]


# -------------------------------------------------
# 9. BUILD TRAIN MASK
# -------------------------------------------------
# Why: loss should only use training nodes

train_mask = y != -1


# -------------------------------------------------
# 10. CREATE GRAPH OBJECT
# -------------------------------------------------

data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

data.train_mask = train_mask


print(data)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Feature dimension:", data.num_features)
