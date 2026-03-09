import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# -------------------------------------------------
# 1. LOAD RAW DATA
# -------------------------------------------------
# Why:
# The test nodes live in the same graph as the training nodes.
# So we must rebuild the full graph, not just load the test table.

train_df = pd.read_csv("data/train_compressed.csv.gz")
test_df = pd.read_csv("data/test.csv")
labels_df = pd.read_csv("data/labels.csv")
edges_df = pd.read_csv("data/edges.csv")


# -------------------------------------------------
# 2. EXTRACT NODE IDS
# -------------------------------------------------
# Why:
# The column "Unnamed: 0" is actually the node identifier.

train_ids = train_df["Unnamed: 0"].values
test_ids = test_df["Unnamed: 0"].values
all_ids = np.concatenate([train_ids, test_ids])

node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_ids)}


# -------------------------------------------------
# 3. BUILD FEATURE MATRIX
# -------------------------------------------------
# Why:
# The neural network should only see gene features.
# We remove:
# - "Unnamed: 0" = node id
# - "is_perturbed" = training-only hint, unavailable in test

train_x_df = train_df.drop(columns=["Unnamed: 0", "is_perturbed"])
test_x_df = test_df.drop(columns=["Unnamed: 0"])

x_df = pd.concat([train_x_df, test_x_df], axis=0)
x = torch.tensor(x_df.values, dtype=torch.float)


# -------------------------------------------------
# 4. BUILD LABEL VECTOR
# -------------------------------------------------
# Why:
# Labels are known only for training nodes.
# We keep -1 for unlabeled test nodes.

num_nodes = len(all_ids)
y = torch.full((num_nodes,), -1, dtype=torch.long)

label_id_col = labels_df.columns[0]
for _, row in labels_df.iterrows():
    node_id = row[label_id_col]
    class_label = row["cell_type"]
    y[node_id_to_idx[node_id]] = int(class_label)


# -------------------------------------------------
# 5. BUILD EDGE INDEX
# -------------------------------------------------
# Why:
# Edges are given in node IDs.
# We map them to internal row indices 0,1,2,...,N-1.

src = edges_df["source"].map(node_id_to_idx)
dst = edges_df["target"].map(node_id_to_idx)

valid_edges = src.notna() & dst.notna()

edge_index_np = np.vstack([
    src[valid_edges].astype(int).values,
    dst[valid_edges].astype(int).values
])

edge_index = torch.tensor(edge_index_np, dtype=torch.long)


# -------------------------------------------------
# 6. BUILD TEST MASK
# -------------------------------------------------
# Why:
# The graph has 4700 nodes, but only 940 belong to the test set.
# The mask tells us which nodes to extract predictions for.

test_mask = torch.zeros(num_nodes, dtype=torch.bool)
for node_id in test_ids:
    test_mask[node_id_to_idx[node_id]] = True


# -------------------------------------------------
# 7. CREATE GRAPH OBJECT
# -------------------------------------------------

data = Data(x=x, edge_index=edge_index, y=y)
data.test_mask = test_mask


# -------------------------------------------------
# 8. DEFINE THE SAME MODEL ARCHITECTURE
# -------------------------------------------------
# Why:
# Saved weights can only be loaded into the exact same model structure
# used during training.

class GCNBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# -------------------------------------------------
# 9. LOAD THE TRAINED MODEL
# -------------------------------------------------
# Why:
# We now restore the best checkpoint saved during training.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

model = GCNBaseline(
    in_channels=data.num_features,
    hidden_channels=128,
    out_channels=3,
    dropout=0.3,
).to(device)

state_dict = torch.load("best_gcn_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()


# -------------------------------------------------
# 10. PREDICT TEST NODES
# -------------------------------------------------
# Why:
# The model outputs logits for all 4700 nodes.
# We keep only the 940 test-node predictions.

with torch.no_grad():
    logits = model(data.x, data.edge_index)
    test_preds = logits[data.test_mask].argmax(dim=1).cpu().numpy()


# -------------------------------------------------
# 11. SAVE SUBMISSION FILE
# -------------------------------------------------
# Why:
# The competition infrastructure expects a CSV file of predictions.

submission = pd.DataFrame({
    "node_id": test_ids,
    "cell_type": test_preds
})

submission.to_csv("submission.csv", index=False)

print("Saved submission to submission.csv")
print("\nFirst rows:")
print(submission.head())

print("\nPrediction counts:")
print(submission["cell_type"].value_counts())
