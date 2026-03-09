import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# -------------------------------------------------
# 1. REPRODUCIBILITY
# -------------------------------------------------
# Why:
# We fix the seed so results are reproducible.
# This is also important for competition tracking.

SEED = 25
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------
# 2. LOAD RAW DATA
# -------------------------------------------------
# Why:
# We read the train/test/features/labels/edges from disk.

train_df = pd.read_csv("data/train_compressed.csv.gz")
test_df = pd.read_csv("data/test.csv")
labels_df = pd.read_csv("data/labels.csv")
edges_df = pd.read_csv("data/edges.csv")


# -------------------------------------------------
# 3. EXTRACT NODE IDS
# -------------------------------------------------
# Why:
# "Unnamed: 0" is actually the node identifier.
# We need these IDs to align features, labels, and edges.

train_ids = train_df["Unnamed: 0"].values
test_ids = test_df["Unnamed: 0"].values

all_ids = np.concatenate([train_ids, test_ids])


# -------------------------------------------------
# 4. BUILD NODE ID -> INTERNAL INDEX MAP
# -------------------------------------------------
# Why:
# PyTorch Geometric expects nodes indexed as 0,1,2,...,N-1.
# The dataset IDs are not stored in row order, so we build a mapping.

node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_ids)}


# -------------------------------------------------
# 5. BUILD FEATURE MATRIX
# -------------------------------------------------
# Why:
# The model should only see gene-expression features.
# We remove:
# - "Unnamed: 0" (node id)
# - "is_perturbed" (training-only hint, not available in test)

train_x_df = train_df.drop(columns=["Unnamed: 0", "is_perturbed"])
test_x_df = test_df.drop(columns=["Unnamed: 0"])

x_df = pd.concat([train_x_df, test_x_df], axis=0)
x = torch.tensor(x_df.values, dtype=torch.float)


# -------------------------------------------------
# 6. BUILD LABEL VECTOR
# -------------------------------------------------
# Why:
# Training nodes have labels; test nodes do not.
# We create a label vector for all nodes and mark unknown labels with -1.

num_nodes = len(all_ids)
y = torch.full((num_nodes,), -1, dtype=torch.long)

# labels_df first column is the node id column
label_id_col = labels_df.columns[0]

for _, row in labels_df.iterrows():
    node_id = row[label_id_col]
    class_label = row["cell_type"]
    y[node_id_to_idx[node_id]] = int(class_label)


# -------------------------------------------------
# 7. BUILD TRAIN / VAL SPLIT
# -------------------------------------------------
# Why:
# We should not evaluate only on the training set.
# We split the labeled nodes into train and validation sets.
# Stratification keeps class proportions stable.

labeled_indices = np.where(y.numpy() != -1)[0]
labeled_targets = y[labeled_indices].numpy()

train_idx, val_idx = train_test_split(
    labeled_indices,
    test_size=0.2,
    random_state=SEED,
    stratify=labeled_targets,
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True


# -------------------------------------------------
# 8. BUILD EDGE INDEX
# -------------------------------------------------
# Why:
# The graph edges are given in terms of node ids.
# We map them to internal indices.
# We also keep only edges whose endpoints exist in our node set.

src = edges_df["source"].map(node_id_to_idx)
dst = edges_df["target"].map(node_id_to_idx)

valid_edges = src.notna() & dst.notna()

edge_index_np = np.vstack([
    src[valid_edges].astype(int).values,
    dst[valid_edges].astype(int).values
])

edge_index = torch.tensor(edge_index_np, dtype=torch.long)


# -------------------------------------------------
# 9. CREATE THE GRAPH OBJECT
# -------------------------------------------------
# Why:
# PyTorch Geometric works with a Data object containing:
# - node features x
# - graph connectivity edge_index
# - labels y
# - train/validation masks

data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.val_mask = val_mask


# -------------------------------------------------
# 10. DEFINE THE MODEL
# -------------------------------------------------
# Why:
# We start with a simple 2-layer GCN.
# Layer 1 learns a hidden representation.
# Layer 2 maps to the 3 output classes.

class GCNBaseline(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First graph convolution:
        # each node aggregates information from its neighbors
        x = self.conv1(x, edge_index)

        # Non-linearity:
        # allows the model to learn non-linear patterns
        x = F.relu(x)

        # Dropout:
        # helps reduce overfitting
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second graph convolution:
        # produces class logits
        x = self.conv2(x, edge_index)
        return x


# -------------------------------------------------
# 11. CLASS WEIGHTS
# -------------------------------------------------
# Why:
# Class 2 is much smaller than classes 0 and 1.
# Weighted loss tells the model that mistakes on rare classes matter more.

train_targets = y[data.train_mask].numpy()
class_counts = np.bincount(train_targets, minlength=3)

class_weights = torch.tensor(
    [len(train_targets) / (3.0 * c) if c > 0 else 1.0 for c in class_counts],
    dtype=torch.float,
)

print("Class counts (train split):", class_counts.tolist())
print("Class weights:", class_weights.tolist())


# -------------------------------------------------
# 12. INITIALIZE MODEL, OPTIMIZER, LOSS
# -------------------------------------------------
# Why:
# - model: the GCN we want to train
# - optimizer: updates parameters
# - criterion: weighted cross-entropy for imbalanced classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
class_weights = class_weights.to(device)

model = GCNBaseline(
    in_channels=data.num_features,
    hidden_channels=128,
    out_channels=3,
    dropout=0.3,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)


# -------------------------------------------------
# 13. EVALUATION FUNCTION
# -------------------------------------------------
# Why:
# We want reusable code for validation metrics.

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits[mask].argmax(dim=1).cpu().numpy()
        true = data.y[mask].cpu().numpy()

    acc = accuracy_score(true, preds)
    macro_f1 = f1_score(true, preds, average="macro")
    return acc, macro_f1


# -------------------------------------------------
# 14. TRAINING LOOP
# -------------------------------------------------
# Why:
# At each epoch:
# - forward pass
# - compute loss only on train nodes
# - backpropagation
# - validate on val nodes

best_val_f1 = -1.0
best_state = None

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)

    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    train_acc, train_f1 = evaluate(data.train_mask)
    val_acc, val_f1 = evaluate(data.val_mask)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )


# -------------------------------------------------
# 15. LOAD BEST MODEL
# -------------------------------------------------
# Why:
# We keep the checkpoint that had the best validation Macro-F1.

model.load_state_dict(best_state)

# Save best model to disk
torch.save(best_state, "best_gcn_model.pt")

final_train_acc, final_train_f1 = evaluate(data.train_mask)
final_val_acc, final_val_f1 = evaluate(data.val_mask)

print("\n===== Final Best Validation Model =====")
print(f"Train Acc: {final_train_acc:.4f}")
print(f"Train Macro-F1: {final_train_f1:.4f}")
print(f"Val Acc: {final_val_acc:.4f}")
print(f"Val Macro-F1: {final_val_f1:.4f}")
print("Saved best model to best_gcn_model.pt")
