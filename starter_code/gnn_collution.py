import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split

# =====================
# 1. LOAD DATA
# =====================

# Load edges
edges = pd.read_csv("../data/edges.csv")
edge_index = torch.tensor(edges.values.T, dtype=torch.long)

# Load labels
labels_df = pd.read_csv("../data/labels.csv")
y = torch.tensor(labels_df['label'].values, dtype=torch.long)

# Load train features + mask
train_df = pd.read_csv("../data/train_compressed.csv.gz")

X = train_df.drop(columns=['mask']).values
mask = train_df['mask'].values

X = torch.tensor(X, dtype=torch.float)

# =====================
# 2. CREATE GRAPH
# =====================

data = Data(x=X, edge_index=edge_index, y=y)

# =====================
# 3. TRAIN / VAL SPLIT
# =====================

train_idx, val_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42
)

train_idx = torch.tensor(train_idx)
val_idx = torch.tensor(val_idx)

mask_tensor = torch.tensor(mask)

# =====================
# 4. MODEL
# =====================

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(X.shape[1], 128, len(torch.unique(y)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# =====================
# 5. TRAINING LOOP
# =====================

for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    # 🔥 USE MASK SMARTLY
    weights = torch.where(mask_tensor == 1, 0.5, 1.0)
    loss = (criterion(out[train_idx], y[train_idx]) * weights[train_idx]).mean()

    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = out[val_idx].argmax(dim=1)
        acc = (val_pred == y[val_idx]).float().mean()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Acc: {acc:.4f}")

# =====================
# 6. TEST PREDICTION
# =====================

test_df = pd.read_csv("../data/test.csv")
test_X = torch.tensor(test_df.values, dtype=torch.float)

# Combine train + test for graph inference
all_X = torch.cat([X, test_X], dim=0)

# Extend graph (important assumption: test nodes isolated or appended)
data_all = Data(x=all_X, edge_index=edge_index)

model.eval()
with torch.no_grad():
    out = model(data_all.x, data_all.edge_index)

# Extract test predictions
test_preds = out[len(X):].argmax(dim=1).numpy()

# =====================
# 7. SAVE SUBMISSION
# =====================

submission = pd.DataFrame({
    "id": np.arange(len(test_preds)),
    "label": test_preds
})

submission.to_csv("../submissions/submission_Idrees_Bhat.csv", index=False)

print("✅ Submission file created!")
