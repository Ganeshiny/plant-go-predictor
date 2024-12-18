import torch
from preprocessing.create_batch_dataset import PDB_Dataset
from torch_geometric.loader import DataLoader
from model import GCN
from sklearn.model_selection import train_test_split
import numpy as np
from focal_loss import FocalLoss
from utils import calculate_class_weights, save_alpha_weights, load_alpha_weights
import pickle
import json

# Constants
THRESHOLD = 0.5
BATCH_SIZE = 64
EPOCHS = 120
LEARNING_RATE = 0.0001 
BEST_MODEL_PATH = f'model_and_weight_files/model_weights_{EPOCHS}_epochs_{BATCH_SIZE}_5_layers.pth'
PATH = "model_and_weight_files/model.pth"
CLASS_WEIGHT_PATH = "model_and_weight_files/alpha_weights.pkl"
MODEL_INFO_PATH = "model_and_weight_files/model_info_5_layers.json"  # Path to save model info

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Dataset Setup
root = 'preprocessing/data/structure_files/tmp_cmap_files'
annot_file = 'preprocessing/data/pdb2go.tsv'
num_shards = 20

torch.manual_seed(12345)
pdb_protBERT_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process", model="protBERT")
dataset = pdb_protBERT_dataset.shuffle()

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=12345)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Calculate alpha
#alpha = calculate_class_weights(dataset, device)
#save_alpha_weights(alpha, CLASS_WEIGHT_PATH)
alpha = load_alpha_weights(CLASS_WEIGHT_PATH)
print(f"Alpha weights:{alpha}")

# Model Setup
input_size = len(pdb_protBERT_dataset[0].x[0])
hidden_sizes = [1000, 912, 820, 500, 300]
output_size = pdb_protBERT_dataset.num_classes
model = GCN(input_size, hidden_sizes, output_size)
model.to(device)

model_info = {
    "input_size": input_size,
    "hidden_sizes": hidden_sizes,
    "output_size": output_size
}

with open(MODEL_INFO_PATH, 'w') as f:
    json.dump(model_info, f)

torch.save(model.state_dict(), PATH)

# Criterion and Optimizer
criterion = FocalLoss(alpha=alpha)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Logits
        loss = criterion(out, data.y.float())  # Compute multilabel binary loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test(loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = torch.sigmoid(out) > THRESHOLD  # Convert to probabilities and threshold
            correct += (pred == data.y).sum().item()  # Count correct predictions
            total += np.prod(data.y.shape)  # Total number of labels
    return correct / total  # Accuracy across all labels

# Tracking best accuracy
best_test_acc = 0

for epoch in range(1, EPOCHS + 1):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    
    # Save the model if test accuracy improves
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model saved with Test Acc: {test_acc:.4f}")
    
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
