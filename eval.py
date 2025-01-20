import torch
from preprocessing.create_batch_dataset import PDB_Dataset
from torch_geometric.loader import DataLoader
from model import GCN
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 0.5

# Dataset Setup
root = 'preprocessing/data/structure_files/tmp_cmap_files'
annot_file = 'preprocessing/data/pdb2go.tsv'
num_shards = 20

torch.manual_seed(12345)
pdb_protBERT_dataset = PDB_Dataset(root, annot_file, num_shards=num_shards, selected_ontology="biological_process", model="protBERT")
dataset = pdb_protBERT_dataset.shuffle()

# Splitting the dataset into train and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=12345)

# Define model architecture and load pre-trained weights
input_size = len(dataset[0].x[0])
hidden_sizes = [1000, 912]
output_size = pdb_protBERT_dataset.num_classes
model = GCN(input_size, hidden_sizes, output_size)
model.load_state_dict(torch.load('model_and_weight_files/model_weights_100_epochs_128_batch_size_2_layers.pth'))
model.to(device)
model.eval()

# Create DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Predictions
all_labels = []
all_predictions = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        outputs = model(data.x, data.edge_index, data.batch)
        predictions = torch.sigmoid(outputs).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_predictions.append(predictions)
        all_labels.append(labels)

all_predictions = np.concatenate(all_predictions, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Calculate AUROC and AUPR for each label
roc_auc_values = []
pr_auc_values = []
fprs, tprs, precisions, recalls = [], [], [], []

for i in range(output_size):
    fpr, tpr, _ = roc_curve(all_labels[:, i], all_predictions[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_values.append(roc_auc)
    fprs.append(fpr)
    tprs.append(tpr)

    precision, recall, _ = precision_recall_curve(all_labels[:, i], all_predictions[:, i])
    pr_auc = auc(recall, precision)
    pr_auc_values.append(pr_auc)
    precisions.append(precision)
    recalls.append(recall)

# Macro AUROC and AUPR
macro_roc_auc = np.mean(roc_auc_values)
macro_pr_auc = np.mean(pr_auc_values)

# Micro AUROC and AUPR
fpr_micro, tpr_micro, _ = roc_curve(all_labels.ravel(), all_predictions.ravel())
micro_roc_auc = auc(fpr_micro, tpr_micro)

precision_micro, recall_micro, _ = precision_recall_curve(all_labels.ravel(), all_predictions.ravel())
micro_pr_auc = auc(recall_micro, precision_micro)

# Plot AUROC and AUPR Violin Plots
auc_df = pd.DataFrame({
    'Metric': ['ROC AUC'] * len(roc_auc_values) + ['PR AUC'] * len(pr_auc_values),
    'Value': roc_auc_values + pr_auc_values
})

plt.figure(figsize=(12, 6))
sns.violinplot(x='Metric', y='Value', data=auc_df)
plt.title('Distribution of AUROC and AUPR')
plt.savefig('auroc_aupr_violin_plot.png')
plt.show()

# Plot ROC and PR Curves
plt.figure(figsize=(12, 6))

# Plot ROC Curves
plt.subplot(1, 2, 1)
for i in range(output_size):
    plt.plot(fprs[i], tprs[i], label=f'Label {i} (AUC = {roc_auc_values[i]:.2f})')
plt.plot(fpr_micro, tpr_micro, label=f'Micro Average (AUC = {micro_roc_auc:.2f})', linestyle='--', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (Macro AUC = {macro_roc_auc:.2f})')
plt.legend(loc='best')

# Plot PR Curves
plt.subplot(1, 2, 2)
for i in range(output_size):
    plt.plot(recalls[i], precisions[i], label=f'Label {i} (AUC = {pr_auc_values[i]:.2f})')
plt.plot(recall_micro, precision_micro, label=f'Micro Average (AUC = {micro_pr_auc:.2f})', linestyle='--', color='red')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (Macro AUC = {macro_pr_auc:.2f})')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('roc_pr_curves.png')
plt.show()

# Compute Confusion Matrix: Calculate TP, FP, TN, FN for each label and sum them
TP, FP, TN, FN = 0, 0, 0, 0
for i in range(output_size):
    predicted = (all_predictions[:, i] >= threshold).astype(int)
    actual = all_labels[:, i]
    
    tp = np.sum((predicted == 1) & (actual == 1))
    fp = np.sum((predicted == 1) & (actual == 0))
    tn = np.sum((predicted == 0) & (actual == 0))
    fn = np.sum((predicted == 0) & (actual == 1))

    TP += tp
    FP += fp
    TN += tn
    FN += fn

# Final aggregated confusion matrix
conf_matrix = np.array([[TP, FP], [FN, TN]])

# Plot aggregated confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Positive', 'Negative'])
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title('Aggregated Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
