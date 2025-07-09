import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Read pickle file 
df = pd.read_pickle("./data/processed/processed_data.pkl")

# Define features and target variable
features = ["H", "R", "OPS", "RBI", "SLG", "HR", "BA", "PA", "OBP", "AB"]
target = "All Star"

X = df[features].copy()
X.fillna(0, inplace=True)
y = df[target].values

# Split data in training (70%), validation (15%), and testing (15%) sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42)

# Standarize features
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_val_scaled = scalar.transform(X_val)
X_test_scaled = scalar.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # Add an extra dimension for compatibility
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1) # Add an extra dimension for compatibility 
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1) # Add an extra dimension for compatibility

# Convert data to DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# Define the neural netword model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(features), 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.sigmoid(self.fc4(x))
        return x

# Instantiate the model
model = Net()

# Due to class imbalance in the dataset, compute the ratio of non-All-Star to All-Star
# to use as a weight that penalizes the minority All-Star class more during training
class_count = np.bincount(y)
pos = class_count[0] / class_count[1]
class_weights = torch.tensor(pos, dtype=torch.float32)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.001)

tr = []  # training loss over epochs
tr_acc = []  # training accuracy over epochs
val = []  # validation losses over epochs
val_acc = []  # validation accuracy over epochs

forward_passes = 0  # number of forward propogations done
backward_passes = 0  # number of backward passes done
loss_over_forward = []  # tracking the loss over the forward propogation
loss_over_backward = []  # tracking the loss over the backward propogation

num_epochs = 80
for epoch in range(num_epochs):
    #### Training Phase ###
    model.train()  # places model in training mode
    running_tr_loss = 0.0  # tracking training loss in this epoch
    correct_tr = 0  # tracking correct training predictions in this epoch
    total_tr = 0  # track total training predictions in this epoch

    for inputs, labels in train_loader:
        outputs = model(inputs)  # performs forward pass to compute logits
        forward_passes += 1
        tr_loss = criterion(outputs, labels)  # computes loss between logits and true labels

        optimizer.zero_grad()  # clears old gradients 
        tr_loss.backward()  # computes gradients of the loss wrt to model parameters, backward pass
        backward_passes += 1
        optimizer.step()  # updates model parameters using computed gradients

        running_tr_loss += tr_loss.item()

        # Comptuing Accuracy
        with torch.no_grad():
            preds = torch.sigmoid(outputs)  # convert logits into predictions
            preds_class = (preds > 0.5).float()  # binary classifcation
            correct_tr += (preds_class == labels).sum().item()
            total_tr += labels.size(0)
    
    avg_tr_loss = running_tr_loss / len(train_loader)  # compute the average loss during training
    train_accuracy = correct_tr / total_tr  # compute training accuracy
    tr.append(avg_tr_loss)
    tr_acc.append(train_accuracy)

    ### Validation Phase ###
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)

            running_val_loss += val_loss.item()

            preds = torch.sigmoid(outputs)
            preds_class = (preds > 0.5).float()
            correct_val += (preds_class == labels).sum().item()
            total_val += labels.size(0)
    
    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val.append(avg_val_loss)
    val_acc.append(val_accuracy)

    loss_over_forward.append((forward_passes, avg_tr_loss, avg_val_loss))
    loss_over_backward.append((backward_passes, avg_tr_loss, avg_val_loss))

    print(f"Epoch {epoch+1:02d} | tr_loss: {avg_tr_loss:.4f} | val_loss: {avg_val_loss:.4f} | "f"tr_acc: {train_accuracy:.4f} | val_acc: {val_accuracy:.4f}")

# Evaluate on test data
with torch.no_grad():
    outputs = model(X_test_tensor)
    probs = torch.sigmoid(outputs)
    predicted_labels = (probs >= 0.5).float()
    accuracy = (predicted_labels == y_test_tensor).float().mean()
    print("Test Accuracy:", accuracy.item())

# Confusion Matrix 
y_true = y_test_tensor.detach().numpy().flatten().astype(int)
y_pred = predicted_labels.detach().numpy().flatten().astype(int)
conf_matrix = confusion_matrix(y_true, y_pred)

# Normalized
with np.errstate(all='ignore'):
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

# Plot Train vs Val Loss 
x_axis = np.arange(len(tr))
plt.figure()
plt.plot(x_axis, tr, label='Train Loss', color='blue')
plt.plot(x_axis, val, label='Validation Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/SGD_train_val_loss.png")
plt.close()

# Plot Confusion Matrix (Raw) 
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_xticklabels(['No', 'Yes'], rotation=45, ha='right')
disp.ax_.set_yticklabels(['No', 'Yes'])
plt.title('Confusion Matrix (Raw Counts)')
plt.tight_layout()
plt.savefig("outputs/SGD_confusion_matrix_raw.png")
plt.close()

# Plot Confusion Matrix (Normalized) 
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized)
disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_xticklabels(['No', 'Yes'], rotation=45, ha='right')
disp.ax_.set_yticklabels(['No', 'Yes'])
plt.title('Confusion Matrix (Normalized)')
plt.tight_layout()
plt.savefig("outputs/SGD_confusion_matrix_normalized.png")
plt.close()

# ROC Curve 
fpr, tpr, _ = roc_curve(y_true, probs.numpy())
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/SGD_roc_curve.png")
plt.close()

# Precision–Recall Curve 
precision, recall, _ = precision_recall_curve(y_true, probs.numpy())
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.tight_layout()
plt.savefig("outputs/SGD_precision_recall_curve.png")
plt.close()

fwd_x, fwd_tr, fwd_val = zip(*loss_over_forward)
bwd_x, bwd_tr, bwd_val = zip(*loss_over_backward)

plt.figure()
plt.plot(fwd_x, fwd_tr, label="Train Loss vs Forward", color="blue")
plt.plot(fwd_x, fwd_val, label="Val Loss vs Forward", color="green")
plt.plot(bwd_x, bwd_tr, linestyle="--", label="Train Loss vs Backward", color="red")
plt.plot(bwd_x, bwd_val, linestyle="--", label="Val Loss vs Backward", color="orange")
plt.xlabel("Number of Propagations")
plt.ylabel("Loss")
plt.title("Train/Val Loss vs Forward/Backward Propagations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/SGD_epoch_loss_vs_propagations.png")
plt.close()

