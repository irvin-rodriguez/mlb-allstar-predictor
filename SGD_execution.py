import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Read the CSV file
file = pd.read_csv("data/2008_2024_FULL_DATASET.csv")

# Define features and target variable
stats_of_interest = ['PA', 'R', 'H', '2B', 'HR', 'RBI', 'BB', 'SB', 'BA', 'OBP', 'SLG', 'OPS']
X = file[stats_of_interest].values
y = file['All Star'].values

X = file[stats_of_interest].copy()
X.fillna(0, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for compatibility with model
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for compatibility with model

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(stats_of_interest), 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


# Instantiate the model
model = Net()

# Define loss function and optimizer
# Account for imbalance in data
class_counts = np.bincount(y)
pos = class_counts[0] / class_counts[1]
class_weights = torch.tensor(pos, dtype=torch.float)

criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Convert data to DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

tr = []
te = []
count = 0
epoch = 0
for epoch in range(80):
    epoch += 1
    running_tr_loss = 0
    count_tr = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        tr_loss = criterion(outputs, labels)
        tr_loss.backward()  # gets gradients
        optimizer.step()
        running_tr_loss += tr_loss.item()
        count_tr += 1
    
    test_loss = 0
    count_te = 0
    for inputs, labels in test_loader:
        
        q_pred = model(inputs)
        test_loss += criterion(labels, q_pred).item()
        count_te += 1
    print(f"Epoch {epoch+1}, tr_loss: {running_tr_loss / count_tr}, test_loss: {test_loss / count_te},")
    tr.append(running_tr_loss / count_tr)
    te.append(test_loss / count_te)


# Evaluation
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted_labels = (outputs >= 0.5).float()
    accuracy = (predicted_labels == y_test_tensor).float().mean()
    print("Test Accuracy:", accuracy.item())

# Compute confusion matrix
y_true = y_test_tensor.detach().numpy().flatten()
y_pred = predicted_labels.detach().numpy().flatten()
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# plot train vs test loss
x_axis = np.linspace(0, len(tr), len(tr))
plt.plot(x_axis, tr, label='Train Loss', color='blue')
plt.plot(x_axis, te, label='Test Loss', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.show()

# Plot confusion matrix
y_true = y_test_tensor.numpy().flatten().astype(int)
y_pred = predicted_labels.numpy().flatten().astype(int)
conf_matrix = confusion_matrix(y_true, y_pred)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]


cell_labels = [['No', 'Yes'], ['No', 'Yes']]

# Plotting the non-normalized confusion matrix with custom labels
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_xticklabels(cell_labels[1], rotation=45, ha='right')
disp.ax_.set_yticklabels(cell_labels[0])
plt.title('Confusion Matrix (Non-Normalized)')
plt.show()

# Plotting the normalized confusion matrix with custom labels
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_normalized)
disp.plot(cmap=plt.cm.Blues)
disp.ax_.set_xticklabels(cell_labels[1], rotation=45, ha='right')
disp.ax_.set_yticklabels(cell_labels[0])
plt.title('Confusion Matrix (Normalized)')
plt.show()