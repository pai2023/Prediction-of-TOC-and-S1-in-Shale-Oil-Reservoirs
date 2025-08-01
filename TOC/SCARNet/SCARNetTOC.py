import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import copy

# **1️⃣ Device detection (GPU/CPU)**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **2️⃣ Read data**
labeled_data = pd.read_excel(r"C:\Users\User\Desktop\test\calculated_curves\training_and_validation_data.xlsx",
                             sheet_name="Training_Set")
unlabeled_data = pd.read_excel(r"C:\Users\User\Desktop\test\calculated_curves\training_and_validation_data.xlsx",
                               sheet_name="Normalized_unlabeled")

# Processing labeled data
X_labeled = labeled_data.iloc[:, 1:-1].values
y_labeled = labeled_data.iloc[:, -1].values

# Handling unlabeled data
X_unlabeled = unlabeled_data.iloc[:, 1:].values

# Convert to PyTorch tensor & send to GPU
X_labeled = torch.tensor(X_labeled, dtype=torch.float32).to(device)
y_labeled = torch.tensor(y_labeled, dtype=torch.float32).unsqueeze(1).to(device)
X_unlabeled = torch.tensor(X_unlabeled, dtype=torch.float32).to(device)

# Split training/test sets
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.1, random_state=42)
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# **3️⃣ Define the CNN structure**

class CNNRegressor(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=5):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size, padding=2)
        self.bn1 = nn.BatchNorm1d(num_filters)

        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.pool1 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size, padding=2)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)

        self.conv4 = nn.Conv1d(num_filters * 4, num_filters * 8, kernel_size, padding=2)
        self.bn4 = nn.BatchNorm1d(num_filters * 8)
        self.pool2 = nn.MaxPool1d(2)


        self.fc1 = nn.Linear(num_filters * 8 * 2, 128)
        self.fc2 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: (B, 1, 8)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# **4️⃣ Initialize CNN**
input_size = X_train.shape[1]
model = CNNRegressor(input_size).to(device)

# **5️⃣ Define the loss function and optimizer**
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# **6️⃣ Training Parameters**
num_epochs = 1000
r2_train_scores, rmse_train_scores = [], []
r2_test_scores, rmse_test_scores = [], []
losses = []



# Define storage variables for the best model
best_model_train = None
best_model_test = None
best_model_overall = None
best_r2_train = -np.inf
best_r2_test = -np.inf
best_score = -np.inf
best_model_supervised = None
best_r2_supervised = -np.inf

def train_model(num_epochs, model, train_dataloader, X_train, y_train, X_test, y_test):
    global best_model_train, best_model_test, best_model_overall, best_model_supervised
    global best_r2_train, best_r2_test, best_r2_supervised, best_score


    num_labeled = len(y_labeled)
    supervised_mask = torch.zeros_like(y_train, dtype=torch.bool)
    supervised_mask[:num_labeled] = True

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_dataloader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train).cpu().numpy()
            y_true_train = y_train.cpu().numpy()
            r2_train = r2_score(y_true_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))

            y_pred_test = model(X_test).cpu().numpy()
            y_true_test = y_test.cpu().numpy()
            r2_test = r2_score(y_true_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

            y_supervised_pred = y_pred_train[supervised_mask.cpu().numpy()]
            y_supervised_true = y_true_train[supervised_mask.cpu().numpy()]
            r2_supervised = r2_score(y_supervised_true, y_supervised_pred)

        overall_score = 0.2 * r2_train + 0.5 * r2_test + 0.3 * r2_supervised

        # Save the model with the best Test R²
        if r2_test > best_r2_test:
            best_r2_test = r2_test
            best_model_test = copy.deepcopy(model)
            torch.save(best_model_test.state_dict(), 'best_test_model.pth')
            print(f"✅ [Updated] Test R² improved to {r2_test:.4f} — best_test_model.pth saved.")

        # Save the best model with Train R²
        if r2_train > best_r2_train:
            best_r2_train = r2_train
            best_model_train = copy.deepcopy(model)
            torch.save(best_model_train.state_dict(), 'best_train_model.pth')
            print(f"✅ [Updated] Train R² improved to {r2_train:.4f} — best_train_model.pth saved.")

        # Save the model with the best Supervised R² (based only on the original label part)
        if r2_supervised > best_r2_supervised:
            best_r2_supervised = r2_supervised
            best_model_supervised = copy.deepcopy(model)
            torch.save(best_model_supervised.state_dict(), 'best_supervised_model.pth')
            print(f"✅ [Updated] Supervised R² improved to {r2_supervised:.4f} — best_supervised_model.pth saved.")

        # Save the model with the highest overall score
        if overall_score > best_score:
            best_score = overall_score
            best_model_overall = copy.deepcopy(model)
            torch.save(best_model_overall.state_dict(), 'best_overall_model.pth')
            print(f"✅ [Updated] Overall Score improved to {overall_score:.4f} — best_overall_model.pth saved.")

        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}, Supervised R²: {r2_supervised:.4f}, Score: {overall_score:.4f}')


# **Start training**
train_model(num_epochs, model, train_dataloader, X_train, y_train, X_test, y_test)

# **8️⃣ Semi-supervised self-training phase**
self_training_iterations = 3

for iteration in range(self_training_iterations):
    model.eval()
    with torch.no_grad():
        predictions = model(X_unlabeled).cpu().numpy()
        confidence = np.abs(predictions).flatten()

    # Select the top 10% of data with the highest confidence level
    sorted_confidences = np.sort(confidence)
    dynamic_threshold = sorted_confidences[int(0.9 * len(sorted_confidences))]

    high_confidence_indices = np.where(confidence >= dynamic_threshold)[0]

    if len(high_confidence_indices) == 0:
        print(f'Iteration {iteration + 1}: No high confidence samples found, stopping self-training.')
        break

    # Select high-confidence data
    X_high_confidence = X_unlabeled[high_confidence_indices]
    y_high_confidence = torch.tensor(predictions[high_confidence_indices], dtype=torch.float32).to(device)

    # Expand the training dataset
    X_train = torch.cat((X_train, X_high_confidence), dim=0)
    y_train = torch.cat((y_train, y_high_confidence), dim=0)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print(f'Iteration {iteration + 1}: Adding {len(high_confidence_indices)} pseudo-labels for retraining.')

    # **Load the best model before each training session**
    if best_model_overall is not None:
        model.load_state_dict(best_model_overall.state_dict())

    # Retrain the model
    train_model(num_epochs, model, train_dataloader, X_train, y_train, X_test, y_test)

# 9️⃣ Save the best model
torch.save(best_model_test.state_dict(), 'best_test_model-TOC.pth')
torch.save(best_model_train.state_dict(), 'best_train_model-TOC.pth')
torch.save(best_model_overall.state_dict(), 'best_overall_model-TOC.pth')


# **🔟 Visualization results**
plt.plot(range(len(losses)), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

print(f'Best R² on test set: {best_r2_test:.4f}')
