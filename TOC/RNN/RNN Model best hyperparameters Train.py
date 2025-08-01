import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from matplotlib import rcParams

# Automatic detection device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_excel(r"C:\Users\User\Desktop\test\calculated_curves\training_and_validation_data.xlsx", sheet_name="Training_Set")

X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Convert to PyTorch tensor and move to device (GPU/CPU)
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

dataset = TensorDataset(X, y)

class RNNRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNRegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

hidden_sizes = [8, 16, 32, 64, 128]
num_layers_list = [1, 2, 3]
learning_rates = [0.0001, 0.001, 0.01, 0.02, 0.05]
batch_size = [8, 16, 32, 64, 128]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_mse = float('inf')
best_params = {}
for hidden_size in hidden_sizes:
    for num_layers in num_layers_list:
        for lr in learning_rates:
            mse_scores = []
            for train_index, val_index in kf.split(dataset):
                train_subset = torch.utils.data.Subset(dataset, train_index)
                val_subset = torch.utils.data.Subset(dataset, val_index)

                train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)


                model = RNNRegression(input_size=X.size(1), hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                model.train()
                for epoch in range(100):
                    for features, labels in train_loader:
                        features, labels = features.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(features.unsqueeze(1))
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                model.eval()
                val_predictions = []
                val_labels = []
                with torch.no_grad():
                    for features, labels in val_loader:
                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features.unsqueeze(1))
                        val_predictions.append(outputs)
                        val_labels.append(labels)

                val_predictions = torch.cat(val_predictions).cpu().numpy()
                val_labels = torch.cat(val_labels).cpu().numpy()
                mse = mean_squared_error(val_labels, val_predictions)
                mse_scores.append(mse)

            avg_mse = np.mean(mse_scores)
            print(f'Hidden size: {hidden_size}, Num layers: {num_layers}, Learning rate: {lr}, MSE: {avg_mse}')

            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'learning_rate': lr
                }

print(f'Best parameters found: {best_params}, with MSE: {best_mse}')

# Retrain the model using the best hyperparameters
best_model = RNNRegression(input_size=X.size(1), hidden_size=best_params['hidden_size'], num_layers=best_params['num_layers'], output_size=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

best_model.train()
for epoch in range(50):
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)  
        optimizer.zero_grad()
        outputs = best_model(features.unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Final model trained with best hyperparameters.")
