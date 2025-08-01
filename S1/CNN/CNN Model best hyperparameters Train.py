import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load saved X and Y data
X = torch.load("X_well_log_data.pt")
Y = torch.load("Y_labels.pt")

# Create a TensorDataset combining input data X and labels Y.
dataset = TensorDataset(X, Y)


# CNN model definition
class CNNRegression(nn.Module):
    def __init__(self, input_size, conv_filters, kernel_size, num_conv_layers, output_size):
        super(CNNRegression, self).__init__()
        layers = []

        # Dynamically create multiple convolutional layers
        for i in range(num_conv_layers):
            if i == 0:  # First layer convolution, input_size as input channel
                layers.append(nn.Conv1d(1, conv_filters, kernel_size, padding='same'))  # Input channel is 1
            else:
                layers.append(nn.Conv1d(conv_filters, conv_filters, kernel_size, padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))  # Use max pooling

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(conv_filters * (input_size // (2 ** num_conv_layers)), output_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Parameter space of hyperparameter search
conv_filters_list = [16, 32, 64]  # Different numbers of convolution kernels
kernel_size_list = [1, 3, 5, 7, 11]  # Convolution kernel size
num_conv_layers_list = [1, 2, 3]  # Number of convolutional layers
learning_rates = [0.00001, 0.0001, 0.001, 0.01]
batch_sizes = [16, 32, 64, 128]  # Batch size
optimizers = ['adam', 'sgd', 'rmsprop']  # Different optimizers
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_mse = float('inf')
best_params = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter search
for conv_filters in conv_filters_list:
    for kernel_size in kernel_size_list:
        for num_conv_layers in num_conv_layers_list:
            for lr in learning_rates:
                mse_scores = []

                for train_index, val_index in kf.split(dataset):
                    train_subset = torch.utils.data.Subset(dataset, train_index)
                    val_subset = torch.utils.data.Subset(dataset, val_index)

                    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

                    # Initialize the model
                    model = CNNRegression(input_size=X.size(1), conv_filters=conv_filters,
                                          kernel_size=kernel_size, num_conv_layers=num_conv_layers, output_size=1).to(
                        device)

                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    # Training the model
                    model.train()
                    for epoch in range(50):
                        for features, labels in train_loader:
                            features, labels = features.to(device), labels.to(device)
                            optimizer.zero_grad()
                            outputs = model(features.unsqueeze(1))  # Convolution inputs require additional dimensions
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                    # Verify the model
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
                print(
                    f'Conv filters: {conv_filters}, Kernel size: {kernel_size}, Conv layers: {num_conv_layers}, Learning rate: {lr}, MSE: {avg_mse}')

                # Update optimal parameters
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = {
                        'conv_filters': conv_filters,
                        'kernel_size': kernel_size,
                        'num_conv_layers': num_conv_layers,
                        'learning_rate': lr
                    }

print(f'Best parameters found: {best_params}, with MSE: {best_mse}')

# Retrain CNN models using optimal hyperparameters
best_model = CNNRegression(input_size=X.size(1), conv_filters=best_params['conv_filters'],
                           kernel_size=best_params['kernel_size'], num_conv_layers=best_params['num_conv_layers'],
                           output_size=1).to(device)
optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train CNN models using optimal parameters
best_model.train()
for epoch in range(50):
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(features.unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("CNN model with best hyperparameters is trained.")
