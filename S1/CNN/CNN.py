import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Comprehensive optimized CNN model
class OptimizedCNN(nn.Module):
    def __init__(self, input_size, conv_filters, kernel_size, num_conv_layers, output_size, conv_dropout_rate=0.2,
                 fc_dropout_rate=0.4):
        super(OptimizedCNN, self).__init__()
        layers = []

# Dynamically create multiple convolutional layers and add Dropout (convolutional layer Dropout is conv_dropout_rate)
        for i in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels=1 if i == 0 else conv_filters,
                                    out_channels=conv_filters,
                                    kernel_size=kernel_size,
                                    padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))  # Maximum pooling
            layers.append(nn.Dropout(conv_dropout_rate))  # Dropout for the convolution layer is set to conv_dropout_rate

        self.conv = nn.Sequential(*layers)

        # Add a fully connected layer (fully connected layer Dropout is fc_dropout_rate)
        self.fc = nn.Sequential(
            nn.Linear(conv_filters * (input_size // (2 ** num_conv_layers)), 100),
            nn.ReLU(),
            nn.Dropout(fc_dropout_rate),  # Dropout for the fully connected layer is set to fc_dropout_rate
            nn.Linear(100, output_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Load saved tensor data
X = torch.load("X_well_log_data.pt")  # Input data X (d, n)
Y = torch.load("Y_labels.pt")  # Label data Y (d, 1)

# Wrap the data into a PyTorch TensorDataset
dataset = TensorDataset(X, Y)

# Define hyperparameters
conv_filters = 64
kernel_size = 7
num_conv_layers = 3
conv_dropout_rate = 0.2
fc_dropout_rate = 0.4
output_size = 1
learning_rate = 0.001

# K-fold cross-validation settings
kfold = KFold(n_splits=5, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Store evaluation metrics for each fold
fold_metrics = []

# K-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"Fold {fold + 1}")

    # Split the training set and validation set
    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Initialize the model
    model = OptimizedCNN(input_size=X.size(1), conv_filters=conv_filters, kernel_size=kernel_size,
                         num_conv_layers=num_conv_layers, output_size=output_size,
                         conv_dropout_rate=conv_dropout_rate, fc_dropout_rate=fc_dropout_rate).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    model.train()
    num_epochs = 1000
    best_val_loss = float('inf')
    best_model_state = None  # Initialize the optimal model state

    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            # Gradient zeroing
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(features.unsqueeze(1))
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Output the average loss for each epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Verify the model
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features.unsqueeze(1))
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Collect all predicted values and actual labels for calculating evaluation metrics.
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Calculate validation loss
    val_loss /= len(val_loader)
    print(f"Fold {fold + 1} Validation Loss: {val_loss:.4f}")

    # Flatten all predictions and labels
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)

    # Calculate MSE, RMSE, and R²
    mse = mean_squared_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_predictions)

    print(f"Fold {fold + 1} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Store the evaluation results for each fold
    fold_metrics.append({
        "fold": fold + 1,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "val_loss": val_loss
    })

    # If the validation loss of the current fold is less than the previously saved minimum loss, save the current model.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # Save the state dictionary of the current best model

# Save the final best model
if best_model_state is not None:
    torch.save(best_model_state, 'best_model.pth')
    print("The best model has been saved.")

# Output the evaluation results for each fold
for metrics in fold_metrics:
    print(f"Fold {metrics['fold']} - MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R²']:.4f}")

# Load the best model parameters (state_dict)
best_model = OptimizedCNN(input_size=X.size(1), conv_filters=conv_filters, kernel_size=kernel_size,
                          num_conv_layers=num_conv_layers, output_size=output_size,
                          conv_dropout_rate=conv_dropout_rate, fc_dropout_rate=fc_dropout_rate).to(device)

best_model.load_state_dict(torch.load('best_model.pth'))

# Use the best model to make predictions on the training set.
best_model.eval()
train_predictions = []
train_labels = []

with torch.no_grad():
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        outputs = best_model(features.unsqueeze(1))
        train_predictions.append(outputs.cpu().numpy())
        train_labels.append(labels.cpu().numpy())

# Flatten all training set predictions and labels
train_predictions = np.concatenate(train_predictions).ravel()
train_labels = np.concatenate(train_labels).ravel()

# Generate DataFrame, store predicted values and actual values in table
results_df = pd.DataFrame({
    'True Values': train_labels,
    'Predicted Values': train_predictions
})

# Save data to Excel file
results_df.to_excel('train_predictions_vs_true_values.xlsx', index=False)
print("The training set predictions and true values have been saved to the ‘train_predictions_vs_true_values.xlsx’ file.")

# Drawing regression graphs
sns.regplot(x=train_labels, y=train_predictions, ci=95, color='#007CB9',
            scatter_kws={'s': 150, 'color': 'red', 'edgecolor': 'white'})
plt.tick_params(axis='both', direction='in', labelsize=24)
plt.xlabel(r'$\mathrm{S1}_{\mathrm{mea}}$(%)', fontsize=24)  # Actual value
plt.ylabel(r'$\mathrm{S1}_{\mathrm{CNN}}$(%)', fontsize=24)  # Predicted value
plt.title("Regression Plot for Best CNN Model on Training Set", fontsize=24)
plt.show()

# Print R² on the training set
train_r2 = r2_score(train_labels, train_predictions)
print(f'Training Set R²: {train_r2:.4f}')
