import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

# Define the CNN model class (consistent with the model structure you used during training)
class OptimizedCNN(nn.Module):
    def __init__(self, input_size, conv_filters, kernel_size, num_conv_layers, output_size, conv_dropout_rate=0.2, fc_dropout_rate=0.4):
        super(OptimizedCNN, self).__init__()
        layers = []

        # Dynamically create multiple convolutional layers and add Dropout
        for i in range(num_conv_layers):
            layers.append(nn.Conv1d(in_channels=1 if i == 0 else conv_filters,
                                    out_channels=conv_filters,
                                    kernel_size=kernel_size,
                                    padding="same"))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))  # Max pooling
            layers.append(nn.Dropout(conv_dropout_rate))  # Dropout in Convolution Layers

        self.conv = nn.Sequential(*layers)

        # Add a fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_filters * (input_size // (2 ** num_conv_layers)), 100),
            nn.ReLU(),
            nn.Dropout(fc_dropout_rate),  # Dropout in the fully connected layer
            nn.Linear(100, output_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load saved tensor data (test set data)
X_test = torch.load("X_test_well_log_data.pt")  # Processed test set input tensors
Y_test = torch.load("Y_test_labels.pt")  # Processed test set label tensor

# Create DataLoader
test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model parameters (consistent with your training)
input_size = X_test.size(1)
conv_filters = 64
kernel_size = 7
num_conv_layers = 3
conv_dropout_rate = 0.2
fc_dropout_rate = 0.4
output_size = 1

# Instantiate the model and load the saved weights
model = OptimizedCNN(input_size, conv_filters, kernel_size, num_conv_layers, output_size, conv_dropout_rate, fc_dropout_rate)
model_path = r"C:\Users\User\Desktop\test\CNN-BiLSTM\best_model.pth"  # Modify the path for your CNN model
model.load_state_dict(torch.load(model_path))

# Make predictions using the loaded model
model.eval()
y_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.unsqueeze(1)
        outputs = model(X_batch)
        y_preds.extend(outputs.numpy())

# Convert the predicted values to a NumPy array and flatten them
y_preds_np = np.array(y_preds).flatten()

# Create a DataFrame to store actual values and predicted values
df_results = pd.DataFrame({
    'True Values': Y_test.cpu().numpy().flatten(),
    'Predicted Values': y_preds_np
})

# Save to Excel file
df_results.to_excel('cnn_test_predictions_output.xlsx', index=False)
print("The CNN model test set prediction values have been successfully saved to an Excel file!")
