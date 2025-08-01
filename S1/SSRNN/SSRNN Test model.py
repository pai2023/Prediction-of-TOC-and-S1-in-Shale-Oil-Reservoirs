import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score


# Define your RNN model class (consistent with the structure when saving the model)
class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Define model parameters (must be consistent with those saved when the model was saved)
input_size = 8
hidden_size = 32
num_layers = 2

# Instantiate the model and load the saved weights
model = RNNRegressor(input_size, hidden_size, num_layers)
model_path = r"C:\Users\User\Desktop\test\CNN-BiLSTM-AT\rnn_model-all-S1.pth"
model.load_state_dict(torch.load(model_path))

# Read Excel data
new_data = pd.read_excel(r"C:\Users\User\Desktop\test\calculated_curves\S1\training_and_validation_data-S1.xlsx", sheet_name="Validation_Set")
X11 = new_data.loc[:40, "DEN":"SH"].values
y_true = new_data.loc[:40, 'S1'].values

# Convert to tensor
X_tensor = torch.tensor(X11, dtype=torch.float32)
X_tensor = X_tensor.unsqueeze(1)  # Add time dimension
y_tensor = torch.tensor(y_true, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Make predictions using the loaded model
model.eval()
y_preds = []
with torch.no_grad():
    for X_batch, _ in dataloader:
        outputs = model(X_batch)
        y_preds.extend(outputs.numpy())

# Convert the predicted values to a NumPy array and flatten them
y_preds = np.array(y_preds).flatten()

y_preds_np = y_preds.numpy() if isinstance(y_preds, torch.Tensor) else y_preds
y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true

# Create a DataFrame to store actual values and predicted values
df_results = pd.DataFrame({
    'True Values': y_true_np,
    'Predicted Values': y_preds_np
})

# Save to Excel file
df_results.to_excel('predictions_output-all-S1.xlsx', index=False)
print("The predicted values have been successfully saved to the Excel file!")


