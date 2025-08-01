import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Read training set / validation set / unlabeled set ===
file_path = r"D:\Result\Train data\training_and_validation_data-S1.xlsx"
train_df = pd.read_excel(file_path, sheet_name="Training_Set")
val_df = pd.read_excel(file_path, sheet_name="Validation_Set")
unlabeled_df = pd.read_excel(file_path, sheet_name="Normalized_unlabeled")

# Remove depth column
X_train = torch.tensor(train_df.iloc[:, 1:-1].values, dtype=torch.float32)
y_train = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(val_df.iloc[:, 1:-1].values, dtype=torch.float32)
y_val = torch.tensor(val_df.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)
X_unlabeled = torch.tensor(unlabeled_df.iloc[:, 1:].values, dtype=torch.float32)

# Reserve test set (randomly select 20% from the training set as test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Data Set Loader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

# === 2. Defining the Model ===
class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Initialize the model
input_size = X_train.shape[1]
hidden_size = 32
num_layers = 3
model = RNNRegressor(input_size, hidden_size, num_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
# Initialize optimal records
best_r2_train = -np.inf
best_model = None
best_preds = {}

# ✅ Training function: Only “training set R²” is used as the saving standard.
def train_model(num_epochs):
    global best_model, best_r2_train, best_preds
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.view(X_batch.size(0), 1, -1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            # Training set and test set
            X_train_seq = X_train.view(X_train.size(0), 1, -1)
            X_test_seq = X_test.view(X_test.size(0), 1, -1)

            y_pred_train = model(X_train_seq).numpy()
            y_pred_test = model(X_test_seq).numpy()

            r2_train = r2_score(y_train.numpy(), y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train.numpy(), y_pred_train))
            r2_test = r2_score(y_test.numpy(), y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test.numpy(), y_pred_test))

            # Validation set
            X_val_seq = X_val.view(X_val.size(0), 1, -1)
            y_pred_val = model(X_val_seq).numpy()

        # Storage verification set predictions
        if r2_train > best_r2_train:
            best_r2_train = r2_train
            best_model = model.state_dict()
            best_preds = {
                'train': (y_train.numpy(), y_pred_train),
                'test': (y_test.numpy(), y_pred_test),
                'val': (y_val.numpy(), y_pred_val)  # Add predictions to the validation set
            }
            print(f"✅ Epoch {epoch + 1}: Best updated. R² (train/test): {r2_train:.4f}/{r2_test:.4f}")


# ✅ Start training
num_epochs = 1200
train_model(num_epochs)

# === 6. Self-training iteration ===
for iteration in range(2):
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        preds = model(X_unlabeled.unsqueeze(1)).numpy()
        confidence = np.abs(preds).flatten()
    threshold = np.percentile(confidence, 90)
    indices = np.where(confidence >= threshold)[0]
    if len(indices) == 0:
        print("⚠️ No high-confidence pseudo labels, stop self-training")
        break
    X_pseudo = X_unlabeled[indices]
    y_pseudo = torch.tensor(preds[indices], dtype=torch.float32)

    X_train = torch.cat([X_train, X_pseudo], dim=0)
    y_train = torch.cat([y_train, y_pseudo], dim=0)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    print(f"🔁 From training round {iteration + 1}: Add {len(indices)} pseudo-label samples.")
    train_model(num_epochs=300)

# === 7. Save results ===
def save_predictions(y_true, y_pred, name):
    df = pd.DataFrame({
        'True Values': y_true.ravel(),
        'Predicted Values': y_pred.ravel()
    })
    df.to_excel(f'predicted_{name}.xlsx', index=False)

save_predictions(*best_preds['train'], 'train')
save_predictions(*best_preds['val'], 'val')
save_predictions(*best_preds['test'], 'test')
torch.save(best_model, 'best_rnn_model_val_fixed.pth')

print("✅ The best model and prediction data have been saved.")

# Visualization verification set
sns.regplot(x=best_preds['val'][0].ravel(), y=best_preds['val'][1].ravel(), ci=95, color='#007CB9',
            scatter_kws={'s': 100, 'color': 'red', 'edgecolor': 'white'})
plt.xlabel(r'$\mathrm{S1}_{\mathrm{mea}}$(%)', fontsize=16)
plt.ylabel(r'$\mathrm{S1}_{\mathrm{RNN}}$(%)', fontsize=16)
plt.title("Validation Set Prediction")
plt.tight_layout()
plt.show()
