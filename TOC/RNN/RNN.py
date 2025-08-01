import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns  # 添加这一行导入 seaborn 库

labeled_data = pd.read_excel(r"C:\Users\User\Desktop\FY1训练数据.xlsx", sheet_name="Normalized_TOC")


X_labeled = labeled_data.iloc[:, :-1].values
y_labeled = labeled_data.iloc[:, -1].values

X_labeled = torch.tensor(X_labeled, dtype=torch.float32)
y_labeled = torch.tensor(y_labeled, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.1, random_state=30)
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNRegressor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


input_size = 8
hidden_size = 32
num_layers = 3
model = RNNRegressor(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

r2_train_scores = []
rmse_train_scores = []
r2_test_scores = []
rmse_test_scores = []
losses = []

# 保存最佳 R² 及对应模型的变量
best_r2_train = float('-inf')
best_model = None

num_epochs = 1800
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_dataloader:
        X_batch = X_batch.unsqueeze(1)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train.unsqueeze(1)).detach().numpy()
        y_true_train = y_train.numpy()
        r2_train = r2_score(y_true_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_true_train, y_pred_train))

        y_pred_test = model(X_test.unsqueeze(1)).detach().numpy()
        y_true_test = y_test.numpy()
        r2_test = r2_score(y_true_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_true_test, y_pred_test))

    # Track the maximum value of R² in the training set and save the corresponding model
    if r2_train > best_r2_train:
        best_r2_train = r2_train
        best_model = model.state_dict().copy()
        # Save the prediction data at that time
        best_y_pred_train = y_pred_train
        best_y_pred_test = y_pred_test
        best_y_true_train = y_true_train
        best_y_true_test = y_true_test

    r2_train_scores.append(r2_train)
    rmse_train_scores.append(rmse_train)
    r2_test_scores.append(r2_test)
    rmse_test_scores.append(rmse_test)
    losses.append(epoch_loss / len(train_dataloader))

    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train R2: {r2_train:.4f}, Train RMSE: {rmse_train:.4f}, Test R2: {r2_test:.4f}, Test RMSE: {rmse_test:.4f}')

# ---- Training completed, save the best model and its corresponding data ----
y_true_all = np.concatenate((best_y_true_train, best_y_true_test))
y_pred_all = np.concatenate((best_y_pred_train, best_y_pred_test))
y_true_all = np.ravel(y_true_all)
y_pred_all = np.ravel(y_pred_all)

# Save data to Excel file
data = {
    'True Values': y_true_all,
    'Predicted Values': y_pred_all
}
df = pd.DataFrame(data)
file_path = 'predicted_values_best-all-TOC.xlsx'
df.to_excel(file_path, index=False)
print(f'File saved at {file_path}')

# Save the best model
torch.save(best_model, 'rnn_model-all-TOC.pth')
print("Best model saved as 'best_rnn_model.pth'")

# Generate regression graph
sns.regplot(x=y_true_all, y=y_pred_all, ci=95, color='#007CB9',
            scatter_kws={'s': 150, 'color': 'red', 'edgecolor': 'white'})
plt.tick_params(axis='both', direction='in', labelsize=24)
plt.xlabel(r'$\mathrm{TOC}_{\mathrm{mea}}$(%)', fontsize=24)
plt.ylabel(r'$\mathrm{TOC}_{\mathrm{RNN}}$(%)', fontsize=24)
plt.show()

print(f'Best R² on train set: {best_r2_train:.4f}')
