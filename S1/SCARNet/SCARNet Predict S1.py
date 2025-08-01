import pandas as pd
import torch
import torch.nn as nn

# **1️⃣ Device detection (GPU/CPU)**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **2️⃣ Define the CNN structure (must be the same as during training)**
class CNNRegressor(nn.Module):
    def __init__(self, input_size, num_filters=32, kernel_size=3):
        super(CNNRegressor, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)

        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.pool1 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)

        self.conv4 = nn.Conv1d(num_filters * 4, num_filters * 8, kernel_size, padding=1)
        self.bn4 = nn.BatchNorm1d(num_filters * 8)
        self.pool2 = nn.MaxPool1d(2)
        # The final length is 2. Calculate the input dimension after flattening.
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

# **3️⃣ Load the trained model**
input_size = 8
model = CNNRegressor(input_size).to(device)
model.load_state_dict(torch.load(r"D:\Result\S1\SCARNet\best_overall_model-S1.pth"))
model.eval()  # Set to inference mode

# **4️⃣ Read new data**
new_data = pd.read_excel(r"D:\Result\Training Data\Batch Normalization + Unlabeled Data.xlsx", sheet_name="New_Data_Normalized")

# **5️⃣ Preprocess data**
X_new = new_data.iloc[:, 1:].values
X_new = torch.tensor(X_new, dtype=torch.float32).to(device)

# **6️⃣ Make predictions**
with torch.no_grad():
    y_pred = model(X_new)  
y_pred = y_pred.cpu().numpy()

# **7️⃣ Save prediction results**
df_predictions = pd.DataFrame(y_pred, columns=["Predicted_Y"])
df_predictions.to_excel(r"D:\Result\S1\SCARNet\predictions-S1-unlabled.xlsx", index=False)

print("✅ Prediction completed, results saved to predictions.xlsx")
