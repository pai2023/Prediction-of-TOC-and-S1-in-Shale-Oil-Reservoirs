import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 1. Load unlabeled data (including depth columns)
unlabeled_file = r"D:\Result\Train data\training_and_validation_data.xlsx"
unlabeled_df = pd.read_excel(unlabeled_file, sheet_name="111")

# Extract data
depth_values = unlabeled_df.iloc[:, 0].values
X_unlabeled = unlabeled_df.iloc[:, 1:].values  

# 2. Data preprocessing (exactly the same as during training)
X_unlabeled_tensor = torch.tensor(X_unlabeled, dtype=torch.float32)


# 3. Define the model structure (must be exactly the same as during training)
class RNNRegressor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNRegressor, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 4. Model parameter configuration (must be consistent with training)
input_size = 8
hidden_size = 32
num_layers = 3

# 5. Load the best model
try:
    model = RNNRegressor(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(r"D:\Result\S1\SSRNN\best_rnn_model_val.pth"))  # Load the best saved model
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully! Current model parameters：")
    print(f"- Input dimension: {input_size}")
    print(f"- Hidden layer dimension: {hidden_size}")
    print(f"- Number of RNN layers: {num_layers}")
except Exception as e:
    print(f"Model loading failed! Please check：{str(e)}")
    exit()

# 6. Create DataLoader (supports big data batch prediction)
batch_size = 256
dataset = TensorDataset(X_unlabeled_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 7. Execute forecast
predictions = []
try:
    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            inputs = batch[0].unsqueeze(1)  # Add time step dimension [batch, 1, features]
            batch_pred = model(inputs)
            predictions.append(batch_pred.numpy())
    predictions = np.concatenate(predictions, axis=0).squeeze()
    print(f"Prediction complete! A total of {len(predictions)} samples were processed.")
except RuntimeError as e:
    print(f"Errors occurring during the prediction process：{str(e)}")
    exit()

# 8. Construct the result DataFrame
results_df = pd.DataFrame({
    'Depth(m)': depth_values,
    'Predicted_S1(%)': predictions  # Modify column names based on actual conditions
})

# 9. Data validation
if len(depth_values) != len(predictions):
    print("Warning: The depth of data does not match the number of prediction results!")
else:
    print("Data validation passed, depth corresponds to prediction results one-to-one.")

# 10. Save results to Excel (including depth column)
try:
    output_path = 'SSRNN_unlabeled_prediction.xlsx'
    results_df.to_excel(output_path, index=False)

    # Using openpyxl for format optimization
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
        workbook = writer.book
        worksheet = workbook.active

        # Set column width
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 18

        # Set number format
        for row in worksheet.iter_rows(min_row=2, max_col=2, max_row=len(results_df) + 1):
            for cell in row:
                cell.number_format = '0.000'

    print(f"The prediction results have been saved to：{output_path}")
except Exception as e:
    print(f"File save failed：{str(e)}")

# 11. Visualization of depth-predicted value curves
plt.figure(figsize=(12, 6))
plt.plot(results_df['Depth(m)'], results_df['Predicted_S1(mg/g)'],
         color='#2E86C1', linewidth=1.2, linestyle='-')

# Reverse Y-axis (increasing depth direction)
plt.gca().invert_yaxis()

# Set axis format
plt.xlabel('Depth (m)', fontsize=12, labelpad=10)
plt.ylabel('Predicted S1 (mg/g)', fontsize=12, labelpad=10)
plt.title('TOC Prediction Profile', fontsize=14, pad=15)
plt.grid(True, linestyle='--', alpha=0.7)

# Set scale style
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tick_params(axis='y', which='both', direction='in', right=True)
plt.tick_params(axis='x', which='both', direction='in', top=True)

# Save high-definition images
plt.savefig('TOC_depth_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. Statistical Information Output
print("\n预测结果统计：")
print(f"- 最大值: {results_df['Predicted_S1(mg/g)'].max():.3f}")
print(f"- 最小值: {results_df['Predicted_S1(mg/g)'].min():.3f}")
print(f"- 平均值: {results_df['Predicted_S1(mg/g)'].mean():.3f}")
print(f"- 标准差: {results_df['Predicted_S1(mg/g)'].std():.3f}")