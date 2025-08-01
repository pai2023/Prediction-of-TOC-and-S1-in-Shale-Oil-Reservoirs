import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# **1️⃣ Device detection**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# **2️⃣ Read the validation set data**
validation_data = pd.read_excel(r"C:\Users\User\Desktop\test\calculated_curves\S1\training_and_validation_data-S1.xlsx",
                                sheet_name="Validation_Set")

# **3️⃣ Extract data**
depths = validation_data.iloc[:, 0].values  # In-depth column
X_val = validation_data.iloc[:, 1:-1].values  # Feature columns (excluding depth and true values)
y_val = validation_data.iloc[:, -1].values  # Real Label

# 🚨 **Ensure that the data format is correct**
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

# **4️⃣ Load the CNN model**
from SCARNetS1 import CNNRegressor

input_size = X_val.shape[1]
model = CNNRegressor(input_size).to(device)

# 🚨 **Loading weights for different models**
models = {
    "Train Best": "best_train_model-S1.pth",
    "Test Best": "best_test_model-S1.pth",
    "Overall Best": "best_overall_model-S1.pth",
    "Supervised Best": "best_supervised_model.pth"
}

# **5️⃣ Record the prediction results of all models**
results = {"Depth": depths, "True Values": y_val.cpu().numpy().flatten()}
model_scores = {}

best_r2_val = -np.inf
best_model_name = None
best_model = None

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

for model_name, model_path in models.items():
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        y_pred_val = model(X_val).cpu().numpy()

    y_true_val = y_val.cpu().numpy()
    r2_val = r2_score(y_true_val, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
    mae_val = mean_absolute_error(y_true_val, y_pred_val)
    mape_val = mean_absolute_percentage_error(y_true_val, y_pred_val)

    model_scores[model_name] = {"R²": r2_val, "RMSE": rmse_val, "MAE": mae_val, "MAPE": mape_val}

    results[f'Predicted ({model_name})'] = y_pred_val.flatten()

    if r2_val > best_r2_val:
        best_r2_val = r2_val
        best_model_name = model_name
        best_model = model

# **🛠 Handling cases where all models have the same R²**
r2_values = [round(score["R²"], 4) for score in model_scores.values()]
if len(set(r2_values)) == 1:
    print("⚠️ All models have the same R², so choose the model with the lowest RMSE")
    best_model_name = min(model_scores, key=lambda x: (model_scores[x]["RMSE"], model_scores[x]["MAE"]))

    # ✅ Reload model
    best_model = CNNRegressor(input_size).to(device)
    best_model.load_state_dict(torch.load(models[best_model_name]))

# **✅ Save the final best model**
if best_model is not None:
    torch.save(best_model.state_dict(), "best_final_model-S1.pth")

# Add `best_supervised_model.pth` to save
torch.save(best_model.state_dict(), "best_supervised_model.pth")

print(f"\n🎉 The best model ultimately selected：{best_model_name}（R²: {best_r2_val:.4f}）")

# **6️⃣ Save results to Excel**
df_val = pd.DataFrame(results)
output_path = "validation_results_all_models-S1.xlsx"
with pd.ExcelWriter(output_path) as writer:
    df_val.to_excel(writer, sheet_name='Validation', index=False)

# **7️⃣ Save the scoring table**
df_scores = pd.DataFrame.from_dict(model_scores, orient='index')
df_scores.to_excel("model_validation_scores-S1.xlsx")
df_scores.sort_values(by="R²", ascending=False).to_excel("model_validation_scores_sorted-S1.xlsx")

print("\n📊 Scores of each model on the validation set：")
print(df_scores)

print(f"\n✅ The verification results have been saved to {output_path}")
print(f"✅ The scorecard has been saved as 'model_validation_scores-S1.xlsx'")
print(f"✅ The sorted score sheet has been saved as 'model_validation_scores_sorted-S1.xlsx'")
print(f"✅ The final best model has been saved as 'best_final_model-S1.pth'")
print(f"✅ The final best supervised learning model has been saved as 'best_supervised_model.pth'")