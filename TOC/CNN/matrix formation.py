import pandas as pd
import torch

# Read data from Excel files in the training set and test set
train_data = pd.read_excel(r"C:\Users\User\Desktop\test\calculated_curves\training_and_validation_data.xlsx", sheet_name="normalized_train")
test_data = pd.read_excel(r"C:\Users\User\Desktop\test\calculated_curves\training_and_validation_data.xlsx", sheet_name="normalized_test")

# Remove the first column (depth column) from the training set and retain columns 2 through 2 from the end as logging attributes
X_train = train_data.iloc[:, 1:-1].values
Y_train = train_data.iloc[:, -1].values

# Remove the first column (depth column) from the test set and retain columns 2 through 2 from the end as logging attributes
X_test = test_data.iloc[:, 1:-1].values
Y_test = test_data.iloc[:, -1].values

# Convert X_train, Y_train, X_test, and Y_test to PyTorch Tensor format
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

# Save training and test set data as .pt files
torch.save(X_train_tensor, "X_train_well_log_data.pt")
torch.save(Y_train_tensor, "Y_train_labels.pt")

torch.save(X_test_tensor, "X_test_well_log_data.pt")
torch.save(Y_test_tensor, "Y_test_labels.pt")

# Output data shape, ensuring it meets expectations
print(f"Shape of training set input data X: {X_train_tensor.shape}")
print(f"Shape of training set label data Y: {Y_train_tensor.shape}")

print(f"Shape of test set input data X: {X_test_tensor.shape}")
print(f"Shape of test set label data Y: {Y_test_tensor.shape}")

print("The training set and test set data have been successfully saved as .pt files！")
