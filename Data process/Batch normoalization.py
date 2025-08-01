import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

file_path = r"C:\Users\User\Desktop\test\calculated_curves\data.xlsx"
columns_to_normalize = ['GR', 'SP', 'RT', 'M2R6', 'CN', 'DEN', 'AC', 'SH']
scaler_path = "minmax_scaler_from_file.pkl"

df_original = pd.read_excel(file_path, sheet_name='Sheet1')

scaler = MinMaxScaler()
scaler.fit(df_original[columns_to_normalize])
print("✅ Successfully obtain normalized parameters from Sheet1")

joblib.dump(scaler, scaler_path)
print(f"✅ Scaler has been saved as {scaler_path} and can be used for subsequent batch normalization!")

new_df = pd.read_excel(file_path, sheet_name='New_Data')

# Normalize new data
new_df_normalized = new_df.copy()
new_df_normalized[columns_to_normalize] = scaler.transform(new_df[columns_to_normalize])
print("✅ New data normalization complete!")

with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    new_df_normalized.to_excel(writer, sheet_name='New_Data_Normalized', index=False)

print("✅ The normalized results have been saved to the ‘New_Data_Normalized’ form in Excel!")
