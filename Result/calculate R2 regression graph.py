import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
# Read Excel file
file_path = r"D:\Result\S1\SCARNet\optimized_data-S1.xlsx"


data = pd.read_excel(file_path, sheet_name="Sheet1")

# Extract True_Values and Predicted_Values
true_values = data['True Values']
predicted_values = data['Predicted Values']
# Calculate R² (coefficient of determination)
r2 = r2_score(true_values, predicted_values)
print(f"R²: {r2:.4f}")
# Set Seaborn style
sns.set(style="whitegrid")


plt.figure(figsize=(10, 6))
sns.regplot(x=true_values, y=predicted_values, ci=95, color='#007CB9',
            scatter_kws={'s': 150, 'color': 'red', 'edgecolor': 'white', 'alpha': 0.7})
plt.tick_params(axis='both', direction='in', labelsize=12)
plt.xlabel(r'$\mathrm{S1}_{\mathrm{mea}}$(mg/g)', fontsize=18)
plt.ylabel(r'$\mathrm{S1}_{\mathrm{SCARNet}}$(mg/g)', fontsize=18)
plt.title("True vs Predicted Values", fontsize=16)
plt.grid(ls='--', alpha=0.3)
plt.tight_layout()
plt.savefig('S1_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()



