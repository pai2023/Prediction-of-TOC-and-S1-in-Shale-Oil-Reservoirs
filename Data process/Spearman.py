import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


data_file = r"D:\Result\Training Data\Batch Normalization + Unlabeled Data.xlsx"
df = pd.read_excel(data_file, sheet_name="Sheet2")

well_logs = ['GR', 'SP', 'M2R6', 'RT', 'DEN', 'AC', 'CN', 'SH']
target_vars = ['TOC', 'S1']
columns = well_logs + target_vars
df_selected = df[columns]


display_columns = ['S₁' if col == 'S1' else col for col in columns]

# Calculate Spearman's correlation coefficient
corr_matrix, _ = spearmanr(df_selected, axis=0)
corr_df = pd.DataFrame(corr_matrix, index=display_columns, columns=display_columns).round(2)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_df,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5,
    cbar_kws={'shrink': 0.8},
    annot_kws={"size": 20, "fontname": "Times New Roman"}
)
plt.xticks(fontsize=20, fontname='Times New Roman', rotation=45)
plt.yticks(fontsize=20, fontname='Times New Roman', rotation=0)

plt.tight_layout()
plt.savefig('Spearman_Correlation_Analysis.png', dpi=300)
plt.show()
