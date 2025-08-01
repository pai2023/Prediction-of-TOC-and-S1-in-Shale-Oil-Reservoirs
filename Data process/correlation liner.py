import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Data loading
df = pd.read_excel(r"D:\Result\Training Data\Batch Normalization + Unlabeled Data.xlsx", sheet_name="Sheet2")

# Configure drawing parameters
config = {
    "font.family": 'serif',
    "font.size": 20,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
    'axes.unicode_minus': False
}
plt.rcParams.update(config)

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.grid.axis'] = 'y'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['ytick.major.width'] = 1.5

# Define analysis parameters
log_curves = ['GR', 'SP', 'M2R6', 'RT', 'DEN', 'AC', 'CN', 'SH']
# Target parameter settings
target_config = {
    'TOC': {
        'label': 'TOC (%)',
        'filename': 'TOC'
    },
    'S1': {
        'label': r'S$_1$ (mg/g)',
        'filename': 'S1'
    }
}

curve_units = {
    'GR': ('GR', 'API'),
    'SP': ('SP', 'mV'),
    'M2R6': ('M2R6', 'Ω·m'),
    'RT': ('RT', 'Ω·m'),
    'SH': ('SH', 'v/v'),
    'DEN': ('DEN', 'g/cm³'),
    'AC': ('AC', 'μs/ft'),
    'CN': ('CN', 'PU')
}

for target in target_config:
    fig = plt.figure(figsize=(24, 12), dpi=100)

    for idx, curve in enumerate(log_curves, 1):
        ax = plt.subplot(2, 4, idx)
        plt.tick_params(axis='both', direction='in')

        # Drawing regression graphs
        sns.regplot(x=df[curve], y=df[target],
                    ci=95, color='#007CB9',
                    scatter_kws={'s': 150, 'color': 'red', 'edgecolor': 'white'})

        # Set axis labels
        curve_name, unit = curve_units[curve]
        plt.xlabel(f"{curve_name} ({unit})", fontsize=24)
        plt.ylabel(target_config[target]['label'], fontsize=24)

        # Set the y-axis range
        if target == 'TOC':
            plt.ylim(0, 9)
        elif target == 'S1':
            plt.ylim(0, 11)
        # Calculate regression parameters
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[curve], df[target])
        r_squared = r_value ** 2

        plt.text(0.05, 0.95, f"({chr(96 + idx)})",
                 ha='center', va='center',
                 transform=ax.transAxes, fontsize=24)
        plt.text(0.14, 0.86, f'R² = {r_squared:.2f}',
                 ha='center', va='center',
                 transform=ax.transAxes, fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.98,
                        top=0.95, bottom=0.07,
                        wspace=0.2, hspace=0.3)

    plt.savefig(f'{target_config[target]["filename"]}_Correlation_Analysis.png', dpi=300)
    plt.show()