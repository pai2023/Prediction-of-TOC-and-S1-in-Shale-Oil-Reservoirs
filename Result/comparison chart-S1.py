import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Breakage depth range
top_range = (3820, 3900)
bottom_range = (4050, 4150)


x_min = 0
x_max = 7
x_interval = 3


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2
})


true_color = '#2f2f2f'
pred_color = '#d62728'
true_marker = 's'
pred_marker = 'o'

models = {
    'RNN': r"D:\Result\S1\RNN\optimized_data.xlsx",
    'SSRNN': r"D:\Result\S1\SSRNN\optimized_data.xlsx",
    'CNN': r"D:\Result\S1\SCARNet\optimized_data.xlsx",
    'SCARNet': r"D:\Result\S1\CNN\optimized_data.xlsx"
}

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(6.2, 5.8), dpi=300,
                         gridspec_kw={'height_ratios': [1, 1]}, sharex='col')

for idx, (model_name, file_path) in enumerate(models.items()):
    df = pd.read_excel(file_path)
    depth = df['Depth']
    true_values = df['True Values']
    predicted_values = df['Predicted Values']
    rmse = np.sqrt(np.mean((true_values - predicted_values) ** 2))

    top_mask = (depth >= top_range[0]) & (depth <= top_range[1])
    bot_mask = (depth >= bottom_range[0]) & (depth <= bottom_range[1])

    ax_top = axes[0, idx]
    ax_bot = axes[1, idx]


    ax_top.plot(true_values[top_mask], depth[top_mask],
                linestyle='-', marker=true_marker, markersize=3, color=true_color,
                label='True' if idx == 0 else '')
    ax_top.plot(predicted_values[top_mask], depth[top_mask],
                linestyle='--', marker=pred_marker, markersize=3, color=pred_color,
                label='Predicted' if idx == 0 else '')
    ax_top.set_ylim(top_range[1], top_range[0])
    ax_top.set_facecolor("#f2f2f2")
    ax_top.spines['bottom'].set_visible(False)


    ax_bot.plot(true_values[bot_mask], depth[bot_mask],
                linestyle='-', marker=true_marker, markersize=3, color=true_color)
    ax_bot.plot(predicted_values[bot_mask], depth[bot_mask],
                linestyle='--', marker=pred_marker, markersize=3, color=pred_color)
    ax_bot.set_ylim(bottom_range[1], bottom_range[0])
    ax_bot.set_xlim(x_min, x_max)
    ax_bot.set_xticks(np.arange(x_min, x_max + 1e-6, x_interval))
    ax_bot.set_facecolor("#e6f2ff")
    ax_bot.spines['top'].set_visible(False)

    for ax in [ax_top, ax_bot]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    ax_bot.set_xlabel("S₁ (mg/g)", fontname='Times New Roman')
    if idx == 0:

        ax_bot.set_yticks(np.arange(bottom_range[0], bottom_range[1] + 1, 50))

        ax_top.set_yticks(np.arange(top_range[0], top_range[1] + 1, 40))
    else:
        ax_top.set_yticklabels([])
        ax_bot.set_yticklabels([])

    ax_top.set_title(f"{model_name}\nRMSE = {rmse:.2f}", pad=4, fontname='Times New Roman')


    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=0.6)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


fig.text(0.005, 0.5, "Depth (m)", va='center', rotation='vertical',
         fontname='Times New Roman', fontsize=10)


fig.legend(labels=['True', 'Predicted'],
           loc='upper right',
           bbox_to_anchor=(1.05, 0.905),
           frameon=False,
           prop={'family': 'Times New Roman', 'size': 10})


fig.text(0.905, 0.70, "GD12", ha='left', va='center',
         fontname='Times New Roman', fontsize=10)
fig.text(0.905, 0.28, "GD14", ha='left', va='center',
         fontname='Times New Roman', fontsize=10)


plt.subplots_adjust(left=0.08, right=0.88, hspace=0.09, wspace=0.09)
plt.savefig("S1_Comparison_FinalFineTuned-1.png", dpi=600, bbox_inches='tight')
plt.show()
