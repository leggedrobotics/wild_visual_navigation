import os
import pandas as pd
import matplotlib.pyplot as plt
from BaseWVN import WVN_ROOT_DIR
from matplotlib import gridspec
# Parent directory containing the metric folders
parent_folder = os.path.join(WVN_ROOT_DIR, 'results/compare/PretrainorNot')
color_palette = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

# List of metric names corresponding to the folder names
metrics = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]

# Containers for special metrics
overconfidence_data = []
underconfidence_data = []
overconfidence_files = []
underconfidence_files = []
plt.rcParams.update({'font.size': 12})  # Adjust as needed

# Create figure and GridSpec layout
fig = plt.figure(figsize=(8, 10))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.5])

# Metrics to be plotted in each row
row_metrics = [
    ['Friction', 'Stiffness'],
    ['Overconfidence_mean', 'Overconfidence_std'],
    ['Underconfidence_mean', 'Underconfidence_std'],
]
y_labels=[['Mean error','Mean error'],
          ['Value (%)','Value (%)'],
          ['Value (%)','Value (%)'],
          ['Accuracy (%)']]

# Process each metric
for i, row in enumerate(row_metrics):
    for j, metric in enumerate(row):
        ax = plt.subplot(gs[i, j])
        folder_path = os.path.join(parent_folder, metric)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for file_idx, file in enumerate(csv_files):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=None)
            steps = df.iloc[:, 0]
            values = df.iloc[:, -1]
            label = file.rsplit('.csv', 1)[0]
            color = color_palette[file_idx % len(color_palette)]  # Cycle through the color palette
            if metric == 'Overconfidence_mean':
                overconfidence_data.append(values)
                overconfidence_files.append(label)
            elif metric == 'Underconfidence_mean':
                underconfidence_data.append(values)
                underconfidence_files.append(label)
            ax.plot(steps, values, label=label,color=color)  # label using filename
            ax.set_xlabel('Step')
            ax.set_ylabel(y_labels[i][j] )
            ax.set_title(f'{metric}' if 'confidence' not in metric else f'{metric} (in %)')
            ax.legend()

# Process Confidence Mask Accuracy if both metrics are present
if overconfidence_data and underconfidence_data:
    ax = plt.subplot(gs[3, :])  # Span all columns in the last row
    for o_data, u_data, o_label, idx in zip(overconfidence_data, underconfidence_data, overconfidence_files, range(len(overconfidence_files))):
        cma = [100 - o - u for o, u in zip(o_data, u_data)]
        color = color_palette[idx % len(color_palette)]
        ax.plot(steps, cma, label=f'{o_label}',color=color)  # label using overconfidence filename

    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Confidence Mask Accuracy (in %)')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'Overall_Comparison.png'), dpi=300)
plt.close()


folder_path = os.path.join(parent_folder, metric)
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

for file_idx, file in enumerate(csv_files):
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, header=None)
    steps = df.iloc[:, 0]
    values = df.iloc[:, -1]
    label = file.rsplit('.csv', 1)[0]
    color = color_palette[file_idx % len(color_palette)]  # Cycle through the color palette
    if metric == 'Overconfidence_mean':
        overconfidence_data.append(values)
        overconfidence_files.append(label)
    elif metric == 'Underconfidence_mean':
        underconfidence_data.append(values)
        underconfidence_files.append(label)
    ax.plot(steps, values, label=label,color=color)  # label using filename
    ax.set_xlabel('Step')
    ax.set_ylabel(y_labels[i][j] )
    ax.set_title(f'{metric}' if 'confidence' not in metric else f'{metric} (in %)')
    ax.legend()