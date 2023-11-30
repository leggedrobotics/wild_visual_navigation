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
fig = plt.figure(figsize=(12, 14))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.5])

# Metrics to be plotted in each row
row_metrics = [
    ['Friction', 'Stiffness'],
    ['Overconfidence_mean', 'Overconfidence_std'],
    ['Underconfidence_mean', 'Underconfidence_std'],
]

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
            ax.set_ylabel(metric if 'confidence' not in metric else f'{metric} (%)')
            ax.set_title(f'{metric} Metric' if 'confidence' not in metric else f'{metric} Metric (Values in %)')
            ax.legend()

# Process Confidence Mask Accuracy if both metrics are present
if overconfidence_data and underconfidence_data:
    ax = plt.subplot(gs[3, :])  # Span all columns in the last row
    for o_data, u_data, o_label, idx in zip(overconfidence_data, underconfidence_data, overconfidence_files, range(len(overconfidence_files))):
        cma = [100 - o - u for o, u in zip(o_data, u_data)]
        color = color_palette[idx % len(color_palette)]
        ax.plot(steps, cma, label=f'{o_label}',color=color)  # label using overconfidence filename

    ax.set_xlabel('Step')
    ax.set_ylabel('Confidence Mask Accuracy (%)')
    ax.set_title('Confidence Mask Accuracy Over Different Runs (Values in %)')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'Overall_Comparison.png'))
plt.close()
# for metric in metrics:
#     folder_path = os.path.join(parent_folder, metric)
#     csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

#     for file in csv_files:
#         file_path = os.path.join(folder_path, file)
#         df = pd.read_csv(file_path, header=None)
#         steps = df.iloc[:, 0]
#         values = df.iloc[:, -1]
#         label = file.rsplit('.csv', 1)[0]
#         if metric == 'Overconfidence_mean':
#             overconfidence_data.append(values)
#             overconfidence_files.append(label)
#         elif metric == 'Underconfidence_mean':
#             underconfidence_data.append(values)
#             underconfidence_files.append(label)
#         # Plot normal metrics
        
#         plt.figure(figsize=(10, 6))
#         plt.plot(steps, values, label=label)
#         plt.xlabel('Step')
#         plt.ylabel(metric)
#         plt.title(f'{metric} Metric Over Different Runs')
#         plt.legend()
#         # plt.show()
#         plt.savefig(os.path.join(folder_path, f'{metric}.png'))
#         plt.close()

# Process Confidence Mask Accuracy if both metrics are present
# if overconfidence_data and underconfidence_data:
#     plt.figure(figsize=(10, 6))
#     for o_data, u_data, o_label, u_label in zip(overconfidence_data, underconfidence_data, overconfidence_files, underconfidence_files):
#         cma = [100 - o - u for o, u in zip(o_data, u_data)]
#         plt.plot(steps, cma, label=f'{o_label}')  # label combining both filenames

#     plt.xlabel('Step')
#     plt.ylabel('Confidence Mask Accuracy')
#     plt.title('Confidence Mask Accuracy Over Different Runs')
#     plt.legend()
#     os.makedirs(os.path.join(parent_folder, 'Confidence Mask Accuracy'), exist_ok=True)
#     plt.savefig(os.path.join(parent_folder, 'Confidence Mask Accuracy', 'Confidence Mask Accuracy.png'))
#     # plt.show()