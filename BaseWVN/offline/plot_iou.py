import pandas as pd
import matplotlib.pyplot as plt

def read_csv_mean_std(method_name, folder_mean='iou_mean', folder_std='iou_std'):
    """Reads mean and standard deviation CSVs for a given method."""
    mean_path = f'/home/chen/BaseWVN/results/compare/{folder_mean}/{method_name}.csv'
    std_path = f'/home/chen/BaseWVN/results/compare/{folder_std}/{method_name}.csv'
    
    # Read mean and std data
    mean_df = pd.read_csv(mean_path, usecols=[0, 2], header=None, names=['timestep', 'iou_mean'])
    std_df = pd.read_csv(std_path, usecols=[2], header=None, names=['iou_std'])
    
    return mean_df['timestep'], mean_df['iou_mean'], std_df['iou_std']

def plot_with_shades(timestep, mean, std, label,color):
    """Plots a curve with shaded standard deviation."""
    plt.plot(timestep, mean, label=label,color=color)
    plt.fill_between(timestep, mean-std, mean+std, alpha=0.1,color=color)

# Plot settings
plt.figure(figsize=(5, 3))

methods = ['ours', 'fixed_threshold']
# Define methods and their respective colors (RGB)
methods_colors = {
    'ours': (1.0, 0.0, 0.0),  # Red
    # 'RND': (0,139/255,139/255),   # Green
    'fixed_threshold': (31/255,119/255,180/255) # Blue
}

for method in methods:
    timestep, mean, std = read_csv_mean_std(method)
    plot_with_shades(timestep, mean, std, method,methods_colors[method])


plt.xlabel('Step')
plt.ylabel('IOU (%)')
plt.title('IOU over Time by Method')
plt.tight_layout()
plt.legend()
plt.savefig('/home/chen/BaseWVN/results/compare/iou_over_time.pdf')
plt.show()