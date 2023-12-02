import rosbag
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from BaseWVN.config.wvn_cfg import ParamCollection
from BaseWVN import WVN_ROOT_DIR
def read_data(bag_path, topic_name, dims=341):
    data = []
    with rosbag.Bag(bag_path, 'r') as bag:
        total_messages = bag.get_message_count(topic_filters=[topic_name])
        progress_bar = tqdm(total=total_messages, desc=f'Reading {bag_path}')
        for _, msg, _ in bag.read_messages(topics=[topic_name]):
            array_data = np.array(msg.data)
            data.append(array_data[:dims])
            progress_bar.update(1)
        progress_bar.close()
    return np.array(data)

def analyze_range(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return min_vals, max_vals

def plot_heatmap(data, title, vmin, vmax, timestep_limit=5000):
    plt.figure(figsize=(15, 10))
    # Limit to the first 200 timesteps
    data_limited = data[:timestep_limit].T
    sns.heatmap(data_limited, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Dimension')
    plt.show()

def write_comparison_to_file(comparison, file_path):
    with open(file_path, 'w') as file:
        for comp in comparison:
            file.write(f"Dimension {comp['dimension']}:\n")
            file.write(f"  Bag 1 - Min: {comp['min_bag_1']:.2f}, Max: {comp['max_bag_1']:.2f}\n")
            file.write(f"  Bag 2 - Min: {comp['min_bag_2']:.2f}, Max: {comp['max_bag_2']:.2f}\n")
        print(f"Comparison data written to {file_path}")

param=ParamCollection()

# Paths to your ROS bags
bag_path_1 = '/media/chen/UDisk1/vis_rosbag/snow/2022-12-10-15-40-10_anymal-d020-lpc_mission_0.bag'
bag_path_2 = '/media/chen/UDisk1/catkin_rosbag/2021-05-03-19-18-21.bag'
topic_name = '/debug_info'
analyze_folder=os.path.join(WVN_ROOT_DIR,param.offline.analyze_path)
os.makedirs(analyze_folder,exist_ok=True)
if os.path.exists(os.path.join(analyze_folder,'bag_snow.npy')): # if already analyzed
    data_1=np.load(os.path.join(analyze_folder,'bag_snow.npy'))
else:
    data_1 = read_data(bag_path_1, topic_name)
    np.save(os.path.join(analyze_folder,'bag_snow.npy'),data_1)
min_values_1, max_values_1 = analyze_range(data_1)

if os.path.exists(os.path.join(analyze_folder,'bag_catkin.npy')): # if already analyzed
    data_2=np.load(os.path.join(analyze_folder,'bag_catkin.npy'))
else:
    data_2 = read_data(bag_path_2, topic_name)
    np.save(os.path.join(analyze_folder,'bag_catkin.npy'),data_2)
min_values_2, max_values_2 = analyze_range(data_2)

data_1=data_1[:,0:48]
data_2=data_2[:,0:48]

overall_min = min(np.nanmin(data_1), np.nanmin(data_2))
overall_max = max(np.nanmax(data_1), np.nanmax(data_2))

# Plot the heatmaps with the unified range and timestep limit
plot_heatmap(data_1, "Heatmap for ROS Bag 1", overall_min, overall_max)
plot_heatmap(data_2, "Heatmap for ROS Bag 2", overall_min, overall_max)

comparison = []
for i in range(133):
    comparison.append({
        'dimension': i,
        'min_bag_1': min_values_1[i],
        'max_bag_1': max_values_1[i],
        'min_bag_2': min_values_2[i],
        'max_bag_2': max_values_2[i]
    })

output_file_path = os.path.join(analyze_folder, 'bag_comparison.txt')
write_comparison_to_file(comparison, output_file_path)
