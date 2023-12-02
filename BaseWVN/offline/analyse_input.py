import rosbag
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from BaseWVN.config.wvn_cfg import ParamCollection
def read_data(bag_path, topic_name, dims=133):
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
def plot_heatmap(data, title):
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.T, cmap='viridis')  # Transpose to make dimensions run down the y-axis
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

data_1 = read_data(bag_path_1, topic_name)
min_values_1, max_values_1 = analyze_range(data_1)

data_2 = read_data(bag_path_2, topic_name)
min_values_2, max_values_2 = analyze_range(data_2)

plot_heatmap(data_1, "Time-Series Heatmap for ROS Bag 1")
plot_heatmap(data_2, "Time-Series Heatmap for ROS Bag 2")

comparison = []
for i in range(133):
    comparison.append({
        'dimension': i,
        'min_bag_1': min_values_1[i],
        'max_bag_1': max_values_1[i],
        'min_bag_2': min_values_2[i],
        'max_bag_2': max_values_2[i]
    })

output_file_path = 'rosbag_comparison_output.txt'
write_comparison_to_file(comparison, output_file_path)
