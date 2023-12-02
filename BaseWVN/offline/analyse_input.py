import rosbag
import numpy as np

def read_data(bag_path, topic_name, dims=133):
    data = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for _, msg, _ in bag.read_messages(topics=[topic_name]):
            # Adjust the line below according to your message structure
            array_data = np.array(msg.data)
            data.append(array_data[:dims])
    return np.array(data)

def analyze_range(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return min_vals, max_vals

# Paths to your ROS bags
bag_path_1 = '/media/chen/UDisk1/vis_rosbag/snow/2022-12-10-15-40-10_anymal-d020-lpc_mission_0.bag'
bag_path_2 = '/media/chen/UDisk1/catkin_rosbag/2021-05-03-19-18-21.bag'
topic_name = '/debug_info'

# Read and analyze data from both ROS bags separately
data_1 = read_data(bag_path_1, topic_name)
min_values_1, max_values_1 = analyze_range(data_1)

data_2 = read_data(bag_path_2, topic_name)
min_values_2, max_values_2 = analyze_range(data_2)

# Compare the ranges for each dimension
comparison = []
for i in range(133):
    comparison.append({
        'dimension': i,
        'min_bag_1': min_values_1[i],
        'max_bag_1': max_values_1[i],
        'min_bag_2': min_values_2[i],
        'max_bag_2': max_values_2[i]
    })

# Print or process the comparison
# This is a simple print, you can format or visualize this data as needed
for comp in comparison:
    print(f"Dimension {comp['dimension']}:")
    print(f"  Bag 1 - Min: {comp['min_bag_1']}, Max: {comp['max_bag_1']}")
    print(f"  Bag 2 - Min: {comp['min_bag_2']}, Max: {comp['max_bag_2']}")
