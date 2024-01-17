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
    mean_vals = np.mean(data, axis=0)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    std_vals = np.std(data, axis=0)
    return min_vals, max_vals,mean_vals,std_vals

def plot_heatmap(data, title, vmin, vmax,folder, timestep_limit=6000):
    plt.figure(figsize=(15, 10))
    # Limit to the first 200 timesteps
    data_limited = data[:timestep_limit].T
    sns.heatmap(data_limited, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Dimension')
    plt.savefig(os.path.join(folder,title+".png"))
    plt.show()
    
# def write_comparison_to_file(comparison, file_path, feature_name_1, feature_name_2):
#     with open(file_path, 'w') as file:
#         file.write(f"Comparison between '{feature_name_1}' and '{feature_name_2}'\n")
#         for comp in comparison:
#             file.write(f"Dimension {comp['dimension']}:\n")
#             file.write(f"  {feature_name_1} - Min: {comp['min_bag_1']:.2f}, Max: {comp['max_bag_1']:.2f}, Mean: {comp['mean_bag_1']:.2f}\n")
#             file.write(f"  {feature_name_2} - Min: {comp['min_bag_2']:.2f}, Max: {comp['max_bag_2']:.2f}, Mean: {comp['mean_bag_2']:.2f}\n")
#         print(f"Comparison data written to {file_path}")


# param=ParamCollection()

# # Paths to your ROS bags
# bag_path_snow = '/media/chen/UDisk1/vis_rosbag/snow/2022-12-10-15-40-10_anymal-d020-lpc_mission_0.bag'
# feature_name_1 = 'SnowDodo'
# bag_path_foam = '/media/chen/UDisk1/catkin_rosbag/2021-05-03-19-18-21.bag'
# feature_name_2='FoamCC'
# topic_name = '/debug_info'
# analyze_folder=os.path.join(WVN_ROOT_DIR,param.offline.analyze_path)
# os.makedirs(analyze_folder,exist_ok=True)
# if os.path.exists(os.path.join(analyze_folder,'bag_snow.npy')): # if already analyzed
#     data_1=np.load(os.path.join(analyze_folder,'bag_snow.npy'))
# else:
#     data_1 = read_data(bag_path_snow, topic_name)
#     np.save(os.path.join(analyze_folder,'bag_snow.npy'),data_1)
# min_values_1, max_values_1,mean_values_1 = analyze_range(data_1)

# if os.path.exists(os.path.join(analyze_folder,'bag_catkin.npy')): # if already analyzed
#     data_2=np.load(os.path.join(analyze_folder,'bag_catkin.npy'))
# else:
#     data_2 = read_data(bag_path_foam, topic_name)
#     np.save(os.path.join(analyze_folder,'bag_catkin.npy'),data_2)
# min_values_2, max_values_2,mean_values_2 = analyze_range(data_2)

# timesteps=6000
# dim_1=0
# dim_2=133
# data_1=data_1[:,dim_1:dim_2]
# data_2=data_2[:,dim_1:dim_2]
# d_data=abs(data_1[:timesteps])-abs(data_2[:timesteps])

# overall_min = min(np.nanmin(data_1), np.nanmin(data_2))
# overall_max = max(np.nanmax(data_1), np.nanmax(data_2))

# d_min=np.nanmin(d_data)
# d_max=np.nanmax(d_data)
# # Plot the heatmaps with the unified range and timestep limit
# plot_heatmap(data_1, f"Heatmap for ROS Bag {feature_name_1}", overall_min, overall_max,analyze_folder)
# plot_heatmap(data_2, f"Heatmap for ROS Bag {feature_name_2}", overall_min, overall_max,analyze_folder)
# plot_heatmap(d_data, "Heatmap for Difference", d_min, d_max,analyze_folder)
# comparison = []
# for i in range(341):
#     comparison.append({
#         'dimension': i,
#         'min_bag_1': min_values_1[i],
#         'max_bag_1': max_values_1[i],
#         'min_bag_2': min_values_2[i],
#         'max_bag_2': max_values_2[i],
#         'mean_bag_1': mean_values_1[i],
#         'mean_bag_2': mean_values_2[i],
#     })

# output_file_path = os.path.join(analyze_folder, f'comparison_{feature_name_1}_vs_{feature_name_2}.txt')
# write_comparison_to_file(comparison, output_file_path, feature_name_1, feature_name_2)

bag_feature_pairs = [
    ('/media/chen/UDisk1/vis_rosbag/snow/2022-12-10-15-40-10_anymal-d020-lpc_mission_0.bag', 'SnowDodo'),
    ('/media/chen/UDisk1/vis_rosbag/single_test/2023-09-20-09-43-57_anymal-d020-lpc_0-001.bag','HikingDodo'),
    ('/media/chen/UDisk1/catkin_rosbag/2021-05-03-19-18-21.bag', 'FoamCC'),
    
    # Add more bags as needed
]

def process_bag(bag_path, feature_name, topic_name, analyze_folder, dims):
    npy_file_path = os.path.join(analyze_folder, f'bag_{feature_name}.npy')
    if os.path.exists(npy_file_path):
        data = np.load(npy_file_path)
    else:
        data = read_data(bag_path, topic_name, dims)
        np.save(npy_file_path, data)
    return data

def compare_bags(bag_feature_pairs, topic_name, analyze_folder, timesteps, dim_start, dim_end):
    data_dict = {}
    min_max_mean_dict = {}
    
    # Process each bag
    for bag_path, feature_name in bag_feature_pairs:
        data = process_bag(bag_path, feature_name, topic_name, analyze_folder, dim_end)
        data_dict[feature_name] = data[:, dim_start:dim_end]
        min_max_mean_dict[feature_name] = analyze_range(data_dict[feature_name][:timesteps])

    # Write comparison for all bags
    comparison = []
    for i in range(dim_end - dim_start):
        comp_info = {'dimension': i}
        for _, feature_name in bag_feature_pairs:
            min_val, max_val, mean_val,std_val = min_max_mean_dict[feature_name]
            comp_info[f'min_{feature_name}'] = min_val[i]
            comp_info[f'max_{feature_name}'] = max_val[i]
            comp_info[f'mean_{feature_name}'] = mean_val[i]
            comp_info[f'std_{feature_name}'] = std_val[i]
        comparison.append(comp_info)

    output_file_path = os.path.join(analyze_folder, 'bags_comparison.txt')
    write_comparison_to_file(comparison, output_file_path, bag_feature_pairs)

def write_comparison_to_file(comparison, file_path, bag_feature_pairs):
    with open(file_path, 'w') as file:
        file.write("Side-by-Side Comparison\n")
        for comp in comparison:
            file.write(f"Dimension {comp['dimension']}:\n")
            for _, feature_name in bag_feature_pairs:
                file.write(f"  {feature_name} - Min: {comp[f'min_{feature_name}']:.2f}, Max: {comp[f'max_{feature_name}']:.2f}, Mean: {comp[f'mean_{feature_name}']:.2f}, Std: {comp[f'std_{feature_name}']:.2f}\n")
            file.write('\n')
        print(f"Comparison data written to {file_path}")

# Usage

param=ParamCollection()

# Paths to your ROS bags
# bag_path_snow = '/media/chen/UDisk1/vis_rosbag/snow/2022-12-10-15-40-10_anymal-d020-lpc_mission_0.bag'
# feature_name_1 = 'SnowDodo'
# bag_path_foam = '/media/chen/UDisk1/catkin_rosbag/2021-05-03-19-18-21.bag'
# feature_name_2='FoamCC'

analyze_folder=os.path.join(WVN_ROOT_DIR,param.offline.analyze_path)
topic_name = '/debug_info'
timesteps = 14000
dim_1 = 0
dim_2 = 341
compare_bags(bag_feature_pairs, topic_name, analyze_folder, timesteps, dim_1, dim_2)