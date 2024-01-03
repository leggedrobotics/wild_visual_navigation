# import rosbag
# import os
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from BaseWVN.config.wvn_cfg import ParamCollection
# from BaseWVN import WVN_ROOT_DIR
# import deepdish as dd
# def read_data(bag_path, topic_name, dims=341):
#     data = []
#     with rosbag.Bag(bag_path, 'r') as bag:
#         total_messages = bag.get_message_count(topic_filters=[topic_name])
#         progress_bar = tqdm(total=total_messages, desc=f'Reading {bag_path}')
#         for _, msg, _ in bag.read_messages(topics=[topic_name]):
#             array_data = np.array(msg.data)
#             data.append(array_data[:dims])
#             progress_bar.update(1)
#         progress_bar.close()
#     return np.array(data)

# def analyze_range(data):
#     mean_vals = np.mean(data, axis=0)
#     min_vals = np.min(data, axis=0)
#     max_vals = np.max(data, axis=0)
#     std_vals = np.std(data, axis=0)
#     return min_vals, max_vals,mean_vals,std_vals

# def process_bag(bag_path, feature_name, topic_name, analyze_folder, dims):
#     npy_file_path = os.path.join(analyze_folder, f'bag_{feature_name}.npy')
#     if os.path.exists(npy_file_path):
#         data = np.load(npy_file_path)
#     else:
#         data = read_data(bag_path, topic_name, dims)
#         np.save(npy_file_path, data)
#     return data

# def write_comparison_to_file(comparison, file_path, bag_feature_names):
#     with open(file_path, 'w') as file:
#         file.write("Side-by-Side Comparison\n")
#         for comp in comparison:
#             file.write(f"Dimension {comp['dimension']}:\n")
#             for feature_name in bag_feature_names:
#                 file.write(f"  {feature_name} - Min: {comp[f'min_{feature_name}']:.2f}, Max: {comp[f'max_{feature_name}']:.2f}, Mean: {comp[f'mean_{feature_name}']:.2f}, Std: {comp[f'std_{feature_name}']:.2f}\n")
#             file.write('\n')
#         print(f"Comparison data written to {file_path}")

# def compare_bags(bag_feature_names, data_lists ,analyze_folder):
#     data_dict = {}
#     min_max_mean_dict = {}
#     dim_start = 0
#     dim_end = 341
#     # Process each bag
#     for data, feature_name in zip(data_lists, bag_feature_names):
#         data_dict[feature_name] = data
#         min_max_mean_dict[feature_name] = analyze_range(data_dict[feature_name])

#     # Write comparison for all bags
#     comparison = []
#     for i in range(dim_end - dim_start):
#         comp_info = {'dimension': i}
#         for feature_name in bag_feature_names:
#             min_val, max_val, mean_val,std_val = min_max_mean_dict[feature_name]
#             comp_info[f'min_{feature_name}'] = min_val[i]
#             comp_info[f'max_{feature_name}'] = max_val[i]
#             comp_info[f'mean_{feature_name}'] = mean_val[i]
#             comp_info[f'std_{feature_name}'] = std_val[i]
#         comparison.append(comp_info)

#     output_file_path = os.path.join(analyze_folder, 'bags_comparison_gym.txt')
#     write_comparison_to_file(comparison, output_file_path, bag_feature_names)

# if __name__ == '__main__':
#     param=ParamCollection()
#     analyze_folder=os.path.join(WVN_ROOT_DIR,param.offline.analyze_path)
#     bag_path_snow = '/media/chen/UDisk1/vis_rosbag/snow/2022-12-10-15-40-10_anymal-d020-lpc_mission_0.bag'
#     feature_name_1 = 'SnowDodo'
#     topic_name = '/debug_info'
#     snow_data=process_bag(bag_path_snow, feature_name_1, topic_name, analyze_folder, 341)[:200]
#     proprio_snow=snow_data[:,:133]
#     exte_snow=snow_data[:,133:]
    
    
#     feature_name_2='still'
#     gym_path=os.path.join(WVN_ROOT_DIR,param.offline.analyze_path,f'{feature_name_2}.h5')
#     gym_data_ori=dd.io.load(gym_path)
#     proprio_gym=gym_data_ori[0]['policy']
#     exte_gym=gym_data_ori[0]['priv_exte']
#     gym_data=np.concatenate((proprio_gym,exte_gym),axis=1)
    
#     bag_path_cc = '/media/chen/UDisk1/catkin_rosbag/2021-05-03-19-18-21.bag'
#     feature_name_3 = 'FoamCC'
#     topic_name = '/debug_info'
#     cc_data=process_bag(bag_path_cc, feature_name_3, topic_name, analyze_folder, 341)[:200]
#     proprio_cc=cc_data[:,:133]
#     exte_cc=cc_data[:,133:]
    
#     feature_list=[feature_name_1,feature_name_2,feature_name_3]
#     data_list=[snow_data,gym_data,cc_data]
#     compare_bags(feature_list,data_list,analyze_folder)