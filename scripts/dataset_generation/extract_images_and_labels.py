import sys
import os
from pathlib import Path
import time
from tqdm import tqdm
import subprocess
import yaml

from tf_bag import BagTfTransformer
import rospy
import rosparam
from sensor_msgs.msg import Image, CameraInfo
import rosbag

from postprocessing_tools_ros.merging import merge_bags_single, merge_bags_all

# from py_image_proc_cuda import ImageProcCuda
# from cv_bridge import CvBridge

from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.utils import perguia_dataset, ROOT_DIR

sys.path.append(f"{WVN_ROOT_DIR}/wild_visual_navigation_ros/scripts")
from wild_visual_navigation_node import WvnRosInterface

sys.path.append(f"{WVN_ROOT_DIR}/wild_visual_navigation_anymal/scripts")
from anymal_msg_converter_node import anymal_msg_callback

# We need to do the following
# 1. Debayering cam4 -> send via ros and wait for result ("correct params")
# 2. anymal_state_topic -> /wild_visual_navigation_node/robot_state
# 3. Feed into wild_visual_navigation_node ("correct params")
# # Iterate rosbags


def get_bag_info(rosbag_path: str) -> dict:
    # This queries rosbag info using subprocess and get the YAML output to parse the topics
    info_dict = yaml.safe_load(
        subprocess.Popen(["rosbag", "info", "--yaml", rosbag_path], stdout=subprocess.PIPE).communicate()[0]
    )
    return info_dict


class BagTfTransformerWrapper:
    def __init__(self, bag):
        self.tf_listener = BagTfTransformer(bag)

    def waitForTransform(self, parent_frame, child_frame, time, duration):
        return self.tf_listener.waitForTransform(parent_frame, child_frame, time)

    def lookupTransform(self, parent_frame, child_frame, time):
        try:
            return self.tf_listener.lookupTransform(parent_frame, child_frame, time)
        except:
            return (None, None)


for d in perguia_dataset:
    s = os.path.join(ROOT_DIR, d["name"])
    valid_topics = ["/state_estimator/anymal_state", "/alphasense_driver_ros/cam4", "/log/state/desiredRobotTwist"]

    # Merge rosbags if necessary
    rosbags = [
        str(s)
        for s in Path(s).rglob("*.bag")
        if str(s).find("lpc_robot_state") != -1
        or str(s).find("jetson_images") != -1
        or str(s).find("lpc_locomotion") != -1
    ]
    output_bag_wvn = s + "_wvn.bag"
    output_bag_tf = s + "_tf.bag"
    tf_bags = [b for b in rosbags if b.find("lpc_robot_state") != -1]
    if not os.path.exists(output_bag_tf):
        total_included_count, total_skipped_count = merge_bags_single(
            input_bag=tf_bags, output_bag=output_bag_tf, topics="/tf /tf_static", verbose=True
        )
    if not os.path.exists(output_bag_wvn):
        total_included_count, total_skipped_count = merge_bags_single(
            input_bag=rosbags, output_bag=output_bag_wvn, topics=" ".join(valid_topics), verbose=True
        )

    # Setup WVN node
    rospy.init_node("wild_visual_navigation_node")

    running_store_folder = "/media/Data/Datasets/2022_Perugia/day3/mission_data/2022-05-12T09:57:13_mission_0_day_3"
    running_store_folder.replace("2022_Perugia/day3/mission_data", "2022_Perugia/wvn_output/day3")

    rosparam.set_param("wild_visual_navigation_node/mode", "extract_labels")
    rosparam.set_param("'wild_visual_navigation_node/running_store_folder", running_store_folder)

    # for proprioceptive callback
    state_msg_valid = False
    desired_twist_msg_valid = False

    # for camera callback
    # con = "/home/jonfrey/git/anymal_dataset_generation/postprocessing_tools_ros/config/image_proc_cuda/image_proc_cuda_undistorted.yaml"
    # cal = "/home/jonfrey/catkin_ws/src/anymal_rsl/anymal_c_rsl/anymal_cerberus_rsl/anymal_cerberus_rsl/calibration/alphasense/intrinsic/cam4.yaml"
    # import rospkg
    # rospack = rospkg.RosPack()
    # Set config files
    # calib_file = rospack.get_path("image_proc_cuda") + "/config/alphasense_calib_example.yaml"
    # color_calib_file = rospack.get_path("image_proc_cuda") + "/config/alphasense_color_calib_example.yaml"
    # param_file = rospack.get_path("image_proc_cuda") + "/config/pipeline_params_example.yaml"
    # Create image Proc
    # proc = ImageProcCuda(param_file, calib_file, color_calib_file)
    # proc = ImageProcCuda(con, cal)
    # proc.set_flip(True)
    # input_encoding = "bayer_gbrg8"
    # Store the camera calibration after debayering and optional undistortion
    # with open(cal, "r") as f:
    #     cam = yaml.load(f, Loader=yaml.FullLoader)
    #     cam["image_width"] = proc.get_image_width()
    #     cam["image_height"] = proc.get_image_height()
    #     cam["distortion_model"] = proc.get_distortion_model()
    #     cam["distortion_coefficients"]["data"] = proc.get_distortion_coefficients().flatten().tolist()
    #     cam["rectification_matrix"]["data"] = proc.get_rectification_matrix().flatten().tolist()
    # cam["projection_matrix"]["data"] = proc.get_projection_matrix().flatten().tolist()
    # bridge = CvBridge()

    pub = rospy.Publisher("/alphasense_driver_ros/cam4", Image, queue_size=1)
    wvn_ros_interface = WvnRosInterface()
    print("-" * 80)

    print("start loading tf")
    tf_listener = BagTfTransformerWrapper(output_bag_tf)
    wvn_ros_interface.setup_rosbag_replay(tf_listener)
    print("done loading tf")

    info_msg = CameraInfo()
    info_msg.height = 540
    info_msg.width = 720
    info_msg.distortion_model = "plumb_bob"
    info_msg.K = [347.548139773951, 0.0, 342.454373227748, 0.0, 347.434712422309, 271.368057185649, 0.0, 0.0, 1.0]
    info_msg.P = [
        347.548139773951,
        0.0,
        342.454373227748,
        0.0,
        0.0,
        347.434712422309,
        271.368057185649,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    # ---
    # header:
    #   seq: 68053
    #   stamp:
    #     secs: 1652349761
    #     nsecs: 732710573
    #   frame_id: "cam4_sensor_frame"
    # height: 540
    # width: 720
    # distortion_model: "plumb_bob"
    # D: [0.0, 0.0, 0.0, 0.0]
    # K: [347.548139773951, 0.0, 342.454373227748, 0.0, 347.434712422309, 271.368057185649, 0.0, 0.0, 1.0]
    # R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    # P: [347.548139773951, 0.0, 342.454373227748, 0.0, 0.0, 347.434712422309, 271.368057185649, 0.0, 0.0, 0.0, 1.0, 0.0]
    # binning_x: 0
    # binning_y: 0
    # roi:
    #   x_offset: 0
    #   y_offset: 0
    #   height: 0
    #   width: 0
    #   do_rectify: False
    # ---

    rosbag_info_dict = get_bag_info(output_bag_wvn)
    total_msgs = sum([x["messages"] for x in rosbag_info_dict["topics"] if x["topic"] in valid_topics])

    with rosbag.Bag(output_bag_wvn, "r") as bag:
        start_time = rospy.Time.from_sec(bag.get_start_time() + d["start"])
        end_time = rospy.Time.from_sec(bag.get_start_time() + d["stop"])

        with tqdm(
            total=total_msgs,
            desc="Total",
            colour="green",
            position=1,
            bar_format="{desc:<13}{percentage:3.0f}%|{bar:20}{r_bar}",
        ) as pbar:
            for (topic, msg, ts) in bag.read_messages(topics=valid_topics, start_time=start_time, end_time=end_time):
                pbar.update(1)
                st = time.time()
                if topic == "/state_estimator/anymal_state":
                    state_msg = anymal_msg_callback(msg, return_msg=True)
                    state_msg_valid = True

                elif topic == "/log/state/desiredRobotTwist":
                    desired_twist_msg = msg
                    desired_twist_msg_valid = True

                elif topic == "/alphasense_driver_ros/cam4":
                    for i in range(100):
                        pub.publish(msg)
                        try:
                            image_msg = rospy.wait_for_message(
                                "/alphasense_driver_ros/cam4/debayered", Image, timeout=0.1
                            )
                            suc = True
                        except:
                            suc = False
                            pass
                        if suc:
                            if msg.header.stamp == image_msg.header.stamp:
                                break
                        if i >= 99:
                            raise Exception("Timeout waiting for debayerd image message")

                    # if input_encoding is not None:
                    #     msg.encoding = input_encoding
                    # Convert message to image using the msg encoding
                    # img = bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
                    # # Apply image proc pipeline
                    # img = proc.process(img, msg.encoding)
                    # # "img is opencv img"
                    # image_msg = bridge.cv2_to_imgmsg(img, encoding="passthrough")

                    info_msg.header = msg.header
                    wvn_ros_interface.image_callback(image_msg, info_msg)

                if state_msg_valid and desired_twist_msg_valid:
                    wvn_ros_interface.robot_state_callback(state_msg, desired_twist_msg)
                    state_msg_valid = False
                    desired_twist_msg_valid = True

                # print("time: {}, {}".format(time.time()-st, topic))
