import sys
import os
import time
from tqdm import tqdm
import subprocess
import yaml

from tf_bag import BagTfTransformer
import rospy
import rosparam
from sensor_msgs.msg import CameraInfo
import rosbag

from postprocessing_tools_ros.merging import merge_bags_single

# from py_image_proc_cuda import ImageProcCuda
# from cv_bridge import CvBridge

from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.utils import perugia_dataset, ROOT_DIR

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
        except Exception:
            return (None, None)


def do(n, dry_run):
    d = perugia_dataset[n]

    if bool(dry_run):
        print(d)
        return

    s = os.path.join(ROOT_DIR, d["name"])

    valid_topics = ["/state_estimator/anymal_state", "/wide_angle_camera_front/img_out"]

    rosbags = [
        "/home/rschmid/RosBags/6/images.bag",
        "/home/rschmid/RosBags/6/2023-03-02-11-13-08_anymal-d020-lpc_mission_0.bag",
        "/home/rschmid/RosBags/6/2023-03-02-11-13-08_anymal-d020-lpc_mission_1.bag",
    ]

    output_bag_wvn = s + "_wvn.bag"
    output_bag_tf = s + "_tf.bag"

    if not os.path.exists(output_bag_tf):
        total_included_count, total_skipped_count = merge_bags_single(
            input_bag=rosbags,
            output_bag=output_bag_tf,
            topics="/tf /tf_static",
            verbose=True,
        )
    if not os.path.exists(output_bag_wvn):
        total_included_count, total_skipped_count = merge_bags_single(
            input_bag=rosbags,
            output_bag=output_bag_wvn,
            topics=" ".join(valid_topics),
            verbose=True,
        )

    # Setup WVN node
    rospy.init_node("wild_visual_navigation_node")

    mission = s.split("/")[-1]

    running_store_folder = f"/home/rschmid/RosBags/output/{mission}"

    if os.path.exists(running_store_folder):
        print("Folder already exists, but proceeding!")
        # return

    rosparam.set_param("wild_visual_navigation_node/mode", "extract_labels")
    rosparam.set_param("wild_visual_navigation_node/running_store_folder", running_store_folder)

    # for supervision callback
    state_msg_valid = False
    # desired_twist_msg_valid = False

    wvn_ros_interface = WvnRosInterface()
    print("-" * 80)

    print("start loading tf")
    tf_listener = BagTfTransformerWrapper(output_bag_tf)
    wvn_ros_interface.setup_rosbag_replay(tf_listener)
    print("done loading tf")

    # Höngg new
    info_msg = CameraInfo()
    info_msg.height = 1080
    info_msg.width = 1440
    info_msg.distortion_model = "equidistant"
    info_msg.D = [
        0.4316922809468283,
        0.09279900476637248,
        -0.4010909691803734,
        0.4756163338479413,
    ]
    info_msg.K = [
        575.6050407221768,
        0.0,
        745.7312198525915,
        0.0,
        578.564849365178,
        519.5207040671075,
        0.0,
        0.0,
        1.0,
    ]
    info_msg.P = [
        575.6050407221768,
        0.0,
        745.7312198525915,
        0.0,
        0.0,
        578.564849365178,
        519.5207040671075,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    info_msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    rosbag_info_dict = get_bag_info(output_bag_wvn)
    total_msgs = sum([x["messages"] for x in rosbag_info_dict["topics"] if x["topic"] in valid_topics])
    total_time_img = 0
    total_time_state = 0
    n = 0

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
            for topic, msg, ts in bag.read_messages(topics=None, start_time=start_time, end_time=end_time):
                if rospy.is_shutdown():
                    return
                pbar.update(1)
                st = time.time()
                if topic == "/state_estimator/anymal_state":
                    state_msg = anymal_msg_callback(msg, return_msg=True)
                    state_msg_valid = True

                elif topic == "/wide_angle_camera_front/img_out":
                    image_msg = msg
                    # print("Received /wide_angle_camera_front/img_out")

                    info_msg.header = msg.header
                    camera_options = {}
                    camera_options["name"] = "wide_angle_camera_front"
                    camera_options["use_for_training"] = True

                    info_msg.header = msg.header
                    try:
                        wvn_ros_interface.image_callback(image_msg, info_msg, camera_options)
                    except Exception as e:
                        print("Bad image_callback", e)

                    total_time_img += time.time() - st
                    # print(f"image time: {total_time_img} , state time: {total_time_state}")
                    # print("add image")
                if state_msg_valid:
                    try:
                        wvn_ros_interface.robot_state_callback(state_msg, None)
                    except Exception as e:
                        print("Bad robot_state callback ", e)

                    state_msg_valid = False
                    total_time_state += time.time() - st

    print("Finished with converting the dataset")
    rospy.signal_shutdown("stop the node")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0, help="Store data")
    parser.add_argument("--dry_run", type=int, default=0, help="Store data")
    args = parser.parse_args()

    do(args.n, args.dry_run)
