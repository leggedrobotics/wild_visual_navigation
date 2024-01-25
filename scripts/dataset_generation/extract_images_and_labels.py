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

    valid_topics = [
        "/state_estimator/anymal_state",
        "/alphasense_driver_ros/cam4",
        "/log/state/desiredRobotTwist",
    ]

    # Merge rosbags if necessary
    rosbags = [
        str(s)
        for s in Path(s).rglob("*.bag")
        if str(s).find("lpc_robot_state") != -1
        or str(s).find("jetson_images") != -1
        or str(s).find("lpc_locomotion") != -1
    ]
    try:
        rosbags.sort(key=lambda x: int(x.split("/")[-1][-5]))
    except Exception:
        pass

    output_bag_wvn = s + "_wvn.bag"
    output_bag_tf = s + "_tf.bag"
    tf_bags = [b for b in rosbags if b.find("lpc_robot_state") != -1]

    # jetson locomotion robot
    if not os.path.exists(output_bag_tf):
        total_included_count, total_skipped_count = merge_bags_single(
            input_bag=tf_bags,
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

    # 2022-05-12T11:56:13_mission_0_day_3
    # extraction_store_folder = f"/media/Data/Datasets/2022_Perugia/wvn_output/day3/{mission}"
    extraction_store_folder = f"/media/matias/datasets/2022_Perugia/wvn_output/day3/{mission}"

    if os.path.exists(extraction_store_folder):
        print(f"Stopped because folder already exists: {extraction_store_folder}")
        return

    rosparam.set_param("wild_visual_navigation_node/mode", "extract_labels")
    rosparam.set_param("wild_visual_navigation_node/extraction_store_folder", extraction_store_folder)

    # for supervision callback
    state_msg_valid = False
    desired_twist_msg_valid = False

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
    info_msg.K = [
        347.548139773951,
        0.0,
        342.454373227748,
        0.0,
        347.434712422309,
        271.368057185649,
        0.0,
        0.0,
        1.0,
    ]
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
                pbar.update(1)
                st = time.time()
                if topic == "/state_estimator/anymal_state":
                    state_msg = anymal_msg_callback(msg, return_msg=True)
                    state_msg_valid = True

                elif topic == "/log/state/desiredRobotTwist":
                    desired_twist_msg = msg
                    desired_twist_msg_valid = True

                elif topic == "/alphasense_driver_ros/cam4":
                    N = 1000
                    for i in range(N):
                        pub.publish(msg)
                        # Change this for service call
                        try:
                            image_msg = rospy.wait_for_message(
                                "/alphasense_driver_ros/cam4/debayered",
                                Image,
                                timeout=0.01,
                            )
                            suc = True
                        except Exception:
                            suc = False
                            pass
                        if suc:
                            if msg.header.stamp == image_msg.header.stamp:
                                break
                        if i >= N - 1:
                            raise Exception("Timeout waiting for debayered image message")

                    info_msg.header = msg.header
                    try:
                        wvn_ros_interface.image_callback(image_msg, info_msg)
                    except Exception as e:
                        tqdm.write("Bad image_callback", e)

                    total_time_img += time.time() - st
                    # print(f"image time: {total_time_img} , state time: {total_time_state}")
                    tqdm.write("add image")
                if state_msg_valid and desired_twist_msg_valid:
                    try:
                        wvn_ros_interface.robot_state_callback(state_msg, desired_twist_msg)
                    except Exception as e:
                        tqdm.write("Bad robot_state callback ", e)

                    state_msg_valid = False
                    desired_twist_msg_valid = True
                    total_time_state += time.time() - st
                    tqdm.write("add supervision")

    print("Finished with converting the dataset")
    rospy.signal_shutdown("stop the node")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2, help="Store data")
    parser.add_argument("--dry_run", type=int, default=0, help="Store data")
    args = parser.parse_args()
    print(args.n)
    do(args.n, args.dry_run)
