import argparse
import os

"""
python3 /media/Data/Datasets/2022_Perugia/dataset_info.py
"""

# ROOT_DIR = "/media/Data/Datasets/2022_Perugia"
ROOT_DIR = "/media/matias/datasets/2022_Perugia"
perugia_dataset = [
    {
        "name": "day3/mission_data/2022-05-12T09:45:07_mission_0_day_3",
        "env": "hilly",
        "mode": "test",
        "nr": 0,
        "start": 80,
        "stop": 390,
        "comment": "emergency button by accident",
    },
    {
        "name": "day3/mission_data/2022-05-12T09:57:13_mission_0_day_3",
        "env": "hilly",
        "mode": "test",
        "nr": 1,
        "start": 50,
        "stop": 430,
        "comment": "fall",
    },
    {
        "name": "day3/mission_data/2022-05-12T10:18:16_mission_0_day_3",
        "env": "hilly",
        "mode": "train",
        "nr": 0,
        "start": 44,
        "stop": 680,
        "comment": "",
    },
    {
        "name": "day3/mission_data/2022-05-12T10:34:03_mission_0_day_3",
        "env": "hilly",
        "mode": "test",
        "nr": 2,
        "start": 42,
        "stop": 450,
        "comment": "",
    },
    {
        "name": "day3/mission_data/2022-05-12T10:45:20_mission_0_day_3",
        "env": "hilly",
        "mode": "test",
        "nr": 3,
        "start": 26,
        "stop": 269,
        "comment": "",
    },
    {
        "name": "day3/mission_data/2022-05-12T11:44:56_mission_0_day_3",
        "env": "forest",
        "mode": "test",
        "nr": 0,
        "start": 42,
        "stop": 590,
        "comment": "",
    },
    {
        "name": "day3/mission_data/2022-05-12T11:56:13_mission_0_day_3",
        "env": "forest",
        "mode": "train",
        "nr": 0,
        "start": 45,
        "stop": 585,
        "comment": "559 stuck with tree (traversability)",
    },
    {
        "name": "day3/mission_data/2022-05-12T12:08:09_mission_0_day_3",
        "env": "forest",
        "mode": "test",
        "nr": 1,
        "start": 55,
        "stop": 530,
        "comment": "",
    },
    {
        "name": "day3/mission_data/2022-05-12T15:36:30_mission_0_day_3",
        "env": "grassland",
        "mode": "test",
        "nr": 0,
        "start": 0,
        "stop": 696,
        "comment": "walk in the end in high flowers, and close to small trees ",
    },
    {
        "name": "day3/mission_data/2022-05-12T15:52:37_mission_0_day_3",
        "env": "grassland",
        "mode": "test",
        "nr": 1,
        "start": 0,
        "stop": 1000,
        "comment": "walking on grass and gravel road; anomoly walking close to river ",
    },
    {
        "name": "day3/mission_data/2022-05-12T17:36:33_mission_0_day_3",
        "env": "grassland",
        "mode": "train",
        "nr": 0,
        "start": 55,
        "stop": 1160,
        "comment": "features some brown rough soil; robot falls down",
    },
    {
        "name": "day3/mission_data/2022-05-12T18:21:23_mission_0_day_3",
        "env": "grassland",
        "mode": "test",
        "nr": 2,
        "start": 80,
        "stop": 750,
        "comment": "",
    },
]


def dataset_play(env="forest", mode="train", nr=0, rate=1.0, ignore_tf=False, other_args=""):
    """
    Play a dataset.
    """
    dataset = perugia_dataset
    for d in dataset:
        if d["env"] == env and d["mode"] == mode and d["nr"] == nr:
            break
    else:
        raise ValueError("No dataset found for env={} and mode={}".format(env, mode))
    start = d["start"]
    stop = d["stop"]
    duration = stop - start
    bags = os.path.join(ROOT_DIR, d["name"]) + "/*.bag"
    # comment = d["comment"]
    if bool(ignore_tf):
        bags += " /tf:=/tf_trash"
    cmd = f"rosbag play -s {start} -u {duration} -r {rate} {other_args} {bags}"

    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Play a dataset.""")
    parser.add_argument("--env", type=str, default="forest")
    parser.add_argument("--mode", required=False, type=str, default="train")
    parser.add_argument("--nr", type=int, default=0)
    parser.add_argument("--rate", required=False, default=1.0, type=float)
    parser.add_argument("--ignore_tf", required=False, type=int, default=0)
    parser.add_argument("--other_args", required=False, type=str, default="--clock")

    args = parser.parse_args()
    dataset_play(
        env=args.env,
        mode=args.mode,
        nr=args.nr,
        rate=args.rate,
        ignore_tf=args.ignore_tf,
        other_args=args.other_args,
    )
