import argparse
import os

"""
python3 /media/Data/Datasets/2022_Perugia/dataset_info.py
"""

# ROOT_DIR = "/media/Data/Datasets/2022_Perugia"
# ROOT_DIR = "/home/rschmid/RosBags/South_Africa2"
# ROOT_DIR = "/home/rschmid/RosBags/6_proc"
# perugia_dataset = [
#     {
#         "name": "6",
#         "env": "hilly",
#         "mode": "test",
#         "nr": 0,
#         "start": 0,  # Start and stop time is in seconds
#         "stop": 500,
#         "comment": "",
#     },

ROOT_DIR = "/home/rschmid/RosBags/uetliberg_small"
perugia_dataset = [
    {
        "name": "uetliberg_small",
        "env": "hilly",
        "mode": "test",
        "nr": 0,
        "start": 0,  # Start and stop time is in seconds
        "stop": 700,
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
    comment = d["comment"]
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
        env=args.env, mode=args.mode, nr=args.nr, rate=args.rate, ignore_tf=args.ignore_tf, other_args=args.other_args
    )
