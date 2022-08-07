from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.affordance_generator import KalmanFilter
from os.path import join
import os
import torch
import torch.nn as nn


class AffordanceGenerator:
    def __init__(self):
        """Generates affordance signals/labels from different sources

        Args:
            mode (str): Technique used to generate the signal

        Returns:
            None
        """
        
        self._kalman_filter_ = KalmanFilter(
            dim_state=1, dim_control=1, dim_meas=1, outlier_rejection=False, outlier_delta=1.0, outlier_loss="huber"
        )
        self._kalman_filter_.init_process_model(proc_model=torch.eye(1) * 1, proc_cov=torch.eye(1) * 0.01)
        self._kalman_filter_.init_meas_model(meas_model=torch.eye(1), meas_cov=torch.eye(1) * 100)

        self._state = torch.FloatTensor([0.0])[None]
        self._cov = torch.FloatTensor([0.1])[None]

    def update_with_velocities(
        self, current_velocity: torch.tensor, desired_velocity: torch.tensor, max_velocity: float = 1.0
    ):
        # Get dimensionality of input
        N = current_velocity.shape[-1]

        # Compute discrepancy
        # current_velocity = (current_velocity / max_velocity).clip(-1, 1)
        # desired_velocity = (desired_velocity / max_velocity).clip(-1, 1)
        # raw_error = (torch.nn.functional.mse_loss(current_velocity, desired_velocity))  / 2

        raw_error = ((torch.nn.functional.mse_loss(current_velocity, desired_velocity)) / max_velocity).clip(0, 1)

        # Filtering stage
        self._state, self._cov = self._kalman_filter_(self._state, self._cov, raw_error)

        # Clip output
        s = nn.Sigmoid()
        self._affordance = 1.0 - self._state  # Clipping the output of the KF is a bit sketchy
        self._affordance_conf = torch.tensor([0.9]) # 1.0 - self._cov

        # Return
        return self._affordance, self._affordance_conf, raw_error


def run_affordance_generator():
    """Projects 3D points to example images and returns an image with the projection"""

    from wild_visual_navigation.learning.dataset import TwistDataset
    import matplotlib.pyplot as plt
    import pandas as pd

    # Prepare dataset
    root = str(os.path.join(WVN_ROOT_DIR, "results/arche_loop"))
    current_filename = "current_robot_twist.csv"
    desired_filename = "desired_robot_twist.csv"
    data = TwistDataset(root, current_filename, desired_filename, seq_size=1, velocities=["vx", "vy"])

    # Prepare affordance generator
    ag = AffordanceGenerator()

    # Saved data list
    saved_data = []

    # Iterate the dataset
    for i in range(data.size):
        # Get samples
        t, curr, des = data[i]

        # Update affordance
        aff, aff_cov, raw_error = ag.update_with_velocities(curr, des, max_velocity=0.8)
        saved_data.append([t.item(), curr.norm().item(), des.norm().item(), aff.item(), aff_cov.item(), raw_error.item()])

    df = pd.DataFrame(saved_data, columns=["ts", "curr", "des", "aff", "aff_cov", "raw_err"])

    df["aff_upper"] = df["aff"] + df["aff_cov"]
    df["aff_lower"] = df["aff"] - df["aff_cov"]
    plt.plot(df["ts"], df["curr"], label="Current twist", color="b")
    plt.plot(df["ts"], df["des"], label="Desired twist", color="k")
    plt.plot(df["ts"], df["aff"], label="Affordance", color="r")
    plt.plot(df["ts"], df["aff_cov"], label="Affordance Cov", color="m")
    # plt.plot(df["ts"], df["raw_err"], label="Raw error", color="g")
    # plt.plot(df["ts"], (1.0 - df["aff"]) - df["raw_err"], label="Affordance", color="c")
    # plt.fill_between(df["ts"], 0.0, df["aff_cov"], alpha=0.2, label="Aff bounds (1$\sigma$)", color="r")
    # plt.fill_between(df["ts"], df["aff_lower"], df["aff_upper"], alpha=0.3, label="Aff bounds (1$\sigma$)", color="r")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_affordance_generator()
