from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.affordance_generator import KalmanFilter
from os.path import join
import os
import torch
import torch.nn as nn


class AffordanceGenerator:
    def __init__(
        self,
        device: str = "cuda",
        kf_process_cov: float = 0.01,
        kf_meas_cov: float = 10,
        kf_outlier_rejection: str = "none",
        kf_outlier_rejection_delta: float = 1.0,
        sigmoid_slope: float = 15,
        sigmoid_cutoff: float = 0.2,
        unaffordable_thr: float = 0.1,
    ):
        """Generates affordance signals/labels from different sources

        Args:
            device (str): Device used to load the torch models

        Returns:
            None
        """
        self.device = device

        # Setup Kalman Filter to smooth signals
        D = 1
        self._kalman_filter_ = KalmanFilter(
            dim_state=D,
            dim_control=D,
            dim_meas=D,
            outlier_rejection=kf_outlier_rejection,
            outlier_delta=kf_outlier_rejection_delta,
        )

        self._kalman_filter_.init_process_model(proc_model=torch.eye(D) * 1, proc_cov=torch.eye(D) * kf_process_cov)
        self._kalman_filter_.init_meas_model(meas_model=torch.eye(D), meas_cov=torch.eye(D) * kf_meas_cov)

        # Initial states
        self._state = torch.FloatTensor([0.0] * D).to(self.device)
        self._cov = (torch.eye(D) * 0.1).to(self.device)

        # Move Kalman filter to device
        self._kalman_filter_.to(self.device)

        # Setup sigmoid to stretch output
        self._sigmoid_slope = sigmoid_slope
        self._sigmoid_cutoff = sigmoid_cutoff

        # Save param to classify unaffordable cases
        self._unaffordable_thr = unaffordable_thr

    def update_with_velocities(
        self, current_velocity: torch.tensor, desired_velocity: torch.tensor, max_velocity: float = 1.0
    ):
        """Generates an affordance signal using velocity tracking error

        Args:
            current_velocity (torch.tensor): Current estimated velocity
            desired_velocity (torch.tensor): Desired velocity (command)
            max_velocity (float): Max velocity (magnitude) to scale the error
            sigmoid_slope (float): Slope for the sigmoid to binarie tracking error. Larger values is more binary. Default: 10
            sigmoid_cutoff (float): Determines for which velocity tracking error we consider 0.5 affordance. Default: 0.2

        Returns:
            affordance (torch.tensor): Estimated affordance
            affordance_var (torch.tensor): Variance of the estimated affordance
        """

        # Get dimensionality of input
        N = current_velocity.shape[-1]

        # Compute discrepancy
        error = (torch.nn.functional.mse_loss(current_velocity, desired_velocity)) / max_velocity

        # Filtering stage
        with torch.no_grad():
            self._state, self._cov = self._kalman_filter_(self._state, self._cov, error)
        error = self._state

        # Note: The way we use the sigmoid is a bit hacky
        # We use negative argument to revert sigmoid (smaller errors -> 1.0) and stretch the errors
        self._affordance = torch.sigmoid(-(self._sigmoid_slope * (error - self._sigmoid_cutoff)))
        self._affordance_var = torch.tensor([1.0]).to(
            self._affordance.device
        )  # This needs to be improved, the KF can help

        # Apply threshold to detect hard obstacles
        self._is_unaffordable = (self._affordance < self._unaffordable_thr).item()

        # Return
        return self._affordance, self._affordance_var, self._is_unaffordable

    @property
    def affordance(self):
        return self._affordance

    @property
    def affordance_varidence(self):
        return self._affordance_var

    @property
    def unaffordable_thr(self):
        return self._unaffordable_thr


def run_affordance_generator():
    """Projects 3D points to example images and returns an image with the projection"""

    from wild_visual_navigation.learning.dataset import TwistDataset
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Prepare dataset
    root = str(os.path.join(WVN_ROOT_DIR, "assets/twist_measurements"))
    current_filename = "current_robot_twist_short.csv"
    desired_filename = "desired_robot_twist_short.csv"
    data = TwistDataset(root, current_filename, desired_filename, seq_size=1, velocities=["vx", "vy"])

    # Prepare affordance generator
    ag = AffordanceGenerator(
        kf_process_cov=0.1,
        kf_meas_cov=1000,
        kf_outlier_rejection="huber",
        kf_outlier_rejection_delta=0.5,
        sigmoid_slope=25,
        sigmoid_cutoff=0.5,
        unaffordable_thr=0.1,
    )

    # Saved data list
    saved_data = []

    # Iterate the dataset
    for i in range(data.size):
        # Get samples
        t, curr, des = data[i]

        # Update affordance
        aff, aff_cov, is_unaff = ag.update_with_velocities(curr[0], des[0], max_velocity=0.8)
        saved_data.append(
            [t.item(), curr.norm().item(), des.norm().item(), aff.item(), aff_cov.item(), is_unaff.item()]
        )

    df = pd.DataFrame(saved_data, columns=["ts", "curr", "des", "aff", "aff_cov", "is_unaffordable"])

    df["aff_upper"] = df["aff"] + df["aff_cov"]
    df["aff_lower"] = df["aff"] - df["aff_cov"]
    plt.plot(df["ts"], df["curr"], label="Current twist", color="m")
    plt.plot(df["ts"], df["des"], label="Desired twist", color="k")
    plt.plot(df["ts"], df["aff"], label="Affordance", color="b")
    # plt.plot(df["ts"], df["aff_cov"], label="Affordance Cov", color="m")
    plt.plot(df["ts"], np.ones(df["ts"].shape) * ag.unaffordable_thr, label="Unaffordable Thr", color="r")
    # plt.fill_between(df["ts"], df["aff_lower"], df["aff_upper"], alpha=0.3, label="1$\sigma$", color="b")
    plt.fill_between(
        df["ts"], df["is_unaffordable"] * 0, df["is_unaffordable"], alpha=0.3, label="Unaffordable", color="k"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_affordance_generator()
