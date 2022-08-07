from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.affordance_generator import KalmanFilter
from os.path import join
import os
import torch
import torch.nn as nn


class AffordanceGenerator:
    def __init__(self, device: str = "cuda"):
        """Generates affordance signals/labels from different sources

        Args:
            mode (str): Technique used to generate the signal

        Returns:
            None
        """
        self.device = device

        D = 1
        self._kalman_filter_ = KalmanFilter(
            dim_state=D,
            dim_control=D,
            dim_meas=D,
            outlier_rejection="huber",
            outlier_delta=1.0,
        )

        self._kalman_filter_.init_process_model(proc_model=torch.eye(D) * 1, proc_cov=torch.eye(D) * 0.01)
        self._kalman_filter_.init_meas_model(meas_model=torch.eye(D), meas_cov=torch.eye(D) * 1)

        self._state = torch.FloatTensor([0.0] * D).to(self.device)
        self._cov = (torch.eye(D) * 0.1).to(self.device)

        # Initialize Kalman filter in device
        self._kalman_filter_.to(self.device)

    def update_with_velocities(
        self,
        current_velocity: torch.tensor,
        desired_velocity: torch.tensor,
        max_velocity: float = 1.0,
        sigmoid_slope: float = 15,
        sigmoid_cutoff: float = 0.2,
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
        s = nn.Sigmoid()
        # Note: we use negative argument to revert sigmoid (smaller errors -> 1.0)
        self._affordance = s(-(sigmoid_slope * (error - sigmoid_cutoff))) 
        self._affordance_var = torch.tensor([1.0]).to(self._affordance.device)  # This needs to be improved, the KF can help

        # Return
        return self._affordance, self._affordance_var

    @property
    def affordance(self):
        return self._affordance

    @property
    def affordance_varidence(self):
        return self._affordance_var


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
        aff, aff_cov = ag.update_with_velocities(curr[0], des[0], max_velocity=0.8)
        saved_data.append([t.item(), curr.norm().item(), des.norm().item(), aff.item(), aff_cov.item()])

    df = pd.DataFrame(saved_data, columns=["ts", "curr", "des", "aff", "aff_cov"])

    df["aff_upper"] = df["aff"] + (1 - df["aff_cov"])
    df["aff_lower"] = df["aff"] - (1 - df["aff_cov"])
    plt.plot(df["ts"], df["curr"], label="Current twist", color="b")
    plt.plot(df["ts"], df["des"], label="Desired twist", color="k")
    plt.plot(df["ts"], df["aff"], label="Affordance", color="r")
    plt.plot(df["ts"], df["aff_cov"], label="Affordance Cov", color="m")
    plt.fill_between(df["ts"], df["aff_lower"], df["aff_upper"], alpha=0.3, label="1$\sigma$", color="b")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_affordance_generator()
