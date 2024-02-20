#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.traversability_estimator.graphs import DistanceWindowGraph
from wild_visual_navigation.traversability_estimator.nodes import TwistNode
from wild_visual_navigation.utils import KalmanFilter
from liegroups import SE3
import os
import torch


class SupervisionGenerator:
    def __init__(
        self,
        device: str,
        kf_process_cov,
        kf_meas_cov,
        kf_outlier_rejection,
        kf_outlier_rejection_delta,
        sigmoid_slope,
        sigmoid_cutoff,
        untraversable_thr,
        time_horizon,
        graph_max_length,
    ):
        """Generates traversability signals/labels from different sources

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

        # Save param to classify untraversable cases
        self._untraversable_thr = untraversable_thr

        # Future graph
        self._time_horizon = time_horizon
        self._graph_twist = DistanceWindowGraph(max_distance=graph_max_length, edge_distance=0.0)

    def get_velocity_selection_matrix(self, velocities: list):
        S = []
        if "vx" in velocities:
            S.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if "vy" in velocities:
            S.append([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        if "vz" in velocities:
            S.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        if "wx" in velocities:
            S.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        if "wy" in velocities:
            S.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        if "wz" in velocities:
            S.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        return torch.tensor(S)

    def update_velocity_tracking(
        self,
        current_velocity: torch.tensor,
        desired_velocity: torch.tensor,
        max_velocity: float = 1.0,
        velocities: list = ["vx", "vy", "vz", "wx", "wy", "wz"],
    ):
        """Generates an traversability signal using velocity tracking error

        Args:
            current_velocity (torch.tensor): Current estimated velocity
            desired_velocity (torch.tensor): Desired velocity (command)
            max_velocity (float): Max velocity (magnitude) to scale the error

        Returns:
            traversability (torch.tensor): Estimated traversability
            traversability_var (torch.tensor): Variance of the estimated traversability
        """

        S = self.get_velocity_selection_matrix(velocities).to(self.device)

        # Compute discrepancy
        error = (torch.nn.functional.mse_loss(S @ current_velocity, S @ desired_velocity)) / max_velocity

        # Filtering stage
        with torch.no_grad():
            self._state, self._cov = self._kalman_filter_(self._state, self._cov, error)
        error = self._state

        # Note: The way we use the sigmoid is a bit hacky
        # We use negative argument to revert sigmoid (smaller errors -> 1.0) and stretch the errors
        self._traversability = torch.sigmoid(-(self._sigmoid_slope * (error - self._sigmoid_cutoff)))
        self._traversability_var = torch.tensor([1.0]).to(
            self._traversability.device
        )  # This needs to be improved, the KF can help

        # Apply threshold to detect hard obstacles
        self._is_untraversable = (self._traversability < self._untraversable_thr).item()

        # Return
        self._traversability = torch.clamp(self._traversability, min=0.001, max=1.0)
        return self._traversability, self._traversability_var, self._is_untraversable

    def update_pose_prediction(
        self,
        timestamp: float,
        current_pose_in_world: torch.tensor,
        current_velocity: torch.tensor,
        desired_velocity: torch.tensor,
        velocities: list = ["vx", "vy", "vz", "wx", "wy", "wz"],
    ):
        # Save in twist graph
        self._graph_twist.add_node(
            TwistNode(
                timestamp=timestamp,
                pose_in_world=current_pose_in_world,
                desired_twist=desired_velocity,
                current_twist=current_velocity,
            )
        )
        # Get all nodes within the time horizon
        nodes = self._graph_twist.get_nodes_within_timespan(t_ini=(timestamp - self._time_horizon), t_end=timestamp)

        # Integration
        predicted_pose_in_world = nodes[0].base_pose_in_world

        for node_t, node_tm1 in zip(nodes[1:], nodes[:-1]):
            dt = node_t.timestamp - node_tm1.timestamp
            v_tm1 = node_tm1.desired_twist
            predicted_pose_in_world = predicted_pose_in_world @ SE3.exp(v_tm1 * dt).as_matrix()

        # Measure pose error
        S = self.get_velocity_selection_matrix(velocities).to(self.device)
        error = (S @ SE3.from_matrix(current_pose_in_world.inverse() @ predicted_pose_in_world).log()).norm()

        self._traversability = torch.sigmoid(-(self._sigmoid_slope * (error - self._sigmoid_cutoff)))
        self._traversability_var = torch.tensor([1.0]).to(self._traversability.device)

        # Apply threshold to detect hard obstacles
        self._is_untraversable = (self._traversability < self._untraversable_thr).item()

        # Return
        self._traversability = torch.clamp(self._traversability, min=0.001, max=1.0)
        return self._traversability, self._traversability_var, self._is_untraversable

    @property
    def traversability(self):
        return self._traversability

    @property
    def traversability_var(self):
        return self._traversability_var

    @property
    def untraversable_thr(self):
        return self._untraversable_thr


def run_supervision_generator():
    """Projects 3D points to example images and returns an image with the projection"""

    from wild_visual_navigation.supervision_generator import TwistDataset
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Prepare dataset
    root = str(os.path.join(WVN_ROOT_DIR, "assets/twist_measurements"))
    current_filename = "current_robot_twist.csv"
    desired_filename = "desired_robot_twist.csv"
    data = TwistDataset(root, current_filename, desired_filename, seq_size=1)

    # Prepare traversability generator
    ag = SupervisionGenerator(
        device="cpu",
        kf_process_cov=0.1,
        kf_meas_cov=1000,
        kf_outlier_rejection="huber",
        kf_outlier_rejection_delta=0.5,
        sigmoid_slope=30,
        sigmoid_cutoff=0.2,
        untraversable_thr=0.05,
        time_horizon=0.05,
        graph_max_length=1,
    )

    # Saved data list
    saved_data = []

    # Iterate the dataset
    for i in range(data.size):
        # Get samples
        t, curr, des = data[i]

        # Update traversability
        trav, trav_cov, is_untrav = ag.update_velocity_tracking(
            curr[0], des[0], max_velocity=0.8, velocities=["vx", "vy"]
        )
        saved_data.append(
            [
                t.item(),
                curr.norm().item(),
                des.norm().item(),
                trav.item(),
                trav_cov.item(),
                is_untrav,
            ]
        )

    df = pd.DataFrame(
        saved_data,
        columns=["ts", "curr", "des", "trav", "trav_cov", "is_untraversable"],
    )
    df["ts"] = df["ts"] - df["ts"][0]

    df["trav_upper"] = df["trav"] + df["trav_cov"]
    df["trav_lower"] = df["trav"] - df["trav_cov"]

    fig, axs = plt.subplots(2, 1, sharex=True)
    # Top plot
    axs[0].plot(df["ts"], df["curr"], label="Current twist", color="tab:orange")
    axs[0].plot(df["ts"], df["des"], label="Desired twist", color="k", linewidth=1.5)
    axs[0].set_ylabel("Velocity [m/s]")
    axs[0].set_title("Velocity tracking")
    axs[0].legend(loc="upper right")

    # Bottom plot
    axs[1].plot(df["ts"], df["trav"], label="Traversability", color="tab:blue", linewidth=2)
    axs[1].plot(
        df["ts"],
        np.ones(df["ts"].shape) * ag.untraversable_thr,
        label="Untraversable Thr",
        color="r",
        linestyle="dashed",
    )
    axs[1].fill_between(
        df["ts"],
        df["is_untraversable"] * 0,
        df["is_untraversable"],
        alpha=0.3,
        label="Untraversable",
        color="k",
        linewidth=0.0,
    )
    axs[1].set_ylabel("Traversability")
    axs[1].set_title("Traversability")
    axs[1].legend(loc="upper right")
    # plt.plot(df["ts"], df["trav_cov"], label="Traversability Cov", color="m")
    # plt.fill_between(df["ts"], df["trav_lower"], df["trav_upper"], alpha=0.3, label="1$\sigma$", color="b")

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    print("done")


if __name__ == "__main__":
    run_supervision_generator()
