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
        self.kalman_filter_ = KalmanFilter(dim_state=1, dim_control=1, dim_meas=1)

    def update_with_velocities(
        self, current_velocity: torch.tensor, target_velocity: torch.tensor, max_velocity: float = 1.0
    ):
        # Get dimensionality of input
        N = current_velocity.shape[-1]

        # Compute velocity difference
        error = torch.nn.functional.mse_loss(current_velocity, target_velocity) / max_velocity

        # Clip output
        self._affordance = 1.0 - error.clip(0, 1)
        self._affordance_var = torch.FloatTensor(1).to(current_velocity.device)

        return self._affordance, self._affordance_var


def run_affordance_generator():
    """Projects 3D points to example images and returns an image with the projection"""

    import matplotlib.pyplot as plt


if __name__ == "__main__":
    run_image_projector()
