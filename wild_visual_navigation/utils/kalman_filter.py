#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import torch
import torch.nn as nn


class KalmanFilter(nn.Module):
    def __init__(
        self,
        dim_state: int = 1,
        dim_control: int = 1,
        dim_meas: int = 1,
        outlier_rejection: str = "none",
        outlier_delta: float = 1.0,
    ):
        super().__init__()

        # Store dimensions
        self.dim_state = dim_state
        self.dim_control = dim_control
        self.dim_meas = dim_meas

        # Prediction model
        self.proc_model = nn.Parameter(torch.eye(dim_state))
        self.proc_cov = nn.Parameter(torch.eye(dim_state))
        self.control_model = nn.Parameter(torch.eye(dim_state, dim_control))

        # Measurement model
        self.meas_model = nn.Parameter(torch.eye(dim_meas, dim_state))
        self.meas_cov = nn.Parameter(torch.eye(dim_meas, dim_meas))

        # Helper
        self.eye = nn.Parameter(torch.eye(dim_state, dim_state), requires_grad=False)

        # Outlier rejection
        self.outlier_rejection = outlier_rejection
        self.outlier_delta = outlier_delta

    def init_process_model(self, proc_model=None, proc_cov=None, control_model=None):
        # Initialize process model
        if proc_model is not None:
            assert (
                self.proc_model.shape == proc_model.shape
            ), f"Desired state doesn't match {self.proc_model.shape} (expected) != {proc_model.shape} (new)"
            self.proc_model = nn.Parameter(proc_model)

        # Initialize process model covariance
        if proc_cov is not None:
            assert (
                self.proc_cov.shape == proc_cov.shape
            ), f"Desired state doesn't match {self.proc_cov.shape} (expected) != {proc_cov.shape} (new)"
            self.proc_cov = nn.Parameter(proc_cov)

        # Initialize control
        if control_model is not None:
            assert (
                self.control_model.shape == control_model.shape
            ), f"Desired state doesn't match {self.control_model.shape} (expected) != {control_model.shape} (new)"
            self.control_model = nn.Parameter(control_model)

    def init_meas_model(self, meas_model=None, meas_cov=None):
        # Initialize measurement model
        if meas_model is not None:
            assert (
                self.meas_model.shape == meas_model.shape
            ), f"Desired state doesn't match {self.meas_model.shape} (expected) != {meas_model.shape} (new)"
            self.meas_model = nn.Parameter(meas_model)

        # Initialize measurement model covariance
        if meas_cov is not None:
            assert (
                self.meas_cov.shape == meas_cov.shape
            ), f"Desired state doesn't match {self.meas_cov.shape} (expected) != {meas_cov.shape} (new)"
            self.meas_cov = nn.Parameter(meas_cov)

    def prediction(self, state, state_cov, control: torch.tensor = None):
        # Update prior
        if control is None:
            state = self.proc_model @ state
        else:
            state = self.proc_model @ state + self.control_model @ control

        # Update covariance
        state_cov = self.proc_model @ state_cov @ self.proc_model.t() + self.proc_cov

        return state, state_cov

    def correction(self, state: torch.tensor, state_cov: torch.tensor, meas: torch.tensor):
        # Innovation
        innovation = meas - self.meas_model @ state

        # Get outlier rejection weight
        outlier_weight = self.get_outlier_weight(innovation, self.meas_cov)

        # Innovation covariance
        innovation_cov = self.meas_model @ state_cov @ self.meas_model.t() + self.meas_cov

        # Kalman gain
        kalman_gain = outlier_weight * state_cov @ self.meas_model.t() @ innovation_cov.inverse()

        # Updated state
        state = state + (kalman_gain @ innovation)
        state_cov = (self.eye - kalman_gain @ self.meas_model) @ state_cov

        return state, state_cov

    def get_outlier_weight(self, error: torch.tensor, cov: torch.tensor):
        if self.outlier_rejection != "none":
            # Compute residual
            r = torch.sqrt(error.t() @ cov.inverse() @ error)

            # Apply outlier rejection strategy
            if self.outlier_rejection == "hard":
                weight = torch.tensor([0.0]) if r.item() >= self.outlier_delta else torch.tensor([1.0])
            elif self.outlier_rejection == "huber":
                # Prepare Huber loss
                abs_r = r.abs()
                weight = 1.0 if abs_r <= self.outlier_delta else (self.outlier_delta / abs_r).item()
                return weight
            else:
                print(f"Outlier rejection due to invalid option outlier_rejection [{self.outlier_rejection}].")
                return 1.0
        else:
            return 1.0

    def forward(self, state, state_cov, meas, control=None):
        state, state_cov = self.prediction(state, state_cov, control)
        state, state_cov = self.correction(state, state_cov, meas)
        return state, state_cov


def run_kalman_filter():
    """Tests Kalman Filter"""

    import matplotlib.pyplot as plt

    # Normal KF
    kf1 = KalmanFilter(dim_state=1, dim_control=1, dim_meas=1, outlier_rejection="none")
    kf1.init_process_model(proc_model=torch.eye(1) * 1, proc_cov=torch.eye(1) * 0.5)
    kf1.init_meas_model(meas_model=torch.eye(1), meas_cov=torch.eye(1) * 2)

    # Outlier-robust KF
    kf2 = KalmanFilter(
        dim_state=1,
        dim_control=1,
        dim_meas=1,
        outlier_rejection="huber",
        outlier_delta=0.5,
    )
    kf2.init_process_model(proc_model=torch.eye(1) * 1, proc_cov=torch.eye(1) * 0.5)
    kf2.init_meas_model(meas_model=torch.eye(1), meas_cov=torch.eye(1) * 2)

    N = 300
    # Default signal
    t = torch.linspace(0, 10, N)
    T = 5
    x = torch.sin(t * 2 * torch.pi / T)

    # Noises
    # Salt and pepper
    salt_pepper_noise = torch.rand(N)
    min_v = 0.05
    max_v = 1.0 - min_v
    salt_pepper_noise[salt_pepper_noise >= max_v] = 1.0
    salt_pepper_noise[salt_pepper_noise <= min_v] = -1.0
    salt_pepper_noise[torch.logical_and(salt_pepper_noise > min_v, salt_pepper_noise < max_v)] = 0.0
    # White noise
    white_noise = torch.rand(N) / 2

    # Add noise to signal
    x_noisy = x + salt_pepper_noise + white_noise

    # Arrays to store the predictions
    x_e = torch.zeros(2, N)
    x_cov = torch.zeros(2, N)

    # Initial estimate and covariance
    e = torch.FloatTensor([0])
    cov = torch.FloatTensor([0.1])

    # Initial value
    x_e[0, 0] = x_e[1, 0] = e
    x_cov[0, 0] = x_cov[1, 0] = cov

    # Run
    for i in range(1, N):
        # Get sample
        s = x_noisy[i]

        # Get new estimate and cov
        e1, cov1 = kf1(x_e[0, i - 1][None], x_cov[0, i - 1][None], s[None])
        e2, cov2 = kf2(x_e[1, i - 1][None], x_cov[1, i - 1][None], s[None])

        # Save predictions
        x_e[0, i] = e1
        x_cov[0, i] = cov1
        x_e[1, i] = e2
        x_cov[1, i] = cov2

    # Convert to numpy
    t_np = t.numpy()
    x_noisy_np = x_noisy.numpy()
    x_e_np = x_e.detach().numpy()
    x_cov_np = x_cov.detach().numpy()

    # Plot
    plt.plot(t_np, x_noisy_np, label="Noisy signal", color="k")
    plt.plot(t_np, x_e_np[0], label="Filtered", color="r")
    plt.plot(t_np, x_e_np[1], label="Filtered - w/outlier rejection", color="b")
    plt.plot(t_np, x_cov_np[1], label="Cov - w/outlier rejection", color="g")
    plt.fill_between(
        t_np,
        x_e_np[0] - x_cov_np[0],
        x_e_np[0] + x_cov_np[0],
        alpha=0.3,
        label="Confidence bounds (1$\sigma$)",
        color="r",
    )
    plt.fill_between(
        t_np,
        x_e_np[1] - x_cov_np[1],
        x_e_np[1] + x_cov_np[1],
        alpha=0.3,
        label="Confidence bounds (1$\sigma$)",
        color="b",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_kalman_filter()
