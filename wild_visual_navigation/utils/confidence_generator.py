#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.utils import KalmanFilter
from wild_visual_navigation import WVN_ROOT_DIR
import torch
import os
from collections import deque


class ConfidenceGenerator(torch.nn.Module):
    def __init__(
        self,
        std_factor,
        method,
        log_enabled: bool = False,
        log_folder: str = f"{WVN_ROOT_DIR}/results",
    ):
        """Returns a confidence value for each number

        Args:
            std_factor (float, optional): _description_. Defaults to 0.7.
        """
        super(ConfidenceGenerator, self).__init__()
        self.std_factor = std_factor

        self.log_enabled = log_enabled
        self.log_folder = log_folder

        mean = torch.zeros(1, dtype=torch.float32)
        var = torch.ones((1, 1), dtype=torch.float32)
        std = torch.ones(1, dtype=torch.float32)
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.var = torch.nn.Parameter(var, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

        if method == "kalman_filter":
            kf_process_cov = 0.2
            kf_meas_cov = 1.0
            D = 1
            self._kalman_filter = KalmanFilter(
                dim_state=D,
                dim_control=D,
                dim_meas=D,
            )
            self._kalman_filter.init_process_model(proc_model=torch.eye(D) * 1, proc_cov=torch.eye(D) * kf_process_cov)
            self._kalman_filter.init_meas_model(meas_model=torch.eye(D), meas_cov=torch.eye(D) * kf_meas_cov)
            self._update = self.update_kalman_filter
            self._reset = self.reset_kalman_filter
        elif method == "running_mean":
            running_n = torch.zeros(1, dtype=torch.float64)
            running_sum = torch.zeros(1, dtype=torch.float64)
            running_sum_of_squares = torch.zeros(1, dtype=torch.float64)

            self.running_n = torch.nn.Parameter(running_n, requires_grad=False)
            self.running_sum = torch.nn.Parameter(running_sum, requires_grad=False)
            self.running_sum_of_squares = torch.nn.Parameter(running_sum_of_squares, requires_grad=False)

            # self.running_n = running_n.to("cuda")
            # self.running_sum = running_sum.to("cuda")
            # self.running_sum_of_squares = running_sum_of_squares.to("cuda")

            self._update = self.update_running_mean
            self._reset = self.reset_running_mean
        elif method == "latest_measurement":
            self._update = self.update_latest_measurement
            self._reset = self.reset_latest_measurement
        elif method == "moving_average":
            window_size = 5
            self.data_window = deque(maxlen=window_size)
            self._update = self.update_moving_average
            self._reset = self.reset_moving_average
        else:
            raise ValueError("Unknown method")

    def update_latest_measurement(self, x: torch.Tensor, x_positive: torch.Tensor):
        # Then the confidence is computed as the distance to the center of the Gaussian given factor*sigma
        self.mean[0] = x_positive.mean()
        self.std[0] = x_positive.std()
        return self.inference_without_update(x)

    def reset_latest_measurement(self, x: torch.Tensor, x_positive: torch.Tensor):
        self.mean[0] = 0
        self.var[0] = 1
        self.std[0] = 1

    def reset_moving_average(self, x: torch.Tensor, x_positive: torch.Tensor):
        self.mean[0] = 0
        self.var[0] = 1
        self.std[0] = 1

    def update_running_mean(self, x: torch.tensor, x_positive: torch.tensor):
        # We assume the positive samples' loss follows a Gaussian distribution
        # We estimate the parameters empirically
        self.running_n += x_positive.numel()
        self.running_sum += x_positive.sum()
        self.running_sum_of_squares += (x_positive**2).sum()

        self.mean[0] = self.running_sum[0] / self.running_n
        self.var[0] = self.running_sum_of_squares / self.running_n - self.mean**2
        self.std[0] = torch.sqrt(self.var)

        if x.device != self.mean.device:
            return torch.zeros_like(x)

        shifted_mean = self.mean + self.std * self.std_factor
        std_fac = 1
        interval_min = max(shifted_mean - std_fac * self.std, 0)
        interval_max = shifted_mean + std_fac * self.std
        x = torch.clip(x, interval_min, interval_max)
        confidence = 1 - ((x - interval_min) / (interval_max - interval_min))

        return confidence.type(torch.float32)

    def update_moving_average(self, x: torch.tensor, x_positive: torch.tensor):
        self.data_window.append(x_positive)

        data_window_tensor = list(self.data_window)
        data_window_tensor = torch.cat(data_window_tensor, dim=0)

        self.mean[0] = data_window_tensor.mean()
        self.std[0] = data_window_tensor.std()

        x = torch.clip(x, self.mean - 2 * self.std, self.mean + 2 * self.std)
        confidence = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

        return confidence.type(torch.float32)

    def update_kalman_filter(self, x: torch.tensor, x_positive: torch.tensor):
        # Kalman Filter implementation
        if x_positive.shape[0] != 0:
            mean, var = self._kalman_filter(self.mean, self.var, x_positive.mean())
            self.var[0, 0] = var[0, 0]
            self.mean[0] = mean[0]

            assert not torch.isnan(self.mean).any(), "Nan Value in mean detected"
        self.std[0] = torch.sqrt(self.var)[0, 0]

        # Then the confidence is computed as the distance to the center of the Gaussian given factor*sigma
        confidence = torch.exp(-(((x - self.mean) / (self.std * self.std_factor)) ** 2) * 0.5)
        confidence[x < self.mean] = 1.0

        return confidence.type(torch.float32)

    def update(
        self,
        x: torch.tensor,
        x_positive: torch.tensor,
        step: int,
        log_step: bool = False,
    ):
        """Input a tensor with multiple error predictions.
        Returns the estimated confidence score within 2 standard deviations based on the running mean and variance.

        Args:
            x (torch.tensor): BS,N
        Returns:
            (torch.tensor): BS,N
        """
        output = self._update(x, x_positive)
        # Save data to disk

        if self.log_enabled and log_step:
            base_folder = self.log_folder + "/confidence_generator"
            os.makedirs(base_folder, exist_ok=True)

            with torch.no_grad():
                torch.save(
                    {
                        "x": x.cpu(),
                        "x_positive": x_positive.cpu(),
                        "mean": self.mean.cpu(),
                        "std": self.std.cpu(),
                    },
                    os.path.join(base_folder, f"samples_{step:06}.pt"),
                )

        return output

    def inference_without_update(self, x: torch.tensor):

        if x.device != self.mean.device:
            return torch.zeros_like(x)
        shifted_mean = self.mean + self.std * self.std_factor
        std_fac = 1
        interval_min = max(shifted_mean - std_fac * self.std, torch.zeros_like(self.std))
        interval_max = shifted_mean + std_fac * self.std
        x = torch.clip(x, interval_min, interval_max)
        confidence = 1 - ((x - interval_min) / (interval_max - interval_min))

        return confidence.type(torch.float32)

    def forward(self, x: torch.tensor):
        return self.update(x)

    def reset(self):
        self._reset()

    def reset_running_mean(self):
        self.running_n[0] = 0
        self.running_sum[0] = 0
        self.running_sum_of_squares[0] = 0

    def reset_kalman_filter(self):
        self.mean[0] = 0
        self.var[0] = 1
        self.std[0] = 1

    def get_dict(self):
        return {"mean": self.mean, "var": self.var, "std": self.std}


if __name__ == "__main__":
    cg = ConfidenceGenerator(std_factor=0.5, method="latest_measurement")
    for i in range(1000):
        inp = torch.rand(10) * 10
        res = cg.update(inp, inp, step=i)
        # print("inp ", inp, " res ", res, "std", cg.std)
