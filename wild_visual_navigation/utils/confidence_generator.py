from wild_visual_navigation.utils import KalmanFilter
import torch


class ConfidenceGenerator(torch.nn.Module):
    def __init__(self, std_factor: float = 0.7, use_kalman_filter: bool = True):
        """Returns a confidence value for each number

        Args:
            std_factor (float, optional): _description_. Defaults to 2.0.
            device (str, optional): _description_. Defaults to "cpu".
        """
        super(ConfidenceGenerator, self).__init__()
        self.std_factor = std_factor

        self.mean = torch.zeros(1, dtype=torch.float32)
        self.var = torch.ones(1, dtype=torch.float32)
        self.std = torch.ones(1, dtype=torch.float32)

        if use_kalman_filter:
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
            self.update = self.update_kalman_filter
            self.reset = self.reset_kalman_filter
        else:
            running_n = torch.zeros(1, dtype=torch.float64)
            running_sum = torch.zeros(1, dtype=torch.float64)
            running_sum_of_squares = torch.zeros(1, dtype=torch.float64)

            self.running_n = torch.nn.Parameter(running_n, requires_grad=False)
            self.running_sum = torch.nn.Parameter(running_sum, requires_grad=False)
            self.running_sum_of_squares = torch.nn.Parameter(running_sum_of_squares, requires_grad=False)
            self.update = self.update_running_mean
            self.reset = self.reset_running_mean

    def update_running_mean(self, x: torch.tensor, x_positive: torch.tensor):
        # We assume the positive samples' loss follows a Gaussian distribution
        # We estimate the parameters empirically
        self.running_n += x_positive.numel()
        self.running_sum += x_positive.sum()
        self.running_sum_of_squares += (x_positive**2).sum()

        self.mean = self.running_sum / self.running_n
        self.var = self.running_sum_of_squares / self.running_n - self.mean**2
        self.std = torch.sqrt(self.var)

        # Then the confidence is computed as the distance to the center of the Gaussian given factor*sigma
        confidence = torch.exp(-(((x - self.mean) / (2 * self.std * self.std_factor)) ** 2))

        return confidence.type(torch.float32)

    def update_kalman_filter(self, x: torch.tensor, x_positive: torch.tensor):
        # Kalman Filter implementation
        if x_positive.shape[0] != 0:
            self.mean = self.mean.to(x.device)
            self.var = self.var.to(x.device)
            self.mean, self.var = self._kalman_filter(self.mean, self.var, x_positive.mean())
            self.var = self.var[0]
            self.std = torch.sqrt(self.var)
            assert torch.isnan(self.mean).any() == False, "Nan Value in mean detected"
        # Then the confidence is computed as the distance to the center of the Gaussian given factor*sigma
        confidence = torch.exp(-(((x - self.mean) / (2 * self.std * self.std_factor)) ** 2))

        return confidence.type(torch.float32)

    def update(self, x: torch.tensor, x_positive: torch.tensor):
        """Input a tensor with multiple error predictions.
        Returns the estimated confidence score within 2 standard deviations based on the running mean and variance.

        Args:
            x (torch.tensor): BS,N
        Returns:
            (torch.tensor): BS,N
        """
        pass

    def inference_without_update(self, x: torch.tensor):
        if x.device != self.mean.device:
            return torch.zeros_like(x)

        confidence = torch.exp(-(((x - self.mean) / (2 * self.std * self.std_factor)) ** 2))
        return confidence.type(torch.float32)

    def forward(self, x: torch.tensor):
        return self.update(x)

    def reset(self):
        pass

    def reset_running_mean(self):
        self.running_n[0] = 0
        self.running_sum[0] = 0
        self.running_sum_of_squares[0] = 0

    def reset_kalman_filter(self):
        self.mean = torch.zeros(1, dtype=torch.float32)
        self.var = torch.ones(1, dtype=torch.float32)
        self.std = torch.ones(1, dtype=torch.float32)


if __name__ == "__main__":
    cg = ConfidenceGenerator()
    for i in range(100000):
        inp = (
            torch.rand(
                10,
            )
            * 10
        )
        res = cg.update(inp, inp)
        print("inp ", inp, " res ", res, "std", cg.std)
