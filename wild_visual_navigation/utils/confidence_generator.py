import torch


class ConfidenceGenerator(torch.nn.Module):
    def __init__(self, std_factor: float = 0.7):
        """Returns a confidence value for each number

        Args:
            std_factor (float, optional): _description_. Defaults to 2.0.
            device (str, optional): _description_. Defaults to "cpu".
        """
        super(ConfidenceGenerator, self).__init__()
        running_n = torch.zeros(1, dtype=torch.float64)
        running_sum = torch.zeros(1, dtype=torch.float64)
        running_sum_of_squares = torch.zeros(1, dtype=torch.float64)

        self.running_n = torch.nn.Parameter(running_n, requires_grad=False)
        self.running_sum = torch.nn.Parameter(running_sum, requires_grad=False)
        self.running_sum_of_squares = torch.nn.Parameter(running_sum_of_squares, requires_grad=False)

        self.std_factor = std_factor
        self.mean = torch.zeros(1, dtype=torch.float64)
        self.var = torch.ones(1, dtype=torch.float64)
        self.std = torch.ones(1, dtype=torch.float64)

    def update(self, x: torch.tensor):
        """Input a tensor with multiple error predictions.
        Returns the estimated confidence score within 2 standard deviations based on the running mean and variance.

        Args:
            x (torch.tensor): BS,N
        Returns:
            (torch.tensor): BS,N
        """
        self.running_n += x.numel()
        self.running_sum += x.sum()
        self.running_sum_of_squares += (x**2).sum()

        self.mean = self.running_sum / self.running_n
        self.var = self.running_sum_of_squares / self.running_n - self.mean**2
        self.std = torch.sqrt(self.var)

        uncertainty = (
            (x - self.mean).clip(min=-(self.std * self.std_factor), max=(self.std * self.std_factor))
            + self.std * self.std_factor
        ) / (2 * self.std * self.std_factor)
        confidence = 1 - uncertainty
        return confidence.type(torch.float32)

    def inference_without_update(self, x: torch.tensor):
        if x.device != self.mean.device:
            return torch.zeros_like(x)

        uncertainty = (
            (x - self.mean).clip(min=-(self.std * self.std_factor), max=(self.std * self.std_factor))
            + self.std * self.std_factor
        ) / (2 * self.std * self.std_factor)
        confidence = 1 - uncertainty
        return confidence.type(torch.float32)

    def forward(self, x: torch.tensor):
        return self.update(x)

    def reset(self):
        self.running_n[0] = 0
        self.running_sum[0] = 0
        self.running_sum_of_squares[0] = 0


if __name__ == "__main__":
    cg = ConfidenceGenerator()
    for i in range(100000):
        inp = (
            torch.rand(
                10,
            )
            * 10
        )
        res = cg.update(inp)
        print("inp ", inp, " res ", res, "std", cg.std)
