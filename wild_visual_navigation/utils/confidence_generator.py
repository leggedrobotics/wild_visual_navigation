import torch


class ConfidenceGenerator:
    def __init__(self, std_factor: float = 0.7, device: str = "cpu"):
        """Returns a confidence value for each number

        Args:
            std_factor (float, optional): _description_. Defaults to 2.0.
            device (str, optional): _description_. Defaults to "cpu".
        """

        self.device = device
        self.running_n = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.running_sum = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.running_sum_of_sqaures = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.std_factor = std_factor
        self.mean = torch.zeros(1, dtype=torch.float64, device=self.device)
        self.var = torch.ones(1, dtype=torch.float64, device=self.device)
        self.std = torch.ones(1, dtype=torch.float64, device=self.device)

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
        self.running_sum_of_sqaures += (x**2).sum()

        self.mean = self.running_sum / self.running_n
        self.var = self.running_sum_of_sqaures / self.running_n - self.mean**2
        self.std = torch.sqrt(self.var)

        uncertainty = (
            (x - self.mean).clip(min=-(self.std * self.std_factor), max=(self.std * self.std_factor))
            + self.std * self.std_factor
        ) / (2 * self.std * self.std_factor)
        confidence = 1 - uncertainty
        return confidence.type(torch.float32)


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
