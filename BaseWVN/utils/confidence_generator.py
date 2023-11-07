import torch
import os
from BaseWVN import WVN_ROOT_DIR
class ConfidenceGenerator(torch.nn.Module):
    def __init__(
        self,
        std_factor: float = 0.5,
        method: str = "running_mean",
        log_enabled: bool = False,
        log_folder: str = "/tmp",
        device: str = "cpu",
    ):
        """Returns a confidence value for each number

        Args:
            std_factor (float, optional): _description_. Defaults to 2.0.
            device (str, optional): _description_. Defaults to "cpu".
        """
        super(ConfidenceGenerator, self).__init__()
        self.device = device
        self.std_factor = std_factor

        self.log_enabled = log_enabled
        self.log_folder = log_folder

        mean = torch.zeros(1, dtype=torch.float32)
        var = torch.ones(1, dtype=torch.float32)
        std = torch.ones(1, dtype=torch.float32)
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.var = torch.nn.Parameter(var, requires_grad=False)
        self.std = torch.nn.Parameter(std, requires_grad=False)

        if method == "running_mean":
            running_n = torch.zeros(1, dtype=torch.float64)
            running_sum = torch.zeros(1, dtype=torch.float64)
            running_sum_of_squares = torch.zeros(1, dtype=torch.float64)

            self.running_n = running_n.to(self.device)
            self.running_sum = running_sum.to(self.device)
            self.running_sum_of_squares = running_sum_of_squares.to(self.device)

            self._update = self.update_running_mean
            self._reset = self.reset_running_mean
        else:
            raise ValueError(f"Method {method} not implemented")
    
    def update_running_mean(self, x: torch.tensor, x_positive: torch.tensor):
        # We assume the positive samples' loss follows a Gaussian distribution
        # We estimate the parameters empirically
        self.running_n += x_positive.numel()
        self.running_sum += x_positive.sum()
        self.running_sum_of_squares += (x_positive**2).sum()

        self.mean[0] = self.running_sum[0] / self.running_n
        self.var[0] = self.running_sum_of_squares / self.running_n - self.mean**2
        self.std[0] = torch.sqrt(self.var)

        if x.device != self.device:
            x=x.to(self.device)
        
        # Then the confidence is computed as the distance to the center of the Gaussian given factor*sigma
        confidence = torch.exp(-(((x - self.mean) / (self.std * self.std_factor)) ** 2) * 0.5)
        confidence[x < self.mean] = 1.0

        return confidence.type(torch.float32)
    
    def reset(self):
        self._reset()

    def reset_running_mean(self):
        self.running_n[0] = 0
        self.running_sum[0] = 0
        self.running_sum_of_squares[0] = 0
    
    def get_dict(self):
        return {"mean": self.mean, "var": self.var, "std": self.std}
    
    def forward(self, x: torch.tensor):
        return self.update(x)
    
    def update(self, x: torch.tensor, x_positive: torch.tensor, step: int, log_step: bool = False):
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
            base_folder = WVN_ROOT_DIR+self.log_folder + "/confidence_generator"
            os.makedirs(base_folder, exist_ok=True)

            with torch.no_grad():
                torch.save(
                    {"x": x.cpu(), "x_positive": x_positive.cpu(), "mean": self.mean.cpu(), "std": self.std.cpu()},
                    os.path.join(base_folder, f"samples_{step:06}.pt"),
                )

        return output
    
    def inference_without_update(self, x: torch.tensor):
        if x.device != self.mean.device:
            x = x.to(self.mean.device)
  
        confidence = torch.exp(-(((x - self.mean) / (self.std * self.std_factor)) ** 2) * 0.5)

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
        res = cg.update(inp, inp,i)
        print("inp ", inp, " res ", res, "std", cg.std)