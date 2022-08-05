import torch
import torch.nn as nn

class KalmanFilter(nn.Module):
    def __init__(self, dim_state: int = 1, dim_control: int = 1, dim_meas: int = 1):
        super().__init__()
        # Store dimensions
        self.dim_state = dim_state
        self.dim_control = dim_control
        self.dim_meas = dim_meas
        # Prediction model
        self.proc_model = nn.Parameter(torch.eye(dim_state))
        self.proc_cov = nn.Parameter(torch.eye(dim_state))
        self.control_model = nn.Parameter(torch.eye(dim_state, dim_control))
        # Correction model
        self.meas_model = nn.Parameter(torch.eye(dim_meas, dim_state))
        self.meas_cov = nn.Parameter(torch.eye(dim_meas, dim_meas))
        # Helper
        self.eye = nn.Parameter(torch.eye(dim_state, dim_state), requires_grad=False)

    def init_process_model(self, proc_model, proc_cov, control_model):
        # Initialize process model
        assert (
            self.proc_model.shape == proc_model.shape
        ), f"Desired state doesn't match {self.proc_model.shape} (expected) != {proc_model.shape} (new)"
        self.proc_model = nn.Parameter(proc_model)

        # Initialize process model covariance
        assert (
            self.proc_cov.shape == proc_cov.shape
        ), f"Desired state doesn't match {self.proc_cov.shape} (expected) != {proc_cov.shape} (new)"
        self.proc_cov = nn.Parameter(proc_cov)

        # Initialize control
        assert (
            self.control_model.shape == control_model.shape
        ), f"Desired state doesn't match {self.control_model.shape} (expected) != {control_model.shape} (new)"
        self.control_model = nn.Parameter(control_model)

    def init_meas_model(self, meas_model, meas_cov):
        # Initialize measurement model
        assert (
            self.meas_model.shape == meas_model.shape
        ), f"Desired state doesn't match {self.meas_model.shape} (expected) != {meas_model.shape} (new)"
        self.meas_model = nn.Parameter(meas_model)

        # Initialize measurement model covariance
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

    def correction(self, state, state_cov, meas: torch.tensor):
        # Innovation
        innovation = meas - self.meas_model @ state
        # Innovation covariance
        innovation_cov = self.meas_model @ state_cov @ self.meas_model.t() + self.meas_cov
        # Kalman gain
        kalman_gain = state_cov @ self.meas_model.t() @ innovation_cov.inverse()

        # Updated state
        state = state + kalman_gain @ innovation
        state_cov = (self.eye - kalman_gain @ self.meas_model) @ state_cov

        return state, state_cov

    def forward(self, state, state_cov, meas, control=None):
        state, state_cov = self.prediction(state, state_cov, control)
        state, state_cov = self.correction(state, state_cov, meas)
        return state, state_cov

def run_kalman_filter():
    """Tests differentiable Kalman Filter"""

    import matplotlib.pyplot as plt
    import seaborn as sns

    kf = KalmanFilter(dim_state=1, dim_control=1, dim_meas=1)
    N = 100

    t = torch.linspace(0, 10, N)
    T = 10
    x = torch.sin(t * 2 * torch.pi / T)
    x_noisy = x + torch.rand(N) / 4

    x_e = torch.zeros(N)
    x_cov = torch.zeros(N)

    e = torch.FloatTensor([0])
    cov = torch.FloatTensor([0.1])

    for i in range(100):
        # Get sample
        s = x_noisy[i]

        # Get new estimate and cov
        e, cov = kf(e, cov, s)

        x_e[i] = e
        x_cov[i] = cov

    t_np = t.numpy()
    x_noisy_np = x_noisy.numpy()
    x_e_np = x_e.detach().numpy()
    x_cov_np = x_cov.detach().numpy()

    plt.plot(t_np, x_noisy_np)
    plt.plot(t_np, x_e_np)
    plt.fill_between(t_np, x_e_np - x_cov_np, x_e_np + x_cov_np, alpha=.3)
    plt.show()

    # Optimize Kalman filter
    optimizer = torch.optim.SGD(kf.parameters(), lr=0.002)    
    for i in range(100):
        # Get sample
        s = x_noisy[i]

        # Get new estimate and cov
        e_new, cov_new = kf(e, cov, s)

        x_e[i] = e.detach()
        x_cov[i] = cov.detach()
    
        loss = torch.nn.functional.mse_loss(e, x[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        e, cov = e_new.detach(), cov_new.detach()

    t_np = t.numpy()
    x_noisy_np = x_noisy.numpy()
    x_e_np = x_e.detach().numpy()
    x_cov_np = x_cov.detach().numpy()

    plt.plot(t_np, x_noisy_np)
    plt.plot(t_np, x_e_np)
    plt.fill_between(t_np, x_e_np - x_cov_np, x_e_np + x_cov_np, alpha=.3)
    plt.show()

if __name__ == "__main__":
    run_kalman_filter()
