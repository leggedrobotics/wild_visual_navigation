from wild_visual_navigation.utils import ConfidenceGenerator
import matplotlib.pyplot as plt
import torch


def generate_traversability_test_signal(N=500, T=10, events=[3, 8], event_length=0.5, device=torch.device("cpu")):
    t = torch.linspace(0, T, N, device=device)
    x = torch.zeros((N))

    if not isinstance(event_length, list):
        event_length = [event_length] * len(events)

    for e, l in zip(events, event_length):
        half_event = l / 2
        ini_event = e
        end_event = e + l
        x += torch.sigmoid(30 * (t - ini_event)) - torch.sigmoid(30 * (t - end_event))
    return t, x


def test_confidence_generator():
    device = torch.device("cpu")
    # Design a long 1D traversability signal
    t, x = generate_traversability_test_signal(
        N=1000, T=30, events=[3, 8, 12], event_length=[1.0, 1.0, 10], device=device
    )

    # Noises
    # Salt and pepper
    salt_pepper_noise = torch.rand(x.shape, device=device)
    min_v = 0.05
    max_v = 0.1 - min_v
    salt_pepper_noise[salt_pepper_noise >= max_v] = 1.0
    salt_pepper_noise[salt_pepper_noise <= min_v] = -1.0
    salt_pepper_noise[torch.logical_and(salt_pepper_noise > min_v, salt_pepper_noise < max_v)] = 0.0
    # White noise
    white_noise = 0.2 * torch.rand(x.shape, device=device)

    # Add noise to signal
    ws = 0.0  # salt and pepper noise weight
    wn = 1.0  # white noise weight
    x_noisy = torch.clip(x + ws * salt_pepper_noise + wn * white_noise, 0.0, 1.0)

    # Arrays to store the predictions
    x_mean = torch.zeros(x.shape, device=device)
    x_std = torch.zeros(x.shape, device=device)
    x_conf = torch.zeros(x.shape, device=device)

    # Confidence generator
    cg = ConfidenceGenerator().to(device)

    # Run
    for i in range(x.shape[0]):
        # Get sample
        s = x_noisy[i]

        # Run confidence generator
        conf = cg.update(s)

        # Get mean and confidence
        x_mean[i] = cg.mean[0]
        x_std[i] = cg.std[0]
        x_conf[i] = conf[0]

    # Convert to numpy
    t_np = t.cpu().numpy()
    x_np = x.cpu().numpy()
    x_noisy_np = x_noisy.cpu().numpy()
    x_mean_np = x_mean.detach().cpu().numpy()
    x_std_np = x_std.detach().cpu().numpy()
    x_conf_np = x_conf.detach().cpu().numpy()

    # Plot
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle("Confidence generator")

    # Signal subplot
    axs[0].plot(t_np, x_np, label="Original signal", color="k")
    axs[0].plot(t_np, x_noisy_np, label="Noisy signal", color="r")
    axs[0].plot(t_np, x_mean_np, label="Filtered", color="b")
    axs[0].fill_between(
        t_np,
        x_mean_np - x_std_np,
        x_mean_np + x_std_np,
        alpha=0.3,
        label="Filtered (1$\sigma$)",
        color="b",
    )
    axs[0].set_ylabel("Signals")
    axs[0].legend(loc="upper right")

    # Confidence subplot
    axs[1].plot(t_np, x_conf_np, label="Confidence", color="k")
    axs[1].fill_between(
        t_np,
        0,
        x_conf_np,
        alpha=0.3,
        color="k",
    )
    axs[1].set_ylabel("Confidence")
    axs[1].legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    test_confidence_generator()
