from wild_visual_navigation.utils import ConfidenceGenerator
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def generate_traversability_test_signal(N=500, T=10, events=[3, 8], event_length=0.5, device=torch.device("cpu")):
    t = torch.linspace(0, T, N, device=device)
    x = torch.zeros((N))

    if not isinstance(event_length, list):
        event_length = [event_length] * len(events)

    for e, l in zip(events, event_length):
        half_event = l / 2
        ini_event = e
        end_event = e + l
        x += torch.sigmoid(10 * (t - ini_event)) - torch.sigmoid(10 * (t - end_event))
    return t, x


def test_confidence_generator():
    device = torch.device("cpu")
    N = 1000

    # Design a long 1D traversability signal
    t, x = generate_traversability_test_signal(N=N, T=30, events=[3, 12], event_length=[5.0, 10], device=device)

    # Noises
    # Salt and pepper
    salt_pepper_noise = torch.rand(x.shape, device=device)
    min_v = 0.05
    max_v = 0.1 - min_v
    salt_pepper_noise[salt_pepper_noise >= max_v] = 1.0
    salt_pepper_noise[salt_pepper_noise <= min_v] = -1.0
    salt_pepper_noise[torch.logical_and(salt_pepper_noise > min_v, salt_pepper_noise < max_v)] = 0.0

    # White noise
    white_noise = 0.3 * (torch.rand(x.shape, device=device) - 0.5)

    # Add noise to signal
    ws = 0.0  # salt and pepper noise weight
    wn = 1.0  # white noise weight
    x_noisy = x + ws * salt_pepper_noise + wn * white_noise

    # positive samples
    x_is_positive = x > 0.7
    x_noisy_positive = x.clone()
    x_noisy_positive[~x_is_positive] = torch.nan

    # Add more noise to simulate predictions we don't know
    more_noise = torch.rand(x.shape, device=device) - 0.5
    x_noisy += more_noise * ~x_is_positive

    # Arrays to store the predictions
    x_mean = torch.zeros((3, N), device=device)
    x_std = torch.zeros((3, N), device=device)
    x_conf = torch.zeros((3, N), device=device)

    # Naive confidence generator
    cg = ConfidenceGenerator().to(device)
    # Confidence generator for positive samples
    cg2 = ConfidenceGenerator().to(device)

    # Run
    for i in range(x.shape[0]):
        # Simulate reconstruccion error
        s = F.mse_loss(x_noisy[i], x[i])
        is_pos = x_is_positive[i]

        # Run confidence generator
        conf = cg.update(s)
        # Get mean and confidence
        x_mean[0, i] = cg.mean[0]
        x_std[0, i] = cg.std[0]
        x_conf[0, i] = conf[0]

        # Run confidence generator
        if is_pos:
            # Get mean and confidence
            conf2 = cg2.update(s)
            x_mean[1, i] = cg2.mean[0]
            x_std[1, i] = cg2.std[0]
            x_conf[1, i] = conf2[0]

            # Get confidence from sigmoid method
            conf = cg2.update_sigmoid(s, slope=1, cutoff=0.2)
            x_conf[2, i] = conf[0]

    # Convert to numpy
    t_np = t.cpu().numpy()
    x_np = x.cpu().numpy()
    x_noisy_np = x_noisy.cpu().numpy()
    x_noisy_positive_np = x_noisy_positive.cpu().numpy()
    x_mean_np = x_mean.detach().cpu().numpy()
    x_std_np = x_std.detach().cpu().numpy()
    x_conf_np = x_conf.detach().cpu().numpy()

    # Plot
    fig, axs = plt.subplots(4, sharex=True)
    fig.suptitle("Confidence generator")

    # Signal subplot
    axs[0].plot(t_np, x_np, label="Target signal", color="k")
    axs[0].plot(t_np, x_noisy_np, label="Reconstructed signal", color="r")
    axs[0].plot(t_np, x_noisy_positive_np, label="Noisy signal - positive samples", color=(0.5, 0, 0), marker=".")
    axs[0].set_ylabel("Signals")
    axs[0].legend(loc="upper right")

    # Confidence subplot
    axs[1].plot(t_np, x_mean_np[0], label="Confidence mean", color="b")
    axs[1].fill_between(
        t_np,
        x_mean_np[0] - x_std_np[0],
        x_mean_np[0] + x_std_np[0],
        alpha=0.3,
        label="Confidence mean (1$\sigma$)",
        color="b",
    )
    axs[1].plot(t_np, x_conf_np[0], label="Confidence", color="k")
    axs[1].fill_between(
        t_np,
        0,
        x_conf_np[0],
        alpha=0.3,
        color="k",
    )
    axs[1].set_ylabel("Confidence")
    axs[1].legend(loc="upper right")

    # Confidence subplot
    axs[2].plot(t_np, x_mean_np[1], label="Filtered mean - positive", color="m")
    axs[2].fill_between(
        t_np,
        x_mean_np[1] - x_std_np[1],
        x_mean_np[1] + x_std_np[1],
        alpha=0.3,
        label="Filtered mean (1$\sigma$) - positive",
        color="m",
    )
    axs[2].plot(t_np, x_conf_np[1], label="Confidence2", color="k")
    axs[2].fill_between(
        t_np,
        0,
        x_conf_np[1],
        alpha=0.3,
        color="k",
    )
    axs[2].set_ylabel("Confidence2")
    axs[2].legend(loc="upper right")

    # COnfidence suplot sigmoid
    axs[3].plot(t_np, x_conf_np[2], label="Confidence3", color="k")
    axs[3].fill_between(
        t_np,
        0,
        x_conf_np[2],
        alpha=0.3,
        color="k",
    )
    axs[3].set_ylabel("Confidence - sigmoid")
    axs[3].legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    test_confidence_generator()
