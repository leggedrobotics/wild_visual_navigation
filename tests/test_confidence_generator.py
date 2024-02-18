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
        # half_event = l / 2
        ini_event = e
        end_event = e + l
        x += torch.sigmoid(10 * (t - ini_event)) - torch.sigmoid(10 * (t - end_event))
    return t, x


def test_confidence_generator():
    from wild_visual_navigation.utils.testing import make_results_folder
    from wild_visual_navigation.visu import get_img_from_fig
    from os.path import join

    # Create test directory
    outpath = make_results_folder("test_confidence_generator")

    device = torch.device("cpu")
    N = 1000
    sigma_factor = 0.5

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
    white_noise = 0.1 * (torch.rand(x.shape, device=device) - 0.5)

    # Add noise to signal
    ws = 0.0  # salt and pepper noise weight
    wn = 1.0  # white noise weight
    x_noisy = x + ws * salt_pepper_noise + wn * white_noise

    # positive samples
    x_is_positive = x > 0.7
    x_noisy_positive = x.clone()
    x_noisy_positive[~x_is_positive] = torch.nan

    # Add more noise to simulate predictions we don't know
    more_noise = 1.0 * (torch.rand(x.shape, device=device) - 0.5) + 1.0
    x_noisy += more_noise * ~x_is_positive

    # Add constant bias
    bias = t * 0.1
    x += bias
    x_noisy += bias
    x_noisy_positive += bias

    # Arrays to store the predictions
    loss = torch.zeros((3, N), device=device)
    loss_mean = torch.zeros((3, N), device=device)
    loss_std = torch.zeros((3, N), device=device)
    conf = torch.zeros((3, N), device=device)

    # Naive confidence generator
    cg = ConfidenceGenerator(std_factor=sigma_factor, method="latest_measurement").to(device)

    # Run
    for i in range(x.shape[0]):
        # Simulate reconstruccion error
        s = F.mse_loss(x_noisy[i], x[i])
        is_pos = x_is_positive[i]
        loss[0, i] = s

        # Run confidence generator
        if is_pos:
            c = cg.update(s, s[None], step=i)
        else:
            c = cg.update(s, torch.tensor([]), step=i)  # masking tensors returns empty tensor

        # Get mean and confidence
        loss_mean[0, i] = cg.mean[0]
        loss_std[0, i] = cg.std[0]
        conf[0, i] = c[0]

    # Convert to numpy
    t_np = t.cpu().numpy()
    loss_np = loss.cpu().numpy()
    x_np = x.cpu().numpy()
    x_noisy_np = x_noisy.cpu().numpy()
    x_noisy_positive_np = x_noisy_positive.cpu().numpy()
    loss_mean_np = loss_mean.detach().cpu().numpy()
    loss_std_np = loss_std.detach().cpu().numpy()
    conf_np = conf.detach().cpu().numpy()

    # Plot
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle("Confidence generator")

    # Signal subplot
    axs[0].plot(t_np, x_noisy_np, label="Reconstructed signal", color="r")
    axs[0].plot(t_np, x_np, label="Target signal", color="k")
    axs[0].plot(
        t_np,
        x_noisy_positive_np,
        label="Positive samples",
        color=(0.5, 0, 0),
        marker=".",
    )
    axs[0].set_ylabel("Signals")
    axs[0].legend(loc="upper right")

    # Naive loss distribution mean
    axs[1].fill_between(
        t_np,
        t_np * 0.0,
        loss_mean_np[0] + sigma_factor * loss_std_np[0],
        alpha=0.2,
        label=f"Loss mean $\pm{sigma_factor}\sigma$",
        color="b",
    )

    axs[1].fill_between(
        t_np,
        t_np * 0.0,
        loss_mean_np[0] + loss_std_np[0],
        alpha=0.6,
        label="Loss mean $\pm\sigma$",
        color="b",
    )

    axs[1].plot(t_np, loss_np[0], label="Loss", color="k")
    axs[1].plot(t_np, loss_mean_np[0], label="Loss mean", color="b")
    axs[0].set_ylabel("Loss distribution")
    axs[1].legend(loc="upper right")

    # Naive Confidence
    axs[2].plot(t_np, conf_np[0], label="Confidence", color="k")
    axs[2].fill_between(
        t_np,
        0,
        conf_np[0],
        alpha=0.3,
        color="k",
    )
    axs[2].set_ylabel("Confidence")
    axs[2].legend(loc="upper right")

    img = get_img_from_fig(fig)
    img.save(
        join(
            outpath,
            "confidence_generator_test.png",
        )
    )


if __name__ == "__main__":
    test_confidence_generator()
