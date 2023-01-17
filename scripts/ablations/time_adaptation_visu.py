import pickle
from wild_visual_navigation import WVN_ROOT_DIR
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

with open(os.path.join(WVN_ROOT_DIR, "scripts/ablations/time_adaptation/time_adaptation_steps_done.pkl"), "rb") as f:
    res = pickle.load(f)


auroc_gt = {}
auroc_gt["forest"] = np.zeros((10, 500))
auroc_gt["grassland"] = np.zeros((10, 500))
auroc_gt["hilly"] = np.zeros((10, 500))
auroc_prop = {}
auroc_prop["forest"] = np.zeros((10, 500))
auroc_prop["grassland"] = np.zeros((10, 500))
auroc_prop["hilly"] = np.zeros((10, 500))

for data in res:
    percentage, steps = data["percentage"], data["steps"]
    auroc_gt[data["scene"]][int(percentage / 10), int(steps / 10)] = data["results"][0]["test_auroc_gt_image"]
    auroc_prop[data["scene"]][int(percentage / 10), int(steps / 10)] = data["results"][0][
        "test_auroc_proprioceptive_image"
    ]


def plot_time(title, data):
    label_y = ["Percentage  " + str(k) for k in range(10, 101, 10)]
    label_x = ["Steps " + str(j) for j in range(100, data.shape[1] * 100, 100)]
    fig, ax = plt.subplots()

    im = ax.imshow(data, cmap=sns.color_palette("RdYlBu", as_cmap=True))

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(label_x))[::2])
    ax.set_yticks(np.arange(len(label_y))[::2])
    # ... and label them with the respective list entries
    ax.set_xticklabels(label_x[::2])
    ax.set_yticklabels(label_y[::2])

    # Rotate the tick labels and set their alignment.
    ax.invert_yaxis()
    # ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="left", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    # for i in range(len(label_x)):
    #   for j in range(len(label_y)):
    #     text = ax.text(
    #       i,
    #       j,
    #       data[j, i],
    #       ha="center",
    #       va="center",
    #       color="w",
    #       fontdict={"backgroundcolor": (0, 0, 0, 0.2)},
    #     )

    ax.set_title(title)
    plt.show()


plot_time("Time/Data-Adaptation AUCROC GT", auroc_gt["forest"][1:, :50])
plot_time("Time/Data-Adaptation AUCROC prop", auroc_prop["forest"][1:, :50])
