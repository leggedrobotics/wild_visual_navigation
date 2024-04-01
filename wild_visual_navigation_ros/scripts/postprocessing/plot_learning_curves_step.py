import matplotlib.pyplot as plt
from wild_visual_navigation.visu import paper_colors as pc
from pathlib import Path
import pandas as pd

from wild_visual_navigation import WVN_ROOT_DIR

PALETTE = [[r, g, b] for r, g, b in pc.paper_colors_rgb_f.values()]

experiment_folder = f"{WVN_ROOT_DIR}/wild_visual_navigation_ros/output"
cases = {}

run_dirs = sorted(list(Path(experiment_folder).iterdir()))
for d in run_dirs:
    if not d.is_dir():
        continue
    features, sampling, run = d.name.split("_")
    label = f"{features.upper()}-{sampling.upper()}"
    try:
        cases[label].append(d)
    except Exception:
        cases[label] = [d]


# Matplotlib config
cm = 1 / 2.54
plot_width = 8.89 * cm
plot_height = 4 * cm
plt.rcParams["font.size"] = 8
n_colors = 10

fig, ax = plt.subplots(1, 1, figsize=(plot_width, plot_height), constrained_layout=False, dpi=300)
# Axes
ax.set_axisbelow(True)
# ax.set_aspect("equal")
ax.grid(which="major", color=(0.8, 0.8, 0.8), linewidth=0.7)
ax.grid(which="minor", color=(0.9, 0.9, 0.9), linestyle=":", linewidth=0.5)
ax.minorticks_on()

# Plotting
for i, c in enumerate(cases):
    print(c, len(cases[c]))
    case_df = pd.DataFrame()
    dfs = []

    for path in cases[c]:
        state_file = list(path.glob("*.csv"))[0]
        df = pd.read_csv(state_file).drop_duplicates(subset="stamp")
        df["stamp"] = df["stamp"] - df["stamp"][0]

        valid_idx = (df["stamp"] < 110) & (df["loss_total"] >= 0.0)
        df = df[valid_idx]
        # df = df.set_index(df["stamp"])
        df = df.set_index(df["step"])
        # joined_indices = case_df.index.union(df.index).drop_duplicates()
        # case_df = case_df.reindex(index=joined_indices)
        # case_df = case_df.interpolate(method="index")
        dfs.append(df)

    for j, df in enumerate(dfs):
        # df_j = df.reindex(index=case_df.index)
        # df_j = df_j.interpolate(method="index")
        # case_df[j] = df_j["loss_total"]
        case_df[j] = df["loss_total"]

    # Compute runs
    # case_df = case_df.dropna()
    stamp = case_df.index
    loss_mean = case_df.mean(axis=1)
    loss_std = case_df.std(axis=1)

    # Plot
    ax.plot(stamp, loss_mean, label=c, color=PALETTE[i])
    plt.fill_between(
        stamp,
        loss_mean - 2 * loss_std,
        loss_mean + 2 * loss_std,
        alpha=0.3,
        # label="Confidence bounds (1$\sigma$)",
        color=PALETTE[i],
    )
ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
# ax.margins(x=0.01, y=0.01)
plt.legend()

fig.set_tight_layout(True)
fig.savefig(f"{experiment_folder}/empirical_learning_curves.pdf")
