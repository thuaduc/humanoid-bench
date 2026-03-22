import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

api = wandb.Api()

entity = "thuaduc24042001-technical-university-of-munich"
project = "Benchmark final"

tasks = {
    "balance-simple": {
        "runs": {
            "EGNN": f"{entity}/{project}/3d66q5ev",
            "MLP": f"{entity}/{project}/maipl69r",
        },
        "expected_return": 800
    },
    "reach": {
        "runs": {
            "EGNN": f"{entity}/{project}/xdf00320",
            "MLP": f"{entity}/{project}/4ulvg7u4",
        },
        "expected_return": 12000
    },
    # "push": {
    #     "runs": {
    #         "EGNN": f"{entity}/{project}/a1ssq6a9",
    #         "MLP": f"{entity}/{project}/o9jz7yhn",
    #     },
    #     "expected_return": 700
    # },
    # "run": {
    #     "runs": {
    #         "EGNN": f"{entity}/{project}/qbvhkgzl",
    #         "MLP": f"{entity}/{project}/4lz2zx4j",
    #     },
    #     "expected_return": 700
    # },
    # "slide": {
    #     "runs": {
    #         "EGNN": f"{entity}/{project}/yg3u4fzr",
    #         "MLP": f"{entity}/{project}/x818db6p    ",
    #     },
    #     "expected_return": 700
    # },
    # "hurdle": {
    #     "runs": {
    #         "EGNN": f"{entity}/{project}/4izpgftd",
    #         "MLP": f"{entity}/{project}/lg5ep5fo",
    #     },
    #     "expected_return": 600
    # },
}

metrics = [
    "eval_avg_return",
]

window = 1 # adjust to taste

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
})

os.makedirs("plots", exist_ok=True)

modes = ["step"]

for task, task_data in tasks.items():
    runs = task_data["runs"]
    expected_return = task_data["expected_return"]
    
    for metric in metrics:
        for x_mode in modes:
            fig, ax = plt.subplots(figsize=(6, 3.2))

            for label, run_path in runs.items():
                run = api.run(run_path)

                keys = ["_step", "_timestamp", metric]
                history = run.history(keys=keys)
                df = pd.DataFrame(history).dropna()

                # Running average
                df[f"{metric}_avg"] = (
                    df[metric]
                    .rolling(window=window, min_periods=1)
                    .mean()
                )

                if x_mode == "step":
                    x = df["_step"]
                    xlabel = "Training step"
                    suffix = "step"
                else:
                    # Convert to hours for readability
                    x = (df["_timestamp"] - df["_timestamp"].iloc[0]) / 3600.0
                    xlabel = "Wall-clock time (hours)"
                    suffix = "time"

                ax.plot(
                    x,
                    df[f"{metric}_avg"],
                    linewidth=2.0,
                    label=label,
                )

            ax.axhline(y=expected_return, linestyle='--', color='gray', linewidth=1.5)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            ax.set_title(f"{task} – {metric}", pad=6)

            if x_mode == "step":
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

            # Clean paper look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.margins(x=0)

            fig.tight_layout()
            fig.savefig(f"plots/{task}_{metric}_{suffix}.png")
            plt.close(fig)
