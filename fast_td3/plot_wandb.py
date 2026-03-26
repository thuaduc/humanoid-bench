import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()

entity = "thuaduc24042001-technical-university-of-munich"
project = "Benchmark"

# Hurdle
# runs = {
#     "EGNN": f"{entity}/{project}/4izpgftd",
#     "MLP": f"{entity}/{project}/lg5ep5fo",
#     "Transformer": f"{entity}/{project}/hgjzglyh",
# }

# Slide
runs = {
    "EGNN": f"{entity}/{project}/yg3u4fzr",
    "MLP": f"{entity}/{project}/x818db6p",
    "Transformer": f"{entity}/{project}/8h36jn4d",
}

metrics = [
    "eval_avg_return",
    "env_rewards",
    "buffer_rewards",
    "qf_max",
]

window = 100  # adjust to taste

for metric in metrics:
    plt.figure()

    for label, run_path in runs.items():
        run = api.run(run_path)

        history = run.history(keys=["_step", metric])
        df = pd.DataFrame(history).dropna()

        df[f"{metric}_avg"] = (
            df[metric]
            .rolling(window=window, min_periods=1)
            .mean()
        )

        plt.plot(
            df["_step"],
            df[f"{metric}_avg"],
            label=label,
        )

    plt.xlabel("Timestep")
    plt.ylabel(metric)
    plt.title(f"{metric} (running avg, window={window})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{metric}.png")
