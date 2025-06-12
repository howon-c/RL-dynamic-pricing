import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import numpy as np

# You can customize these as needed
DATASETS = {
    "grid_small_dynamic_4_4": "Grid (4x4)",
    "nyc_small": "NYC"
}
ALGORITHMS = ["PPO", "SAC", "TD3"]
COLORS = {
    "PPO": "tab:blue", "SAC": "tab:green", "TD3": "tab:orange"}
LINETYPES = {"grid_small_dynamic_4_4": "-", "nyc_small": "--"}

def find_log_files(base_dir):
    all_logs = {ds:{} for ds in DATASETS}
    for dataset_key in DATASETS:
        base = Path(base_dir) / dataset_key
        # Recursively search for ALL train_log_50.csv under the dataset
        matches = list(base.rglob("train_log_50.csv"))
        print(f"\nSearching in {base}, found {len(matches)} train_log_50.csv files.")
        for m in matches:
            m_str = str(m)
            for algo in ALGORITHMS:
                # Match the algo name anywhere in the path, case sensitive
                if f"{algo}_" in m_str:
                    all_logs[dataset_key][algo] = m
                    print(f"  Found for {algo} in {dataset_key}: {m}")
    return all_logs

import numpy as np

def moving_average_and_std(x, window=5):
    """Compute moving average and std with given window size."""
    ma = np.convolve(x, np.ones(window)/window, mode='valid')
    std = np.array([np.std(x[max(0, i-window+1):i+1]) for i in range(len(x))])[window-1:]
    return ma, std

def plot_curves(logs_dict):
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7*n_datasets, 5), sharey=False)
    if n_datasets == 1:
        axes = [axes]
    for idx, (dataset, logs) in enumerate(logs_dict.items()):
        ax = axes[idx]
        for algo, path in logs.items():
            df = pd.read_csv(path)
            if {'epoch','income','expense'}.issubset(df.columns):
                reward = df['income'] - df['expense']
                epochs = df['epoch'].values
                # Smooth and compute std
                reward_ma, reward_std = moving_average_and_std(reward.values, window=5)
                # Align x for moving average
                x_ma = epochs[len(epochs) - len(reward_ma):]
                # Choose color for known algos or random otherwise
                color = COLORS[algo.split('_')[0]] if algo.split('_')[0] in COLORS else None
                ax.plot(x_ma, reward_ma, label=algo, color=color)
                ax.fill_between(x_ma, reward_ma-reward_std, reward_ma+reward_std, alpha=0.2, color=color)
        # Set axis label and title for each subplot
        ax.set_xlabel("Epoch")
        ylabel = "Daily reward (Smoothed)" if "grid" in dataset else "Weekly reward (Smoothed)"
        ax.set_ylabel(ylabel)
        ax.set_title(DATASETS[dataset])
        ax.legend()
        ax.grid(True)
        if "grid" in dataset:
            ax.set_xlabel("Simulation ticks (x 1440)")
        else:
            ax.set_xlabel("Simulation ticks (x 10080)")
        
    fig.suptitle("Training Reward Curves (Smoothed, with std shading)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # Set the base results directory here:
    BASE_DIR = "results"
    logs = find_log_files(BASE_DIR)
    plot_curves(logs)
