import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reads the CSV written live by rl/callbacks.py's RewardCurveCallback
# during rl/train_real.py (or rl/train.py). Previously this data was
# hand-copied from a console log into literal Python lists since no
# tensorboard_log/CSV logging was configured for that run -- run
# train_real.py again (now that it logs this CSV) before using this
# script for a fresh curve.
DEFAULT_CSV = "results/training_reward_curve_real.csv"


def load_curve(csv_path):
    timesteps, ep_rew_mean = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            timesteps.append(int(row["timestep"]))
            ep_rew_mean.append(float(row["ep_rew_mean"]))
    return timesteps, ep_rew_mean


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    timesteps, ep_rew_mean = load_curve(csv_path)

    plt.figure(figsize=(8, 4.5))
    plt.plot(timesteps, ep_rew_mean, marker="o", color="#2E75B6")
    plt.xlabel("Training timestep")
    plt.ylabel("Mean episode reward")
    plt.title("PPO Training Reward Curve (real Mininet network)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/training_reward_curve.png", dpi=150)
    plt.close()

    print("Saved: results/training_reward_curve.png")


if __name__ == "__main__":
    main()
