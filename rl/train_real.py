import logging
import os
import signal
import sys

# Make sure the project root (parent of this rl/ directory) is on
# sys.path so `env.real_congestion_env` resolves regardless of the
# working directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.real_congestion_env import RealCongestionEnv
from rl.model_registry import save_versioned
from rl.callbacks import RewardCurveCallback
from common.network import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

TOTAL_TIMESTEPS = 3000

# A SIGTERM (e.g. from a process supervisor, or an aborted
# dashboard-triggered run) bypasses Python's normal try/finally
# unwinding by default. Routing it through the same handler SIGINT
# already uses turns it into a KeyboardInterrupt, so the try/finally
# below (and RealCongestionEnv.close()) still runs instead of leaking
# the live Mininet network and iperf3 processes.
signal.signal(signal.SIGTERM, signal.default_int_handler)

# Create environment (live Mininet network -- each step takes real
# wall-clock time, so this uses a much smaller timestep budget than
# train.py's synthetic-env run, and a smaller rollout buffer so PPO
# actually gets multiple policy updates within that budget).
env = RealCongestionEnv()

try:
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        verbose=1,
        tensorboard_log="results/tb_logs/",
    )

    # Train (~3000 steps * 0.5s/step + reset overhead ~= 25-30 minutes)
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=RewardCurveCallback("results/training_reward_curve_real.csv"),
    )

    # Save model (fixed path for existing loaders, plus a timestamped
    # copy + metadata sidecar -- see rl/model_registry.py)
    save_versioned(model, "models/ppo_real_congestion", {
        "env": "RealCongestionEnv (live Mininet)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_steps": 256,
        "batch_size": 64,
    })

    logger.info("Training complete (real network)")
finally:
    env.close()
