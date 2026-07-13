import logging
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.augmented_congestion_env import AugmentedCongestionEnv
from rl.model_registry import save_versioned
from rl.callbacks import RewardCurveCallback
from common.network import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

signal.signal(signal.SIGTERM, signal.default_int_handler)

TOTAL_TIMESTEPS = 3000

# Trains a *separate* PPO model against the state-augmented env --
# models/ppo_real_congestion.zip (trained by train_real.py against
# the plain RealCongestionEnv) is untouched by this script. Both
# models coexist and get compared in evaluate_real.py/
# evaluate_multi_trial.py's 3-arm comparison (ppo_agent vs
# ppo_hybrid_tree vs cubic_baseline).
env = AugmentedCongestionEnv()

try:
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        verbose=1,
        tensorboard_log="results/tb_logs/",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=RewardCurveCallback("results/training_reward_curve_augmented.csv"),
    )

    save_versioned(model, "models/ppo_real_congestion_augmented", {
        "env": "AugmentedCongestionEnv (RealCongestionEnv + regime-classifier state augmentation)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_steps": 256,
        "batch_size": 64,
    })

    logger.info("Training complete (augmented env)")
finally:
    env.close()
