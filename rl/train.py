import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.congestion_env import CongestionEnv
from rl.model_registry import save_versioned
from rl.callbacks import RewardCurveCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOTAL_TIMESTEPS = 10000

# Create environment
env = CongestionEnv()

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="results/tb_logs/",
)

# Train
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=RewardCurveCallback("results/training_reward_curve.csv"),
)

# Save model (fixed path for existing loaders, plus a timestamped
# copy + metadata sidecar -- see rl/model_registry.py)
save_versioned(model, "models/ppo_congestion", {
    "env": "CongestionEnv (synthetic)",
    "total_timesteps": TOTAL_TIMESTEPS,
})

logger.info("Training complete")
