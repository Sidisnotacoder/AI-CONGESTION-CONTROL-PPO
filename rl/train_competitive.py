import logging
import os
import signal
import sys

# Make sure the project root (parent of this rl/ directory) is on
# sys.path so `env.competitive_congestion_env` resolves regardless of the
# working directory this script is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.competitive_congestion_env import CompetitiveCongestionEnv, COMPOSITIONS
from rl.model_registry import save_versioned
from rl.callbacks import RewardCurveCallback
from common.network import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Attempt 1 (3000 timesteps, 3 opponent compositions) measured 0.613s/
# timestep in practice (1884s for 3072 steps) and, on evaluation, showed a
# real but small improvement in mixed-competition throughput fairness
# (Cubic's advantage narrowed from 1.67x to 1.20x) alongside a regression
# in the homogeneous case (687ms -> 1020ms RTT vs the original solo-trained
# model) -- consistent with an under-trained run (its own reward curve was
# still improving, not plateaued, at cutoff: -485 -> -428 over 12
# iterations). This run targets a much larger, user-approved wall-clock
# budget (4.5-5 hours) at the same 0.613s/timestep rate: 26000 timesteps
# -> ~4.43 hours, leaving real margin under the 5-hour ceiling rather than
# targeting it exactly (real runs vary +/-10-15% with system load/reset
# overhead). See docs/COMPETITIVE_RETRAINING_STATUS.md for the full
# before/after numbers and reasoning.
TOTAL_TIMESTEPS = 26000

# See rl/train_real.py for why.
signal.signal(signal.SIGTERM, signal.default_int_handler)

# Live Mininet network, 4 senders (learner + 3 opponents whose composition
# is resampled every reset() -- see env/competitive_congestion_env.py) --
# each step still takes real wall-clock time, same budget as train_real.py
# since STEP_INTERVAL_S is shared across all senders in one step() call,
# not per-sender.
env = CompetitiveCongestionEnv()

try:
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        verbose=1,
        tensorboard_log="results/tb_logs/",
    )

    logger.info("Training against %d opponent compositions: %s", len(COMPOSITIONS), COMPOSITIONS)
    logger.info("Target: %d timesteps, expected ~%.1f hours at 0.613s/timestep", TOTAL_TIMESTEPS, TOTAL_TIMESTEPS * 0.613 / 3600)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=RewardCurveCallback("results/training_reward_curve_competitive.csv"),
    )

    # New path -- models/ppo_real_congestion.zip (the original solo-trained
    # model) is left untouched, so there's an honest before/after to compare
    # via rl/evaluate_competitive.py.
    save_versioned(model, "models/ppo_competitive", {
        "env": "CompetitiveCongestionEnv (4-node Mininet, mixed ppo/hybrid/cubic opponents)",
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_steps": 256,
        "batch_size": 64,
        "opponent_compositions": COMPOSITIONS,
    })

    logger.info("Training complete (competitive multi-flow)")
finally:
    env.close()
