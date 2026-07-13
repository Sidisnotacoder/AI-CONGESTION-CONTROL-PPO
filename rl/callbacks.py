import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class RewardCurveCallback(BaseCallback):
    """Appends (timestep, ep_rew_mean) to a CSV on every rollout, so
    plot_training_curve.py has a real, reproducible data source
    instead of numbers hand-copied from a console log (the previous
    state of rl/plot_training_curve.py -- there was no persisted
    training-curve data at all).

    Computes ep_rew_mean directly from self.model.ep_info_buffer,
    the same way OnPolicyAlgorithm.dump_logs() does -- reading it from
    self.model.logger instead (the more obvious-looking approach)
    doesn't work here: dump_logs() records "rollout/ep_rew_mean" into
    the logger *after* collect_rollouts() (where _on_rollout_end
    fires), so a callback reading the logger at _on_rollout_end always
    sees the previous iteration's value, or nothing on the first one.

    Note: if the env's episodes never terminate/truncate (true of
    env/congestion_env.py's synthetic env, which runs forever until
    the training loop's total_timesteps cap), ep_info_buffer stays
    empty and this CSV legitimately has no rows -- there's no
    "ep_rew_mean" to report without completed episodes. This is
    expected for that env, not a bug; env/real_congestion_env.py
    truncates every MAX_STEPS_PER_EPISODE steps, so train_real.py's
    curve does get populated.
    """

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w") as f:
            f.write("timestep,ep_rew_mean\n")

    def _on_rollout_end(self):
        if self.model.ep_info_buffer and len(self.model.ep_info_buffer) > 0:
            ep_rew_mean = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            with open(self.csv_path, "a") as f:
                f.write(f"{self.num_timesteps},{ep_rew_mean}\n")

    def _on_step(self):
        return True
