"""State-augmented wrapper around RealCongestionEnv.

Composes a RealCongestionEnv instance rather than modifying it --
env/real_congestion_env.py is untouched by this file. On every
step()/reset(), the regime classifier's 4-class probability vector
(rl/regime_classifier.py, trained on data/raw/network_metrics.csv) is
concatenated onto the base env's raw [rtt, cwnd, throughput]
observation, growing the observation space from Box(shape=(3,)) to
Box(shape=(7,)). This is the "state augmentation" half of the
decision-tree-ensemble methodology augmentation; rl/train_augmented.py
trains a separate PPO model against this env, coexisting with (not
replacing) the original model trained directly against
RealCongestionEnv.
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.real_congestion_env import RealCongestionEnv
from rl.regime_classifier import RegimeModel

DEFAULT_REGIME_MODEL_PATH = "models/regime_classifier.joblib"


class AugmentedCongestionEnv(gym.Env):
    def __init__(self, start_rate_mbps=None, regime_model_path=DEFAULT_REGIME_MODEL_PATH):
        super().__init__()
        self._base = RealCongestionEnv(start_rate_mbps=start_rate_mbps)
        self._regime_model = RegimeModel(regime_model_path)

        base_low, base_high = self._base.observation_space.low, self._base.observation_space.high
        n_regime_classes = len(self._regime_model.classes)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_low, np.zeros(n_regime_classes)]),
            high=np.concatenate([base_high, np.ones(n_regime_classes)]),
            dtype=np.float32,
        )
        self.action_space = self._base.action_space

    def _augment(self, state, loss_pct):
        features = np.concatenate([np.asarray(state, dtype=np.float32), [loss_pct]])
        regime_probs = self._regime_model.regime_probs(features)
        return np.concatenate([state, regime_probs]).astype(np.float32)

    def reset(self, seed=None, options=None):
        state, info = self._base.reset(seed=seed, options=options)
        return self._augment(state, info.get("loss_pct", 0.0)), info

    def step(self, action):
        state, reward, terminated, truncated, info = self._base.step(action)
        augmented_state = self._augment(state, info.get("loss_pct", 0.0))
        return augmented_state, reward, terminated, truncated, info

    def render(self):
        self._base.render()

    def close(self):
        self._base.close()
