"""Single-agent training environment with real competing traffic.

The deployed PPO model (models/ppo_real_congestion.zip) was trained only
against RealCongestionEnv -- one flow, alone, against a fixed bottleneck,
never sharing the link with anyone. Live multi-flow dashboard testing this
session found that model's behavior once it *does* share a link is
essentially undefined extrapolation (e.g. always choosing "increase," every
single step, even at 2000ms+ RTT) -- it's not a considered decision, the
model has simply never seen these states.

CompetitiveCongestionEnv composes MultiFlowCongestionEnv (untouched) to fix
that: it exposes a single-agent gym.Env interface for one learner (always
sender0) while 3 opponent senders run scripted/frozen policies drawn from a
small set of per-episode-randomized compositions, so the resulting policy
has actually seen contention -- including a genuinely uncapped, aggressive
Cubic competitor -- before being asked to handle it live.
"""

import os
import random
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.multi_flow_congestion_env import MultiFlowCongestionEnv
from rl.policy_utils import resolve_action

DEFAULT_OPPONENT_MODEL_PATH = "models/ppo_real_congestion.zip"
DEFAULT_REGIME_MODEL_PATH = "models/regime_classifier.joblib"

# Each entry is the policy for senders 1/2/3 (sender0 is always the
# learner, driven by whatever action SB3's rollout collection provides --
# see the module docstring in env/multi_flow_congestion_env.py for why the
# learner can't literally *be* a hybrid-blended policy mid-training: SB3's
# on-policy loop calls policy.predict() directly, there's no hook for
# injecting a tree blend into that). A composition is resampled every
# reset(), so over many episodes the learner sees a genuinely varied
# population -- not just "always exactly one Cubic peer," which is all the
# first (3-composition) attempt covered:
#   - pure Cubic pressure (worst case: outnumbered 3-to-1 by uncapped peers)
#   - homogeneous PPO-only and hybrid-only peer groups (no Cubic at all --
#     reinforces good behavior among cooperative peers, not just adversarial
#     ones)
#   - 2:1 Cubic-outnumbered mixes (harsher than attempt 1's 1-Cubic-of-4)
#   - "idle" opponents: held at a low, non-competing rate rather than
#     genuinely uncapped -- approximates a near-solo scenario without
#     rebuilding a 1-sender topology. Added specifically because attempt 1's
#     evaluation showed a real regression in the homogeneous case (RTT
#     687ms -> 1020ms) versus the original solo-trained model -- some
#     forgetting of solo-flow restraint likely happened because *every*
#     attempt-1 episode had real contention present. Mixing in low-contention
#     episodes should help preserve that original behavior alongside the new
#     competitive one.
COMPOSITIONS = [
    ["ppo", "ppo", "cubic"],
    ["hybrid", "hybrid", "cubic"],
    ["ppo", "hybrid", "cubic"],
    ["cubic", "cubic", "cubic"],
    ["ppo", "ppo", "ppo"],
    ["hybrid", "hybrid", "hybrid"],
    ["ppo", "cubic", "cubic"],
    ["hybrid", "cubic", "cubic"],
    ["idle", "idle", "idle"],
]

# "idle" behaves exactly like "cubic" from an action-resolution standpoint
# (no agent intervention, always "maintain") -- only its starting rate
# differs, so it doesn't need its own branch in rl/policy_utils.resolve_action.
_START_RATE_OVERRIDE = {"cubic": 100.0, "idle": 2.0}
_RESOLVE_POLICY_ALIAS = {"idle": "cubic"}


class CompetitiveCongestionEnv(gym.Env):
    def __init__(self, bw=10, delay="20ms", max_steps=40,
                 opponent_model_path=DEFAULT_OPPONENT_MODEL_PATH,
                 regime_model_path=DEFAULT_REGIME_MODEL_PATH):
        super().__init__()

        # Imported lazily (not at module level) so this file -- and
        # anything that merely imports it, e.g. a future test -- doesn't
        # require stable_baselines3/torch to already be importable before
        # any training actually starts.
        from stable_baselines3 import PPO
        from rl.regime_classifier import RegimeModel

        # Frozen: loaded once, never updated during training. A *live*,
        # simultaneously-training opponent (true self-play) would be a
        # non-stationary target for PPO's own on-policy updates -- a real
        # training-stability risk not worth taking on for a first pass at
        # this fix.
        self._opponent_model = PPO.load(opponent_model_path)
        self._regime_model = RegimeModel(regime_model_path)

        # Same observation/action space as RealCongestionEnv -- the
        # resulting model stays a drop-in replacement for every existing
        # loader (evaluate_real.py, dashboard/backend.py, etc.).
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([5000, 5000, 100]), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Built once, reused across every episode via MultiFlowCongestionEnv
        # .start() (restarts iperf3 flows only, no Mininet rebuild) --
        # exactly the pattern RealCongestionEnv already uses for the same
        # reason: rebuilding the network every episode would dominate
        # wall-clock time. n_senders is fixed at 4 for the env's lifetime
        # because every COMPOSITIONS entry has exactly 3 opponents.
        self._env = MultiFlowCongestionEnv(
            n_senders=4, bw=bw, delay=delay, max_steps=max_steps,
            start_rates_mbps=[float(MultiFlowCongestionEnv.START_RATE_MBPS)] * 4,
        )

        self._labels = ["ppo", "ppo", "ppo", "ppo"]  # sender0 entry unused, overwritten each reset()
        self._last_obs = [None] * 4
        self._last_loss = [0.0] * 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        opponents = list(random.choice(COMPOSITIONS))
        random.shuffle(opponents)
        self._labels = ["ppo"] + opponents  # index 0 is never read as a policy (learner overrides it)

        self._env.start_rates_mbps = [
            _START_RATE_OVERRIDE.get(label, float(MultiFlowCongestionEnv.START_RATE_MBPS))
            for label in self._labels
        ]
        results = self._env.start()
        for i, (obs, info) in enumerate(results):
            self._last_obs[i] = obs
            self._last_loss[i] = info.get("loss_pct", 0.0)

        return results[0][0], results[0][1]

    def step(self, action):
        actions = [action]
        for i in range(1, 4):
            resolve_policy = _RESOLVE_POLICY_ALIAS.get(self._labels[i], self._labels[i])
            opp_action, _ = resolve_action(
                resolve_policy, self._opponent_model, self._regime_model,
                self._last_obs[i], self._last_loss[i],
            )
            actions.append(opp_action)

        results = self._env.step(actions)
        for i, (obs, _reward, _terminated, _truncated, info) in enumerate(results):
            self._last_obs[i] = obs
            self._last_loss[i] = info.get("loss_pct", 0.0)

        return results[0]

    def render(self):
        pass

    def close(self):
        self._env.close()
