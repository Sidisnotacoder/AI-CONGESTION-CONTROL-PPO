"""Background-thread live session driving a MultiFlowCongestionEnv.

dashboard/backend.py's Flask routes talk to exactly one of these at a time
(only one Mininet network may exist): /live/start constructs it and starts
its thread, /live/stream drains its row queue as SSE, /live/control writes
into its locked control state, /live/stop signals it to end. Only this
class's own background thread ever calls into the env (Mininet/tc calls
aren't meant to be touched concurrently from two threads) -- the Flask
request threads only ever read/write plain, lock-guarded Python state.
"""

import logging
import queue
import threading

from env.multi_flow_congestion_env import MultiFlowCongestionEnv
from rl.policy_utils import VALID_POLICIES, resolve_action

logger = logging.getLogger(__name__)


def _row_for_step(step, sender_id, policy, obs, reward, info, action, extra):
    rtt, cwnd, throughput = obs
    components = info.get("reward_components", {})
    return {
        "sender_id": sender_id, "policy": policy,
        "step": step, "rtt_ms": float(rtt), "cwnd": float(cwnd), "throughput_mbps": float(throughput),
        "reward": float(reward), "rate_mbps": info.get("rate_mbps"),
        "action": action,
        "prob_decrease": extra.get("prob_decrease"), "prob_maintain": extra.get("prob_maintain"),
        "prob_increase": extra.get("prob_increase"), "value_estimate": extra.get("value_estimate"),
        "reward_throughput_term": components.get("throughput"),
        "reward_loss_term": components.get("loss_penalty"),
        "reward_rtt_term": components.get("rtt_penalty"),
        "regime_predicted_class": extra.get("regime_predicted_class"),
        "regime_confidence": extra.get("regime_confidence"),
        "tree_direction": extra.get("tree_direction"),
    }


class LiveSession:
    def __init__(self, n_senders, policies, bw, delay, queue_size, max_steps,
                 model, regime_model, on_finished=None, queue_disc="fifo"):
        if len(policies) != n_senders:
            raise ValueError("policies must have exactly n_senders entries")
        for p in policies:
            if p not in VALID_POLICIES:
                raise ValueError(f"unknown policy {p!r}, must be one of {VALID_POLICIES}")
        if "hybrid" in policies and regime_model is None:
            raise ValueError("regime classifier not trained -- run scripts/train_regime_classifier.py")

        self.n_senders = n_senders
        self.policies = list(policies)
        self.model = model
        self.regime_model = regime_model
        self.on_finished = on_finished

        # Cubic arms start at a rate cap well above the bottleneck (same
        # convention as RealCongestionEnv's cubic_baseline: only the real
        # bottleneck link governs, no agent-side cap in the way).
        start_rates = [
            100.0 if p == "cubic" else MultiFlowCongestionEnv.START_RATE_MBPS
            for p in policies
        ]
        self.env = MultiFlowCongestionEnv(
            n_senders=n_senders, bw=bw, delay=delay, queue_size=queue_size,
            max_steps=max_steps, start_rates_mbps=start_rates, queue_disc=queue_disc,
        )

        self.row_queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending_bw = None
        self._pending_delay = None
        self._pending_queue_disc = None
        self._pending_burst = None
        self._human_actions = {i: 1 for i, p in enumerate(policies) if p == "human"}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_obs = [None] * n_senders
        self._last_loss = [0.0] * n_senders

    def start(self):
        self._thread.start()

    def set_bandwidth(self, value):
        with self._lock:
            self._pending_bw = value

    def set_delay(self, value):
        with self._lock:
            self._pending_delay = value

    def set_queue_disc(self, value):
        with self._lock:
            self._pending_queue_disc = value

    def inject_burst(self, duration_s, rate_mbps):
        with self._lock:
            self._pending_burst = (duration_s, rate_mbps)

    def set_human_action(self, sender_id, action):
        if not (0 <= sender_id < self.n_senders) or self.policies[sender_id] != "human":
            raise ValueError(f"sender {sender_id} is not a human-controlled sender in this session")
        with self._lock:
            self._human_actions[sender_id] = action

    def stop(self):
        self._stop_event.set()

    def join(self, timeout=None):
        self._thread.join(timeout=timeout)

    def _resolve_action(self, sender_id, human_actions_snapshot):
        policy = self.policies[sender_id]
        obs = self._last_obs[sender_id]
        loss_pct = self._last_loss[sender_id]
        human_action = human_actions_snapshot.get(sender_id)
        return resolve_action(policy, self.model, self.regime_model, obs, loss_pct, human_action=human_action)

    def _run(self):
        try:
            results = self.env.start()
            for i, (obs, info) in enumerate(results):
                self._last_obs[i] = obs
                self._last_loss[i] = info.get("loss_pct", 0.0)
                row = _row_for_step(0, i, self.policies[i], obs, 0.0, info, None, {})
                self.row_queue.put(row)

            step = 0
            while not self._stop_event.is_set():
                with self._lock:
                    bw, delay, qdisc, burst = self._pending_bw, self._pending_delay, self._pending_queue_disc, self._pending_burst
                    self._pending_bw = self._pending_delay = self._pending_queue_disc = self._pending_burst = None
                    human_actions_snapshot = dict(self._human_actions)

                if bw is not None:
                    self.env.set_bandwidth(bw)
                if delay is not None:
                    self.env.set_delay(delay)
                if qdisc is not None:
                    self.env.set_queue_disc(qdisc)
                if burst is not None:
                    self.env.inject_burst(*burst)

                actions, extras = [], []
                for i in range(self.n_senders):
                    action, extra = self._resolve_action(i, human_actions_snapshot)
                    actions.append(action)
                    extras.append(extra)

                results = self.env.step(actions)
                step += 1
                episode_done = False
                for i, (obs, reward, terminated, truncated, info) in enumerate(results):
                    self._last_obs[i] = obs
                    self._last_loss[i] = info.get("loss_pct", 0.0)
                    row = _row_for_step(step, i, self.policies[i], obs, reward, info, actions[i], extras[i])
                    self.row_queue.put(row)
                    episode_done = episode_done or terminated or truncated
                if episode_done:
                    break

            self.row_queue.put({"__event__": "done"})
        except Exception as e:
            logger.exception("live session failed")
            self.row_queue.put({"__event__": "error", "message": str(e)})
        finally:
            self.env.close()
            # Runs whether the episode finished naturally (max_steps hit)
            # or was stopped/errored -- the only place that reliably knows
            # "this session is truly done," so it's also where the backend
            # releases the one-session-at-a-time lock (see backend.py's
            # live_start/_on_session_finished).
            if self.on_finished is not None:
                self.on_finished()
