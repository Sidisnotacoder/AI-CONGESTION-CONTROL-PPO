import logging
import os
import sys
import time
from functools import partial

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.network import BottleneckTopo, parse_ss_output, wait_for_estab_socket, FAIL_MODE

logger = logging.getLogger(__name__)


class RealCongestionEnv(gym.Env):
    """Gym env driven by a live Mininet network. Actions adjust h1's
    own uplink rate (via tc, same mechanism validated in
    scripts/collect_dataset.py); state comes from real ss -ti stats
    on the resulting iperf3 flow crossing the s1<->s2 bottleneck.

    One Mininet network is kept running for the lifetime of the env
    (created in __init__, torn down in close()) since rebuilding it
    every episode would dominate wall-clock time. reset() only
    restarts the iperf3 server/client, which is fast.
    """

    MIN_RATE_MBPS = 1
    MAX_RATE_MBPS = 20
    RATE_STEP_MBPS = 1
    START_RATE_MBPS = 5
    STEP_INTERVAL_S = 0.5
    MAX_STEPS_PER_EPISODE = 40
    SERVER_PORT = 5201

    def __init__(self, start_rate_mbps=None):
        super().__init__()

        # Overridable per-episode starting rate. Used to run a plain
        # TCP Cubic baseline (start high, e.g. 100, well above the
        # 10Mbps bottleneck, then never touch it -- only the real
        # bottleneck link governs, same as any other Cubic flow) for
        # comparison against the agent's normal 5Mbps start.
        self.start_rate_mbps = (
            start_rate_mbps if start_rate_mbps is not None else self.START_RATE_MBPS
        )

        if os.geteuid() != 0:
            raise SystemExit(
                "RealCongestionEnv must run as root, e.g.: sudo python3 rl/train_real.py"
            )

        # State: [RTT(ms), CWND, Throughput(Mbps)] -- same shape as
        # env/congestion_env.py so it's a drop-in swap for train.py.
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([5000, 5000, 100]),
            dtype=np.float32,
        )

        # 0 = decrease, 1 = maintain, 2 = increase (uplink rate)
        self.action_space = spaces.Discrete(3)

        self.net = Mininet(
            topo=BottleneckTopo(),
            link=TCLink,
            switch=partial(OVSSwitch, failMode=FAIL_MODE),
            controller=None,
        )
        try:
            self.net.start()

            loss = self.net.pingAll()
            if loss >= 100:
                raise RuntimeError("No connectivity in RealCongestionEnv topology.")

            self.h1 = self.net.get("h1")
            self.h3 = self.net.get("h3")
        except Exception:
            # Anything going wrong after net.start() (pingAll failure,
            # or any future addition to this block) must not leak a
            # live Mininet network/OVS bridges -- tear it down before
            # propagating, same as the explicit pingAll path already did.
            self.net.stop()
            raise

        self.rate_mbps = float(self.START_RATE_MBPS)
        self.step_count = 0

    def _restart_server(self):
        self.h3.cmd("pkill -9 iperf3")
        time.sleep(0.2)
        self.h3.cmd(f"iperf3 -s -p {self.SERVER_PORT} -D")
        time.sleep(0.3)

    def _restart_client(self):
        self.h1.cmd("pkill -9 -f 'iperf3 -c'")
        time.sleep(0.2)
        duration = int(self.MAX_STEPS_PER_EPISODE * self.STEP_INTERVAL_S) + 10
        self.h1.cmd(
            f"iperf3 -c {self.h3.IP()} -p {self.SERVER_PORT} "
            f"-t {duration} > /tmp/real_env_client.log 2>&1 &"
        )
        # Bounded readiness poll instead of a flat sleep: returns as
        # soon as the handshake is actually done (usually much faster
        # than 1s), falls back to that same 1s wait if it never
        # converges within the timeout (never hangs indefinitely).
        if not wait_for_estab_socket(self.h1, timeout=1.0):
            time.sleep(1)

    def _apply_rate(self):
        intf = self.h1.intfList()[0]
        intf.config(bw=self.rate_mbps)

    def _sample_state(self):
        parsed = parse_ss_output(self.h1.cmd("ss -ti"))
        if parsed is None:
            # parse_ss_output already logged the raw output that
            # failed to parse; the zero-state fallback below is a
            # last resort to keep step() returning a valid observation,
            # not a legitimate "throughput dropped to zero" reading --
            # log it here too so a broken monitoring pipeline is
            # visible in this env's own step count, not just once.
            logger.warning(
                "step %d: no data-carrying ESTAB socket found, "
                "returning zero-state observation", self.step_count
            )
            return np.array([0, 0, 0], dtype=np.float32), 0.0
        rtt, cwnd, throughput, loss_pct = parsed
        return np.array([rtt, cwnd, throughput], dtype=np.float32), loss_pct

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.rate_mbps = float(self.start_rate_mbps)
        self.step_count = 0

        self._restart_server()
        self._apply_rate()
        self._restart_client()

        state, loss_pct = self._sample_state()
        return state, {"loss_pct": loss_pct}

    def step(self, action):
        if action == 0:
            self.rate_mbps = max(self.MIN_RATE_MBPS, self.rate_mbps - self.RATE_STEP_MBPS)
        elif action == 2:
            self.rate_mbps = min(self.MAX_RATE_MBPS, self.rate_mbps + self.RATE_STEP_MBPS)
        self._apply_rate()

        time.sleep(self.STEP_INTERVAL_S)  # let the network react before sampling
        state, loss_pct = self._sample_state()
        rtt, cwnd, throughput = state

        # Reward = throughput - loss penalty - latency penalty
        # (Section 8 of the project plan). Each term is also kept
        # separately in reward_components so callers (evaluate_real.py,
        # the dashboard) can show *why* a given reward was earned
        # instead of just the summed total.
        throughput_term = float(throughput)
        loss_term = -2.0 * loss_pct
        rtt_term = -(float(rtt) / 100.0)
        reward = throughput_term + loss_term + rtt_term

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.MAX_STEPS_PER_EPISODE

        info = {
            "rate_mbps": self.rate_mbps,
            "loss_pct": loss_pct,
            "reward_components": {
                "throughput": throughput_term,
                "loss_penalty": loss_term,
                "rtt_penalty": rtt_term,
            },
        }
        return state, reward, terminated, truncated, info

    def render(self):
        logger.info("current rate_mbps=%s", self.rate_mbps)

    def close(self):
        self.h1.cmd("pkill -9 -f 'iperf3 -c'")
        self.h3.cmd("pkill -9 iperf3")
        self.net.stop()
