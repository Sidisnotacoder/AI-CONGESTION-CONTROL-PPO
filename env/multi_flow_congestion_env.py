import logging
import os
import sys
import time
from functools import partial

import numpy as np

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.network import MultiSenderTopo, parse_ss_output, wait_for_estab_socket, FAIL_MODE

logger = logging.getLogger(__name__)


class MultiFlowCongestionEnv:
    """Live orchestrator for N sender hosts sharing one real bottleneck.

    Deliberately not a gym.Env: this drives several independently-policied
    flows at once (for the interactive dashboard), not a single agent being
    trained. RealCongestionEnv is untouched and still used for training and
    single-arm evaluation exactly as before.

    Mirrors RealCongestionEnv's own separation of concerns: this class only
    applies actions and reports observations, it never decides them. Which
    policy (PPO / hybrid / Cubic / human) produced a given action for a
    given sender is entirely the caller's business (dashboard/live_session.py) --
    step() just takes one action int per sender, same as RealCongestionEnv.step()
    takes one action int for its single sender.
    """

    MIN_RATE_MBPS = 1
    RATE_STEP_MBPS = 1
    START_RATE_MBPS = 5
    STEP_INTERVAL_S = 0.5
    BASE_SENDER_PORT = 5201
    BURST_PORT = 5299

    def __init__(self, n_senders=2, bw=10, delay="20ms", queue_size=None,
                 max_steps=40, start_rates_mbps=None, queue_disc="fifo"):
        if os.geteuid() != 0:
            raise SystemExit(
                "MultiFlowCongestionEnv must run as root, e.g.: sudo python3 dashboard/backend.py"
            )

        self.n_senders = n_senders
        self.max_steps = max_steps
        self.start_rates_mbps = (
            list(start_rates_mbps) if start_rates_mbps is not None
            else [float(self.START_RATE_MBPS)] * n_senders
        )
        if len(self.start_rates_mbps) != n_senders:
            raise ValueError("start_rates_mbps must have exactly n_senders entries")

        # TCIntf.config() unconditionally deletes and rebuilds the whole tc
        # qdisc tree from *only* the parameters passed to that specific
        # call -- calling .config(delay=x) alone after an earlier
        # .config(bw=y) silently drops the bandwidth cap (bw defaults to
        # None -> unlimited), and vice versa. Tracking the full current
        # bottleneck config here and always passing all of it on every
        # reconfigure (see _apply_bottleneck_config) is the only way to
        # change one live parameter without silently resetting the others.
        if queue_disc not in ("fifo", "red"):
            raise ValueError(f"queue_disc must be 'fifo' or 'red', got {queue_disc!r}")

        self._bottleneck_bw = bw
        self._bottleneck_delay = delay
        self._bottleneck_queue_size = queue_size
        # "fifo" (default): plain tail-drop, matches every other env in this
        # project. "red": TCIntf's built-in RED (Random Early Detection) --
        # starts probabilistically dropping packets once the queue's
        # *average* occupancy crosses a threshold, rather than only at
        # tail-drop when it's already full, which keeps average queueing
        # delay lower for everyone sharing the link. Exposed as a live-
        # togglable parameter (set_queue_disc) specifically to let a
        # session demonstrate *why* a shared plain FIFO queue lets an
        # uncapped Cubic flow keep RTT just as high for a self-limiting
        # PPO flow as for itself -- RED narrows that gap; it does not fully
        # eliminate it, since it still isn't per-flow fair queueing (every
        # flow's packets face the same drop probability regardless of how
        # much each is individually contributing to the queue).
        self._queue_disc = queue_disc

        # A sender's own tc rate cap is only a meaningful constraint while
        # it stays at or below the real bottleneck's capacity -- letting it
        # climb past that (the original single-flow RealCongestionEnv's
        # MAX_RATE_MBPS=20 with a 10Mbps bottleneck) is harmless solo (no
        # one else to compete with) but breaks the whole "learns to hold
        # back" premise once several senders share one real link: once a
        # sender's cap exceeds the bottleneck, its actual behavior is
        # governed purely by the plain kernel TCP Cubic stack underneath,
        # collapsing it toward indistinguishable-from-cubic regardless of
        # policy. Tracked as a live instance attribute (not a fixed class
        # constant) so it re-tracks set_bandwidth() changes mid-episode.
        self.max_rate_mbps = bw

        self.net = Mininet(
            topo=MultiSenderTopo(n_senders=n_senders, bw=bw, delay=delay, queue_size=queue_size),
            link=TCLink,
            switch=partial(OVSSwitch, failMode=FAIL_MODE),
            controller=None,
        )
        try:
            self.net.start()

            loss = self.net.pingAll()
            if loss >= 100:
                raise RuntimeError("No connectivity in MultiFlowCongestionEnv topology.")

            self.senders = [self.net.get(f"sender{i}") for i in range(n_senders)]
            self.receiver = self.net.get("receiver")
            self.burst_host = self.net.get("burst")

            s1 = self.net.get("s1")
            s2 = self.net.get("s2")
            pairs = s1.connectionsTo(s2)
            if not pairs:
                raise RuntimeError("No s1<->s2 bottleneck link found.")
            self._bottleneck_intf_s1, self._bottleneck_intf_s2 = pairs[0]
        except Exception:
            self.net.stop()
            raise

        self.rate_mbps = list(self.start_rates_mbps)
        self.step_count = 0
        self._burst_process_started = False

    def _sender_port(self, sender_idx):
        return self.BASE_SENDER_PORT + sender_idx

    def start(self):
        """Start iperf3 servers (one per sender + one always-on for burst
        traffic) on the receiver, and one iperf3 client per sender. Returns
        a list of (obs, info) tuples, one per sender, same shape as a single
        RealCongestionEnv.reset()."""
        self.rate_mbps = list(self.start_rates_mbps)
        self.step_count = 0

        self.receiver.cmd("pkill -9 iperf3")
        time.sleep(0.2)
        for i in range(self.n_senders):
            self.receiver.cmd(f"iperf3 -s -p {self._sender_port(i)} -D")
        self.receiver.cmd(f"iperf3 -s -p {self.BURST_PORT} -D")
        time.sleep(0.3)

        for i, sender in enumerate(self.senders):
            sender.cmd("pkill -9 -f 'iperf3 -c'")
        time.sleep(0.2)

        duration = int(self.max_steps * self.STEP_INTERVAL_S) + 20
        for i, sender in enumerate(self.senders):
            intf = sender.intfList()[0]
            intf.config(bw=self.rate_mbps[i])
            sender.cmd(
                f"iperf3 -c {self.receiver.IP()} -p {self._sender_port(i)} "
                f"-t {duration} > /tmp/multi_flow_sender{i}.log 2>&1 &"
            )

        for sender in self.senders:
            if not wait_for_estab_socket(sender, timeout=1.0):
                time.sleep(1)

        return [self._sample_sender(i) for i in range(self.n_senders)]

    def _sample_sender(self, idx):
        sender = self.senders[idx]
        parsed = parse_ss_output(sender.cmd("ss -ti"))
        if parsed is None:
            logger.warning(
                "sender%d step %d: no data-carrying ESTAB socket found, "
                "returning zero-state observation", idx, self.step_count,
            )
            return np.array([0, 0, 0], dtype=np.float32), {"loss_pct": 0.0}
        rtt, cwnd, throughput, loss_pct = parsed
        return np.array([rtt, cwnd, throughput], dtype=np.float32), {"loss_pct": loss_pct}

    def step(self, actions):
        """actions: list[int] of length n_senders (0=decrease, 1=maintain,
        2=increase), already decided by the caller for each sender.
        Returns a list of (obs, reward, terminated, truncated, info), one
        per sender, mirroring RealCongestionEnv.step()'s return shape."""
        if len(actions) != self.n_senders:
            raise ValueError("actions must have exactly n_senders entries")

        for i, sender in enumerate(self.senders):
            action = actions[i]
            if action == 0:
                self.rate_mbps[i] = max(self.MIN_RATE_MBPS, self.rate_mbps[i] - self.RATE_STEP_MBPS)
            elif action == 2:
                self.rate_mbps[i] = min(self.max_rate_mbps, self.rate_mbps[i] + self.RATE_STEP_MBPS)
            sender.intfList()[0].config(bw=self.rate_mbps[i])

        time.sleep(self.STEP_INTERVAL_S)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        results = []
        for i in range(self.n_senders):
            state, extra = self._sample_sender(i)
            rtt, cwnd, throughput = state
            loss_pct = extra["loss_pct"]

            throughput_term = float(throughput)
            loss_term = -2.0 * loss_pct
            rtt_term = -(float(rtt) / 100.0)
            reward = throughput_term + loss_term + rtt_term

            info = {
                "rate_mbps": self.rate_mbps[i],
                "loss_pct": loss_pct,
                "reward_components": {
                    "throughput": throughput_term,
                    "loss_penalty": loss_term,
                    "rtt_penalty": rtt_term,
                },
            }
            results.append((state, reward, terminated, truncated, info))
        return results

    def _apply_bottleneck_config(self):
        """Re-applies the *full* tracked bottleneck config (bw + delay +
        queue_size) to both sides of the s1<->s2 link in one call each --
        see the __init__ comment on why a partial .config() call would
        silently reset whichever parameter isn't passed."""
        opts = {"bw": self._bottleneck_bw, "delay": self._bottleneck_delay,
                "enable_red": self._queue_disc == "red"}
        if self._bottleneck_queue_size is not None:
            opts["max_queue_size"] = self._bottleneck_queue_size
        self._bottleneck_intf_s1.config(**opts)
        self._bottleneck_intf_s2.config(**opts)

    def set_bandwidth(self, bw_mbps):
        """Live-reconfigure the shared bottleneck's bandwidth. Applied
        immediately (the caller -- dashboard/live_session.py -- is
        responsible for calling this at a tick boundary, same as it applies
        any other pending control action)."""
        self._bottleneck_bw = bw_mbps
        self.max_rate_mbps = bw_mbps
        self._apply_bottleneck_config()
        logger.info("bottleneck bandwidth -> %sMbps (sender rate-cap ceiling follows it)", bw_mbps)

    def set_delay(self, delay):
        """delay: string like '20ms', same format Mininet/TCLink expects."""
        self._bottleneck_delay = delay
        self._apply_bottleneck_config()
        logger.info("bottleneck delay -> %s", delay)

    def set_queue_disc(self, queue_disc):
        """Live-toggle the bottleneck's queueing discipline between plain
        FIFO tail-drop and RED, so a session can demonstrate the fairness
        difference mid-run rather than only as a fixed startup choice."""
        if queue_disc not in ("fifo", "red"):
            raise ValueError(f"queue_disc must be 'fifo' or 'red', got {queue_disc!r}")
        self._queue_disc = queue_disc
        self._apply_bottleneck_config()
        logger.info("bottleneck queue discipline -> %s", queue_disc)

    def inject_burst(self, duration_s=4, rate_mbps=8):
        """Fire a non-blocking iperf3 burst from the dedicated burst host
        to the receiver, creating genuine competing traffic on the shared
        bottleneck without touching any sender's own bookkeeping."""
        self.burst_host.cmd(
            f"iperf3 -c {self.receiver.IP()} -p {self.BURST_PORT} "
            f"-t {duration_s} -b {rate_mbps}M > /tmp/multi_flow_burst.log 2>&1 &"
        )
        logger.info("injected traffic burst: %ss at %sMbps", duration_s, rate_mbps)

    def close(self):
        for sender in self.senders:
            sender.cmd("pkill -9 -f 'iperf3 -c'")
        self.burst_host.cmd("pkill -9 -f 'iperf3 -c'")
        self.receiver.cmd("pkill -9 iperf3")
        self.net.stop()
