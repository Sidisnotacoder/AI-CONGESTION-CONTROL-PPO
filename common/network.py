"""Shared Mininet topology and ss -ti parsing logic.

Previously this BottleneckTopo/parse_ss_output pair was copy-pasted
across env/real_congestion_env.py, scripts/collect_dataset.py, and
(a slightly different inline variant) scripts/monitor.py -- with
env/real_congestion_env.py and scripts/collect_dataset.py already
byte-for-byte identical, proving the drift risk was real rather than
hypothetical. This module is the single source of truth; every caller
imports from here instead.
"""

import logging
import re
import time

from mininet.topo import Topo

logger = logging.getLogger(__name__)

# OVSSwitch's default failMode is 'secure' (empty flow table, drops
# everything with no controller). Passing controller=None to Mininet()
# does not fix this by itself -- switches must be created with
# failMode='standalone' so they behave as plain L2 learning switches.
FAIL_MODE = "standalone"


def configure_logging(level=logging.INFO):
    """logging.basicConfig(), plus undoing a global side effect of
    importing mininet.log: it calls logging.setLoggerClass(MininetLogger)
    at import time, and MininetLogger.__init__ attaches its own
    private, unformatted StreamHandler to *every* logger constructed
    afterward -- not just Mininet's own "mininet"-named one. Left
    alone, that means any logger.warning() call anywhere in a process
    that has imported mininet prints twice: once raw via that private
    per-logger handler, once formatted via whatever basicConfig() adds
    to root (both fire because Logger.callHandlers walks the private
    handler *and* propagates to root's).

    Fix: reset the logger class so anything constructed from this
    point on is a plain logging.Logger (no auto-attached handler), and
    strip the private handler from anything already constructed under
    MininetLogger (e.g. common.network's own module-level `logger`
    above, built at import time -- before any caller gets a chance to
    call this function). Mininet's own "mininet"-named logger keeps
    its private handler (so its unformatted "*** ..." output still
    looks the way Mininet intends) but gets propagate=False, so its
    messages go out through exactly that one handler instead of also
    hitting root's -- the same double-print problem, just the other
    direction (Mininet's handler plus root's, instead of a private
    handler plus root's).

    Call this instead of logging.basicConfig() directly in any script
    that also drives Mininet."""
    logging.setLoggerClass(logging.Logger)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("mininet").propagate = False
    for name, existing in list(logging.root.manager.loggerDict.items()):
        if name == "mininet" or not isinstance(existing, logging.Logger):
            continue
        private_handler = getattr(existing, "ch", None)
        if private_handler is not None and private_handler in existing.handlers:
            existing.removeHandler(private_handler)


class BottleneckTopo(Topo):
    def build(self, bw=10, delay="20ms"):
        h1 = self.addHost("h1")
        h2 = self.addHost("h2")
        h3 = self.addHost("h3")
        s1 = self.addSwitch("s1")
        s2 = self.addSwitch("s2")
        self.addLink(h1, s1)
        self.addLink(h2, s1)
        self.addLink(h3, s2)
        self.addLink(s1, s2, bw=bw, delay=delay)


def parse_ss_output(output):
    """Return (rtt_ms, cwnd, throughput_mbps, loss_pct) for the ESTAB
    socket actually carrying data (largest bytes_sent). ss -ti reports
    more than one socket per iperf3 run (control channel + data
    stream); mixing fields across sockets pairs the wrong values
    together, so every field here comes from the same block.

    Returns None if no data-carrying ESTAB socket could be found --
    callers should treat this as "monitoring pipeline broke", not as
    "the flow legitimately went to zero", and log accordingly instead
    of silently substituting a zero-valued reading.
    """
    blocks = re.split(r"\n(?=\S)", output)
    estab_blocks = [b for b in blocks if b.startswith("ESTAB")]
    if not estab_blocks:
        logger.warning("parse_ss_output: no ESTAB socket in ss -ti output: %r", output)
        return None

    def bytes_sent_of(block):
        m = re.search(r"bytes_sent:(\d+)", block)
        return int(m.group(1)) if m else -1

    block = max(estab_blocks, key=bytes_sent_of)
    if bytes_sent_of(block) <= 0:
        logger.warning("parse_ss_output: ESTAB socket has no bytes_sent: %r", block)
        return None

    rtt_match = re.search(r"rtt:(\d+\.\d+)", block)
    cwnd_match = re.search(r"cwnd:(\d+)", block)
    rate_match = re.search(r"delivery_rate\s+([\d.]+)([KMGkmg]?)bps", block)
    segs_match = re.search(r"segs_out:(\d+)", block)
    retrans_match = re.search(r"retrans:\d+/(\d+)", block)

    unit_to_mbps = {"": 1e-6, "K": 1e-3, "M": 1, "G": 1e3}

    rtt = float(rtt_match.group(1)) if rtt_match else 0
    cwnd = int(cwnd_match.group(1)) if cwnd_match else 0
    throughput = (
        float(rate_match.group(1)) * unit_to_mbps[rate_match.group(2).upper()]
        if rate_match else 0
    )

    loss_pct = 0.0
    if retrans_match and segs_match:
        segs_out = int(segs_match.group(1))
        total_retrans = int(retrans_match.group(1))
        if segs_out > 0:
            loss_pct = 100.0 * total_retrans / segs_out

    return rtt, cwnd, throughput, loss_pct


def wait_for_estab_socket(host, timeout=3.0, poll_interval=0.1):
    """Poll `ss -ti` on `host` until a data-carrying ESTAB socket shows
    up (i.e. the TCP handshake has actually completed), instead of a
    flat time.sleep() that either wastes time waiting past readiness
    or -- under load/CI jitter -- samples before the handshake is
    done. Returns True once ready, False if `timeout` elapses first
    (callers should fall back to their previous fixed-sleep behavior
    in that case, not hang indefinitely -- a bounded readiness check,
    not a replacement guarantee)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if parse_ss_output(host.cmd("ss -ti")) is not None:
            return True
        time.sleep(poll_interval)
    logger.warning(
        "wait_for_estab_socket: no ESTAB socket on %s after %.1fs, giving up",
        getattr(host, "name", host), timeout,
    )
    return False
