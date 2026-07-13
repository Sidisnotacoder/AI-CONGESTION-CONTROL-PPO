import logging
import os
import csv
import sys
import time
from functools import partial
from datetime import datetime

from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink

if os.geteuid() != 0:
    raise SystemExit(
        "collect_dataset.py must be run as root, e.g.: sudo python3 collect_dataset.py"
    )

# Anchored to this script's own location rather than a relative path,
# so it always resolves to <project_root>/data/raw/... regardless of
# whether this is launched from the project root or from scripts/
# (a relative "../data/..." silently wrote outside the project
# entirely when run from the wrong directory).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from common.network import BottleneckTopo, parse_ss_output, wait_for_estab_socket, configure_logging, FAIL_MODE

CSV_FILE = os.path.join(_PROJECT_ROOT, "data", "raw", "network_metrics.csv")

configure_logging()
logger = logging.getLogger(__name__)

# How often to sample each active sender (seconds). Matches the
# project plan's ~100ms sampling target. Override with
# CNP_SAMPLE_INTERVAL for a quick smoke test instead of hand-editing
# this file (e.g. CNP_SAMPLE_INTERVAL=0.5 CNP_SCENARIO_DURATION=30
# sudo -E python3 scripts/collect_dataset.py).
SAMPLE_INTERVAL = float(os.environ.get("CNP_SAMPLE_INTERVAL", 0.1))
_SCENARIO_DURATION_OVERRIDE = os.environ.get("CNP_SCENARIO_DURATION")

# Traffic scenarios (Section 5 of the project plan): baseline/medium/high
# load plus a bursty pattern, all crossing the s1<->s2 bottleneck link
# (h1, h2 are senders on s1; h3 is the receiver on s2).
#
# iperf3's own "-b" target-bitrate pacing isn't reliably enforced in
# this environment (a single flow saturates the 10Mbps bottleneck
# regardless of -b), so real load differentiation comes from
# "uplink_bw_mbps": a genuine tc-level cap applied to each sender's
# own link to the switch (see set_uplink_bw()), independent of iperf3.
#
# duration=1500s per scenario (~25 min each, ~100 min total) targets
# ~85k total rows, close to the plan's ~100k-sample goal. For a quick
# smoke test instead, set CNP_SCENARIO_DURATION=30 CNP_SAMPLE_INTERVAL=0.5.
_DEFAULT_DURATION = int(_SCENARIO_DURATION_OVERRIDE) if _SCENARIO_DURATION_OVERRIDE else 1500
SCENARIOS = [
    {"name": "baseline_low", "senders": ["h1"],       "duration": _DEFAULT_DURATION, "uplink_bw_mbps": 2,   "pattern": "steady"},
    {"name": "medium_load",  "senders": ["h1", "h2"], "duration": _DEFAULT_DURATION, "uplink_bw_mbps": 6,   "pattern": "steady"},
    {"name": "high_load",    "senders": ["h1", "h2"], "duration": _DEFAULT_DURATION, "uplink_bw_mbps": 50,  "pattern": "steady"},
    {"name": "bursty",       "senders": ["h1"],       "duration": _DEFAULT_DURATION, "uplink_bw_mbps": 100, "pattern": "bursty"},
]

# iperf3's server handles one client at a time by default, so
# concurrent multi-flow scenarios (medium/high load) need one server
# port per sender -- otherwise the second client's connection just
# queues behind the first and there's no real contention on the link.
SENDER_PORTS = {"h1": 5201, "h2": 5202}


def set_uplink_bw(host, bw_mbps):
    """Cap a host's own link to its switch via tc (TCIntf.config), so
    load actually differs between scenarios regardless of whether
    iperf3's own -b pacing is honored."""
    intf = host.intfList()[0]
    intf.config(bw=bw_mbps)


def sample_hosts(writer, scenario_name, hosts):
    for h in hosts:
        parsed = parse_ss_output(h.cmd("ss -ti"))
        if parsed is None:
            # parse_ss_output already logged the raw output; also flag
            # it here with scenario/host context since a whole host
            # silently producing zero rows for a scenario has bitten
            # this pipeline before (see run_steady's server-restart
            # comment below).
            logger.warning(
                "scenario=%s host=%s: skipping sample, no parseable ESTAB socket",
                scenario_name, h.name,
            )
            continue
        rtt, cwnd, throughput, loss_pct = parsed
        writer.writerow([
            datetime.now(), scenario_name, h.name,
            rtt, cwnd, throughput, loss_pct,
        ])


def run_steady(writer, scenario, h3, sender_hosts):
    for h in sender_hosts:
        h.cmd("pkill -9 -f 'iperf3 -c'")
        set_uplink_bw(h, scenario["uplink_bw_mbps"])
    time.sleep(0.2)

    for h in sender_hosts:
        h.cmd(
            f"iperf3 -c {h3.IP()} -p {SENDER_PORTS[h.name]} "
            f"-t {scenario['duration'] + 5} "
            f"> /tmp/iperf_{h.name}_{scenario['name']}.log 2>&1 &"
        )
    # Bounded readiness poll instead of a flat sleep: returns as soon
    # as every sender's handshake is actually done, falls back to the
    # same 1s wait if any host doesn't converge within the timeout.
    if not all(wait_for_estab_socket(h, timeout=1.0) for h in sender_hosts):
        time.sleep(1)

    # Step-counted rather than wall-clock-deadline-based: if the
    # machine suspends/sleeps mid-run, time.time() jumps forward on
    # resume and a "while time.time() < end_time" check would read as
    # already-expired, silently truncating the whole scenario to
    # whatever ran before the sleep. Counting iterations is immune to
    # that -- it just resumes wherever it left off.
    num_steps = max(1, round(scenario["duration"] / SAMPLE_INTERVAL))
    for _ in range(num_steps):
        sample_hosts(writer, scenario["name"], sender_hosts)
        time.sleep(SAMPLE_INTERVAL)

    for h in sender_hosts:
        h.cmd("pkill -9 -f 'iperf3 -c'")


def run_bursty(writer, scenario, h3, sender_hosts):
    burst_on = 5
    burst_off = 3
    burst_on_steps = max(1, round(burst_on / SAMPLE_INTERVAL))
    burst_off_steps = max(1, round(burst_off / SAMPLE_INTERVAL))
    total_steps = max(1, round(scenario["duration"] / SAMPLE_INTERVAL))

    step = 0
    while step < total_steps:
        for h in sender_hosts:
            h.cmd("pkill -9 -f 'iperf3 -c'")
            set_uplink_bw(h, scenario["uplink_bw_mbps"])
            h.cmd(
                f"iperf3 -c {h3.IP()} -p {SENDER_PORTS[h.name]} "
                f"-t {burst_on} "
                f"> /tmp/iperf_{h.name}_{scenario['name']}.log 2>&1 &"
            )
        time.sleep(0.5)

        for _ in range(burst_on_steps):
            if step >= total_steps:
                break
            sample_hosts(writer, scenario["name"], sender_hosts)
            time.sleep(SAMPLE_INTERVAL)
            step += 1

        for h in sender_hosts:
            h.cmd("pkill -9 -f 'iperf3 -c'")

        for _ in range(burst_off_steps):
            if step >= total_steps:
                break
            sample_hosts(writer, scenario["name"], sender_hosts)
            time.sleep(SAMPLE_INTERVAL)
            step += 1


def main():
    topo = BottleneckTopo()
    net = Mininet(
        topo=topo,
        link=TCLink,
        switch=partial(OVSSwitch, failMode=FAIL_MODE),
        controller=None,
    )
    net.start()

    # Warm up ARP and confirm the switches are actually forwarding
    # before starting any traffic -- this is what pingall did manually
    # in earlier testing; skipping it left connections failing with
    # "unable to send control message: Bad file descriptor".
    loss = net.pingAll()
    logger.info("pingAll loss: %s%%", loss)
    if loss >= 100:
        net.stop()
        raise SystemExit("No connectivity between hosts -- aborting before collecting garbage data.")

    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    try:
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "scenario", "host",
                "rtt_ms", "cwnd", "throughput_mbps", "loss_pct",
            ])

            h3 = net.get("h3")

            for scenario in SCENARIOS:
                logger.info(
                    "=== Scenario: %s (%ds, %s) ===",
                    scenario["name"], scenario["duration"], scenario["pattern"],
                )
                sender_hosts = [net.get(name) for name in scenario["senders"]]

                # Restart the server(s) fresh for every scenario. A
                # server whose previous client was SIGKILLed can be
                # left in a state where it silently refuses the next
                # client on that port (observed: h1 produced zero rows
                # in medium_load after its clean port-5201 server from
                # baseline_low was reused without restarting).
                h3.cmd("pkill -9 iperf3")
                time.sleep(0.2)
                ports_needed = {SENDER_PORTS[name] for name in scenario["senders"]}
                for port in ports_needed:
                    h3.cmd(f"iperf3 -s -p {port} -D")
                time.sleep(0.5)

                if scenario["pattern"] == "bursty":
                    run_bursty(writer, scenario, h3, sender_hosts)
                else:
                    run_steady(writer, scenario, h3, sender_hosts)

                f.flush()
                logger.info("    done.")

            h3.cmd("pkill -9 iperf3")
    except KeyboardInterrupt:
        logger.warning("Interrupted -- stopping network.")
    finally:
        net.stop()

    logger.info("Dataset collection complete -> %s", CSV_FILE)


if __name__ == "__main__":
    main()

# Full-scale run (current settings): ~100 minutes total, ~85k rows
# across all four scenarios (per project plan Section 12's ~100k
# target). For a quick smoke test instead, set SAMPLE_INTERVAL back to
# 0.5 and each scenario's "duration" back to 30.
