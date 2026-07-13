import argparse
import logging
import os
import subprocess
import csv
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from common.network import parse_ss_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Anchored to this script's own location rather than a relative path,
# so it always resolves to <project_root>/data/raw/... regardless of
# which directory this is launched from. Written to a distinctly-named
# file (not network_metrics.csv) so a manual monitor.py run can no
# longer silently truncate scripts/collect_dataset.py's dataset if
# both happen to be run against the same project checkout.
DEFAULT_CSV_FILE = os.path.join(_PROJECT_ROOT, "data", "raw", "monitor_metrics.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live ss -ti monitor for a Mininet host's network namespace."
    )
    parser.add_argument(
        "--pid", type=int, required=True,
        help="PID of a process inside the target Mininet host's namespace "
             "(e.g. the host's shell PID from Mininet's own `ps`/`intf` output).",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Seconds between samples (default: 1.0).",
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV_FILE,
        help=f"Output CSV path (default: {DEFAULT_CSV_FILE}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # mnexec needs root to attach to another process's network namespace.
    # Run this script itself with `sudo python3 monitor.py --pid ...` --
    # if it isn't already root, sudo can't prompt for a password from
    # inside the subprocess loop below and every call silently fails,
    # which is why RTT/CWND/THR all showed 0 before.
    if os.geteuid() != 0:
        raise SystemExit(
            "monitor.py must be run as root, e.g.: sudo python3 monitor.py --pid <pid>"
        )

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "rtt_ms", "cwnd", "throughput_mbps"])

        logger.info("Collecting metrics for PID %d -> %s", args.pid, args.csv)

        while True:
            try:
                cmd = f"mnexec -a {args.pid} ss -ti"
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=5,
                )

                output = result.stdout
                if result.returncode != 0 or result.stderr:
                    logger.warning("mnexec/ss stderr: %s", result.stderr.strip())

                parsed = parse_ss_output(output)
                rtt, cwnd, throughput = (parsed[0], parsed[1], parsed[2]) if parsed else (0, 0, 0)

                timestamp = datetime.now()
                writer.writerow([timestamp, rtt, cwnd, throughput])
                f.flush()

                logger.info("RTT=%.2fms | CWND=%s | THR=%.2fMbps", rtt, cwnd, throughput)

                time.sleep(args.interval)

            except KeyboardInterrupt:
                logger.info("Stopped")
                break

            except subprocess.TimeoutExpired:
                logger.error("mnexec/ss call timed out after 5s, retrying")
                time.sleep(args.interval)

            except Exception as e:
                logger.error("Unexpected error: %s", e)
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
