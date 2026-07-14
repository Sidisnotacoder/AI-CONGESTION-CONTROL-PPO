"""Documentation dataset: N trials each of 4 fixed 4-sender configurations,
saved to CSV for later write-up/graphing use. Reuses run_session()/
summarize() from rl/evaluate_competitive.py directly rather than
re-implementing the same live-session-driving logic a third time.
"""

import csv
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from rl.evaluate_competitive import run_session, summarize
from rl.regime_classifier import RegimeModel
from common.network import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

N_TRIALS = 10
MODEL_PATH = "models/ppo_competitive.zip"
REGIME_MODEL_PATH = "models/regime_classifier.joblib"
SUMMARY_CSV = "results/documentation_runs_summary.csv"
PER_STEP_CSV = "results/documentation_runs_per_step.csv"

CONFIGS = {
    "cubic_only": ["cubic", "cubic", "cubic", "cubic"],
    "ppo_only": ["ppo", "ppo", "ppo", "ppo"],
    "hybrid_only": ["hybrid", "hybrid", "hybrid", "hybrid"],
    "2ppo_2hybrid": ["ppo", "ppo", "hybrid", "hybrid"],
}


def main():
    os.makedirs("results", exist_ok=True)
    model = PPO.load(MODEL_PATH)
    regime_model = RegimeModel(REGIME_MODEL_PATH) if os.path.exists(REGIME_MODEL_PATH) else None

    with open(SUMMARY_CSV, "w", newline="") as sf, open(PER_STEP_CSV, "w", newline="") as pf:
        summary_writer = csv.writer(sf)
        summary_writer.writerow([
            "config", "trial", "sender_id", "policy",
            "avg_rtt_ms", "avg_throughput_mbps", "avg_loss_pct", "avg_reward",
        ])
        per_step_writer = csv.writer(pf)
        per_step_writer.writerow([
            "config", "trial", "sender_id", "policy", "step",
            "rtt_ms", "throughput_mbps", "loss_pct", "reward",
        ])

        for config_name, policies in CONFIGS.items():
            logger.info("=== %s: %d trials ===", config_name, N_TRIALS)
            for trial in range(N_TRIALS):
                per_sender_rows = run_session(policies, model, regime_model)
                for sender_id, sender_rows in enumerate(per_sender_rows):
                    policy = policies[sender_id]
                    s = summarize(sender_rows)
                    summary_writer.writerow([
                        config_name, trial, sender_id, policy,
                        s["rtt_ms"], s["throughput_mbps"], s["loss_pct"], s["reward"],
                    ])
                    for step, r in enumerate(sender_rows):
                        per_step_writer.writerow([
                            config_name, trial, sender_id, policy, step,
                            r["rtt_ms"], r["throughput_mbps"], r["loss_pct"], r["reward"],
                        ])
                sf.flush()
                pf.flush()
                logger.info("  trial %d done", trial)

    logger.info("Summary (one row per sender per trial) -> %s", SUMMARY_CSV)
    logger.info("Per-step (full raw data) -> %s", PER_STEP_CSV)


if __name__ == "__main__":
    main()
