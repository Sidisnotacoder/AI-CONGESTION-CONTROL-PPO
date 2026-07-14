"""Old (solo-trained) vs new (competitive-trained) PPO, evaluated in the
exact two scenarios this session's live dashboard testing already
characterized manually: mixed competition (where the old model lost badly)
and homogeneous groups (where the old model was already ahead). Answers
"did retraining actually help" with real numbers, not assumption -- same
multi-trial-averaging methodology used ad hoc earlier this session,
scripted here so it's repeatable.
"""

import json
import logging
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.multi_flow_congestion_env import MultiFlowCongestionEnv
from rl.policy_utils import resolve_action
from rl.regime_classifier import RegimeModel
from common.network import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

N_TRIALS = 3
MAX_STEPS = 40
RESULTS_CSV = "results/competitive_eval.csv"
STATS_JSON = "results/competitive_eval_stats.json"
REGIME_MODEL_PATH = "models/regime_classifier.joblib"

MODELS = {
    "old_solo_trained": "models/ppo_real_congestion.zip",
    "new_competitive_trained": "models/ppo_competitive.zip",
}


def run_session(policies, model, regime_model, bw=10, delay="20ms", max_steps=MAX_STEPS):
    """One live MultiFlowCongestionEnv session. Every non-cubic sender in
    `policies` is driven by the same `model` via resolve_action -- this is
    how we ask "how does *this* PPO checkpoint do here," not a multi-model
    comparison within one session. Returns a list (one per sender) of
    per-step dicts."""
    n_senders = len(policies)
    env = MultiFlowCongestionEnv(
        n_senders=n_senders, bw=bw, delay=delay, max_steps=max_steps,
        start_rates_mbps=[100.0 if p == "cubic" else MultiFlowCongestionEnv.START_RATE_MBPS for p in policies],
    )
    try:
        results = env.start()
        last_obs = [r[0] for r in results]
        last_loss = [r[1].get("loss_pct", 0.0) for r in results]
        per_sender_rows = [[] for _ in range(n_senders)]

        terminated = truncated = False
        while not (terminated or truncated):
            actions = [resolve_action(policies[i], model, regime_model, last_obs[i], last_loss[i])[0]
                       for i in range(n_senders)]
            step_results = env.step(actions)
            for i, (obs, reward, terminated, truncated, info) in enumerate(step_results):
                last_obs[i] = obs
                last_loss[i] = info.get("loss_pct", 0.0)
                per_sender_rows[i].append({
                    "rtt_ms": float(obs[0]), "throughput_mbps": float(obs[2]),
                    "loss_pct": info.get("loss_pct", 0.0), "reward": float(reward),
                })
        return per_sender_rows
    finally:
        env.close()


def summarize(rows):
    n = len(rows)
    return {
        "rtt_ms": sum(r["rtt_ms"] for r in rows) / n,
        "throughput_mbps": sum(r["throughput_mbps"] for r in rows) / n,
        "loss_pct": sum(r["loss_pct"] for r in rows) / n,
        "reward": sum(r["reward"] for r in rows) / n,
    }


def average(summaries):
    n = len(summaries)
    return {k: sum(s[k] for s in summaries) / n for k in summaries[0]}


def main():
    os.makedirs("results", exist_ok=True)
    regime_model = RegimeModel(REGIME_MODEL_PATH) if os.path.exists(REGIME_MODEL_PATH) else None

    csv_rows = []  # (group, scenario, policy_label, trial, rtt, throughput, loss, reward)
    all_summaries = {}

    for group_name, model_path in MODELS.items():
        model = PPO.load(model_path)

        # Scenario 1: mixed -- 3x this model (labeled "ppo") + 1x cubic,
        # the exact composition that broke the old model.
        logger.info("=== %s: mixed (3xppo + 1xcubic), %d trials ===", group_name, N_TRIALS)
        mixed_ppo_summaries, mixed_cubic_summaries = [], []
        for trial in range(N_TRIALS):
            rows = run_session(["ppo", "ppo", "ppo", "cubic"], model, regime_model)
            for i, policy in enumerate(["ppo", "ppo", "ppo", "cubic"]):
                s = summarize(rows[i])
                (mixed_ppo_summaries if policy == "ppo" else mixed_cubic_summaries).append(s)
                csv_rows.append([group_name, "mixed", policy, trial, s["rtt_ms"], s["throughput_mbps"], s["loss_pct"], s["reward"]])
            logger.info("  trial %d done", trial)

        # Scenario 2: homogeneous -- 4x this model, never mixed with cubic.
        logger.info("=== %s: homogeneous (4xppo), %d trials ===", group_name, N_TRIALS)
        homog_summaries = []
        for trial in range(N_TRIALS):
            rows = run_session(["ppo", "ppo", "ppo", "ppo"], model, regime_model)
            for i in range(4):
                s = summarize(rows[i])
                homog_summaries.append(s)
                csv_rows.append([group_name, "homogeneous", "ppo", trial, s["rtt_ms"], s["throughput_mbps"], s["loss_pct"], s["reward"]])
            logger.info("  trial %d done", trial)

        all_summaries[group_name] = {
            "mixed_ppo": average(mixed_ppo_summaries),
            "mixed_cubic": average(mixed_cubic_summaries),
            "homogeneous": average(homog_summaries),
        }

    # Cubic homogeneous baseline -- shared reference point, only needs
    # running once (doesn't depend on either PPO checkpoint).
    logger.info("=== cubic_baseline: homogeneous (4xcubic), %d trials ===", N_TRIALS)
    cubic_homog_summaries = []
    for trial in range(N_TRIALS):
        rows = run_session(["cubic", "cubic", "cubic", "cubic"], None, None)
        for i in range(4):
            s = summarize(rows[i])
            cubic_homog_summaries.append(s)
            csv_rows.append(["cubic_baseline", "homogeneous", "cubic", trial, s["rtt_ms"], s["throughput_mbps"], s["loss_pct"], s["reward"]])
        logger.info("  trial %d done", trial)
    all_summaries["cubic_baseline"] = {"homogeneous": average(cubic_homog_summaries)}

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "scenario", "policy_label", "trial", "avg_rtt_ms", "avg_throughput_mbps", "avg_loss_pct", "avg_reward"])
        writer.writerows(csv_rows)

    with open(STATS_JSON, "w") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info("=== Summary (avg over %d trials each) ===", N_TRIALS)
    for group, scenarios in all_summaries.items():
        for scenario, stats in scenarios.items():
            logger.info(
                "%-24s %-12s rtt=%7.1fms  thr=%6.2fMbps  loss=%5.2f%%  reward=%7.2f",
                group, scenario, stats["rtt_ms"], stats["throughput_mbps"], stats["loss_pct"], stats["reward"],
            )

    logger.info("Per-trial results -> %s", RESULTS_CSV)
    logger.info("Averaged stats -> %s", STATS_JSON)


if __name__ == "__main__":
    main()
