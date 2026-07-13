import logging
import os
import signal
import sys
import csv

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.real_congestion_env import RealCongestionEnv
from rl.policy_utils import predict_with_distribution, predict_hybrid
from rl.regime_classifier import RegimeModel
from common.network import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# See rl/train_real.py for why: turns SIGTERM into the same
# KeyboardInterrupt path the try/finally blocks below already handle,
# so env.close() still runs instead of leaking the Mininet network.
signal.signal(signal.SIGTERM, signal.default_int_handler)

RESULTS_CSV = "results/ppo_vs_cubic_eval.csv"
REGIME_MODEL_PATH = "models/regime_classifier.joblib"

CSV_HEADER = [
    "run", "step", "rtt_ms", "cwnd", "throughput_mbps", "reward", "rate_mbps",
    "action", "prob_decrease", "prob_maintain", "prob_increase", "value_estimate",
    "reward_throughput_term", "reward_loss_term", "reward_rtt_term",
    "regime_predicted_class", "regime_confidence", "tree_direction",
]


def run_episode(env, writer, run_name, policy_fn):
    obs, info = env.reset()
    rows = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        loss_pct = info.get("loss_pct", 0.0)
        action, extra = policy_fn(obs, loss_pct)
        obs, reward, terminated, truncated, info = env.step(action)
        rtt, cwnd, throughput = obs
        components = info.get("reward_components", {})
        row = [
            run_name, step, rtt, cwnd, throughput, reward, info["rate_mbps"],
            action,
            extra.get("prob_decrease", ""), extra.get("prob_maintain", ""), extra.get("prob_increase", ""),
            extra.get("value_estimate", ""),
            components.get("throughput", ""), components.get("loss_penalty", ""), components.get("rtt_penalty", ""),
            extra.get("regime_predicted_class", ""), extra.get("regime_confidence", ""), extra.get("tree_direction", ""),
        ]
        writer.writerow(row)
        rows.append(row)
        step += 1
    return rows


def summarize(name, rows):
    n = len(rows)
    avg_rtt = sum(r[2] for r in rows) / n
    avg_thr = sum(r[4] for r in rows) / n
    avg_reward = sum(r[5] for r in rows) / n
    logger.info(
        "%-12s n=%-4d avg_rtt=%8.1fms avg_throughput=%6.2fMbps avg_reward=%8.2f",
        name, n, avg_rtt, avg_thr, avg_reward,
    )


def _regime_info(regime_model, obs, loss_pct):
    if regime_model is None:
        return {}
    full_features = np.concatenate([np.asarray(obs, dtype=np.float32), [loss_pct]])
    regime_probs = regime_model.regime_probs(full_features)
    top_idx = int(np.argmax(regime_probs))
    return {
        "regime_predicted_class": regime_model.classes[top_idx],
        "regime_confidence": float(regime_probs[top_idx]),
    }


def main():
    os.makedirs("results", exist_ok=True)
    model = PPO.load("models/ppo_real_congestion")

    regime_model = None
    if os.path.exists(REGIME_MODEL_PATH):
        regime_model = RegimeModel(REGIME_MODEL_PATH)
    else:
        logger.warning(
            "%s not found -- skipping ppo_hybrid_tree arm. "
            "Run scripts/train_regime_classifier.py first (after collecting "
            "data/raw/network_metrics.csv) to enable it.",
            REGIME_MODEL_PATH,
        )

    def agent_policy(obs, loss_pct):
        action, probs, value = predict_with_distribution(model, obs)
        extra = {
            "prob_decrease": probs[0], "prob_maintain": probs[1], "prob_increase": probs[2],
            "value_estimate": value,
        }
        extra.update(_regime_info(regime_model, obs, loss_pct))
        return action, extra

    def cubic_policy(obs, loss_pct):
        # TCP Cubic uses no learned policy -- there is nothing to
        # report here, unlike agent_policy's real probabilities/value.
        return 1, {}

    def hybrid_policy(obs, loss_pct):
        action, blended_probs, _, tree_direction = predict_hybrid(model, regime_model, obs, loss_pct)
        extra = {
            "prob_decrease": blended_probs[0], "prob_maintain": blended_probs[1], "prob_increase": blended_probs[2],
            "tree_direction": tree_direction,
        }
        extra.update(_regime_info(regime_model, obs, loss_pct))
        return action, extra

    runs = [("ppo_agent", lambda: RealCongestionEnv(), agent_policy)]
    if regime_model is not None:
        runs.append(("ppo_hybrid_tree", lambda: RealCongestionEnv(), hybrid_policy))
    # Fresh network for the baseline run -- start rate high (100Mbps,
    # well above the 10Mbps bottleneck) and always "maintain" (action
    # 1), so no external rate shaping ever kicks in: only the real
    # bottleneck link and TCP Cubic's own congestion control govern
    # the flow, same as any plain Cubic connection.
    runs.append(("cubic_baseline", lambda: RealCongestionEnv(start_rate_mbps=100), cubic_policy))

    all_rows = {}
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for run_name, env_factory, policy_fn in runs:
            logger.info("=== Running %s ===", run_name)
            run_env = env_factory()
            try:
                all_rows[run_name] = run_episode(run_env, writer, run_name, policy_fn)
                f.flush()
            finally:
                run_env.close()

    logger.info("=== Summary ===")
    for run_name, rows in all_rows.items():
        summarize(run_name, rows)
    logger.info("Full per-step results -> %s", RESULTS_CSV)


if __name__ == "__main__":
    main()
