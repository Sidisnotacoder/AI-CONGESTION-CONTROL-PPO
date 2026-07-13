import itertools
import json
import logging
import os
import signal
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats as scipy_stats
from stable_baselines3 import PPO
from env.real_congestion_env import RealCongestionEnv
from rl.policy_utils import predict_hybrid
from rl.regime_classifier import RegimeModel
from common.network import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# See rl/train_real.py for why.
signal.signal(signal.SIGTERM, signal.default_int_handler)

N_TRIALS = 5
RESULTS_CSV = "results/multi_trial_eval.csv"
PER_STEP_CSV = "results/multi_trial_per_step.csv"
STATS_JSON = "results/multi_trial_stats.json"
REGIME_MODEL_PATH = "models/regime_classifier.joblib"


def run_episode(env, policy_fn):
    obs, info = env.reset()
    rtts, thrs, rewards = [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        loss_pct = info.get("loss_pct", 0.0)
        action = policy_fn(obs, loss_pct)
        obs, reward, terminated, truncated, info = env.step(action)
        rtt, cwnd, throughput = obs
        rtts.append(rtt)
        thrs.append(throughput)
        rewards.append(reward)
    return rtts, thrs, rewards


def mean(xs):
    # float(...) here, not just at the two call sites that write to
    # JSON: xs elements trace back to RealCongestionEnv's np.float32
    # observations (rtt, cwnd, throughput = obs), so sum(xs)/len(xs)
    # comes out numpy.float32 too -- fine for CSV (str()'d either
    # way) but json.dump() raises TypeError on a bare numpy.float32,
    # which is exactly what broke the stats_out write below the first
    # time this ran end-to-end.
    return float(sum(xs) / len(xs))


def ci95(xs):
    # Exact small-sample t-interval (scipy.stats.t), not the earlier
    # normal-approximation workaround -- matters specifically at
    # N_TRIALS=5, where the normal approximation understates the
    # interval width relative to the true t-distribution.
    m = mean(xs)
    n = len(xs)
    if n <= 1:
        return m, m, m
    sem = scipy_stats.sem(xs)
    lo, hi = scipy_stats.t.interval(0.95, df=n - 1, loc=m, scale=sem)
    return m, float(lo), float(hi)


def welch_t_test(a, b):
    result = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def run_trials(run_name, env_factory, policy_fn, per_step_writer):
    trial_rtts, trial_thrs = [], []
    env = env_factory()
    try:
        for trial in range(N_TRIALS):
            rtts, thrs, rewards = run_episode(env, policy_fn)
            for step, (rtt, thr, reward) in enumerate(zip(rtts, thrs, rewards)):
                per_step_writer.writerow([run_name, trial, step, rtt, thr, reward])
            trial_rtts.append(mean(rtts))
            trial_thrs.append(mean(thrs))
            logger.info(
                "%s trial %d: avg_rtt=%.1fms avg_thr=%.2fMbps",
                run_name, trial, mean(rtts), mean(thrs),
            )
    finally:
        env.close()
    return trial_rtts, trial_thrs


def main():
    os.makedirs("results", exist_ok=True)
    model = PPO.load("models/ppo_real_congestion")
    agent_policy = lambda obs, loss_pct: int(model.predict(obs, deterministic=True)[0])
    cubic_policy = lambda obs, loss_pct: 1

    regime_model = None
    if os.path.exists(REGIME_MODEL_PATH):
        regime_model = RegimeModel(REGIME_MODEL_PATH)
    else:
        logger.warning(
            "%s not found -- skipping ppo_hybrid_tree arm. "
            "Run scripts/train_regime_classifier.py first.",
            REGIME_MODEL_PATH,
        )

    def hybrid_policy(obs, loss_pct):
        action, _, _, _ = predict_hybrid(model, regime_model, obs, loss_pct)
        return action

    runs = [("ppo_agent", RealCongestionEnv, agent_policy)]
    if regime_model is not None:
        runs.append(("ppo_hybrid_tree", RealCongestionEnv, hybrid_policy))
    runs.append(("cubic_baseline", lambda: RealCongestionEnv(start_rate_mbps=100), cubic_policy))

    trial_results = {}  # run_name -> (rtts, thrs)

    with open(RESULTS_CSV, "w", newline="") as summary_f, \
         open(PER_STEP_CSV, "w", newline="") as per_step_f:
        summary_writer = csv.writer(summary_f)
        summary_writer.writerow(["run", "trial", "avg_rtt_ms", "avg_throughput_mbps", "avg_reward"])
        per_step_writer = csv.writer(per_step_f)
        per_step_writer.writerow(["run", "trial", "step", "rtt_ms", "throughput_mbps", "reward"])

        for run_name, env_factory, policy_fn in runs:
            logger.info("=== %s: %d trials ===", run_name, N_TRIALS)
            rtts, thrs = run_trials(run_name, env_factory, policy_fn, per_step_writer)
            trial_results[run_name] = (rtts, thrs)
            for trial, (r, t) in enumerate(zip(rtts, thrs)):
                summary_writer.writerow([run_name, trial, r, t, ""])

    logger.info("=== Statistical comparison (n=%d trials each, exact Welch's t-test) ===", N_TRIALS)
    stats_out = {}
    run_names = list(trial_results.keys())
    for metric_name, idx, unit in [("rtt_ms", 0, "ms"), ("throughput_mbps", 1, "Mbps")]:
        stats_out[metric_name] = {}
        for run_name in run_names:
            m, lo, hi = ci95(trial_results[run_name][idx])
            stats_out[metric_name][run_name] = {"mean": m, "ci95_low": lo, "ci95_high": hi}
            logger.info("%s: %s=%.1f%s (95%% CI [%.1f, %.1f])", metric_name, run_name, m, unit, lo, hi)

        stats_out[metric_name]["pairwise_welch_t_test"] = {}
        for a, b in itertools.combinations(run_names, 2):
            t, p = welch_t_test(trial_results[a][idx], trial_results[b][idx])
            key = f"{a}_vs_{b}"
            stats_out[metric_name]["pairwise_welch_t_test"][key] = {"t": t, "p": p}
            logger.info("%s: %s  t=%.2f p=%.4f", metric_name, key, t, p)

    with open(STATS_JSON, "w") as f:
        json.dump(stats_out, f, indent=2)

    logger.info("Per-trial results -> %s", RESULTS_CSV)
    logger.info("Per-step results -> %s", PER_STEP_CSV)
    logger.info("Statistics -> %s", STATS_JSON)


if __name__ == "__main__":
    main()
