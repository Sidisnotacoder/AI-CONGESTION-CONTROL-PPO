"""results/*.csv -> dashboard/static/replay_data.json

Reads the per-step CSV produced by rl/evaluate_real.py (ppo_agent,
ppo_hybrid_tree if trained, cubic_baseline -- see that file's
CSV_HEADER) and the stats JSON produced by rl/evaluate_multi_trial.py,
and combines them into one JSON blob the dashboard frontend can either
fetch live (dashboard.html) or have inlined at build time
(build_artifact.py -> replay_artifact.html). Safe to run with only
ppo_vs_cubic_eval.csv present -- the multi-trial stats section is
simply omitted if that file doesn't exist yet.
"""

import csv
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_CSV = os.path.join(_PROJECT_ROOT, "results", "ppo_vs_cubic_eval.csv")
STATS_JSON = os.path.join(_PROJECT_ROOT, "results", "multi_trial_stats.json")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "replay_data.json")

NUMERIC_FIELDS = {
    "step", "rtt_ms", "cwnd", "throughput_mbps", "reward", "rate_mbps", "action",
    "prob_decrease", "prob_maintain", "prob_increase", "value_estimate",
    "reward_throughput_term", "reward_loss_term", "reward_rtt_term",
    "regime_confidence", "tree_direction",
}


def _coerce(key, value):
    if value == "":
        return None
    if key in NUMERIC_FIELDS:
        try:
            return float(value)
        except ValueError:
            return None
    return value


def load_eval_csv(csv_path):
    runs = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            run = row["run"]
            runs.setdefault(run, []).append(
                {k: _coerce(k, v) for k, v in row.items() if k != "run"}
            )
    return runs


def main():
    if not os.path.exists(EVAL_CSV):
        raise SystemExit(
            f"{EVAL_CSV} not found. Run: sudo python3 rl/evaluate_real.py first."
        )

    data = {"runs": load_eval_csv(EVAL_CSV)}

    if os.path.exists(STATS_JSON):
        with open(STATS_JSON) as f:
            data["multi_trial_stats"] = json.load(f)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(data, f)

    print(f"Wrote {OUT_JSON} ({sum(len(v) for v in data['runs'].values())} rows across {len(data['runs'])} runs)")


if __name__ == "__main__":
    main()
