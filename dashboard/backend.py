"""Live-mode SSE server for the dashboard.

Must be launched with `sudo python3 dashboard/backend.py` from an
interactive terminal -- RealCongestionEnv needs root for Mininet, and
this environment has no NOPASSWD sudo (confirmed during project
setup), so it cannot elevate itself or be started headlessly. Binds to
127.0.0.1 only; the frontend (dashboard/static/dashboard.html) is
served from this same process, so there's no CORS to configure.

Only one live episode may run at a time (a second Mininet network
can't safely coexist with the first) -- a `/stream` request made while
one is already in flight gets HTTP 409, not a silently queued second
network.
"""

import json
import logging
import os
import sys
import threading

from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.real_congestion_env import RealCongestionEnv
from rl.policy_utils import predict_with_distribution, predict_hybrid
from rl.regime_classifier import RegimeModel
from common.network import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
REGIME_MODEL_PATH = "models/regime_classifier.joblib"

app = Flask(__name__, static_folder=None)
_episode_lock = threading.Lock()

_model = None
_regime_model = None


def _get_model():
    global _model
    if _model is None:
        _model = PPO.load("models/ppo_real_congestion")
    return _model


def _get_regime_model():
    global _regime_model
    if _regime_model is None and os.path.exists(REGIME_MODEL_PATH):
        _regime_model = RegimeModel(REGIME_MODEL_PATH)
    return _regime_model


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "dashboard.html")


@app.route("/replay-data.json")
def replay_data():
    return send_from_directory(STATIC_DIR, "replay_data.json")


def _row_for_step(step, obs, reward, info, action, extra):
    rtt, cwnd, throughput = obs
    components = info.get("reward_components", {})
    return {
        "step": step, "rtt_ms": float(rtt), "cwnd": float(cwnd), "throughput_mbps": float(throughput),
        "reward": float(reward), "rate_mbps": info.get("rate_mbps"),
        "action": action,
        "prob_decrease": extra.get("prob_decrease"), "prob_maintain": extra.get("prob_maintain"),
        "prob_increase": extra.get("prob_increase"), "value_estimate": extra.get("value_estimate"),
        "reward_throughput_term": components.get("throughput"),
        "reward_loss_term": components.get("loss_penalty"),
        "reward_rtt_term": components.get("rtt_penalty"),
        "regime_predicted_class": extra.get("regime_predicted_class"),
        "regime_confidence": extra.get("regime_confidence"),
        "tree_direction": extra.get("tree_direction"),
    }


def _regime_extra(regime_model, obs, loss_pct):
    if regime_model is None:
        return {}
    import numpy as np
    features = np.concatenate([np.asarray(obs, dtype=np.float32), [loss_pct]])
    probs = regime_model.regime_probs(features)
    top = int(np.argmax(probs))
    return {"regime_predicted_class": regime_model.classes[top], "regime_confidence": float(probs[top])}


def _run_live_episode(run_name):
    """Generator yielding SSE-formatted strings for one live episode."""
    model = _get_model()
    regime_model = _get_regime_model()

    if run_name == "cubic_baseline":
        env = RealCongestionEnv(start_rate_mbps=100)
    else:
        env = RealCongestionEnv()

    try:
        obs, info = env.reset()
        step = 0
        terminated = truncated = False
        while not (terminated or truncated):
            loss_pct = info.get("loss_pct", 0.0)

            if run_name == "ppo_agent":
                action, probs, value = predict_with_distribution(model, obs)
                extra = {"prob_decrease": probs[0], "prob_maintain": probs[1], "prob_increase": probs[2], "value_estimate": value}
                extra.update(_regime_extra(regime_model, obs, loss_pct))
            elif run_name == "ppo_hybrid_tree":
                if regime_model is None:
                    raise RuntimeError("regime classifier not trained -- run scripts/train_regime_classifier.py")
                action, blended_probs, _, tree_direction = predict_hybrid(model, regime_model, obs, loss_pct)
                extra = {"prob_decrease": blended_probs[0], "prob_maintain": blended_probs[1], "prob_increase": blended_probs[2], "tree_direction": tree_direction}
                extra.update(_regime_extra(regime_model, obs, loss_pct))
            elif run_name == "cubic_baseline":
                action, extra = 1, {}
            else:
                raise ValueError(f"unknown run: {run_name}")

            obs, reward, terminated, truncated, info = env.step(action)
            row = _row_for_step(step, obs, reward, info, action, extra)
            yield f"data: {json.dumps(row)}\n\n"
            step += 1

        yield "event: done\ndata: {}\n\n"
    except Exception as e:
        logger.exception("live episode failed")
        yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    finally:
        env.close()


@app.route("/stream")
def stream():
    run_name = request.args.get("run", "ppo_agent")
    if run_name not in ("ppo_agent", "ppo_hybrid_tree", "cubic_baseline"):
        return jsonify({"error": f"unknown run {run_name}"}), 400

    if not _episode_lock.acquire(blocking=False):
        return jsonify({"error": "a live episode is already running -- only one Mininet network can exist at a time"}), 409

    def generate():
        try:
            yield from _run_live_episode(run_name)
        finally:
            _episode_lock.release()

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    if os.geteuid() != 0:
        raise SystemExit("dashboard/backend.py must run as root: sudo python3 dashboard/backend.py")
    app.run(host="127.0.0.1", port=8765, threaded=True, debug=False, use_reloader=False)
