"""Live-mode SSE + control server for the dashboard.

Must be launched with `sudo python3 dashboard/backend.py` from an
interactive terminal -- MultiFlowCongestionEnv needs root for Mininet, and
this environment has no NOPASSWD sudo (confirmed during project setup), so
it cannot elevate itself or be started headlessly. Binds to 127.0.0.1 only;
the frontend (dashboard/static/dashboard.html) is served from this same
process, so there's no CORS to configure.

Only one live session may run at a time (a second Mininet network can't
safely coexist with the first) -- a `/live/start` request made while one is
already in flight gets HTTP 409, not a silently queued second network.

Replaces the earlier single-arm `/stream?run=...` endpoint: that endpoint's
three fixed policies are now just the n_senders=1 special case of the
general session below (e.g. policies=["ppo"]), so there is exactly one live
code path instead of two that could drift apart.
"""

import json
import logging
import os
import sys
import threading

from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from rl.regime_classifier import RegimeModel
from common.network import configure_logging, MAX_SENDERS
from dashboard.live_session import LiveSession, VALID_POLICIES

configure_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
REGIME_MODEL_PATH = "models/regime_classifier.joblib"
# ppo_competitive.zip (trained with real competing traffic present, see
# docs/COMPETITIVE_RETRAINING_STATUS.md) supersedes ppo_real_congestion.zip
# (solo-trained only) as the live "ppo"/"hybrid" model -- same fixed-path
# convention rl/model_registry.py already uses elsewhere: the fixed path is
# always "whichever is current," with timestamped copies preserved for
# history, not swapped into the live path.
MODEL_PATH = "models/ppo_competitive"

app = Flask(__name__, static_folder=None)
_episode_lock = threading.Lock()

_model = None
_regime_model = None
_session = None  # the one active LiveSession, or None between/before sessions


def _get_model():
    global _model
    if _model is None:
        _model = PPO.load(MODEL_PATH)
    return _model


def _get_regime_model():
    global _regime_model
    if _regime_model is None and os.path.exists(REGIME_MODEL_PATH):
        _regime_model = RegimeModel(REGIME_MODEL_PATH)
    return _regime_model


def _on_session_finished():
    """Called from the session's own background thread once its episode
    is truly over (natural end, error, or explicit stop) -- the single
    place that clears _session and frees the one-network-at-a-time lock,
    so callers of /live/stop don't also need to release it themselves."""
    global _session
    _session = None
    if _episode_lock.locked():
        _episode_lock.release()


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "dashboard.html")


@app.route("/replay-data.json")
def replay_data():
    return send_from_directory(STATIC_DIR, "replay_data.json")


@app.route("/live/start", methods=["POST"])
def live_start():
    global _session
    body = request.get_json(force=True, silent=True) or {}

    n_senders = body.get("n_senders")
    policies = body.get("policies")
    bw = body.get("bw", 10)
    delay = body.get("delay", "20ms")
    queue_size = body.get("queue_size")
    max_steps = body.get("max_steps", 40)
    queue_disc = body.get("queue_disc", "fifo")

    if not isinstance(n_senders, int) or not (1 <= n_senders <= MAX_SENDERS):
        return jsonify({"error": f"n_senders must be an int between 1 and {MAX_SENDERS}"}), 400
    if not isinstance(policies, list) or len(policies) != n_senders:
        return jsonify({"error": "policies must be a list with exactly n_senders entries"}), 400
    if any(p not in VALID_POLICIES for p in policies):
        return jsonify({"error": f"each policy must be one of {VALID_POLICIES}"}), 400
    if queue_disc not in ("fifo", "red"):
        return jsonify({"error": "queue_disc must be 'fifo' or 'red'"}), 400

    if not _episode_lock.acquire(blocking=False):
        return jsonify({"error": "a live session is already running -- only one Mininet network can exist at a time"}), 409

    try:
        session = LiveSession(
            n_senders=n_senders, policies=policies, bw=bw, delay=delay,
            queue_size=queue_size, max_steps=max_steps, queue_disc=queue_disc,
            model=_get_model(), regime_model=_get_regime_model(),
            on_finished=_on_session_finished,
        )
    except Exception as e:
        _episode_lock.release()
        logger.exception("failed to start live session")
        return jsonify({"error": str(e)}), 400

    _session = session
    _session.start()
    return jsonify({
        "status": "started",
        "senders": [{"id": i, "policy": p} for i, p in enumerate(policies)],
    })


@app.route("/live/stream")
def live_stream():
    session = _session
    if session is None:
        return jsonify({"error": "no live session running -- call POST /live/start first"}), 404

    def generate():
        while True:
            row = session.row_queue.get()
            event = row.pop("__event__", None)
            if event == "done":
                yield "event: done\ndata: {}\n\n"
                break
            elif event == "error":
                yield f"event: error\ndata: {json.dumps({'message': row.get('message')})}\n\n"
                break
            else:
                yield f"data: {json.dumps(row)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/live/control", methods=["POST"])
def live_control():
    session = _session
    if session is None:
        return jsonify({"error": "no live session running"}), 404

    body = request.get_json(force=True, silent=True) or {}
    ctrl_type = body.get("type")

    try:
        if ctrl_type == "bandwidth":
            session.set_bandwidth(float(body["value"]))
        elif ctrl_type == "delay":
            session.set_delay(str(body["value"]))
        elif ctrl_type == "queue_disc":
            value = body.get("value")
            if value not in ("fifo", "red"):
                return jsonify({"error": "queue_disc value must be 'fifo' or 'red'"}), 400
            session.set_queue_disc(value)
        elif ctrl_type == "burst":
            session.inject_burst(float(body.get("duration", 4)), float(body.get("rate_mbps", 8)))
        elif ctrl_type == "human_action":
            session.set_human_action(int(body["sender_id"]), int(body["action"]))
        else:
            return jsonify({"error": f"unknown control type {ctrl_type!r}"}), 400
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"status": "ok"})


@app.route("/live/stop", methods=["POST"])
def live_stop():
    session = _session
    if session is None:
        return jsonify({"error": "no live session running"}), 404
    session.stop()
    session.join(timeout=10)
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    if os.geteuid() != 0:
        raise SystemExit("dashboard/backend.py must run as root: sudo python3 dashboard/backend.py")
    app.run(host="127.0.0.1", port=8765, threaded=True, debug=False, use_reloader=False)
