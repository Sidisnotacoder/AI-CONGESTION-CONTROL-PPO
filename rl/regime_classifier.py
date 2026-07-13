"""Decision-tree-ensemble methodology augmentation for PPO congestion control.

Two models, both trained from data/raw/network_metrics.csv (the
baseline_low/medium_load/high_load/bursty labeled dataset collected by
scripts/collect_dataset.py):

- A GradientBoostingClassifier that classifies the current network
  regime from [rtt_ms, cwnd, throughput_mbps, loss_pct]. Its
  class-probability output is what env/augmented_congestion_env.py
  concatenates onto RealCongestionEnv's raw 3-value observation --
  state augmentation, not a replacement for the raw signal.

- A GradientBoostingRegressor that predicts a continuous "recommended
  direction" in [-1, 1] (decrease..increase), trained on a per-row
  heuristic label (_aimd_style_direction_label below) that encodes
  classical AIMD-style logic (increase when clean/low-latency,
  decrease under loss or high relative latency). The regressor is a
  data-driven, regime-conditional generalization of that heuristic --
  described honestly as that, not as a novel published algorithm.
  rl/policy_utils.py's predict_hybrid() blends this with PPO's own
  action distribution rather than using either alone.

An ensemble (boosted trees), not a single decision tree, is the
deliberate "more complex decision tree" -- trading extra compute for
more robustness to noisy live ss -ti readings than one tree gives.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["rtt_ms", "cwnd", "throughput_mbps", "loss_pct"]
REGIME_CLASSES = ["baseline_low", "medium_load", "high_load", "bursty"]


def _aimd_style_direction_label(df):
    """Weak-supervision label for the direction regressor: classical
    AIMD logic (no loss -> probe upward; any loss -> back off), with
    the "high latency" cutoff computed per-scenario (75th percentile
    of that scenario's own RTT) rather than one fixed global threshold,
    since baseline/medium/high/bursty have very different normal RTT
    ranges (see README.md's per-scenario avg RTT table)."""
    high_rtt_cutoff = df.groupby("scenario")["rtt_ms"].transform(lambda s: s.quantile(0.75))
    direction = np.zeros(len(df))
    direction[(df["loss_pct"] > 0) | (df["rtt_ms"] > high_rtt_cutoff)] = -1.0
    direction[(df["loss_pct"] == 0) & (df["rtt_ms"] <= df.groupby("scenario")["rtt_ms"].transform("median"))] = 1.0
    return direction


def train(csv_path):
    """Fit both models on csv_path (data/raw/network_metrics.csv).
    Returns (classifier, regressor, held_out_accuracy)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=FEATURE_COLUMNS + ["scenario"])

    X = df[FEATURE_COLUMNS].values
    y_class = df["scenario"].values
    y_direction = _aimd_style_direction_label(df)

    X_train, X_test, yc_train, yc_test, yd_train, _ = train_test_split(
        X, y_class, y_direction, test_size=0.2, random_state=0, stratify=y_class,
    )

    classifier = GradientBoostingClassifier(random_state=0)
    classifier.fit(X_train, yc_train)
    held_out_accuracy = classifier.score(X_test, yc_test)

    # Regressor trains on [raw features + regime class probabilities]
    # -- the same feature layout predict_hybrid() builds at inference
    # time -- so the regime signal is available to both models, not
    # just the classifier.
    regime_probs_train = classifier.predict_proba(X_train)
    regressor_features = np.concatenate([X_train, regime_probs_train], axis=1)
    regressor = GradientBoostingRegressor(random_state=0)
    regressor.fit(regressor_features, yd_train)

    return classifier, regressor, held_out_accuracy


def save(classifier, regressor, path):
    joblib.dump({"classifier": classifier, "regressor": regressor, "classes": list(classifier.classes_)}, path)


def load(path):
    return joblib.load(path)


class RegimeModel:
    """Inference-time wrapper used by AugmentedCongestionEnv and
    predict_hybrid(): loads the joblib bundle once, exposes
    regime_probs(obs) and direction(obs) against raw [rtt, cwnd,
    throughput, loss_pct]-shaped input."""

    def __init__(self, path):
        bundle = load(path)
        self.classifier = bundle["classifier"]
        self.regressor = bundle["regressor"]
        self.classes = bundle["classes"]

    def regime_probs(self, features):
        """features: array-like of shape (4,) = [rtt, cwnd, throughput, loss_pct].
        Returns an array of shape (4,) of class probabilities, ordered
        to match self.classes (== the classifier's own .classes_ order,
        not necessarily REGIME_CLASSES' order)."""
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return self.classifier.predict_proba(features)[0]

    def direction(self, features):
        """Returns a scalar in [-1, 1] (decrease..increase). The
        underlying GradientBoostingRegressor is trained on labels in
        {-1, 0, 1} but is a regressor, not a classifier -- nothing
        stops it from predicting outside that range (e.g. 1.03), which
        would break policy_utils.predict_hybrid()'s probability
        construction (1 - abs(direction) going negative). Clip here,
        once, rather than relying on every caller to remember to."""
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        regime_probs = self.classifier.predict_proba(features)
        regressor_features = np.concatenate([features, regime_probs], axis=1)
        raw = float(self.regressor.predict(regressor_features)[0])
        return max(-1.0, min(1.0, raw))
