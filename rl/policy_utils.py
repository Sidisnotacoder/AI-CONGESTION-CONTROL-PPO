"""Introspection helpers for SB3 PPO policies.

evaluate_real.py and evaluate_multi_trial.py previously only ever
called model.predict(obs, deterministic=True)[0] -- the chosen action
id, nothing else. SB3's ActorCriticPolicy also exposes the categorical
action-probability distribution and the critic's value estimate for
the same observation; neither was ever read anywhere in this codebase.
Both are needed for the dashboard's explainability panel (action
confidence, value-vs-reward tracking) and for the hybrid PPO+decision
tree blending in rl/regime_classifier.py.
"""

import numpy as np
import torch


def predict_with_distribution(model, obs):
    """Return (action, probs, value_estimate) for a single observation.

    action: int, the same argmax action model.predict(deterministic=True)
        would return.
    probs: list[float] of length n_actions -- the policy's action
        probabilities (index 0/1/2 == decrease/maintain/increase for
        this project's Discrete(3) action space).
    value_estimate: float, the critic's value estimate for this
        observation.
    """
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(obs))
        distribution = model.policy.get_distribution(obs_tensor)
        probs = distribution.distribution.probs.cpu().numpy()[0]
        value = model.policy.predict_values(obs_tensor).cpu().numpy()[0][0]

    action = int(np.argmax(probs))
    # .tolist() converts numpy.float32 elements to plain Python floats.
    # Matters beyond style: evaluate_real.py's csv.writer tolerates
    # numpy scalars (stringifies anything), but dashboard/backend.py's
    # SSE stream calls json.dumps() on this same data, and a bare
    # numpy.float32 raises TypeError there -- caught the hard way via
    # a live-mode run that fell back to replay data mid-demo.
    return action, probs.tolist(), float(value)


def predict_hybrid(ppo_model, regime_model, obs, loss_pct, base_alpha=0.7):
    """Blend PPO's action distribution with a decision-tree ensemble's
    recommended direction -- a mixture-of-experts gate, not a novel
    published algorithm. When PPO's own top-action probability is
    high (confident), its distribution dominates; when PPO is
    uncertain, the tree's recommendation is weighted more heavily.

    ppo_model predicts against the plain (unaugmented) 3-value
    [rtt, cwnd, throughput] observation, same as predict_with_distribution.
    regime_model (rl/regime_classifier.RegimeModel) predicts a
    continuous "recommended direction" in [-1, 1] (decrease..increase)
    from the raw [rtt, cwnd, throughput, loss_pct] state -- loss_pct
    isn't part of obs itself (RealCongestionEnv's observation space is
    3-valued), so it's passed separately here from the env's info dict.

    Returns (action, blended_probs, ppo_probs, tree_direction) --
    blended_probs and ppo_probs are both plain list[float] (see
    predict_with_distribution's docstring for why that matters for
    JSON serialization, not just CSV).
    """
    action, ppo_probs, _ = predict_with_distribution(ppo_model, obs)
    ppo_probs_arr = np.asarray(ppo_probs)  # back to ndarray for the arithmetic below

    full_features = np.concatenate([np.asarray(obs, dtype=np.float32), [loss_pct]])
    tree_direction = regime_model.direction(full_features)  # in [-1, 1]

    # Turn the scalar direction into a 3-way distribution over
    # decrease/maintain/increase, symmetric around "maintain".
    tree_probs = np.array([
        max(0.0, -tree_direction),
        1.0 - abs(tree_direction),
        max(0.0, tree_direction),
    ])
    tree_probs = tree_probs / tree_probs.sum()

    ppo_confidence = float(np.max(ppo_probs_arr))
    alpha = base_alpha * ppo_confidence + (1 - base_alpha) * 0.5  # low PPO confidence -> lean on tree
    blended_probs = alpha * ppo_probs_arr + (1 - alpha) * tree_probs
    blended_probs = blended_probs / blended_probs.sum()

    hybrid_action = int(np.argmax(blended_probs))
    return hybrid_action, blended_probs.tolist(), ppo_probs, tree_direction


VALID_POLICIES = ("ppo", "hybrid", "cubic", "human")


def regime_extra(regime_model, obs, loss_pct):
    """Regime classifier's predicted class + confidence for a single
    observation, or {} if no classifier is loaded (e.g. before
    scripts/train_regime_classifier.py has run) -- shared by every
    resolve_action() caller that wants to show/log it."""
    if regime_model is None:
        return {}
    features = np.concatenate([np.asarray(obs, dtype=np.float32), [loss_pct]])
    probs = regime_model.regime_probs(features)
    top = int(np.argmax(probs))
    return {"regime_predicted_class": regime_model.classes[top], "regime_confidence": float(probs[top])}


def resolve_action(policy, model, regime_model, obs, loss_pct, human_action=None):
    """Single shared per-sender policy dispatch: ppo/hybrid/cubic/human ->
    (action, extra_info). Originally duplicated between
    dashboard/live_session.py and env/competitive_congestion_env.py;
    factored here for the same reason common/network.py consolidated
    BottleneckTopo/parse_ss_output -- one place, not two that can drift.

    extra_info carries whatever the caller might want to show or log
    (action probabilities, critic value estimate, tree direction, regime
    classification) -- callers that don't care (e.g. resolving an
    opponent's action during RL training) can just ignore it.

    human_action: the most recently received manual action for this
    sender (already resolved by the caller -- this function doesn't know
    about per-session human-action state), defaulting to 1 (maintain) if
    none has been set yet.
    """
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown policy {policy!r}, must be one of {VALID_POLICIES}")

    if policy == "ppo":
        action, probs, value = predict_with_distribution(model, obs)
        extra = {"prob_decrease": probs[0], "prob_maintain": probs[1], "prob_increase": probs[2], "value_estimate": value}
    elif policy == "hybrid":
        action, blended_probs, _, tree_direction = predict_hybrid(model, regime_model, obs, loss_pct)
        extra = {"prob_decrease": blended_probs[0], "prob_maintain": blended_probs[1], "prob_increase": blended_probs[2], "tree_direction": tree_direction}
    elif policy == "cubic":
        action, extra = 1, {}
    else:  # human
        action, extra = (human_action if human_action is not None else 1), {}

    if policy in ("ppo", "hybrid"):
        extra.update(regime_extra(regime_model, obs, loss_pct))
    return action, extra
