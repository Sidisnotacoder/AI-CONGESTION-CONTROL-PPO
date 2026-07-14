# AI-Based Adaptive Congestion Control — Complete Project State

**Purpose of this document**: raw material for a later report / paper / presentation —
not a polished write-up itself. Every number here is from a real, executed run
(no simulated or estimated figures), with the source file noted so it can be
re-verified or re-plotted later. See `data/` for underlying CSVs/JSON and
`figures/` for supporting images/charts.

**Repository**: branch `hybrid-ml-approach`, pushed to
`https://github.com/Sidisnotacoder/AI-CONGESTION-CONTROL-PPO` (fork; the
original repo owner `vibin56-debug` hasn't granted write access to this
account). `git log` on that branch has the full, granular commit history
behind everything summarized here.

---

## 1. What this project is

A reinforcement-learning approach to TCP congestion control. Instead of a
fixed algorithm like TCP Cubic deciding how fast to send data, a PPO
(Proximal Policy Optimization) agent makes that decision dynamically from
live network telemetry, on a real (not simulated-in-software) Mininet
network with genuine `iperf3` traffic flows.

**Core mechanics** (unchanged throughout this project's evolution):
- **Topology**: sender host(s) and a receiver connected through a
  bandwidth/delay-constrained bottleneck link (10Mbps / 20ms by default).
- **Observation** (every 0.5s): `[RTT (ms), CWND, Throughput (Mbps)]`, read
  directly off the sending host's real socket via `ss -ti`.
- **Action**: `Discrete(3)` — decrease / maintain / increase the sender's
  own uplink rate cap, enforced via Linux `tc` traffic shaping, stepping
  1Mbps at a time.
- **Reward**: `throughput − 2×loss% − RTT/100` — rewards throughput,
  penalizes loss and latency, so a good policy holds back just enough to
  avoid bufferbloat (queueing delay) rather than blindly maximizing send rate.

This document covers the project's full evolution: the original solo-flow
benchmark, a decision-tree-hybrid methodology, an interactive live
multi-flow dashboard, the empirical findings that dashboard produced, and a
competitive retraining effort that followed directly from those findings.

---

## 2. Original result: solo PPO vs. solo Cubic

The foundational, still-valid result. A PPO agent and plain TCP Cubic each
run **alone** (no competing traffic) against the same physical bottleneck,
in separate episodes, compared with proper statistics (5 trials, Welch's
t-test, 95% confidence intervals — `rl/evaluate_multi_trial.py`).

| Metric | PPO agent | PPO + decision-tree hybrid | TCP Cubic |
|---|---|---|---|
| Mean RTT | 235.1ms (95% CI [88.3, 381.8]) | **153.1ms** (95% CI [42.3, 263.9]) | 333.7ms (95% CI [141.9, 525.5]) |
| Mean throughput | 9.07Mbps (95% CI [8.81, 9.33]) | 9.22Mbps (95% CI [9.17, 9.27]) | 9.45Mbps (95% CI [9.32, 9.58]) |

- Hybrid-vs-Cubic RTT difference: t=−2.26, **p=0.061** (close to
  conventional significance despite n=5).
- Both PPO variants show a small, statistically significant throughput cost
  vs. Cubic (p=0.011 plain PPO, p=0.006 hybrid) — consistent with the core
  thesis: trade a little throughput for a lot less queueing delay.
- The hybrid arm didn't just beat plain PPO on RTT — it also *recovered*
  some of PPO's throughput cost (9.22 vs 9.07Mbps), so the two objectives
  weren't strictly in tension for the hybrid arm specifically.
- **Caveat**: n=5 trials is small; confidence intervals are correspondingly
  wide. This describes these particular runs honestly, not a
  high-confidence settled conclusion.

Source: `data/multi_trial_stats.json`, `data/multi_trial_eval.csv`,
`data/multi_trial_per_step.csv`. Figures: `figures/rtt_vs_time.png`,
`figures/throughput_vs_time.png`, `figures/summary_comparison.png`.

---

## 3. Methodology augmentation: the decision-tree hybrid

### 3.1 How PPO itself decides (mechanism, not just outcome)

The policy is a small feedforward network (Stable-Baselines3 default
`MlpPolicy`, two 64-unit hidden layers) with two heads:
- **Actor**: 3 logits → softmax → probability distribution over
  {decrease, maintain, increase}.
- **Critic**: a single scalar `V(s)`, an estimate of expected future reward
  — used only during training (to compute advantages for the policy
  gradient update), not for choosing actions at inference.

**Input**: just the current instant's `[RTT, CWND, Throughput]` — no
history or memory of prior steps.

**Action selection at inference is deterministic**: `action = argmax(probs)`.
The same input always produces the same output given fixed weights. (During
*training*, actions are sampled stochastically from the same distribution,
which is how PPO explores.) This matters for interpreting variance seen in
live testing (§5): it isn't the policy "rolling dice," it's a deterministic
function reacting to genuinely different real-network readings step to
step, with small early differences compounding through TCP's own nonlinear
dynamics.

**Learning**: no hand-coded rules anywhere. PPO runs real episodes,
observes `(state, action, reward)` sequences, and nudges the network's
weights via gradient descent so actions that led to better-than-expected
reward (positive advantage, per the critic's baseline) become more probable
in similar future states. This is why a model's behavior outside its
training distribution can look strange — there's no fallback logic, only
whatever the learned function happens to extrapolate to.

### 3.2 The decision-tree ensemble

Two gradient-boosted-tree models (`rl/regime_classifier.py`), trained on
**81,725 real, labeled network-state readings** collected across four
traffic scenarios (baseline-low, medium-load, high-load, bursty) via
`scripts/collect_dataset.py`:
1. A **classifier** predicting which of the 4 regimes the current
   `[rtt, cwnd, throughput, loss]` reading resembles. Held-out accuracy:
   **99.8%** (up from ~82% on an earlier, much smaller smoke-test sample —
   the regimes are highly separable once there's enough real data).
2. A **regressor** outputting a continuous "recommended direction" in
   `[-1, 1]`, trained on a per-scenario AIMD-style heuristic label (back
   off under loss or above-median RTT for that regime; probe upward when
   clean) — a data-driven generalization of classical AIMD logic, not a
   novel algorithm.

### 3.3 The blend ("hybrid")

`rl/policy_utils.py::predict_hybrid` / `resolve_action`:
1. Get PPO's own action distribution.
2. Convert the tree's scalar direction into a matching 3-way distribution.
3. `alpha = 0.7 × PPO's own confidence + 0.3 × 0.5` — high PPO confidence →
   PPO's view dominates; low confidence → the tree's recommendation gets
   real weight.
4. `final = alpha × PPO_probs + (1−alpha) × tree_probs`, take the argmax.

A confidence-gated mixture-of-experts, honestly described as that — not a
published or novel algorithm. Evaluated as a **third arm alongside**, never
replacing, plain PPO and Cubic throughout this project.

---

## 4. The interactive live multi-flow dashboard

Built to move beyond static plots/solo-episode replay into a real,
configurable network lab. Two modes:
- **Replay mode**: self-contained, zero-network-call build
  (`dashboard/static/replay_artifact.html`) for sharing/screenshotting —
  animates pre-recorded episodes.
- **Live mode** (`dashboard/backend.py`, Flask + Server-Sent Events):
  drives a real Mininet episode and streams telemetry to the browser as it
  happens.

**8 sections** (tabbed UI, light-themed): Replay & Compare; Configure & Run
(build a real 1-4 sender Mininet network, any policy mix, live); Live
Multi-Flow Metrics (per-sender live charts); Live Control (drag
bandwidth/delay, flip queue discipline, inject traffic bursts — all
mid-episode); Animated Topology Diagram; Human vs AI Manual Override (drive
a flow's rate by hand against AI-controlled ones); Scrub Any Recorded Run
(any finished live session becomes replayable, exactly like the built-in
arms); Feel the Latency (a click-and-respond widget whose lag is driven by
real RTT data).

Figures: `figures/dashboard_home_overview.png`,
`figures/dashboard_replay_compare.png`,
`figures/dashboard_topology_diagram.png`,
`figures/dashboard_live_multiflow_metrics.png`.

---

## 5. Key empirical finding: solo restraint doesn't transfer to real competition

This is the project's most important finding — and it came directly from
building something (the multi-flow dashboard) that let policies actually
compete on one shared link, which nothing before this could test.

**The problem**: `models/ppo_real_congestion.zip` (the original model
behind §2's results) was trained *only* against a solo flow — zero
competing traffic, ever. Live-tested against real competition, its
behavior (e.g. always choosing "increase," every step, even at 2000ms+
RTT) was consistent with pure out-of-distribution extrapolation, not a
considered decision — it had simply never seen these states.

**The mechanism, once diagnosed**: RTT is a property of the *shared FIFO
queue*, not something one flow controls unilaterally. If a competing flow
(Cubic) won't reciprocate restraint, a self-limiting flow backing off
doesn't lower shared queueing delay — it just cedes throughput for no
delay benefit. This is a known, real problem in networking research (it's
exactly why "TCP-friendliness" — not starving other flows — is a hard
design requirement for any congestion-control scheme, not a nice-to-have).
Even Google's BBR hit an analogous fairness problem after real-world
deployment, addressed only in BBRv2/v3.

**Mixed competition** (e.g. 3×PPO + 1×Cubic sharing one link, original
model): Cubic often won on throughput, sometimes badly; RTT ended up
nearly identical for everyone regardless of policy.

**Homogeneous groups** (4×PPO vs. 4×Cubic, never mixed): PPO/hybrid *did*
show a real, repeatable RTT advantage over Cubic — restraint helps when
the whole group practices it. (Later reconfirmed at much higher statistical
power — see §7.)

### 5.1 Active Queue Management (RED) investigation

Tested whether a fairer network-side queueing discipline (RED — Random
Early Detection, vs. plain FIFO tail-drop) could fix the mixed-competition
problem without retraining anything. Real result, properly averaged over 3
trials per condition (3×PPO + 1×Cubic, same bottleneck):

| | avg RTT | avg throughput | Cubic's loss% |
|---|---|---|---|
| FIFO | ~1036ms (both) | PPO 1.86Mbps / Cubic 3.75Mbps | 0.33% |
| RED | ~991-995ms (both) | PPO 2.35Mbps / Cubic 2.22Mbps | 0.61% |

RED genuinely shifted **throughput fairness** (PPO's average edged past
Cubic's, a real flip from FIFO) — driven by Cubic's loss rate nearly
doubling under RED (it's correctly being identified as taking a
disproportionate share and dropped more). But RED barely moved **RTT**
(~5% lower, not the ~40% a single earlier unaveraged trial had suggested —
an important correction made honestly once properly averaged). **Verdict**:
RED helps fairness, not latency, at this bandwidth with default
(untuned-for-this-link) thresholds — the delay problem stayed unsolved by
a network-side fix alone, motivating the retraining effort in §6.

---

## 6. Competitive retraining

**Approach**: retrain PPO with real opponent senders present during
training — mirroring MIT's **Remy** (the closest intellectual sibling to
this project: also a computer-generated rate-control policy, but trained
against a *diverse distribution of network conditions including competing
traffic*, the exact ingredient the original model lacked).

**Architecture** (`env/competitive_congestion_env.py`): a single-agent
`gym.Env` wrapping the existing multi-sender live environment. The learner
is always sender0; 3 opponent senders per episode run scripted/frozen
policies (a frozen copy of the original model for "ppo"/"hybrid" opponents,
genuinely uncapped Cubic for "cubic" opponents), sampled per-episode from a
fixed set of compositions — always exactly 4 total senders, so the Mininet
topology never needs rebuilding between episodes (only per-sender policy
labels change).

### 6.1 Attempt 1 — negative/mixed result (kept for the record, not repeated)

3000 timesteps, `n_steps=256`/`batch_size=64` (matching the original
training precedent), 3 opponent compositions (always exactly one Cubic
opponent). Training reward curve barely moved (`ep_rew_mean` −485 → −428
over 12 iterations, clearly not converged).

| | scenario | RTT | throughput |
|---|---|---|---|
| old | mixed (as ppo) | 1017.9ms | 2.02Mbps |
| **new (attempt 1)** | mixed (as ppo) | **1137.4ms** | **2.21Mbps** |
| old | homogeneous 4×ppo | 686.8ms | 2.37Mbps |
| **new (attempt 1)** | homogeneous 4×ppo | **1020.3ms** | 2.37Mbps |

Partial throughput-fairness improvement, but RTT got *worse* everywhere,
and the homogeneous case (where the original model already won) regressed
by 48%. Diagnosed cause: an under-trained run (sparse signal, and every
training episode had contention present — likely diluting the original
solo-restraint behavior with nothing to preserve it).

### 6.2 Attempt 2 — real, verified improvement

Two changes, both targeting attempt 1's specific failure modes:
1. **9 opponent compositions** instead of 3 — added pure-Cubic worst case
   (outnumbered 3-to-1), homogeneous PPO-only/hybrid-only peer groups (no
   Cubic at all), 2:1 Cubic-outnumbered mixes, and an `"idle"` composition
   (opponents held at a low, non-competing rate) specifically to preserve
   original solo-flow behavior.
2. **26,000 timesteps** (~4.4 hours real Mininet training, run overnight,
   user-approved 4.5-5 hour budget) instead of 3000.

Training converged genuinely this time: `ep_rew_mean` reached **−298**,
plateauing in the final ~15 iterations (see
`figures/chart_training_reward_curves.png` — note the honest, non-trivial
optimization texture: a real dip around timestep 12,000-13,000 before
recovering in the final third, not a clean monotonic curve).
`explained_variance` reached 0.27 (up from near-zero at attempt 1's
cutoff) — the critic is actually tracking returns now.

**Full before/after** (3 trials each, evaluated together, same
run/conditions — see `figures/chart_retraining_before_after.png`):

| | scenario | RTT | throughput | loss |
|---|---|---|---|---|
| old | mixed (as ppo) | 1167.6ms | 1.85Mbps | 0.19% |
| **new (attempt 2)** | mixed (as ppo) | **971.6ms** | **2.19Mbps** | **0.13%** |
| old | mixed (cubic opp.) | 1176.2ms | 3.91Mbps | 1.08% |
| **new (attempt 2)** | mixed (cubic opp.) | **972.9ms** | 2.94Mbps | 0.26% |
| old | homogeneous 4×ppo | 882.8ms | 2.36Mbps | 0.11% |
| **new (attempt 2)** | homogeneous 4×ppo | 919.8ms | 2.36Mbps | **0.07%** |
| cubic baseline | homogeneous 4×cubic | 1241.7ms | 2.36Mbps | 1.83% |

**Verdict**: a genuine, across-the-board win in mixed competition — 17%
lower RTT, 18% higher throughput, lower loss than the old model, and
Cubic's throughput advantage narrowed from **2.11× to 1.34×**. Only a small
~4% residual RTT gap in the homogeneous case (vs. attempt 1's 48%
regression) — the `"idle"` composition addition mostly fixed the
catastrophic-forgetting problem.

**A live-dashboard bug found and fixed during this process**: after
retraining, `dashboard/backend.py` was still loading the *old* model path
— every live demo session was silently showing stale, known-bad behavior
even though the underlying retrained model was genuinely better. Caught by
comparing a live session's numbers against the evaluation script's;
verified fixed with a fresh live session afterward.

**A separate visual bug found and fixed**: the animated topology diagram's
flow-line paths all converged to the exact same coordinates through the
bottleneck, so overlapping lines fully occluded each other — a 3×PPO+1×Cubic
session's shared segment misleadingly looked entirely Cubic-colored, when
3 of 4 flows were actually a different color underneath. Fixed with a small
per-sender lane offset.

Source: `data/competitive_eval.csv`, `data/competitive_eval_stats.json`,
`data/training_reward_curve_competitive.csv`,
`data/attempt1_backup/` (attempt 1's preserved raw data),
`data/ppo_competitive_20260713T200226Z.json` (attempt 1 metadata),
`data/ppo_competitive_20260714T004648Z.json` (attempt 2 metadata, includes
exact opponent compositions and git commit hash for reproducibility).

---

## 7. Documentation batch — properly-powered final numbers (n=40 per configuration)

After the retraining fix landed everywhere (including the live dashboard),
a clean, statistically-sized batch was run for documentation purposes: 10
trials × 4 fixed 4-sender configurations
(`rl/collect_documentation_runs.py`), using the final retrained model
throughout. This is the most reliable dataset in this project — 4× the
trial count of any other multi-flow comparison run this session.

| config | n | avg RTT | avg throughput | avg loss | avg reward |
|---|---|---|---|---|---|
| Cubic only (4×) | 40 | 1238.1ms | 2.36Mbps | 1.70% | −13.41 |
| PPO only (4×) | 40 | 839.4ms | 2.38Mbps | 0.08% | −6.17 |
| **Hybrid only (4×)** | 40 | **751.9ms** | 2.38Mbps | 0.10% | **−5.33** |
| 2×PPO + 2×Hybrid | 40 | 871.7ms | 2.38Mbps | 0.08% | −6.50 |

See `figures/chart_documentation_runs.png`.

**Findings**:
- Throughput is essentially identical across all four configurations
  (~2.36-2.38Mbps) — in homogeneous/near-homogeneous groups, everyone gets
  roughly their fair share regardless of policy. All differentiation is in
  RTT and loss.
- `ppo_only`: 32% lower RTT than Cubic; loss drops ~21× (1.70% → 0.08%).
- **`hybrid_only` outperforms `ppo_only`** — 752ms vs. 839ms RTT (~10%
  better), better average reward too. The tree measurably helps even in a
  fully homogeneous group, not just as a fallback for uncertain PPO states.
- `2ppo_2hybrid`: the two policy types perform almost identically to each
  other when mixed together (870ms/2.33Mbps ppo vs. 873ms/2.42Mbps hybrid)
  — two cooperative, restraint-capable policies coexist without either
  dominating the other.

Source: `data/documentation_runs_summary.csv` (161 rows, one per
sender/trial), `data/documentation_runs_per_step.csv` (6401 rows, full raw
per-step data for any deeper analysis).

---

## 8. Honest limitations and caveats (know these before presenting)

- **Small samples in several places**: the original 5-trial solo benchmark
  (§2), the RED investigation (3 trials/condition, §5.1), and the
  retraining before/after (3 trials/condition, §6.2) all carry real
  sampling variance. The n=40 documentation batch (§7) is the most
  statistically solid dataset in this project — prefer citing it where the
  claim overlaps.
- **Individual-sender variance is real and large**: across ~24 individual
  live-session PPO-sender readings gathered informally during this
  session, throughput ranged 0.74-3.14Mbps (44% coefficient of variation)
  for the *same* model under nominally the same configuration — chaotic
  sensitivity to real timing noise, not policy randomness (action
  selection is deterministic at inference). Any single live demo run can
  look unusually good or bad; don't over-interpret one session.
- **Cubic still wins mixed-competition throughput outright** (1.34× even
  after retraining) — the fix narrowed the gap, it didn't reverse it.
  Closing the rest of the way is a harder problem (per-flow fair queueing
  like fq_codel/SFQ, more extensive training, or explicit fairness-aware
  reward shaping — none attempted this session).
- **RED helps fairness, not latency** — don't claim it fixed bufferbloat;
  it didn't, at this bandwidth with default thresholds.
- **The decision tree itself is still trained only on solo-flow data**
  (`data/raw/network_metrics.csv`, from the original single-flow data
  collection) — recollecting competitive-scenario data and retraining
  `rl/regime_classifier.py` was explicitly deferred, not attempted.
- **4-node cap**: Mininet/OVS resource constraints on a single VM, a
  practical scoping choice, not a fundamental limit of the approach.

---

## 9. Repository map (for anyone extending this later)

- `env/real_congestion_env.py` — original single-flow live Mininet env (training/eval for §2)
- `env/multi_flow_congestion_env.py` — N-sender live orchestrator (dashboard + all multi-flow evaluation)
- `env/competitive_congestion_env.py` — single-agent training env with real opponents (§6)
- `rl/policy_utils.py` — `resolve_action`/`predict_hybrid`/`predict_with_distribution` (shared by dashboard, training, and evaluation code)
- `rl/regime_classifier.py` — the decision-tree ensemble (§3.2)
- `rl/train_real.py` / `rl/train_competitive.py` — original / competitive training scripts
- `rl/evaluate_multi_trial.py` / `rl/evaluate_competitive.py` / `rl/collect_documentation_runs.py` — the three evaluation scripts behind §2 / §6 / §7 respectively
- `dashboard/` — the live multi-flow dashboard (Flask backend + tabbed frontend)
- `models/ppo_competitive.zip` — the current, best-performing model (fixed path; dashboard and all evaluation scripts load this)
- `docs/HYBRID_ML_APPROACH.md` — the original hybrid-methodology write-up
- `docs/COMPETITIVE_RETRAINING_STATUS.md` — full retraining log (source for §6 above)
