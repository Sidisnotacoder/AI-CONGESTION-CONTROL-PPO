# Competitive Retraining — Status & Results

**Written**: 2026-07-13 night (pre-run), updated 2026-07-14 morning with
attempt 2's real results.
**Purpose**: full record of this effort, without re-deriving any of the
context below. If you're a future Claude session reading this cold: this
doc is self-contained, but `git log` on `hybrid-ml-approach` and this
session's conversation history have the full derivation if you need it.

## TL;DR (2026-07-14 morning)

Attempt 2 (26000 timesteps, 9 opponent compositions, ~4.4hr run) worked.
Evaluated against the original solo-trained model, same run/conditions:

- **Mixed competition (3×model + 1×Cubic)** — the scenario attempt 1 failed
  to fix: new model gets **17% lower RTT, 18% higher throughput, lower
  loss**, and Cubic's throughput advantage narrowed from **2.11× to 1.34×**.
  A genuine, across-the-board win, not just a smaller version of attempt 1's
  partial one.
- **Homogeneous (4×model, no Cubic)** — where the old model already won:
  new model is only **~4% behind** (919.8ms vs 882.8ms RTT), a small
  residual gap instead of attempt 1's 48% regression. Both still hold a
  strong ~26-29% RTT advantage over pure Cubic here.

See "Attempt 2 results (full)" below for the complete numbers and the
model/artifact locations. `models/ppo_competitive.zip` now holds this
result (fixed path, live loaders pick it up automatically); attempt 1's
model is preserved separately at
`models/ppo_competitive_20260713T200226Z.zip` if ever needed for reference.

## Why this exists

The live multi-flow dashboard (built earlier this session) let us actually
test PPO/hybrid competing against Cubic on a shared bottleneck, instead of
each running solo in separate episodes (the original project's only mode).
Verified findings from that testing:

- **Mixed competition** (e.g. 3×PPO + 1×Cubic sharing one link): Cubic
  often wins on throughput, RTT ends up nearly identical for everyone
  regardless of policy (RTT is a property of the one shared FIFO queue,
  not something one flow controls unilaterally).
- **Homogeneous groups** (4×PPO vs 4×Cubic, never mixed): PPO/hybrid *do*
  show a real, repeatable ~24-26% lower RTT than Cubic at equal throughput.
- Root cause of the mixed-competition failure: `models/ppo_real_congestion.zip`
  was trained **only** against `RealCongestionEnv` (solo, no competing
  traffic, ever). Its behavior once actually sharing a link (always
  choosing "increase," every step, even at 2000ms+ RTT) is consistent with
  pure out-of-distribution extrapolation, not a considered decision.

**The fix in progress**: retrain PPO with real competing senders present
during training (mirroring MIT's **Remy**, the closest intellectual sibling
to this project — also a computer-generated rate-control policy, but
trained against a diverse distribution of network conditions *including
competing traffic*, which is exactly the ingredient our model lacked).

## Attempt 1 — done, real negative/mixed result (don't repeat, but don't ignore)

`env/competitive_congestion_env.py` (a single-agent `gym.Env` wrapping
`MultiFlowCongestionEnv`, learner always at sender0, 3 opponent senders
per episode) + `rl/train_competitive.py`, first pass:
- 3000 timesteps, `n_steps=256`, `batch_size=64` (matching `rl/train_real.py`'s
  existing precedent) — took **1884s (~31 min)**, i.e. **0.613s/timestep**.
- Only 3 opponent compositions: `[ppo,ppo,cubic]`, `[hybrid,hybrid,cubic]`,
  `[ppo,hybrid,cubic]` — always exactly one Cubic opponent.
- Training reward curve barely moved: `ep_rew_mean` −485 → −428 over 12
  rollout iterations, clearly not converged.

Evaluated with `rl/evaluate_competitive.py` (old `ppo_real_congestion` vs
new `ppo_competitive`, 3 trials each, mixed 3×ppo+1×cubic and homogeneous
4×ppo scenarios) — raw numbers preserved in `results/attempt1_backup/`:

| | scenario | RTT | throughput | loss | reward |
|---|---|---|---|---|---|
| old (solo-trained) | mixed (as ppo) | 1017.9ms | 2.02Mbps | 0.12% | −8.39 |
| old (solo-trained) | mixed (cubic opp.) | 1024.5ms | 3.37Mbps | 0.53% | −7.92 |
| **new (attempt 1)** | mixed (as ppo) | **1137.4ms** | **2.21Mbps** | 0.44% | −10.06 |
| **new (attempt 1)** | mixed (cubic opp.) | **1159.5ms** | **2.66Mbps** | 1.44% | −11.82 |
| old (solo-trained) | homogeneous 4×ppo | 686.8ms | 2.37Mbps | 0.08% | −4.67 |
| **new (attempt 1)** | homogeneous 4×ppo | **1020.3ms** | 2.37Mbps | 0.08% | −7.98 |
| cubic baseline | homogeneous 4×cubic | 1237.9ms | 2.36Mbps | 1.74% | −13.51 |

**Verdict**: partial, real improvement in mixed-competition throughput
*fairness* (Cubic's advantage narrowed from 1.67× to 1.20×) — but RTT got
*worse* for everyone in the mixed case, and the homogeneous case (where
the old model was already winning decisively) **regressed** — RTT rose
from 687ms to 1020ms, giving back roughly half its advantage over plain
Cubic, for no throughput gain. Most likely cause: 3000 timesteps (~75
episodes, split 3 ways) is a sparse signal for a harder problem than solo
training, and every episode had contention present, likely diluting the
original good solo-restraint behavior (nothing preserved a "mostly
uncontested" case). This is consistent with, not a refutation of, the
Remy comparison — Remy's actual training is vastly more extensive than a
30-minute real-Mininet run can match.

## Attempt 2 — what changed

Two changes, both directly targeting attempt 1's specific failure modes:

1. **`env/competitive_congestion_env.py`: 3 → 9 opponent compositions.**
   Added: pure-Cubic worst case (`[cubic,cubic,cubic]`, outnumbered 3-to-1),
   homogeneous PPO-only and hybrid-only peer groups (no Cubic at all —
   reinforces cooperative-peer behavior, not just adversarial), 2:1
   Cubic-outnumbered mixes (`[ppo,cubic,cubic]`, `[hybrid,cubic,cubic]`),
   and — specifically targeting the homogeneous-case regression — an
   `["idle","idle","idle"]` composition where opponents are held at a low
   (2Mbps), non-competing rate rather than genuinely uncapped, approximating
   a near-solo scenario without rebuilding a 1-sender topology. `"idle"`
   reuses `resolve_action("cubic", ...)` for its action (identical
   behavior: always "maintain") — only its start rate differs, via a small
   `_START_RATE_OVERRIDE`/`_RESOLVE_POLICY_ALIAS` mapping local to this
   file. No changes needed to `rl/policy_utils.py`.
2. **`rl/train_competitive.py`: `TOTAL_TIMESTEPS` 3000 → 26000** (~4.43
   hours at the measured 0.613s/timestep, leaving real margin under the
   user's 4.5-5 hour ceiling rather than targeting it exactly).

Both changes were smoke-tested (`sudo python3 env/test_competitive_env.py`,
run twice — once after the first plan's implementation, once after the
9-composition expansion) before committing to the long run. Second smoke
test's 3 sampled episodes landed on 3 different compositions
(`[hybrid,cubic,hybrid]`, `[cubic,cubic,cubic]`, `[hybrid,hybrid,hybrid]`),
confirming the expanded sampling genuinely varies, with no errors and
physically sane reward/RTT/throughput trajectories.

`results/attempt1_backup/` holds attempt 1's raw CSVs/JSON (reward curve +
eval results) since the fixed-path result files get overwritten by rerunning
the same scripts — the table above is the durable record either way.

**Training ran**: directly in the foreground in the user's terminal (no
tmux/nohup needed — terminal was kept open all night as planned), started
2026-07-13 ~20:14, finished 2026-07-14 00:46. Completed cleanly: 26112
total timesteps (target 26000, the small overshoot is normal — training
stops at the first `n_steps`-aligned checkpoint at/past the target), 102
rollout iterations, 15751s elapsed (~4h22m, under the 0.613s/timestep
estimate and comfortably inside the 4.5-5hr budget).

**Final training stats**: `ep_rew_mean` reached **-298**, and — the more
important signal — the last ~15 iterations (timesteps 22528-26112) plateau
in a stable -280 to -307 band rather than still trending downward, real
convergence rather than "just ran longer." `explained_variance` reached
0.27 (up from near-zero at attempt 1's cutoff), meaning the critic is
actually tracking returns now. Compare to attempt 1's terminal `ep_rew_mean`
of -428, still clearly improving (not plateaued) at cutoff.

## Attempt 2 results (full) — 2026-07-14

Evaluated with the same `rl/evaluate_competitive.py` (3 trials each, old
`ppo_real_congestion` vs new `ppo_competitive`, evaluated together in the
same run/conditions — the valid basis for comparison, since these are real
network measurements with natural run-to-run variance; the "old" model's
own absolute numbers shifted slightly from attempt 1's separate evaluation
run too, for that reason):

| | scenario | RTT | throughput | loss | reward |
|---|---|---|---|---|---|
| old (solo-trained) | mixed (as ppo) | 1167.6ms | 1.85Mbps | 0.19% | −10.20 |
| old (solo-trained) | mixed (cubic opp.) | 1176.2ms | 3.91Mbps | 1.08% | −10.02 |
| **new (attempt 2)** | mixed (as ppo) | **971.6ms** | **2.19Mbps** | **0.13%** | −7.78 |
| **new (attempt 2)** | mixed (cubic opp.) | **972.9ms** | 2.94Mbps | 0.26% | −7.31 |
| old (solo-trained) | homogeneous 4×ppo | 882.8ms | 2.36Mbps | 0.11% | −6.68 |
| **new (attempt 2)** | homogeneous 4×ppo | 919.8ms | 2.36Mbps | **0.07%** | −6.98 |
| cubic baseline | homogeneous 4×cubic | 1241.7ms | 2.36Mbps | 1.83% | −13.71 |

**Verdict — a real, honest improvement**:
- **Mixed competition** (the scenario attempt 1 failed to fix): the new
  model gets **17% lower RTT** (971.6 vs 1167.6ms), **18% higher
  throughput** (2.19 vs 1.85Mbps), and **lower loss** (0.13% vs 0.19%) than
  the old model — an across-the-board win in the exact scenario this whole
  effort targeted. Cubic's throughput advantage narrowed from **2.11× to
  1.34×** (old model faced 3.91/1.85; new model faces 2.94/2.19). Cubic
  still wins outright, but the gap is much smaller and every other metric
  moved in the right direction too — a genuinely different outcome from
  attempt 1's "narrower gap but worse RTT for everyone."
- **Homogeneous** (where the old model already won): new model is only
  **~4% behind** on RTT (919.8 vs 882.8ms) with identical throughput and
  slightly *lower* loss — a small residual gap, not attempt 1's 48%
  regression (1020.3ms). The `"idle"` composition addition appears to have
  mostly fixed the catastrophic-forgetting problem. Both models still hold
  a strong ~26-29% RTT advantage over pure Cubic here.

**Not yet done / possible next steps** (not attempted this session):
- The residual ~4% homogeneous gap could likely be closed further with
  either more `"idle"`-composition weighting or more total training time.
- Cubic still wins mixed-competition throughput outright (1.34×) — closing
  that the rest of the way is a harder problem, per the earlier discussion
  of AQM/fair-queueing (RED) vs. reward-shaping vs. much longer training
  more in line with Remy's actual compute budget.
- The decision-tree hybrid's tree itself is still trained only on
  solo-flow data (`data/raw/network_metrics.csv`) — recollecting
  competitive-scenario data and retraining `rl/regime_classifier.py` was
  explicitly deferred at the start of this effort, still on the table if
  wanted.
- `dashboard/backend.py` still defaults to `models/ppo_real_congestion.zip`
  for its "ppo"/"hybrid" arms — the live dashboard hasn't been pointed at
  `models/ppo_competitive.zip` yet, so this new model isn't visible there
  until that's wired up (a small change, not done as part of this effort).

## Key files

- `env/competitive_congestion_env.py` — training env, 9 opponent compositions
- `env/test_competitive_env.py` — smoke test (run before trusting a long run)
- `rl/train_competitive.py` — training script, `TOTAL_TIMESTEPS=26000` now
- `rl/evaluate_competitive.py` — old-vs-new comparison (mixed + homogeneous)
- `rl/policy_utils.py::resolve_action` — shared ppo/hybrid/cubic/human
  dispatch, used by both the dashboard and this training/eval code
- `models/ppo_real_congestion.zip` — original solo-trained model (untouched,
  stays the "old" baseline)
- `models/ppo_competitive.zip` — attempt 2's model (current fixed-path
  output; attempt 1's model is preserved only via its timestamped
  `models/ppo_competitive_20260713T200226Z.zip` copy, not the fixed-path one)
- `results/attempt1_backup/` — attempt 1's raw reward curve + eval CSVs/JSON
