# Hybrid ML Approach — What Changed and Why

This document covers the work merged on the `hybrid-ml-approach` branch. It's
written for someone who understands the *original* project (PPO agent vs TCP
Cubic on a live Mininet bottleneck, reward = `throughput - 2*loss% - rtt/100`)
and wants to know what's different now, and why — not a line-by-line code
diff. For that, see `git log` / `git diff main`.

## What did **not** change

The original problem statement was left alone on purpose, so every new
result stays directly comparable to the project's existing baseline:

- The reward formula, the 3-action space (decrease / maintain / increase),
  and the raw `[rtt, cwnd, throughput]` observation space of the live
  environment.
- The bottleneck topology's physical parameters (10 Mbps, 20 ms delay).
- The original PPO model and its training recipe.

Everything below is either (a) making the existing pipeline more trustworthy,
or (b) adding a genuinely new, separately-evaluated third option alongside
PPO and Cubic — never replacing either.

---

## 1. Making the existing results trustworthy

Before touching methodology, a pass went through the whole pipeline looking
for ways it could silently produce misleading numbers without anyone
noticing. The concrete problems found and fixed:

- **A parsing failure used to look identical to a real result.** If the
  live network-stats reader (`ss -ti`) ever returned something the parser
  didn't recognize, the code quietly substituted an all-zero reading —
  which is indistinguishable from "the agent legitimately drove throughput
  to zero." Any run affected by this would have silently corrupted its own
  training or evaluation data with no trace. This now logs a visible
  warning (with the raw unparsed line) instead of failing silently.
- **The reported statistics were approximate when they didn't need to be.**
  The 95% confidence intervals and the PPO-vs-Cubic significance test were
  computed with a hand-rolled normal-distribution approximation, flagged in
  the code as a stopgap. At only 5 trials per arm — a small-sample regime
  where that approximation is at its least reliable — this was swapped for
  the exact method (scipy's t-distribution interval and Welch's t-test).
  The original project's own headline number (RTT: p≈0.069, borderline)
  was exactly the kind of result where using the wrong statistical tool
  could change the conclusion.
- **The training-reward curve in the report was hand-typed from console
  output**, not read from any log. Training now writes real per-step
  reward data to disk as it happens, so that chart (and any future one)
  reflects an actual recorded run instead of numbers copied out of a
  terminal by hand.
- **Evaluation runs weren't versioned against the model that produced
  them.** Every training run now writes a timestamped, permanent copy of
  the model alongside a small record of what produced it (hyperparameters,
  timestep budget, git commit) — so a given set of results can always be
  traced back to the exact model and code that generated them, without
  disturbing the fixed model path the rest of the pipeline already expects.
- **Per-step data from the 5-trial comparison was computed and then
  thrown away**, leaving only trial-level averages. It's now persisted in
  full, which is what makes the dashboard's step-by-step visualization
  possible at all (see §3).
- General robustness: cleaner shutdown of the Mininet network and iperf3
  processes if a run errors out partway through, and consistent logging in
  place of scattered `print()` statements, so a failed run is diagnosable
  after the fact instead of just producing a truncated CSV.

None of this changes what the agent does — it changes whether you can trust
what the pipeline tells you about what the agent did.

---

## 2. New methodology: PPO blended with a decision-tree ensemble

This is the actual new idea in this branch. The question it asks: can a
second, structurally different model — trained directly on real observed
network behavior rather than learned through trial-and-error interaction —
sharpen PPO's decisions, especially in the moments where PPO itself is
uncertain?

**The data.** A dedicated data-collection run drove the live Mininet
bottleneck through four distinct traffic regimes — a light baseline load, a
medium load, a heavy/high load, and a bursty on/off traffic pattern — and
recorded **81,725 real, labeled network-state readings** across them
(`data/raw/network_metrics.csv`).

**The model.** A gradient-boosted decision tree ensemble was trained on
that data to do two things at once:

1. **Classify the current traffic regime** (baseline / medium / high /
   bursty) from the live `[rtt, cwnd, throughput, loss]` reading. On a first
   pass with a small smoke-test sample this classifier was only ~82%
   accurate; once trained on the full 81,725-row dataset, held-out accuracy
   reached **99.8%** — the traffic regimes turn out to be highly
   separable once there's enough real data to learn the boundaries from.
2. **Recommend a rate direction** (a continuous decrease↔increase score),
   trained on classic AIMD-style logic (back off under loss or elevated
   latency, probe upward when clean) but applied per-regime rather than
   with one fixed global threshold — a data-driven generalization of that
   heuristic, not a novel algorithm being claimed as one.

**The blend.** At every decision point, this tree ensemble's recommendation
is combined with PPO's own action probabilities — weighted by how
*confident* PPO is at that moment. When PPO is confident, its own judgment
dominates; when PPO is uncertain, the tree ensemble's recommendation carries
more weight. This is a straightforward mixture-of-experts gate, not a
learned or novel combination rule.

**Crucially, this produces a third thing to test, not a replacement.**
Plain PPO and plain Cubic keep running exactly as before; the hybrid model
is evaluated as a third, independent arm in the same comparison, so its
value is something you can check empirically rather than something asserted
by construction.

---

## 3. Live demonstration dashboard

Previously the only outputs were static matplotlib PNGs and flat CSVs — no
way to see what the agent was actually "thinking" at any given moment, and
nothing that could be shared or replayed without re-running the whole
Mininet pipeline. The dashboard fixes both:

- **Replay mode**: a self-contained, shareable page (no backend, no root,
  no live network needed) that animates a real recorded episode for all
  three arms — PPO, the hybrid, and Cubic.
- **Live mode**: for anyone with the environment set up, a backend process
  drives an actual Mininet episode in real time and streams it to the same
  frontend as it happens.
- **Explainability panel**: at every step, shows PPO's action probabilities
  (how confident it was, not just what it picked), the critic's predicted
  value versus the reward actually received, a breakdown of exactly which
  reward term (throughput / loss / RTT) drove that step's score, and — new
  from §2 — the tree ensemble's regime classification and recommended
  direction, so you can see the two models agree or disagree in real time.

---

## 4. What the new results actually show

From the 5-trial statistical comparison run against the full dataset and
retrained models (`results/multi_trial_stats.json`):

| Metric | PPO agent | **PPO + tree hybrid** | Cubic (baseline) |
|---|---|---|---|
| Mean RTT | 235.1 ms | **153.1 ms** | 333.7 ms |
| Mean throughput | 9.07 Mbps | **9.22 Mbps** | 9.45 Mbps |

- The hybrid arm's RTT-vs-Cubic difference (t = −2.26, p = 0.061) is
  noticeably closer to conventional significance than plain PPO's own
  RTT-vs-Cubic result (t = −1.13, p = 0.29) — despite the same n=5 trial
  count, which is a real constraint on how much confidence to place in any
  of these numbers individually.
- Both PPO variants show a small, statistically significant throughput
  cost relative to Cubic (p = 0.011 for plain PPO, p = 0.006 for the
  hybrid) — consistent with the project's original finding that this
  approach trades a little throughput for a lot less queueing delay.
- Notably, the hybrid arm didn't just lower RTT further than plain PPO —
  it also recovered some of PPO's throughput cost (9.22 vs 9.07 Mbps), so
  in this run the two objectives weren't strictly in tension for the
  hybrid the way they were for PPO alone.
- Caveat that applies to all of the above: n=5 trials per arm is small,
  and the confidence intervals are correspondingly wide (e.g. hybrid RTT's
  95% CI spans roughly 42–264 ms) — these numbers describe this run
  honestly, not a settled, high-confidence conclusion. More trials would
  narrow that.

These are the same figures currently live on the published dashboard.

---

## 5. Where to look

- `README.md` — original project setup and the pre-existing PPO-vs-Cubic
  result.
- `results/multi_trial_stats.json`, `results/multi_trial_eval.csv`,
  `results/multi_trial_per_step.csv` — the numbers behind §4.
- `data/raw/network_metrics.csv` — the 81,725-row labeled dataset (not
  committed to git; regenerate via `scripts/collect_dataset.py`).
- `dashboard/static/replay_artifact.html` — the shareable, self-contained
  dashboard build.
