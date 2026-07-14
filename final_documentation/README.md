# final_documentation/

Raw material for the project report / paper / presentation — assembled so
none of this session's findings need re-deriving later. Start with
**`PROJECT_STATE.md`** — it's the complete narrative (project description,
methodology, every real result with numbers, honest caveats, and a
repository map) and links out to everything below by section.

```
final_documentation/
  README.md            <- this file
  PROJECT_STATE.md      <- start here: the full write-up
  data/                 <- every CSV/JSON result cited in PROJECT_STATE.md
    documentation_runs_summary.csv     n=40-per-config final results (Section 7)
    documentation_runs_per_step.csv    same, full per-step raw data
    competitive_eval.csv / _stats.json  retraining before/after (Section 6.2)
    attempt1_backup/                    attempt 1's preserved raw data (Section 6.1)
    multi_trial_*.{csv,json}            original solo PPO/hybrid/Cubic benchmark (Section 2)
    training_reward_curve*.csv          training curves, both attempts
    ppo_vs_cubic_eval.csv               original single-episode comparison
    ppo_competitive_*.json              model metadata (hyperparams, opponent
                                         compositions, git commit per training run)
  figures/               <- charts and dashboard screenshots, ready to drop into slides
    chart_documentation_runs.png         Section 7's 4-config comparison (generated fresh)
    chart_retraining_before_after.png    Section 6.2's old-vs-new comparison (generated fresh)
    chart_training_reward_curves.png     attempt 1 vs attempt 2 convergence (generated fresh)
    rtt_vs_time.png, throughput_vs_time.png,
    summary_comparison.png, training_reward_curve.png   original solo-benchmark plots
    dashboard_*.png                      live dashboard screenshots (light theme, final UI)
```

Everything in `data/` and `figures/` is real, executed output — nothing
here is a mockup or placeholder.
