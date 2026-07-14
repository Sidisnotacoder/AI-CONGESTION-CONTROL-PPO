from competitive_congestion_env import CompetitiveCongestionEnv

# Mirrors test_real_env.py / test_multi_flow_env.py: no assertions on
# exact values (real network conditions vary run to run), just a sanity
# pass confirming reset()/step() work, opponent composition actually
# varies across episodes, and close() leaves no leaked Mininet state.
env = CompetitiveCongestionEnv()

try:
    for episode in range(3):
        obs, info = env.reset()
        print(f"episode {episode} opponent labels (senders 1-3): {env._labels[1:]}")
        print("  initial state:", obs, info)

        for step in range(5):
            action = 2 if step % 2 == 0 else 0  # alternate increase/decrease
            obs, reward, terminated, truncated, info = env.step(action)
            print(
                "  step", step + 1,
                "action:", action, "state:", obs,
                "reward:", round(reward, 3), "rate:", info["rate_mbps"],
            )
finally:
    env.close()
