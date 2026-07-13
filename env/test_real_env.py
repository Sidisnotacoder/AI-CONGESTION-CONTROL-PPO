from real_congestion_env import RealCongestionEnv

env = RealCongestionEnv()

try:
    obs, _ = env.reset()
    print("Initial state:", obs)

    for i in range(10):

        action = 2 if i % 2 == 0 else 0  # alternate increase/decrease

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            "Step", i + 1,
            "Action:", action,
            "State:", obs,
            "Reward:", round(reward, 3),
            "Rate:", info["rate_mbps"],
        )
finally:
    # Without this, an exception mid-loop (or a Ctrl-C, which raises
    # KeyboardInterrupt the same way) would skip env.close() entirely,
    # leaking the live Mininet network and its iperf3 processes --
    # recoverable only with a manual `sudo mn -c`.
    env.close()
