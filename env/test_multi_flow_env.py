from multi_flow_congestion_env import MultiFlowCongestionEnv

# sender0 alternates decrease/increase (stand-in for an active policy),
# sender1 always maintains (stand-in for a Cubic-baseline arm) -- this
# script doesn't know about PPO/hybrid/human policies at all, same as
# MultiFlowCongestionEnv itself: it only applies actions it's given.
env = MultiFlowCongestionEnv(n_senders=2, bw=10, delay="20ms", max_steps=40)

try:
    results = env.start()
    for i, (obs, info) in enumerate(results):
        print(f"sender{i} initial state:", obs, info)

    for step in range(6):
        actions = [2 if step % 2 == 0 else 0, 1]
        results = env.step(actions)
        for i, (obs, reward, terminated, truncated, info) in enumerate(results):
            print(
                "step", step + 1, "sender", i,
                "action:", actions[i], "state:", obs,
                "reward:", round(reward, 3), "rate:", info["rate_mbps"],
            )

    print("--- lowering bottleneck bandwidth to 3Mbps mid-run ---")
    env.set_bandwidth(3)

    for step in range(6, 12):
        actions = [2 if step % 2 == 0 else 0, 1]
        results = env.step(actions)
        for i, (obs, reward, terminated, truncated, info) in enumerate(results):
            print(
                "step", step + 1, "sender", i,
                "action:", actions[i], "state:", obs,
                "reward:", round(reward, 3), "rate:", info["rate_mbps"],
            )

    print("--- injecting a 4s/8Mbps traffic burst ---")
    env.inject_burst(duration_s=4, rate_mbps=8)

    for step in range(12, 18):
        actions = [1, 1]
        results = env.step(actions)
        for i, (obs, reward, terminated, truncated, info) in enumerate(results):
            print(
                "step", step + 1, "sender", i,
                "action:", actions[i], "state:", obs,
                "reward:", round(reward, 3), "rate:", info["rate_mbps"],
            )
finally:
    # Same reasoning as env/test_real_env.py: without this, an exception
    # mid-loop (or Ctrl-C) would skip env.close() and leak the live
    # Mininet network + iperf3 processes.
    env.close()
