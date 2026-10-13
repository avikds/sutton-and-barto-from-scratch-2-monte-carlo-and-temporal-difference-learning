"""
Sutton and Barto from Scratch 2: Monte Carlo and Temporal-Difference Learning scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Sutton & Barto from scratch 2: Monte Carlo and TD learning demo (chapters 5 and 6 at small scale)."""
import numpy as np


def stick_policy(state, rng):
    return 0 if state[0] >= 20 else 1


def policy_rows(policy, usable_ace):
    rows = []
    for i in range(9, -1, -1):
        row = "".join("S" if policy[i, j, int(usable_ace)] == 0 else "H" for j in range(10))
        rows.append(f"    player {12 + i:2d}: {row}")
    return rows


def greedy_cliff_path(q_table, max_steps=60):
    state, path = 36, [36]
    for _ in range(max_steps):
        state, _, done = cliff_step(state, int(np.argmax(q_table[state])))
        path.append(state)
        if done or state == 36:
            break
    return path


def main():
    # --- Part 1: Monte Carlo prediction on blackjack ---
    print("=== Part 1: MC prediction, stick on 20 or 21 ===")
    rng = np.random.default_rng(0)
    n_pred = 10000
    episodes = [play_blackjack_episode(stick_policy, rng) for _ in range(n_pred)]
    values, counts = mc_prediction_blackjack(episodes, first_visit=True)
    values_ev, counts_ev = mc_prediction_blackjack(episodes, first_visit=False)
    print(f"Episodes: {n_pred}, mean episode length: {np.mean([len(e) for e in episodes]):.3f}")
    print(f"First-visit V(player 20, dealer 10, no usable ace): {values[8, 9, 0]:.4f} from {int(counts[8, 9, 0])} visits")
    print(f"First-visit V(player 13, dealer 2, usable ace):     {values[1, 1, 1]:.4f} from {int(counts[1, 1, 1])} visits")
    print(f"Every-visit table identical to first-visit: {bool(np.allclose(values, values_ev)) and bool(np.array_equal(counts, counts_ev))}")
    print(f"Mean V with usable ace: {values[:, :, 1].mean():.4f} | without: {values[:, :, 0].mean():.4f}")
    print("Player 20 vs dealer A..10 (no usable ace):", np.round(values[8, :, 0], 2).tolist())
    print("Player 15 vs dealer A..10 (no usable ace):", np.round(values[3, :, 0], 2).tolist())

    # --- Part 2: MC control and importance sampling ---
    print("\n=== Part 2: on-policy MC control (epsilon = 0.1) ===")
    rng = np.random.default_rng(1)
    n_ctrl = 50000
    q_table, policy = mc_control_epsilon_soft(n_ctrl, 0.1, rng)
    print(f"Episodes: {n_ctrl}; greedy policy vs dealer A..10 (S = stick, H = hit)")
    print("  no usable ace:")
    for row in policy_rows(policy, False):
        print(row)
    print("  usable ace:")
    for row in policy_rows(policy, True):
        print(row)
    print(f"Q(player 20, dealer 10, no ace): stick {q_table[8, 9, 0, 0]:.4f} | hit {q_table[8, 9, 0, 1]:.4f}")
    print(f"Q(player 16, dealer 10, no ace): stick {q_table[4, 9, 0, 0]:.4f} | hit {q_table[4, 9, 0, 1]:.4f}")

    print("\n--- Off-policy evaluation of (13, 2, usable ace), target = stick on 20/21, behaviour = random ---")
    rng = np.random.default_rng(2)
    ref_returns = []
    for _ in range(50000):
        dealer = [2, draw_card(rng)]
        ref_returns.append(play_blackjack_episode(stick_policy, rng, [1, 2], dealer)[-1][2])
    true_value = float(np.mean(ref_returns))
    print(f"On-policy reference value (50000 episodes): {true_value:.4f}")
    rng = np.random.default_rng(3)
    returns, ratios = off_policy_blackjack_returns(10000, rng)
    ordinary, weighted = importance_sampling_estimates(returns, ratios)
    print(f"Episodes consistent with the target: {int((ratios > 0).sum())} of {len(ratios)}")
    for n in (100, 1000, 10000):
        print(f"  after {n:5d} episodes: ordinary IS {ordinary[n - 1]:+.4f} (err {abs(ordinary[n - 1] - true_value):.4f}) | "
              f"weighted IS {weighted[n - 1]:+.4f} (err {abs(weighted[n - 1] - true_value):.4f})")

    print("\n--- Infinite-variance example (true value 1.0) ---")
    rng = np.random.default_rng(4)
    returns, ratios = infinite_variance_returns(20000, rng)
    ordinary, weighted = importance_sampling_estimates(returns, ratios)
    print(f"Largest importance ratio seen: {ratios.max():.0f}")
    for n in (100, 1000, 5000, 10000, 20000):
        print(f"  after {n:5d} episodes: ordinary IS {ordinary[n - 1]:.4f} | weighted IS {weighted[n - 1]:.4f}")

    # --- Part 3: TD(0) vs MC on the random walk ---
    print("\n=== Part 3: TD(0) vs constant-alpha MC on the random walk ===")
    rng = np.random.default_rng(5)
    n_runs, n_eps = 100, 100
    results = {}
    for method, alpha in (("td", 0.05), ("td", 0.1), ("td", 0.15), ("mc", 0.01), ("mc", 0.02), ("mc", 0.03), ("mc", 0.04)):
        curve = random_walk_rms_curve(method, alpha, n_eps, n_runs, rng)
        results[(method, alpha)] = curve
        print(f"  {method.upper():2s} alpha={alpha:.2f}: RMS after 25 episodes {curve[24]:.4f}, after 100 episodes {curve[99]:.4f}")
    best_td = min(results[k][99] for k in results if k[0] == "td")
    best_mc = min(results[k][99] for k in results if k[0] == "mc")
    print(f"Best TD RMS at 100 episodes {best_td:.4f} < best MC {best_mc:.4f}: {best_td < best_mc}")

    print("\n--- Batch updating on 100 stored episodes ---")
    rng = np.random.default_rng(6)
    stored = [random_walk_episode(rng) for _ in range(100)]
    batch_values = batch_td_prediction(stored, 0.001, 1e-7)
    counts = np.zeros((7, 7))
    reward_sum = np.zeros(7)
    visits = np.zeros(7)
    for ep in stored:
        for s, r, s2 in ep:
            counts[s, s2] += 1
            reward_sum[s] += r
            visits[s] += 1
    P = counts[1:6, 1:6] / visits[1:6, None]
    ce_values = np.linalg.solve(np.eye(5) - P, reward_sum[1:6] / visits[1:6])
    print("Batch TD(0) values A..E:          ", np.round(batch_values[1:6], 4).tolist())
    print("Certainty-equivalence values A..E:", np.round(ce_values, 4).tolist())
    print("True values A..E:                 ", np.round(np.arange(1, 6) / 6.0, 4).tolist())
    print(f"Max |batch TD - certainty equivalence|: {np.max(np.abs(batch_values[1:6] - ce_values)):.6f}")

    # --- Part 4: TD control on cliff walking ---
    print("\n=== Part 4: cliff walking (alpha = 0.5, epsilon = 0.1, 500 episodes, 10 runs) ===")
    rng = np.random.default_rng(7)
    for name, learner in (("Sarsa", sarsa_cliff), ("Q-learning", q_learning_cliff), ("Expected Sarsa", expected_sarsa_cliff)):
        total = np.zeros(500)
        for _ in range(10):
            q_table, episode_rewards = learner(500, 0.5, 0.1, rng)
            total += episode_rewards
        path = greedy_cliff_path(q_table)
        rows = sorted(set(s // 12 for s in path[1:-1]))
        print(f"  {name:15s}: mean reward over last 100 episodes {np.mean(total[-100:] / 10):8.2f} | "
              f"greedy path {len(path) - 1} steps, rows used {rows}, reaches goal {path[-1] == 47}")

    print("\n--- Maximization bias (alpha = 0.1, epsilon = 0.1, 300 runs): fraction of left actions from A ---")
    rng = np.random.default_rng(8)
    double_curve = maximization_bias_experiment(300, 300, 0.1, 0.1, rng, double=True)
    single_curve = maximization_bias_experiment(300, 300, 0.1, 0.1, rng, double=False)
    for n in (1, 10, 25, 50, 100, 200, 300):
        print(f"  episode {n:3d}: Q-learning {single_curve[n - 1]:.3f} | double Q-learning {double_curve[n - 1]:.3f}")
    print(f"Optimal left fraction under epsilon-greedy: {0.05:.3f}")


if __name__ == "__main__":
    main()

