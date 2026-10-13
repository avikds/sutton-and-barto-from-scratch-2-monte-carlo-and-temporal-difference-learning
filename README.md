# Sutton and Barto from Scratch 2: Monte Carlo and Temporal-Difference Learning

Implement the model-free prediction and control methods of Sutton and Barto chapters 5 and 6. Build a blackjack simulator and estimate state values with first-visit and every-visit Monte Carlo, learn a blackjack policy with on-policy epsilon-soft MC control, evaluate a target policy off-policy with ordinary and weighted importance sampling, compare TD(0) with constant-alpha MC on the random walk, then learn cliff walking with Sarsa, Q-learning and Expected Sarsa and remove maximization bias with double Q-learning.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** draw_card
- [x] **2.** hand_value
- [x] **3.** blackjack_step
- [x] **4.** play_blackjack_episode
- [x] **5.** episode_returns
- [x] **6.** mc_prediction_blackjack
- [x] **7.** epsilon_soft_action
- [x] **8.** mc_control_epsilon_soft
- [ ] **9.** importance_sampling_estimates
- [ ] **10.** off_policy_blackjack_returns
- [ ] **11.** infinite_variance_returns
- [ ] **12.** random_walk_episode
- [ ] **13.** td0_update
- [ ] **14.** constant_alpha_mc_update
- [ ] **15.** random_walk_rms_curve
- [ ] **16.** batch_td_prediction
- [ ] **17.** cliff_step
- [ ] **18.** sarsa_cliff
- [ ] **19.** q_learning_cliff
- [ ] **20.** expected_sarsa_cliff
- [ ] **21.** double_q_update
- [ ] **22.** maximization_bias_experiment

---

Built on Deep-ML.
