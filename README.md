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
- [x] **9.** importance_sampling_estimates
- [x] **10.** off_policy_blackjack_returns
- [x] **11.** infinite_variance_returns
- [x] **12.** random_walk_episode
- [x] **13.** td0_update
- [x] **14.** constant_alpha_mc_update
- [x] **15.** random_walk_rms_curve
- [x] **16.** batch_td_prediction
- [x] **17.** cliff_step
- [x] **18.** sarsa_cliff
- [x] **19.** q_learning_cliff
- [x] **20.** expected_sarsa_cliff
- [x] **21.** double_q_update
- [x] **22.** maximization_bias_experiment

## Results

```
=== Part 1: MC prediction, stick on 20 or 21 ===
Episodes: 10000, mean episode length: 1.559
First-visit V(player 20, dealer 10, no usable ace): 0.4562 from 559 visits
First-visit V(player 13, dealer 2, usable ace):     -0.1429 from 7 visits
Every-visit table identical to first-visit: True
Mean V with usable ace: -0.1041 | without: -0.3696
Player 20 vs dealer A..10 (no usable ace): [0.12, 0.63, 0.6, 0.63, 0.67, 0.77, 0.71, 0.81, 0.77, 0.46]
Player 15 vs dealer A..10 (no usable ace): [-0.68, -0.68, -0.6, -0.64, -0.63, -0.55, -0.57, -0.85, -0.6, -0.66]

=== Part 2: on-policy MC control (epsilon = 0.1) ===
Episodes: 50000; greedy policy vs dealer A..10 (S = stick, H = hit)
  no usable ace:
    player 21: SSSSSSSSSS
    player 20: SSSSSSSSSS
    player 19: SSSSSSSSSS
    player 18: SSSSSSSSSS
    player 17: HSSSSSSSSS
    player 16: SSSSSSHSSS
    player 15: HHHSSSSSHH
    player 14: SSSSHHHSHH
    player 13: HSHHSSHHHH
    player 12: HSSSSSHHHH
  usable ace:
    player 21: SSSSSSSSSS
    player 20: SSHSSSSSSS
    player 19: HSSSHSSSSS
    player 18: HHHSSSSSHH
    player 17: SHHSSSHHHH
    player 16: SSSHHSHSHH
    player 15: SHHSSSHHHH
    player 14: HHHSHHSSSH
    player 13: SHHSSSHHHH
    player 12: HSHHSHHHHH
Q(player 20, dealer 10, no ace): stick 0.4352 | hit -0.8350
Q(player 16, dealer 10, no ace): stick -0.5849 | hit -0.5935

--- Off-policy evaluation of (13, 2, usable ace), target = stick on 20/21, behaviour = random ---
On-policy reference value (50000 episodes): -0.2839
Episodes consistent with the target: 1375 of 10000
  after   100 episodes: ordinary IS -0.2800 (err 0.0039) | weighted IS -0.2800 (err 0.0039)
  after  1000 episodes: ordinary IS -0.1240 (err 0.1599) | weighted IS -0.1235 (err 0.1604)
  after 10000 episodes: ordinary IS -0.2792 (err 0.0047) | weighted IS -0.2933 (err 0.0094)

--- Infinite-variance example (true value 1.0) ---
Largest importance ratio seen: 4096
  after   100 episodes: ordinary IS 0.1000 | weighted IS 1.0000
  after  1000 episodes: ordinary IS 0.7300 | weighted IS 1.0000
  after  5000 episodes: ordinary IS 1.3640 | weighted IS 1.0000
  after 10000 episodes: ordinary IS 1.0032 | weighted IS 1.0000
  after 20000 episodes: ordinary IS 0.7922 | weighted IS 1.0000

=== Part 3: TD(0) vs constant-alpha MC on the random walk ===
  TD alpha=0.05: RMS after 25 episodes 0.1101, after 100 episodes 0.0322
  TD alpha=0.10: RMS after 25 episodes 0.0588, after 100 episodes 0.0553
  TD alpha=0.15: RMS after 25 episodes 0.0595, after 100 episodes 0.0692
  MC alpha=0.01: RMS after 25 episodes 0.1879, after 100 episodes 0.0955
  MC alpha=0.02: RMS after 25 episodes 0.1503, after 100 episodes 0.0798
  MC alpha=0.03: RMS after 25 episodes 0.1433, after 100 episodes 0.0923
  MC alpha=0.04: RMS after 25 episodes 0.1296, after 100 episodes 0.1028
Best TD RMS at 100 episodes 0.0322 < best MC 0.0798: True

--- Batch updating on 100 stored episodes ---
Batch TD(0) values A..E:           [0.1612, 0.3256, 0.49, 0.6737, 0.832]
Certainty-equivalence values A..E: [0.1612, 0.3256, 0.49, 0.6738, 0.832]
True values A..E:                  [0.1667, 0.3333, 0.5, 0.6667, 0.8333]
Max |batch TD - certainty equivalence|: 0.000004

=== Part 4: cliff walking (alpha = 0.5, epsilon = 0.1, 500 episodes, 10 runs) ===
  Sarsa          : mean reward over last 100 episodes   -28.42 | greedy path 17 steps, rows used [0, 1, 2], reaches goal True
  Q-learning     : mean reward over last 100 episodes   -47.99 | greedy path 13 steps, rows used [2], reaches goal True
  Expected Sarsa : mean reward over last 100 episodes   -20.94 | greedy path 15 steps, rows used [1, 2], reaches goal True

--- Maximization bias (alpha = 0.1, epsilon = 0.1, 300 runs): fraction of left actions from A ---
  episode   1: Q-learning 0.970 | double Q-learning 0.967
  episode  10: Q-learning 0.957 | double Q-learning 0.270
  episode  25: Q-learning 0.903 | double Q-learning 0.163
  episode  50: Q-learning 0.800 | double Q-learning 0.113
  episode 100: Q-learning 0.427 | double Q-learning 0.127
  episode 200: Q-learning 0.133 | double Q-learning 0.047
  episode 300: Q-learning 0.100 | double Q-learning 0.050
Optimal left fraction under epsilon-greedy: 0.050
```
