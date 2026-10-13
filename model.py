"""
Sutton and Barto from Scratch 2: Monte Carlo and Temporal-Difference Learning

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - draw_card
def draw_card(rng):
    # Draw one card rank from 1 to 13 using the provided NumPy generator.
    # Ace = 1, numbered cards = 2–10, and face cards = 10.
    rank = rng.integers(1, 14)

    # Clip face cards (11, 12, 13) to 10 and return a plain Python int.
    return int(min(rank, 10))

# Step 2 - hand_value
def hand_value(cards):
    # Calculate the hand total with all aces initially counted as 1.
    total = sum(cards)

    # If the hand contains an ace, count one ace as 11 when it does not bust.
    if 1 in cards and total + 10 <= 21:
        return (int(total + 10), True)

    # Otherwise, keep all aces counted as 1.
    return (int(total), False)

# Step 3 - blackjack_step (not yet solved)
# TODO: implement

# Step 4 - play_blackjack_episode (not yet solved)
# TODO: implement

# Step 5 - episode_returns (not yet solved)
# TODO: implement

# Step 6 - mc_prediction_blackjack (not yet solved)
# TODO: implement

# Step 7 - epsilon_soft_action (not yet solved)
# TODO: implement

# Step 8 - mc_control_epsilon_soft (not yet solved)
# TODO: implement

# Step 9 - importance_sampling_estimates (not yet solved)
# TODO: implement

# Step 10 - off_policy_blackjack_returns (not yet solved)
# TODO: implement

# Step 11 - infinite_variance_returns (not yet solved)
# TODO: implement

# Step 12 - random_walk_episode (not yet solved)
# TODO: implement

# Step 13 - td0_update (not yet solved)
# TODO: implement

# Step 14 - constant_alpha_mc_update (not yet solved)
# TODO: implement

# Step 15 - random_walk_rms_curve (not yet solved)
# TODO: implement

# Step 16 - batch_td_prediction (not yet solved)
# TODO: implement

# Step 17 - cliff_step (not yet solved)
# TODO: implement

# Step 18 - sarsa_cliff (not yet solved)
# TODO: implement

# Step 19 - q_learning_cliff (not yet solved)
# TODO: implement

# Step 20 - expected_sarsa_cliff (not yet solved)
# TODO: implement

# Step 21 - double_q_update (not yet solved)
# TODO: implement

# Step 22 - maximization_bias_experiment (not yet solved)
# TODO: implement

