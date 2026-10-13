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

# Step 3 - blackjack_step
def blackjack_step(player_cards, dealer_cards, action, rng):
    # Create new lists so the input hands are never modified.
    player_cards = list(player_cards)
    dealer_cards = list(dealer_cards)

    # Action 1: Player hits and draws one card.
    if action == 1:
        player_cards.append(draw_card(rng))

        player_total, _ = hand_value(player_cards)

        # The player busts if the total exceeds 21.
        if player_total > 21:
            return player_cards, dealer_cards, -1.0, True

        # The player can continue playing.
        return player_cards, dealer_cards, 0.0, False

    # Action 0: Player sticks, so the dealer plays.
    while hand_value(dealer_cards)[0] < 17:
        dealer_cards.append(draw_card(rng))

    player_total, _ = hand_value(player_cards)
    dealer_total, _ = hand_value(dealer_cards)

    # Determine the outcome of the game.
    if dealer_total > 21 or player_total > dealer_total:
        reward = 1.0
    elif player_total == dealer_total:
        reward = 0.0
    else:
        reward = -1.0

    # Sticking ends the episode.
    return player_cards, dealer_cards, float(reward), True

# Step 4 - play_blackjack_episode
def play_blackjack_episode(policy, rng, player_cards=None, dealer_cards=None):
    # Deal the player's initial two cards if no hand is provided.
    if player_cards is None:
        player_cards = [draw_card(rng), draw_card(rng)]
    else:
        # Copy the provided hand to avoid modifying the input list.
        player_cards = list(player_cards)

    # Automatically hit until the player's hand value is at least 12.
    while hand_value(player_cards)[0] < 12:
        player_cards.append(draw_card(rng))

    # Deal the dealer's initial two cards if no hand is provided.
    if dealer_cards is None:
        dealer_cards = [draw_card(rng), draw_card(rng)]
    else:
        # Copy the provided hand to avoid modifying the input list.
        dealer_cards = list(dealer_cards)

    # Record each step as (state, action, reward).
    episode = []
    done = False

    while not done:
        # Construct the state before the player takes an action.
        player_total, usable_ace = hand_value(player_cards)
        state = (int(player_total), int(dealer_cards[0]), bool(usable_ace))

        # Select the player's action using the supplied policy.
        action = policy(state, rng)

        # Apply the action and obtain the updated hands and outcome.
        player_cards, dealer_cards, reward, done = blackjack_step(
            player_cards, dealer_cards, action, rng
        )

        # Store the state, action, and immediate reward.
        episode.append((state, action, float(reward)))

    return episode

# Step 5 - episode_returns
import numpy as np

def episode_returns(rewards, gamma=1.0):
    # Convert rewards to a sequence of floating-point values.
    rewards = np.asarray(rewards, dtype=float)

    # Initialize an array to store the discounted returns.
    T = len(rewards)
    returns = np.empty(T, dtype=float)

    # Compute returns using a single backward pass.
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = float(rewards[t]) + gamma * G
        returns[t] = G

    return returns

# Step 6 - mc_prediction_blackjack
def mc_prediction_blackjack(episodes, first_visit=True):
    # Initialize value and visit-count tables.
    # Indices: [player_sum - 12, dealer_showing - 1, int(usable_ace)]
    values = np.zeros((10, 10, 2), dtype=float)
    counts = np.zeros((10, 10, 2), dtype=float)

    # Process each episode independently.
    for episode in episodes:
        if not episode:
            continue

        # Extract rewards and calculate undiscounted returns.
        rewards = [transition[2] for transition in episode]
        returns = episode_returns(rewards, gamma=1.0)

        # Track visited states for first-visit Monte Carlo.
        visited = set()

        # Update each state's value using the incremental sample average.
        for t, (state, action, reward) in enumerate(episode):
            player_sum, dealer_showing, usable_ace = state

            # Ignore states outside the supported value-table range.
            if not (12 <= player_sum <= 21 and 1 <= dealer_showing <= 10):
                continue

            index = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace)
            )

            # In first-visit MC, update a state only at its first occurrence.
            if first_visit and index in visited:
                continue

            visited.add(index)

            # Increment the number of returns observed for this state.
            counts[index] += 1.0

            # Incremental sample-average update.
            values[index] += (
                returns[t] - values[index]
            ) / counts[index]

    return values, counts

# Step 7 - epsilon_soft_action
def epsilon_soft_action(q_values, epsilon, rng):
    # Draw once to determine whether to explore or exploit.
    u = rng.random()

    if u < epsilon:
        # Explore: choose an action uniformly at random.
        return int(rng.integers(0, len(q_values)))

    # Exploit: choose the greedy action.
    # np.argmax selects the smallest index when values are tied.
    return int(np.argmax(q_values))

# Step 8 - mc_control_epsilon_soft
def mc_control_epsilon_soft(n_episodes, epsilon, rng):
    # Initialize action-value and visit-count tables.
    # Indices: [player_sum - 12, dealer_showing - 1, usable_ace, action]
    q_table = np.zeros((10, 10, 2, 2), dtype=float)
    counts = np.zeros((10, 10, 2, 2), dtype=float)

    # Run the required number of episodes.
    for _ in range(n_episodes):

        # Select actions using the current, continuously updated Q-table.
        def policy(state, r):
            player_sum, dealer_showing, usable_ace = state

            index = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace)
            )

            return epsilon_soft_action(q_table[index], epsilon, r)

        # Generate an episode using the current epsilon-soft policy.
        episode = play_blackjack_episode(policy, rng)

        # Compute undiscounted returns for every time step.
        rewards = [transition[2] for transition in episode]
        returns = episode_returns(rewards, gamma=1.0)

        # Track state-action pairs already encountered in this episode.
        visited = set()

        # Apply first-visit Monte Carlo updates.
        for t, (state, action, reward) in enumerate(episode):
            player_sum, dealer_showing, usable_ace = state

            index = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace)
            )

            action = int(action)
            state_action = (state, action)

            # Skip repeated occurrences of the same state-action pair.
            if state_action in visited:
                continue

            visited.add(state_action)

            # Update the visit count and action-value estimate.
            q_index = index + (action,)
            counts[q_index] += 1.0

            q_table[q_index] += (
                returns[t] - q_table[q_index]
            ) / counts[q_index]

    # Derive the greedy policy from the learned action values.
    policy = np.argmax(q_table, axis=3).astype(int)

    return q_table, policy

# Step 9 - importance_sampling_estimates
def importance_sampling_estimates(returns, ratios):
    # Convert inputs to NumPy float arrays without modifying the originals.
    returns = np.asarray(returns, dtype=float)
    ratios = np.asarray(ratios, dtype=float)

    # Compute cumulative weighted returns and cumulative importance ratios.
    weighted_returns = np.cumsum(ratios * returns)
    cumulative_ratios = np.cumsum(ratios)

    n = len(returns)

    # Ordinary importance sampling: divide by the number of samples.
    ordinary = weighted_returns / np.arange(1, n + 1, dtype=float)

    # Weighted importance sampling: divide by cumulative ratios.
    # Return 0.0 wherever the cumulative ratio sum is zero.
    weighted = np.zeros(n, dtype=float)
    np.divide(
        weighted_returns,
        cumulative_ratios,
        out=weighted,
        where=(cumulative_ratios != 0)
    )

    return ordinary.astype(float), weighted.astype(float)

# Step 10 - off_policy_blackjack_returns
def off_policy_blackjack_returns(n_episodes, rng):
    # Initialize arrays for episode returns and importance-sampling ratios.
    returns = np.zeros(n_episodes, dtype=float)
    ratios = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        # Deal the dealer's showing card and hole card.
        dealer_cards = [2, draw_card(rng)]

        # Behaviour policy: choose stick (0) or hit (1) uniformly at random.
        def behaviour_policy(state, r):
            return int(r.integers(0, 2))

        # Start from player state (13, 2, usable ace).
        episode = play_blackjack_episode(
            behaviour_policy,
            rng,
            player_cards=[1, 2],
            dealer_cards=dealer_cards
        )

        # The episode return is the final reward.
        returns[i] = float(episode[-1][2])

        # The importance ratio is nonzero only if every action agrees
        # with the deterministic target policy.
        matches_target = True

        for state, action, reward in episode:
            player_sum = state[0]

            # Target policy: stick at 20 or 21; hit otherwise.
            target_action = 0 if player_sum in (20, 21) else 1

            if action != target_action:
                matches_target = False
                break

        # Each matching action contributes pi(a|s) / b(a|s) = 1 / 0.5 = 2.
        if matches_target:
            ratios[i] = 2.0 ** len(episode)
        else:
            ratios[i] = 0.0

    return returns, ratios

# Step 11 - infinite_variance_returns
def infinite_variance_returns(n_episodes, rng):
    # Initialize arrays for episode returns and importance-sampling ratios.
    returns = np.zeros(n_episodes, dtype=float)
    ratios = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        n_left = 0

        while True:
            # Choose an action uniformly at random:
            # 0 = left, 1 = right.
            action = int(rng.integers(0, 2))

            # Choosing right terminates the episode with reward 0.
            if action == 1:
                returns[i] = 0.0
                ratios[i] = 0.0
                break

            # Choosing left returns to the same state or terminates
            # successfully with probability 0.1.
            n_left += 1

            if rng.random() < 0.1:
                returns[i] = 1.0

                # The target always chooses left, while the behaviour
                # policy chooses left with probability 0.5.
                ratios[i] = 2.0 ** n_left
                break

            # Otherwise, continue the episode in the same state.

    return returns, ratios

# Step 12 - random_walk_episode
def random_walk_episode(rng):
    # Start at the center state C (state 3).
    state = 3
    episode = []

    # Continue until reaching either terminal state 0 or 6.
    while state not in (0, 6):
        # Move right with probability 0.5; otherwise, move left.
        u = rng.random()
        if u < 0.5:
            next_state = state + 1
        else:
            next_state = state - 1

        # Reward is 1.0 only when entering terminal state 6.
        reward = 1.0 if next_state == 6 else 0.0

        # Record the transition as (state, reward, next_state).
        episode.append((state, float(reward), next_state))

        # Advance to the next state.
        state = next_state

    return episode

# Step 13 - td0_update
def td0_update(values, transitions, alpha, gamma=1.0):
    # Create a copy so the original value table remains unchanged.
    updated_values = np.array(values, dtype=float, copy=True)

    # Process transitions in chronological order, using the latest values.
    for state, reward, next_state in transitions:
        # Calculate the TD(0) target using the current next-state estimate.
        target = reward + gamma * updated_values[next_state]

        # Compute the temporal-difference error.
        td_error = target - updated_values[state]

        # Update the current state's value estimate.
        updated_values[state] += alpha * td_error

    return updated_values

# Step 14 - constant_alpha_mc_update
def constant_alpha_mc_update(values, transitions, alpha, gamma=1.0):
    # Create a copy so the original value table is not modified.
    updated_values = np.array(values, dtype=float, copy=True)

    # Extract rewards and compute the discounted return from every time step.
    rewards = [reward for state, reward, next_state in transitions]
    returns = episode_returns(rewards, gamma=gamma)

    # Apply constant-alpha Monte Carlo updates for every visit,
    # processing transitions in chronological order.
    for t, (state, reward, next_state) in enumerate(transitions):
        G = returns[t]

        # Update the state's value toward the return from this time step.
        updated_values[state] += alpha * (G - updated_values[state])

    return updated_values

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

