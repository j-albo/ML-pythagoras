import gymnasium as gym
import numpy as np
import random
import time

env = gym.make("LunarLander-v3", render_mode="human")

# Discretization settings
num_bins = 10  # Fewer bins = faster learning, but less precision
state_bins = [np.linspace(low, high, num_bins) for low, high in zip(env.observation_space.low, env.observation_space.high)]

# Q-table initialization
q_table = np.zeros([num_bins] * len(env.observation_space.low) + [env.action_space.n])

def discretize_state(state):
    """Convert continuous state into discrete bins."""
    return tuple(np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state)))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay factor
min_epsilon = 0.01
num_episodes = 50000  # Increase for better learning

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    done = False
    total_reward = 0

    while not done:
        # Choose action (exploration-exploitation tradeoff)
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit best known action

        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * q_table[next_state][best_next_action])

        state = next_state
        total_reward += reward

    # Decay exploration rate
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 1000 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

print("Training complete!")

state, _ = env.reset()
state = discretize_state(state)
done = False

while not done:
    action = np.argmax(q_table[state])  # Select best action
    state, reward, done, _, _ = env.step(action)
    state = discretize_state(state)
    env.render()
    time.sleep(0.01)  # Slow down rendering for visibility

env.close()
