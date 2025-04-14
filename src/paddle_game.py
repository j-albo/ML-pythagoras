import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class CatchGameEnv(gym.Env):
    def __init__(self, width=400, height=400):
        super().__init__()
        self.width, self.height = width, height
        self.paddle_width = 80
        self.paddle_x = (width - self.paddle_width) // 2
        self.block_x = np.random.randint(0, width - self.paddle_width)
        self.block_y = 0
        self.block_speed = 5
        self.done = False
        
        # Define action & observation spaces
        self.action_space = spaces.Discrete(3)  # Move left, stay, move right
        self.observation_space = spaces.Box(low=0, high=width, shape=(2,), dtype=np.int32)

        # Initialize pygame
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Catch the Falling Block")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        self.paddle_x = (self.width - self.paddle_width) // 2
        self.block_x = np.random.randint(0, self.width - self.paddle_width)
        self.block_y = 0
        self.done = False
        return np.array([self.paddle_x, self.block_x]), {}

    def step(self, action):
        if action == 0:  # Move left
            self.paddle_x = max(0, self.paddle_x - 20)
        if action == 2:  # Move right
            self.paddle_x = min(self.width - self.paddle_width, self.paddle_x + 20)

        self.block_y += self.block_speed  # Block falls down

        # Check if block reaches bottom
        if self.block_y >= self.height - 20:
            reward = 10 if abs(self.paddle_x - self.block_x) < 40 else -10
            self.done = True
        else:
            reward = 0
        
        return np.array([self.paddle_x, self.block_x]), reward, self.done, False, {}

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.paddle_x, self.height - 20, self.paddle_width, 10))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.block_x, self.block_y, 20, 20))
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

env = CatchGameEnv()

q_table = np.zeros((400, 400, 3))  # Q-values for (paddle_x, block_x, action)
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

for episode in range(5000):
    state, _ = env.reset()
    done = False
    
    while not done:
        paddle_x, block_x = state
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[paddle_x, block_x])  # Exploit

        next_state, reward, done, _, _ = env.step(action)
        next_paddle_x, next_block_x = next_state

        # Q-learning update
        q_table[paddle_x, block_x, action] = (1 - alpha) * q_table[paddle_x, block_x, action] + \
            alpha * (reward + gamma * np.max(q_table[next_paddle_x, next_block_x]))

print("Training complete!")

state, _ = env.reset()
done = False

while not done:
    paddle_x, block_x = state
    action = np.argmax(q_table[paddle_x, block_x])  # Use trained policy
    state, reward, done, _, _ = env.step(action)
    env.render()

env.close()