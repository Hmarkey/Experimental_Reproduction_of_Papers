import numpy as np
import os, sys

BASEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEPATH)
# print(BASEPATH)

class Gridworld:
    def __init__(self, width, height, obstacles, start, goal):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state
    
    def is_hit(self, x, y):
        hit = False
        if x < 0 or x > self.width - 1 or y < 0 or y > self.height - 1:
            # print("hit boundry")
            hit = True
        if (x, y) in self.obstacles:
            # print("hit obstacles")
            hit = True
        return hit

    def step(self, action):
        x, y = self.state
        hit = False
        if action == 'left':
            x -= 1
        elif action == 'right':
            x += 1
        elif action == 'up':
            y -= 1
        elif action == 'down':
            y += 1
        
        if self.is_hit(x, y):
            hit = True
            x, y = self.state
                   
        next_state = (x, y)
        self.state = next_state
        reward = self.get_reward(next_state, hit)
        done = (self.state == self.goal)
        
        return next_state, reward, done, hit

    def get_reward(self, state, hit): # 3.00420971 -2.77342725 -0.06027737
        if state == self.goal:
            return 0.95145913  # Reward for reaching the goal
        elif hit:
            return -1.21074378
        else:
            return -0.00276116  # Small penalty for each step

    def get_valid_actions(self):
        return ['left', 'right', 'up', 'down', 'stand-still']

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.97):
        self.env = env
        self.q_table = np.zeros((env.width, env.height, len(env.get_valid_actions())))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.get_valid_actions())
        else:
            x, y = state
            return self.env.get_valid_actions()[np.argmax(self.q_table[x, y])]

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_index = self.env.get_valid_actions().index(action)
        best_next_action = np.argmax(self.q_table[next_x, next_y])
        td_target = reward + self.discount_factor * self.q_table[next_x, next_y, best_next_action]
        self.q_table[x, y, action_index] += self.learning_rate * (td_target - self.q_table[x, y, action_index])

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def train(self, episodes=100):
        total_steps = 0
        for episode in range(episodes):
            state = self.env.reset()
            steps = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, hit = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                steps += 1
            total_steps += steps
            print(f"episode {episode} step {steps}")
            self.update_epsilon()
        print(f"avg total steps {total_steps / episodes}")

if __name__ == "__main__":
    env = Gridworld(7, 6, [(4, 0), (4, 1), (4, 3), (4, 4), (4,5),(4,6)], (5, 0), (0, 3))
    agent = QLearningAgent(env)
    agent.train()

