import os
import pickle
import time
import numpy as np
from tqdm import tqdm
import gym

ENV_NAME = 'FrozenLake8x8-v1'
# параметры обучения
NUM_EPISODES = 20_000
MAX_STEPS = 100
# параметры эпсилон-жадности
EPSILON = 1
MIN_EPSILON = 0.001
EPSILON_DECAY_RATE = 1 / NUM_EPISODES
# RL-параметры
LR_RATE = 0.5
GAMMA = 0.9


class Agent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    # статический метод?
    def learn(self, num_episodes, max_steps, epsilon, min_epsilon, epsilon_decay_rate, lr_rate, gamma):
        os.system('cls')  # очищает терминал
        print(f'Обучение в {str(self.env.spec)[8:-1]}...\n')

        def epsilon_greedy(state, eps):
            if np.random.uniform(0, 1) < eps:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.q_table[state, :])
            return action

        def decrease_epsilon(eps, min_eps, eps_decay_rate):
            if eps > min_eps:
                eps -= eps_decay_rate
            else:
                eps = min_eps
            return eps

        def update_q_table(state, new_state, reward, action, lr_rate, gamma):
            self.q_table[state, action] = self.q_table[state, action] + lr_rate * (reward + gamma * np.max(self.q_table[new_state, :]) - self.q_table[state, action])

        for episode in tqdm(range(num_episodes), ascii=True, unit='episode'):
            state = self.env.reset()

            step = 0
            while step < max_steps:
                action = epsilon_greedy(state, epsilon)
                new_state, reward, done, info = self.env.step(action)

                if done and (reward == 0):    # провалился в прорубь
                    reward = -10
                elif done and (reward == 1):  # добрался до фрисби
                    reward = 100
                else:                         # наступил на лёд
                    reward = -1

                update_q_table(state, new_state, reward, action, lr_rate, gamma)
                state = new_state
                step += 1

                if done:
                    break

            epsilon = decrease_epsilon(epsilon, min_epsilon, epsilon_decay_rate)

    def test(self, num_episodes, max_steps, show=True):
        print(f'\nPlaying {str(self.env.spec)[8:-1]}...\n')

        wins = 0
        defeats = 0

        for episode in range(num_episodes):
            state = self.env.reset()

            step = 0
            while step < max_steps:
                action = np.argmax(self.q_table[state, :])
                new_state, reward, done, info = self.env.step(action)
                # Windows CLI visualization
                if show:
                    os.system('cls')
                    try:
                        win_rate = wins / (wins + defeats) * 100
                        print(f'Win rate: {win_rate}%')
                        print(f'Wins: {wins}\n')
                        print(f'Defeats: {defeats}\n')
                    except ZeroDivisionError:
                        print('Win rate: 0.0%')
                        print(f'Wins: {wins}\n')
                        print(f'Defeats: {defeats}\n')
                    # show env
                    self.env.render()
                    if done:
                        print('\n\tWIN' if reward == 1 else '\n\tDEFEAT')
                        time.sleep(0.6)

                state = new_state

                if done:
                    if reward == 1:
                        wins += 1
                    else:
                        defeats += 1
                    break

                step += 1

            if show and (step >= MAX_STEPS):
                print('\nTIME IS OVER (steps > 100)')
                time.sleep(1)

        win_rate = wins / (wins + defeats) * 100
        print(f'Win rate: {win_rate}%')

    # статический метод?
    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self.q_table, file)

    # статический метод?
    def load(self, file_name):
        with open(file_name, 'rb') as file:
            self.q_table = pickle.load(file)


def main():
    environment = gym.make(ENV_NAME)

    agent = Agent(environment)
    agent.learn(NUM_EPISODES, MAX_STEPS, EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE, LR_RATE, GAMMA)
    agent.test(NUM_EPISODES, MAX_STEPS)


if __name__ == '__main__':
    main()
