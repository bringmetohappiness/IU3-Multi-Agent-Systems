##################
# LEARNING STAGE #
##################

import gym
import numpy as np
import pickle
import os
from tqdm import tqdm
from time import sleep

# loading Frozen Lake env from gym
env = gym.make('FrozenLake-v0')

os.system('cls')
print(f"\nLearning in {str(env.spec)[8:-1]}...\n")

# constant parameters
TOTAL_EPISODES = 20_000
MAX_STEPS = 100
EPSILON_DECAY_RATE = 1 / TOTAL_EPISODES
MIN_EPSILON = 0.001

# RL parameters
gamma = 0.9
lr_rate = 0.5
epsilon = 1

# Q-table initialization
Q = np.zeros((env.observation_space.n, env.action_space.n))


def epsilon_greedy(state):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action


def learn(state, state2, reward, action):
    # same as formula for Q-learning
    # Q_new[s, a] <- Q_old[s, a] + alpha * (r + gamma * max_a(Q_old[s_new, a]) - Q_old[s, a])

    Q[state, action] = Q[state, action] + lr_rate * \
                       (reward + gamma * np.max(Q[state2, :]) - Q[state, action])


# Start
for episode in tqdm(range(TOTAL_EPISODES), ascii=True, unit="episode"):
    state = env.reset()
    step = 0

    # decreasing epsilon
    if epsilon > MIN_EPSILON:
        epsilon -= EPSILON_DECAY_RATE
    else:
        epsilon = MIN_EPSILON

    # loop within episode
    while step < MAX_STEPS:
        action = epsilon_greedy(state)
        new_state, reward, done, info = env.step(action)

        # debug to see what's happening ("Ctrl + /" - uncomment highlighted code)

        # print("state --action--> new_state")
        # print("  {}       {}         {}".format(state, action, new_state))
        # env.render()

        if done and (reward == 0):
            reward = -10  # fell into the hole
        elif done and (reward == 1):
            reward = 100  # goal achieved
        else:
            reward = -1  # step on ice

        # doing the learning
        learn(state, new_state, reward, action)
        state = new_state
        step += 1

        if done:
            break

# print("\nQ-table:\n", Q)

# save Q-table in file on drive (same directory)
with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)


#################
# PLAYING STAGE #
#################

# comment all below, if you only need to train agent

print(f"\nPlaying {str(env.spec)[8:-1]}...\n")

# load q-table from file
with open("frozenLake_qTable.pkl", 'rb') as f:
    Q = pickle.load(f)

win = 0
defeat = 0

# for visualization on win cmd
show_play = True

# Start
for episode in range(1000):
    state = env.reset()
    step = 0

    while step < MAX_STEPS:
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)

        # Windows CLI visualization
        if show_play:
            os.system('cls')

            try:
                win_rate = win / (win + defeat) * 100
                print("Win rate: {}%".format(win_rate))
                print(f"Wins: {win}\n"
                      f"Defeats: {defeat}\n")
            except ZeroDivisionError:
                print("Win rate: 0.0%")
                print(f"Wins: {win}\n"
                      f"Defeats: {defeat}\n")

            env.render()  # show env

            if done:
                print("\n\tWIN" if reward == 1 else "\n\tDEFEAT")
                sleep(0.6)

        state = new_state

        if done:
            if reward == 1:
                win += 1
            else:
                defeat += 1

            break

        step += 1

    if show_play and (step >= MAX_STEPS):
        print("\nTIME IS OVER (steps > 100)")
        sleep(1)

win_rate = win / (win + defeat) * 100
print("Win rate: {}%".format(win_rate))
