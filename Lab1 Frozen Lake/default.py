#!/usr/bin/env python3
import gym
import sys
import frozen_lake_agents as agents

# параметры обучения
TOTAL_EPISODES = 20_000
MAX_STEPS = 100
# параметры эпсилон-жадности
EPSILON = 1
MIN_EPSILON = 0.001
EPSILON_DECAY_RATE = 1 / TOTAL_EPISODES
# RL-параметры
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.9


env_name = sys.argv[1]
environment = gym.make(env_name)

agent = agents.EpsilonGreedyAgent(environment, EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE)
agent.learn(TOTAL_EPISODES, MAX_STEPS, LEARNING_RATE, DISCOUNT_FACTOR, show=True)
agent.test(TOTAL_EPISODES, MAX_STEPS, show=True)
