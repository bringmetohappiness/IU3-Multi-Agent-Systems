#!/usr/bin/env python
"""Удобный скрипт для изучения доступных сред Gym."""

import gym

ENV_NAME = 'Phoenix-v0'
NUM_STEPS = 2000


env = gym.make(ENV_NAME)
env.reset()
for _ in range(NUM_STEPS):
    env.render()
    env.step(env.action_space.sample())
env.close()
