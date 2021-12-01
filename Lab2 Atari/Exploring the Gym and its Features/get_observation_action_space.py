#!/usr/bin/env python
"""Удобный скрипт для исследования пространств сред Gym."""

import gym
from gym.spaces import *

ENV_NAME = 'Phoenix-v0'


def print_spaces(space):
    print(space)
    if isinstance(space, Box):  # Печатаем нижнюю и верхнюю границу, если это пространство Box
        print(f'\n space.low: {space.low}')
        print(f'\n space.high: {space.high}')


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    print('Размерность наблюдения:')
    print_spaces(env.observation_space)
    print('Размерность действий:')
    print_spaces(env.action_space)
    try:
        print(f'Описание/значение действий: {env.unwrapped.get_action_meanings()}')
    except AttributeError:
        pass
