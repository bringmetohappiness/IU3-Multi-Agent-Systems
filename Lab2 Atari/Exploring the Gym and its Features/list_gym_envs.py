#!/usr/bin/env python
"""Удобный скрипт для вывода списка всех доступных сред Gym."""

from gym import envs

env_names = [spec.id for spec in envs.registry.all()]
for name in sorted(env_names):
    print(name)
