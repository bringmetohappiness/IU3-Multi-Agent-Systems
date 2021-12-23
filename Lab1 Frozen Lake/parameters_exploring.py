import numpy as np
import matplotlib.pyplot as plt
import gym
import frozen_lake_agents as agents

ENV_NAME = 'FrozenLake8x8-v1'
# параметры обучения
TOTAL_EPISODES = 10_000
MAX_STEPS = 100
# параметры эпсилон-жадности
EPSILON = 1
MIN_EPSILON = 0.001
EPSILON_DECAY_RATE = 1 / TOTAL_EPISODES
# RL-параметры
LEARNING_RATE = 0.5
DISCOUNT_FACTOR = 0.9
# количество замеров при одних и тех же параметрах
NUM_SAMPLES = 5


environment = gym.make(ENV_NAME)

# график для learning_rate
lr_rates = [
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
]
lr_rate_win_rates = dict()
for lr_rate in lr_rates:
    for _ in range(NUM_SAMPLES):
        agent = agents.EpsilonGreedyAgent(environment, EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE)
        agent.learn(TOTAL_EPISODES, MAX_STEPS, lr_rate, DISCOUNT_FACTOR)
        win_rate = agent.test(TOTAL_EPISODES, MAX_STEPS)
        lr_rate_win_rates.setdefault(lr_rate, []).append(win_rate)
for lr_rate in lr_rates:
    win_rates = lr_rate_win_rates[lr_rate]
    plt.scatter([lr_rate] * len(win_rates), win_rates)
win_rates_mean = np.array([np.mean(v) for k, v in sorted(lr_rate_win_rates.items())])
win_rates_std = np.array([np.std(v) for k, v in sorted(lr_rate_win_rates.items())])
plt.errorbar(lr_rates, win_rates_mean, yerr=win_rates_std)
plt.title('Зависимость win_rate от learning_rate')
plt.xlabel('learning_rate')
plt.xticks(lr_rates)
plt.ylabel('win_rate')
plt.show()

# график для DISCOUNT_FACTOR
discount_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
discount_factor_win_rates = dict()
for discount_factor in discount_factors:
    for _ in range(NUM_SAMPLES):
        agent = agents.EpsilonGreedyAgent(environment, EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE)
        agent.learn(TOTAL_EPISODES, MAX_STEPS, LEARNING_RATE, discount_factor)
        win_rate = agent.test(TOTAL_EPISODES, MAX_STEPS)
        discount_factor_win_rates.setdefault(discount_factor, []).append(win_rate)
for discount_factor in discount_factors:
    win_rates = discount_factor_win_rates[discount_factor]
    plt.scatter([discount_factor] * len(win_rates), win_rates)
win_rates_mean = np.array([np.mean(v) for k, v in sorted(discount_factor_win_rates.items())])
win_rates_std = np.array([np.std(v) for k, v in sorted(discount_factor_win_rates.items())])
plt.errorbar(discount_factors, win_rates_mean, yerr=win_rates_std)
plt.title('Зависимость win_rate от DISCOUNT_FACTOR')
plt.xlabel('discount_factor')
plt.xticks(discount_factors)
plt.ylabel('win_rate')
plt.show()

# график для TOTAL_EPISODES
total_episodes = [10_000, 11_500, 14_000, 17_000, 20_000, 25_000, 32_000, 40_000]
total_episodes_win_rates = dict()
for total_episode in total_episodes:
    for _ in range(NUM_SAMPLES):
        agent = agents.EpsilonGreedyAgent(environment, EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE)
        agent.learn(total_episode, MAX_STEPS, LEARNING_RATE, DISCOUNT_FACTOR)
        win_rate = agent.test(TOTAL_EPISODES, MAX_STEPS)
        total_episodes_win_rates.setdefault(total_episode, []).append(win_rate)
for total_episode in total_episodes:
    win_rates = total_episodes_win_rates[total_episode]
    plt.scatter([total_episode] * len(win_rates), win_rates)
win_rates_mean = np.array([np.mean(v) for k, v in sorted(total_episodes_win_rates.items())])
win_rates_std = np.array([np.std(v) for k, v in sorted(total_episodes_win_rates.items())])
plt.errorbar(total_episodes, win_rates_mean, yerr=win_rates_std)
plt.title('Зависимость win_rate от TOTAL_EPISODES')
plt.xlabel('total_episodes')
plt.xticks(total_episodes)
plt.ylabel('win_rate')
plt.show()


# Предложение от Дениса, как эти 3 блока записать в одном
# dict_params = {
#     'lr_rates': [
#         0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
#     ],
#     'discount_factors': [
#         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
#     ],
#     'TOTAL_EPISODES': [
#         10_000, 11_500, 14_000, 17_000, 20_000, 25_000, 32_000, 40_000,
#     ],
# }
# for key, value in dict_params.items():
#     lr_rate_win_rates = dict()
#     for i in value:
#         for _ in range(5):
#             agent = agents.EpsilonGreedyAgent(environment, EPSILON, MIN_EPSILON, EPSILON_DECAY_RATE)
#             str_params = [
#                 'TOTAL_EPISODES',
#                 'MAX_STEPS',
#                 'EPSILON_DECAY_RATE',
#                 'LEARNING_RATE',
#                 'DISCOUNT_FACTOR',
#             ]
#             params = [
#                 TOTAL_EPISODES,
#                 MAX_STEPS,
#                 EPSILON_DECAY_RATE,
#                 LEARNING_RATE,
#                 DISCOUNT_FACTOR,
#             ]
#             index = str_params.index(key)
#             params[index] = i
#             agent.learn(*params)
#             win_rate = agent.test(TOTAL_EPISODES, MAX_STEPS)
#             lr_rate_win_rates.setdefault(LEARNING_RATE, []).append(win_rate)
#     for LEARNING_RATE in value:
#         win_rates = lr_rate_win_rates[LEARNING_RATE]
#         plt.scatter([LEARNING_RATE] * len(win_rates), win_rates)
#     win_rates_mean = np.array([np.mean(v) for k, v in sorted(lr_rate_win_rates.items())])
#     win_rates_std = np.array([np.std(v) for k, v in sorted(lr_rate_win_rates.items())])
#     plt.errorbar(value, win_rates_mean, yerr=win_rates_std)
#     plt.title('Зависимость win_rate от learning_rate')
#     plt.xlabel('learning_rate')
#     plt.ylabel('win_rate')
#     plt.show()
