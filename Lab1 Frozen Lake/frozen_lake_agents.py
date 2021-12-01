import abc
import os
import time
from typing import Union

import numpy as np
from tqdm import tqdm


class FrozenLakeAgent:
    """Агент в среде типа FrozenLake."""
    def __init__(self, env):
        self._env = env

    def _render(self, episode):
        """Отображает среду. Опционально используется при обучении и тестировании."""
        os.system('cls')
        print('===========')
        print(f' ЭПИЗОД {episode + 1}')
        print('===========')
        self._env.render()
        time.sleep(0.75)

    # PROBLEM
    # Этот метод в дочерних классах переопределяется 2 раза с разницей в одну строчку.
    # Хорошо бы что-то с этим сделать. Но как?
    @abc.abstractmethod
    def test(self, total_episodes: int, max_steps: int, show: bool = False) -> float:
        """Запускает агента играть в среду total_episodes раз и подсчитывает винрейт.

        Args:
            total_episodes: количество эпизодов игры, которые будет играть агент.
            max_steps: максимальное количество шагов агента в эпизоде.
            show: показывать ли тестирование.

        Returns
            win_rate: винрейт агента.
        """
        raise NotImplementedError()


class RandomAgent(FrozenLakeAgent):
    """Случайно действующий агент."""
    def __init__(self, env):
        super().__init__(env)

    def test(self, total_episodes: int, max_steps: int, show: bool = False) -> float:
        """Смотри родительский класс."""
        if show:
            print(f'\nТестирование в среде {str(self._env.spec)[8:-1]}\n')

        wins = 0
        defeats = 0
        for episode in range(total_episodes):
            self._env.reset()

            if show:
                self._render(episode)

            step = 0
            while step < max_steps:
                # PROBLEM
                # ЕДИНСТВЕННАЯ ИЗМЕНЁННАЯ СТРОЧКА
                action = self._env.action_space.sample()  # действие выбирается случайно
                state, reward, done, info = self._env.step(action)

                if show:
                    self._render(episode)

                if done:
                    if reward == 1:
                        wins += 1
                        if show:
                            print('\nПОБЕДА')
                            time.sleep(0.75)
                    else:
                        defeats += 1
                        if show:
                            print('\nПОРАЖЕНИЕ')
                            time.sleep(0.75)
                    break

                step += 1

            if show and (step >= max_steps):
                print(f'\nВРЕМЯ ВЫШЛО (steps > {max_steps})')
                time.sleep(1)

        win_rate = wins / (wins + defeats) * 100
        if show:
            print(f'Процент побед: {win_rate}%')

        return win_rate


class QTableAgent(FrozenLakeAgent):
    """Агент, способный к обучению через Q-таблицу."""
    def __init__(self, env):
        super().__init__(env)
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    def _update_q_table(self, s, a, r, new_s, lr_rate, y):
        """Обновляет Q-таблицу агента.

        s: state, состояние.
        a: action, действие.
        r: reward, награда.
        new_s: new state, новое состояние.
        lr_rate: скорость обучения.
        y: gamma или discount factor, фактор дисконтирования.
        """
        self.Q[s, a] = self.Q[s, a] + lr_rate * (r + y * np.max(self.Q[new_s, :]) - self.Q[s, a])

    # PROBLEM
    # Этот метод в дочернем классе переопределяется с небольшими изменениями.
    def learn(
            self,
            total_episodes: int,
            max_steps: int,
            lr_rate: Union[int, float],
            discount_factor: Union[int, float],
            show: bool = False
            ) -> None:
        """Обучает агента.

        Args:
            total_episodes: количество эпизодов игры.
            max_steps: максимальное количество шагов агента в эпизоде.
            lr_rate: скорость обучения.
            discount_factor: дисконтирующий фактор.
            show: показывать ли обучение.
        """
        if show:
            os.system('cls')
            print(f'Обучение в среде {str(self._env.spec)[8:-1]}\n')

        for _ in tqdm(range(total_episodes), ascii=True, unit='episode'):
            state = self._env.reset()

            step = 0
            while step < max_steps:
                # PROBLEM
                # ЕДИНСТВЕННАЯ ИЗМЕНЁННАЯ СТРОЧКА ПО СРАВНЕНИЮ С МЕТОДОМ ДОЧЕРНЕГО КЛАССА
                action = np.argmax(self.Q[state, :])  # выбирается лучшее действие
                new_state, reward, done, info = self._env.step(action)

                if done and (reward == 0):    # провалился в прорубь
                    reward = -10
                elif done and (reward == 1):  # добрался до фрисби
                    reward = 100
                else:                         # наступил на лёд
                    reward = -1

                self._update_q_table(state, action, reward, new_state, lr_rate, discount_factor)
                state = new_state
                step += 1

                if done:
                    break

    def test(self, total_episodes: int, max_steps: int, show: bool = False) -> float:
        """Смотри родительский класс."""
        if show:
            print(f'\nТестирование в среде {str(self._env.spec)[8:-1]}\n')

        wins = 0
        defeats = 0
        for episode in range(total_episodes):
            state = self._env.reset()

            if show:
                self._render(episode)

            step = 0
            while step < max_steps:
                # PROBLEM
                # ЕДИНСТВЕННАЯ ИЗМЕНЁННАЯ СТРОЧКА
                action = np.argmax(self.Q[state, :])  # выбирается лучшее действие
                state, reward, done, info = self._env.step(action)

                if show:
                    self._render(episode)

                if done:
                    if reward == 1:
                        wins += 1
                        if show:
                            print('\nПОБЕДА')
                            time.sleep(0.75)
                    else:
                        defeats += 1
                        if show:
                            print('\nПОРАЖЕНИЕ')
                            time.sleep(0.75)
                    break

                step += 1

            if show and (step >= max_steps):
                print(f'\nВРЕМЯ ВЫШЛО (steps > {max_steps})')
                time.sleep(1)

        try:
            win_rate = wins / (wins + defeats) * 100
            if show:
                print(f'Процент побед: {win_rate}%')
        except ZeroDivisionError:
            win_rate = 0
            if show:
                print(f'Процент побед: 0%')

        return win_rate


class EpsilonGreedyAgent(QTableAgent):
    """Эпсилон-жадный агент."""
    def __init__(self, env, eps, min_eps, eps_decay_rate):
        super().__init__(env)
        # параметры эпсилон-жадности
        self._eps = eps
        self._min_eps = min_eps
        self._eps_decay_rate = eps_decay_rate

    def _decrease_epsilon(self):
        """Уменьшает эпсилон."""
        # PROBLEM
        # не лучший способ уменьшить параметр
        if self._eps > self._min_eps:
            self._eps -= self._eps_decay_rate
        else:
            self._eps = self._min_eps

    def _greedy(self, state):
        """Выбирает случайное действие с жадностью эпсилон."""
        if np.random.uniform(0, 1) < self._eps:
            action = self._env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action

    # PROBLEM
    # Этот метод в родительском классе определяется с небольшими изменениями.
    def learn(
            self,
            total_episodes: int,
            max_steps: int,
            lr_rate: Union[int, float],
            discount_factor: Union[int, float],
            show: bool = False
            ) -> None:
        """Смотри родительский класс."""
        if show:
            os.system('cls')
            print(f'Обучение в среде {str(self._env.spec)[8:-1]}\n')

        for _ in tqdm(range(total_episodes), ascii=True, unit='episode'):
            state = self._env.reset()

            step = 0
            while step < max_steps:
                # PROBLEM
                # ЕДИНСТВЕННАЯ ИЗМЕНЁННАЯ СТРОЧКА ПО СРАВНЕНИЮ С МЕТОДОМ РОДИТЕЛЬСКОГО КЛАССА
                # с жадностью эпсилон выбирается случайное действие, иначе лучшее
                action = self._greedy(state)
                new_state, reward, done, info = self._env.step(action)

                if done and (reward == 0):    # провалился в прорубь
                    reward = -10
                elif done and (reward == 1):  # добрался до фрисби
                    reward = 100
                else:                         # наступил на лёд
                    reward = -1

                self._update_q_table(state, action, reward, new_state, lr_rate, discount_factor)
                state = new_state
                step += 1

                if done:
                    break

            self._decrease_epsilon()
