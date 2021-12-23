import cv2
import gym
import gym.spaces
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    """Обёртка для сред, в которых для запуска игры необходимо нажать FIRE.

    В дополнение к нажатию FIRE, эта обёртка проверяет наличие некоторых вырожденных случаев,
    которые присутствуют в некоторых играх.
    """
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """Обёртка, преобразующая наблюдения из 210х160(250х160) RGB в 84х84 в оттенках серого."""
    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, 'Неизвестное разрешение.'
        # колориметрическое преобразования в оттенки серого
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]  # обрезает верх и низ изображения
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ImageToPyTorch(gym.ObservationWrapper):
    """Обёртка изменяет формат наблюдения с HWC на CHW, необходимый PyTorch."""
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Обёртка преобразует значения пикселей из int с диапазоном [0..255] в float [0.0..1.0]."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        # careful! This undoes the memory optimization, use with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    """Обёртка создаёт стек из последовательности кадров и возвращает его в качестве наблюдения."""
    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype)
        self.reset()

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames so it's important
            # to keep lives > 0, so that we only reset once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


def make_env(env_name):
    env = gym.make(env_name)

    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    env = EpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    return env
