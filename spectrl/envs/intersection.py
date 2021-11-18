import gym
import numpy as np


class IntersectionEnv(gym.Env):

    def __init__(self, init_location, stay_prob=0., max_steps=100):

        # Number of North-South agents
        self.n_ns = len(init_location[0])
        # Number of East-West agents
        self.n_ew = len(init_location[1])

        self.n = self.n_ns + self.n_ew

        self.start_position = np.concatenate(init_location)

        obs_spaces = [gym.spaces.Discrete(self.start_position[a] + 1)
                      for a in range(self.n)]

        self.observation_space = gym.spaces.Tuple(obs_spaces)

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(2)
                                              for _ in range(self.n)])
        self.max_steps = max_steps
        self.stay_prob = stay_prob

        self.reset()

    def reset(self):
        self.positions = self.start_position
        self.t = 0
        return self._get_obs()

    def step(self, actions):
        actions = np.array(actions)
        for a in range(len(actions)):
            if np.random.rand() < self.stay_prob:
                actions[a] = 0
        self.positions = np.maximum(0, self.positions - actions)
        self.t += 1

        return self._get_obs(), 0, self.t >= self.max_steps, {}

    def render(self):
        curr_position = self._get_obs()
        print(curr_position[:self.n_ns], curr_position[self.n_ns:])

    def get_sim_state(self):
        return self._get_obs()

    def set_sim_state(self, state):
        self.positions = np.array(state, dtype=np.int)
        return self._get_obs()

    def _get_obs(self):
        return tuple(self.positions)
