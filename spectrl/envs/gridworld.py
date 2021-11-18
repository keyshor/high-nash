'''
Finite state gridworld environment with multiple agents.
'''
import gym
import numpy as np


class GridWorld(gym.Env):
    '''
    Finite 2D grid with each agent occupying a point in the grid.
    '''

    def __init__(self, grid_size, start_positions):
        self.grid_size = np.array(grid_size, dtype=int)
        self.start_positions = start_positions
        self.n = len(self.start_positions)

        obs_spaces = [gym.spaces.Tuple([gym.spaces.Discrete(self.grid_size[0]),
                                        gym.spaces.Discrete(self.grid_size[1])])
                      for _ in range(self.n)]
        self.observation_space = gym.spaces.Tuple(obs_spaces)
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(5)
                                              for _ in range(self.n)])

        self.reset()

    def reset(self):
        self.positions = np.array(self.start_positions, dtype=np.int)
        return self._get_obs()

    def step(self, actions):
        actions = list(actions)
        for a in range(len(actions)):
            if np.random.rand() < 0.2:
                actions[a] = self.action_space.spaces[a].sample()
        action_vec = np.array([self._int_to_vec(action) for action in actions], dtype=np.int)
        self.positions = self.positions + action_vec
        self.positions = np.clip(self.positions, 0, self.grid_size.reshape(1, -1)-1)
        return self._get_obs(), 0, False, {}

    def render(self):
        print(self._get_obs())

    def get_sim_state(self):
        return self._get_obs()

    def set_sim_state(self, state):
        self.positions = np.array(state, dtype=np.int)
        return self._get_obs()

    def _get_obs(self):
        return tuple(map(tuple, self.positions))

    def _int_to_vec(self, action):
        if action == 0:
            return [0, 0]
        elif action == 1:
            return [1, 0]
        elif action == 2:
            return [0, 1]
        elif action == 3:
            return [-1, 0]
        else:
            return [0, -1]
