import gym
import numpy as np


class FiniteMDP(gym.Env):
    '''
    Finite Multi-agent MDPs
    '''

    def __init__(self, transitions, start_state, num_actions, max_steps=100, rewards={}):
        '''
        'transitions' is a dictionary mapping joint actions to transition matrices.
            Eg:- transitions[(0, 0, 1)] = [[0.5, 0.5], [0, 1]] (for 3 agents and 2 states)
        'start_state' is an int specifying the starting state
        'num_actions' is a list specifying the number of actions for each player.
            Eg:- num_actions = [2, 2, 3] (3 agents, last agent's actions are {0, 1, 2})
        'rewards' is a dict mapping state-action pairs to list of rewards.
            Can be empty when using Spectrl specs.
        '''
        self.P = transitions
        self.start_state = start_state
        self.rewards = rewards
        self.max_steps = max_steps
        self.n = len(num_actions)

        # Define action space
        spaces = [gym.spaces.Discrete(na) for na in num_actions]
        self.action_space = gym.spaces.Tuple(spaces)

        # Define observation space
        joint_action = self.action_space.sample()
        self.observation_space = gym.spaces.Discrete(len(self.P[joint_action]))

    def reset(self):
        self.state = self.start_state
        self.t = 0
        return self.state

    def step(self, action):
        next_state = np.random.choice(range(self.observation_space.n),
                                      p=self.P[action][self.state])
        reward = 0.0
        if (self.state, action) in self.rewards:
            reward = self.rewards[(self.state, action)]

        self.state = next_state
        self.t += 1
        return self.state, reward, self.t >= self.max_steps, {}

    def render(self):
        print('s_{}'.format(self.state))

    def get_sim_state(self):
        return self.state

    def set_sim_state(self, state):
        self.state = state
        return self.state
