import gym
import numpy as np

from itertools import product


class SortEnv(gym.Env):

    def __init__(self, num_agents, succ_prob=1.0, depth=3, max_steps=100, deterministic=True):

        self.n = num_agents
        self.start_position = [0]*self.n
        self.depth = depth
        self.max_steps = max_steps
        self.deterministic = deterministic

        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(self.depth)
                                                   for _ in range(self.n)])

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(2)
                                              for _ in range(self.n)])

        self.P = {}
        self.state_ordered_list = [self.start_position]
        self.state_proxy = [0]
        self._make_prob_map(succ_prob)

        # self.myprint()
        self.t = 0
        self.reset()

    def reset(self):
        self.positions = np.array(self.start_position, dtype=np.int)
        self.t = 0
        return self._get_obs()

    def step(self, action):

        action_list = list(action)

        if not self.deterministic:
            for a in range(self.n):
                if np.random.rand() < 0.2:
                    action_list[a] = self.action_space.spaces[a].sample()
        action_tuple = tuple(action_list)

        # print(self.state_ordered_list)
        # print(self.P[action_tuple][self._get_obs()])
        # print(len(self.state_ordered_list), len(self.P[action_tuple][self._get_obs()]))
        next_state_index = np.random.choice(self.state_proxy,
                                            p=self.P[action_tuple][self._get_obs()])
        next_state = self.state_ordered_list[next_state_index]

        self.positions = np.array(next_state, dtype=np.int)
        self.t += 1
        return self._get_obs(), 0, self.t >= self.max_steps, {}

    def render(self):
        print(self._get_obs())

    def get_sim_state(self):
        return self._get_obs()

    def set_sim_state(self, state):
        self.positions = np.array(state, dtype=np.int)
        return self._get_obs()

    def _get_obs(self):
        return tuple(self.positions)

    def _make_prob_map(self, succ_prob):

        depth = self.depth

        # Each agent has two actions
        # Action 0 keeps the agent in its current state
        # Action 1 moves to agent to the next state with probability succ_prob and
        # remains in the current with prob 1-succ_prob
        num_actions_per_agent = 2

        trans_single = {}

        for action in range(num_actions_per_agent):
            trans_single[action] = {}
            for src in range(depth):
                trans_single[action][src] = {}
                for dest in range(depth):
                    if src == depth-1 and src == dest:
                        trans_single[action][src][dest] = 1.0
                    elif action == 0 and src == dest:
                        trans_single[action][src][dest] = 1.0
                    elif action == 0 and src != dest:
                        trans_single[action][src][dest] = 0.0
                    elif action == 1 and dest == src:
                        trans_single[action][src][dest] = 1-succ_prob
                    elif action == 1 and dest == src+1:
                        trans_single[action][src][dest] = succ_prob
                    else:
                        trans_single[action][src][dest] = 0.0

        # for action in trans_single.keys():
        #    for state in trans_single[action].keys():
        #        print("{}, {}, {}".format(action, state, trans_single[action][state]))
        action_space = list(product(range(num_actions_per_agent), repeat=self.n))

        state_space = list(product(range(depth), repeat=self.n))

        transitions = {}
        for action in action_space:
            transitions[action] = {}
            for src in state_space:
                transitions[action][src] = []
                for dest in state_space:
                    prob_val = 1.0
                    for agent in range(self.n):
                        action_i = action[agent]
                        src_i = src[agent]
                        dest_i = dest[agent]
                        prob_val = prob_val * trans_single[action_i][src_i][dest_i]
                    transitions[action][src].append(prob_val)

        self.P = transitions
        self.state_ordered_list = state_space
        self.state_proxy = range(len(self.state_ordered_list))
        # print(state_space)

    def myprint(self):

        for action in self.P.keys():
            for state in self.P[action].keys():
                print("action {}, state {}, transprob {}".format(
                    action, state, self.P[action][state]))


"""
class SortEnv(FiniteMDP):

    def __init__(self, num_agents, succ_prob=1.0, depth=3):

        # Each agent has two actions
        ### Action 0 keeps the agent in its current state
        ### Action 1 moves to agent to the next state with probability succ_prob
        # and remains in the current with prob 1-succ_prob

        num_actions_per_agent = 2

        trans_single = {}
        for action in range(num_actions_per_agent):
            trans_single[action] = {}
            for src in range(depth):
                trans_single[action][src] = {}
                for dest in range(depth):
                    if action == 0 or src == depth-1:
                        trans_single[action][src][dest] = 1.0
                    elif action == 1 and dest == src:
                        trans_single[action][src][dest] = 1.0-succ_prob
                    elif action == 1 and dest == src+1:
                        trans_single[action][src][dest] = succ_prob
                    else:
                        trans_single[action][src][dest] = 0.0

        action_space = list(product(range(num_actions_per_agent), repeat = num_agents))

        state_space = list(product(range(depth), repeat = num_agents))

        transitions = {}
        for action in action_space:
            transitions[action] = {}
            for src in state_space:
                transitions[action][src] = {}
                for dest in state_space:
                    transitions[action][src][dest] = 1
                    for agent in range(num_agents):
                        action_i = action[agent]
                        src_i = src[agent]
                        dest_i = dest[agent]
                        transitions[action][src][dest] \
                            = transitions[action][src][dest] * trans_single[action_i][src_i][dest_i]

        num_states = 0
        start_state = tuple([0]*num_agents)

        id_to_state = {}
        for state in state_space:
            #print(state)
            id_to_state[num_states] = state
            if state == start_state:
                start_state_id = num_states
                print(state, start_state_id)
            num_states += 1

        format_trans = {}
        for action in action_space:
            format_trans[action] = []
            for src_id in range(num_states):
                src = id_to_state[src_id]
                format_trans[action].append([])
                for dest_id in range(num_states):
                    dest = id_to_state[dest_id]
                    format_trans[action][src_id].append(transitions[action][src][dest])


        num_actions = [num_actions_per_agent]*num_agents

        super().__init__(format_trans, start_state_id, num_actions)

        self.state_map = id_to_state

        def render(self):
            print('{}'.format(self.state_map[self.state]))

"""
