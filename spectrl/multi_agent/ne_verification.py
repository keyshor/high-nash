from abc import ABCMeta, abstractmethod
from collections import deque
from spectrl.util.rl import test_policy_mutli, test_policy
from spectrl.util.io import print_new_block
from spectrl.rl.qlearning import QLearningAgent

import gym
import scipy
import nashpy
import itertools
import numpy as np


class FiniteStatePolicy(metaclass=ABCMeta):
    '''
    Metaclass for finite state joint policies.
    '''

    def __init__(self, probabilitic=False, update_first=True, joint_policy=True):
        self.probabilitic = probabilitic
        self.update_first = update_first
        self.joint_policy = joint_policy

    @abstractmethod
    def init_state(self):
        '''
        Returns initial policy state.
        '''
        pass

    @abstractmethod
    def step(self, policy_state, env_state, action):
        '''
        Processes an environment state and outputs next policy state.
        '''
        pass

    @abstractmethod
    def act(self, policy_state, env_state):
        '''
        Returns action to take at the given policy state and environment state.
        '''
        pass

    def reset(self):
        self.state = self.init_state()
        return self.state

    def get_action(self, env_state):
        # update state
        if self.update_first:
            self.state = self.step(self.state, env_state, None)

        # get action
        action = self.act(self.state, env_state)

        # randomly choose action if policy is probabilistic
        if self.probabilitic:
            if self.joint_policy:
                action = tuple([np.random.choice(len(p), p=p) for p in action])
            else:
                action = np.random.choice(len(action), p=action)

        # update state if not done before
        if not self.update_first:
            self.state = self.step(self.state, env_state, action)

        return action


class RandomFiniteStatePolicy(FiniteStatePolicy):

    def __init__(self, action_space, reward_machines):
        self.action_space = action_space
        self.reward_machines = reward_machines
        super().__init__(update_first=False)

    def init_state(self):
        return ()

    def step(self, policy_state, env_state, action):
        return ()

    def act(self, policy_state, env_state):
        return self.action_space.sample()


class FullNashPolicy(FiniteStatePolicy):
    '''
    Full Policy that adds punishment strategies to a Finite State Policy

    fs_policy: FiniteStatePolicy (deterministic, update_first, joint_policy)
    adv_policies: List of Dict[(state, policy_state, rm_state, dev_flag), adv_action_probs]
    num_actions: List of max number of actions for each agent
    '''

    def __init__(self, fs_policy, adv_policies, reward_machines, num_actions, horizon):
        self.fs_policy = fs_policy
        self.adv_policies = adv_policies
        self.reward_machines = reward_machines
        self.num_actions = num_actions
        self.horizon = horizon
        super().__init__(update_first=False)

    def init_state(self):
        fstate = self.fs_policy.init_state()
        rmstate = tuple([rm.init_state for rm in self.reward_machines])
        dev_agent = -1
        return fstate, rmstate, dev_agent, self.horizon

    def step(self, policy_state, env_state, action):
        fstate, rmstates, dev_agent, t = policy_state
        fstate = self.fs_policy.step(fstate, env_state, None)
        rmstates = tuple([rm.step(rmstate, env_state)[0]
                          for rm, rmstate in zip(self.reward_machines, rmstates)])
        if dev_agent == -1:
            f_actions = self.fs_policy.act(fstate, env_state)
            for a in range(len(self.num_actions)):
                if action[a] != f_actions[a]:
                    dev_agent = a
                    break
        return fstate, rmstates, dev_agent, t-1

    def act(self, policy_state, env_state):
        fstate, rmstates, dev_agent, t = policy_state
        if dev_agent != -1:
            adv_state = (env_state, fstate, rmstates[dev_agent], 1)
            try:
                adv_probs = self.adv_policies[dev_agent][(adv_state, t)]
                adv_probs = np.maximum(0., adv_probs)
                adv_probs = adv_probs / np.sum(adv_probs)
                adv_action = self.num_to_tuple(np.random.choice(len(adv_probs), p=adv_probs),
                                               dev_agent)
                dev_action = np.random.choice(self.num_actions[dev_agent])
                return adv_action[:dev_agent] + (dev_action,) + adv_action[dev_agent:]
            except KeyError:
                pass
        next_fstate = self.fs_policy.step(fstate, env_state, None)
        f_actions = self.fs_policy.act(next_fstate, env_state)
        return f_actions

    def num_to_tuple(self, action_num, dev_agent):
        num_ac = self.num_actions[:dev_agent] + self.num_actions[dev_agent+1:]
        action = [0] * len(num_ac)
        for i in range(len(num_ac)):
            j = (len(num_ac) - i) - 1
            action[j] = action_num % num_ac[j]
            action_num = action_num // num_ac[j]
        return tuple(action)

    def random_adv_action(self, dev_agent):
        num_ac = self.num_actions[:dev_agent] + self.num_actions[dev_agent+1:]
        return tuple([np.random.choice(n) for n in num_ac])


class SingleAgentEnv(gym.Env):
    '''
    Single agent env for testing nash equilibrium.
    '''

    def __init__(self, multi_env, joint_policy, agent, horizon):
        self.multi_env = multi_env
        self.joint_policy = joint_policy
        self.a = agent
        self.horizon = horizon
        self.rm = self.joint_policy.reward_machines[self.a]

        self.action_space = self.multi_env.action_space.spaces[self.a]

    def reset(self):
        self.t = 0
        self.rm_state = self.rm.init_state
        self.env_state = self.multi_env.reset()
        self.policy_state = self.joint_policy.init_state()
        return self.env_state, self.policy_state, self.rm_state

    def step(self, action):

        # get reward
        self.rm_state, reward = self.rm.step(self.rm_state, self.env_state)

        # get actions of other agents
        actions = list(self.joint_policy.act(self.policy_state, self.env_state))
        if self.joint_policy.probabilitic:
            actions = [np.random.choice(len(p), p=p) for p in actions]
        actions[self.a] = action
        self.policy_state = self.joint_policy.step(
            self.policy_state, self.env_state, tuple(actions))

        # step in wrapped env
        self.env_state, _, done, info = self.multi_env.step(tuple(actions))

        # terminate when horizon is reached
        self.t += 1
        done = done or self.t >= self.horizon

        return (self.env_state, self.policy_state, self.rm_state), reward, done, info

    def render(self):
        return self.multi_env.render()


def estimate_mdp(env, num_samples):
    '''
    Estimates the MDP represented by env from samples.
    Assumes the ability to reset to any state.
    '''

    # possible actions for each agent
    actions = [list(range(space.n)) for space in env.action_space.spaces]

    # dictionary of probabilities, maps each state and action to dict[state->prob]
    P = {}

    # set of states visited and queue of states yet to be processed
    visited = set()
    queue = deque([])

    # get start state
    env.reset()
    state = env.get_sim_state()
    visited.add(state)
    queue.append(state)

    # record number of sample steps
    steps_taken = 0

    while queue:
        state = queue.popleft()
        P[state] = {}

        for action in itertools.product(*actions):
            probs = {}

            for _ in range(num_samples):
                env.set_sim_state(state)
                env.step(action)
                next_state = env.get_sim_state()

                if next_state not in probs:
                    probs[next_state] = 0
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)
                probs[next_state] += 1

            for s in probs:
                probs[s] = probs[s] / num_samples

            P[state][action] = probs
            steps_taken += num_samples

    return P, steps_taken


def get_product_mdp(env, env_P, fs_policy, rm, agent):
    '''
    Constructs two agent product MDP for computing best punishment.

    Parameters:
        env: multi-agant environment
        env_P: estimated probabilities of env
        fs_policy: finite state joint policy (has to be of the form:
            update_first=True, probabilistic=False, joint_policy=True)
        rm: reward machine corresponding to agent

    Returns:
        P: probabilities of product MDP
        R: reward map
    '''

    # get initial state
    env.reset()
    start_state = (env.get_sim_state(), fs_policy.init_state(), rm.init_state, 0)
    agent_n = env.action_space.spaces[agent].n
    all_n = [space.n for space in env.action_space.spaces]
    adv_n = all_n[:agent] + all_n[agent+1:]
    total_adv_n = np.prod(adv_n)

    # BFS stats (using BFS to compute reachable part of product MDP)
    visited = set([start_state])
    queue = deque([start_state])

    # Probability and reward maps
    P = {}
    R = {}

    while queue:
        (env_state, policy_state, rm_state, dev_flag) = s = queue.popleft()
        next_ps = fs_policy.step(policy_state, env_state, None)
        next_rms, rew = rm.step(rm_state, env_state)

        P[s] = [[{} for _ in range(total_adv_n)] for _ in range(agent_n)]
        R[s] = rew

        for action in env_P[env_state]:
            act1 = action[agent]
            act2_tuple = action[:agent] + action[agent+1:]
            act2 = 0
            for i in range(len(act2_tuple)):
                act2 = (adv_n[i] * act2) + act2_tuple[i]

            next_df = 1
            if dev_flag == 0:
                action = list(fs_policy.act(next_ps, env_state))
                if action[agent] == act1:
                    next_df = 0
                action[agent] = act1
                action = tuple(action)

            for next_es in env_P[env_state][action]:
                next_s = (next_es, next_ps, next_rms, next_df)
                P[s][act1][act2][next_s] = env_P[env_state][action][next_es]

                if next_s not in visited:
                    visited.add(next_s)
                    queue.append(next_s)

    return P, R, start_state


def nash_value_iteration_zero_sum(P, R, s, h, DP={}, adv_policy={}, verbose=False, use_lp=True):

    # read memoized values
    if (s, h) in DP:
        return DP[(s, h)]

    # base case
    if h == 0:
        adv_policy[(s, h)] = np.array([1/len(P[s][0]) for _ in range(len(P[s][0]))])
        DP[(s, h)] = 0.
        return DP[(s, h)]

    # create a matrix game for the current stage
    matrix_game = []
    for p_act1 in P[s]:
        row_vals = []
        for p_act1_act_2 in p_act1:
            val = R[s]
            for next_s in p_act1_act_2:
                next_val = nash_value_iteration_zero_sum(P, R, next_s, h-1, DP, adv_policy,
                                                         verbose=verbose, use_lp=use_lp)
                val += (p_act1_act_2[next_s] * next_val)
            if val:
                row_vals.append(val)
            else:
                row_vals.append(0.)
        matrix_game.append(row_vals)

    # solve the matrix game
    if verbose:
        print('Solving matrix game for [s={}, h={}]'.format(s, h))
    if use_lp:
        DP[(s, h)], adv_policy[(s, h)] = solve_matrix_game_lp(np.array(matrix_game))
    else:
        DP[(s, h)], adv_policy[(s, h)] = solve_matrix_game_nashpy(np.array(matrix_game))
    return DP[(s, h)]


def solve_matrix_game_nashpy(payoff_matrix):
    game = nashpy.Game(payoff_matrix)
    ne_list = list(game.support_enumeration())
    return game[ne_list[0][0], ne_list[0][1]][0], np.array(ne_list[0][1])


def solve_matrix_game_lp(payoff_matrix):
    '''
    Solve zero-sum matrix game using linear programming.
    '''
    # get sizes
    n, m = payoff_matrix.shape

    # objective coeffs
    obj = np.concatenate([[1.], np.zeros((m,))])

    # - w.e + Uq <= 0
    lhs_ineq = np.concatenate([-np.ones((n, 1)), payoff_matrix], axis=1)
    rhs_ineq = np.zeros((n,))

    # q.e = 1
    lhs_eq = np.concatenate([[[0.]], np.ones((1, m))], axis=1)
    rhs_eq = np.array([1.])

    # q >= 0
    bounds = [(None, None)] + ([(0, None)] * m)

    opt = scipy.optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
                                 A_eq=lhs_eq, b_eq=rhs_eq, bounds=bounds,
                                 method='revised simplex')
    return opt.fun, opt.x[1:]


def test_nash_solution(multi_env, joint_policy, horizon, samples):
    if joint_policy is None:
        joint_policy = RandomFiniteStatePolicy(multi_env.action_space,
                                               joint_policy.reward_machines)

    print_new_block('Policy Evaluation')
    _, probs = test_policy_mutli(multi_env, joint_policy, 500, use_rm_reward=True,
                                 max_timesteps=horizon, stateful_policy=True)
    print('Satisfaction Probabilities: {}'.format(probs.tolist()))

    best_responses = []
    for a in range(len(joint_policy.reward_machines)):
        env = SingleAgentEnv(multi_env, joint_policy, a, horizon)
        q_agent = QLearningAgent()
        policy = q_agent.learn(env, 300000)
        _, br_prob = test_policy(env, policy, 500, max_timesteps=horizon)
        best_responses.append(br_prob)
    print('Best Responses: {}'.format(best_responses))
    return {'probs': probs, 'best_responses': best_responses, 'samples': samples}
