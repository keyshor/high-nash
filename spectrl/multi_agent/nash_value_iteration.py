from collections import deque
from spectrl.util.io import print_new_block
from spectrl.multi_agent.ne_verification import estimate_mdp, FiniteStatePolicy

import time
import gambit
import itertools
import numpy as np


class TimeDependentPolicy(FiniteStatePolicy):
    '''
    Policy mapping state-time pair to actions.
    '''

    def __init__(self, policy_map, horizon, reward_machines):
        self.policy_map = policy_map
        self.horizon = horizon
        self.reward_machines = reward_machines
        super().__init__(probabilitic=True, update_first=False)

    def init_state(self):
        return self.horizon, tuple([rm.init_state for rm in self.reward_machines])

    def step(self, policy_state, env_state, action):
        t, rm_states = policy_state
        return t-1, tuple([rm.step(rm_state, env_state)[0]
                           for rm, rm_state in zip(self.reward_machines, rm_states)])

    def act(self, policy_state, env_state):
        t, rm_states = policy_state
        full_state = ((env_state, rm_states), t)
        return self.policy_map[full_state]


def get_full_product_mdp(env, env_P, rms):
    '''
    Constructs product MDP with rewards for all agents.
    Parameters:
        env: multi-agant environment
        env_P: estimated probabilities of env
        rms: list of reward machines corresponding to the agents
    Returns:
        P: probabilities of product MDP
        R: reward map
        start_state: (env_start_state, rm_start_states: Tuple)
    '''

    # get initial state
    env.reset()
    rm_init_states = tuple([rm.init_state for rm in rms])
    start_state = (env.get_sim_state(), rm_init_states)
    all_n = [space.n for space in env.action_space.spaces]

    # BFS stats (using BFS to compute reachable part of product MDP)
    visited = set([start_state])
    queue = deque([start_state])

    # Probability and reward maps
    P = {}
    R = {}

    while queue:
        (env_state, rm_states) = s = queue.popleft()

        # compute next product rm state and rewards
        next_rm_states = []
        rewards = []
        for rm, rm_state in zip(rms, rm_states):
            next_rm_state, rew = rm.step(rm_state, env_state)
            next_rm_states.append(next_rm_state)
            rewards.append(rew)
        next_rm_states = tuple(next_rm_states)

        P[s] = {}
        R[s] = rewards

        for actions in itertools.product(*[range(num_actions) for num_actions in all_n]):
            P[s][actions] = {}
            for next_es in env_P[env_state][actions]:
                next_s = (next_es, next_rm_states)
                P[s][actions][next_s] = env_P[env_state][actions][next_es]

                if next_s not in visited:
                    visited.add(next_s)
                    queue.append(next_s)

    return P, R, start_state


def nash_value_iteration(P, R, s, h, num_actions, DP={}, policy={}, verbose=False):
    # read memoized values
    if (s, h) in DP:
        return DP[(s, h)]

    # base case
    if h == 0:
        DP[(s, h)] = np.zeros((len(num_actions),))
        policy[(s, h)] = [np.ones((n,))*(1/n) for n in num_actions]
        return DP[(s, h)]

    # create a matrix game for the current stage
    matrix_game = [R[s][a] * np.ones(tuple(num_actions))
                   for a in range(len(num_actions))]
    for actions in itertools.product(*[range(n) for n in num_actions]):
        v_s = np.zeros((len(num_actions,)))
        for next_s in P[s][actions]:
            v_s += P[s][actions][next_s] * np.array([float(v) for v in nash_value_iteration(
                P, R, next_s, h-1, num_actions, DP, policy=policy, verbose=verbose)])
        for a in range(len(num_actions)):
            matrix_game[a][actions] += v_s[a]

    # solve the matrix game
    if verbose:
        print('Solving matrix game for [s={}, h={}]'.format(s, h))
    DP[(s, h)], policy[(s, h)] = solve_matrix_game_gambit(matrix_game)
    return DP[(s, h)]


def solve_matrix_game_gambit(matrix_game):
    game = gambit.Game.new_table(matrix_game[0].shape)
    for actions in itertools.product(*[range(n) for n in matrix_game[0].shape]):
        for a in range(len(actions)):
            game[actions][a] = gambit.Rational(matrix_game[a][actions])

    profiles = gambit.nash.simpdiv_solve(game)
    best_p = np.argmax(map(
        lambda p: sum(get_gambit_profile_payoffs(p, game)), profiles))
    best_values = np.array(get_gambit_profile_payoffs(profiles[best_p], game))
    best_actions = [np.array([float(pr) for pr in profiles[best_p][game.players[a]]])
                    for a in range(len(game.players))]
    return best_values, best_actions


def get_gambit_profile_payoffs(profile, game):
    return [profile.payoff(game.players[a]) for a in range(len(game.players))]


def estimate_and_solve(env, specs, pred_fns, hyperparameters, verbose=False):
    start_time = time.time()

    print_new_block('Running Estimate+NVI Baseline')
    print('\nEstimating MDP...')
    reward_machines = [spec.get_rm(pred_fns) for spec in specs]
    P, steps_taken = estimate_mdp(env, hyperparameters.num_verify_samples)
    P, R, start_state = get_full_product_mdp(env, P, reward_machines)
    print('MDP estimation completed in {} secs'.format(time.time() - start_time))

    DP = {}
    policy_map = {}
    num_actions = [space.n for space in env.action_space.spaces]
    nash_start = time.time()

    print('Performing general sum nash value iteration...')
    values = nash_value_iteration(P, R, start_state, hyperparameters.horizon,
                                  num_actions, DP=DP, policy=policy_map, verbose=verbose)
    print('Nash value iteration completed in {} secs'.format(time.time() - nash_start))

    print_new_block('Results')
    print('Values of the agents at start_state: {}'.format([float(v) for v in values.tolist()]))
    print('Total steps taken: {}'.format(steps_taken))
    print('Total time taken: {} secs'.format(time.time() - start_time))
    return TimeDependentPolicy(policy_map, hyperparameters.horizon, reward_machines), steps_taken
