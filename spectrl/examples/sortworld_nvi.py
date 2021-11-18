from spectrl.envs.sortworld import SortEnv
from spectrl.multi_agent.finite_state import HyperParams
from spectrl.multi_agent.nash_value_iteration import estimate_and_solve
from spectrl.multi_agent.ne_verification import test_nash_solution
from spectrl.main.spec_compiler import ev, alw, seq
from spectrl.util.io import parse_command_line_options, print_new_block, save_object
from spectrl.util.rl import get_rollout

import os
import time
import random
import numpy as np

MAX_STEPS = 20
NUM_STEPS = 5000

SEEDS = [907, 651, 985,	221, 182, 53, 152, 142, 224, 386]

# Add predicates


def not_reach(agent, depth):
    def predicate(state):
        return (state[agent] != depth-1)
    return predicate


def reach(agent, depth):
    def predicate(state):
        return (state[agent] == depth-1)
    return predicate


def reach_mid(agent, depth):
    def predicate(state):
        return (state[agent] == depth//2)
    return predicate


def not_reach_mid(agent, depth):
    def predicate(state):
        return (state[agent] < depth//2)
    return predicate


depth = 5


# Add specs
pred_fns = {'a0_reach': reach(0, depth),
            'a1_reach': reach(1, depth),
            'a2_reach': reach(2, depth),
            'a0_not_reach': not_reach(0, depth),
            'a1_not_reach': not_reach(1, depth),
            'a2_not_reach': not_reach(2, depth),
            'a0_reach_mid': reach_mid(0, depth),
            'a1_reach_mid': reach_mid(1, depth),
            'a2_reach_mid': reach_mid(2, depth),
            'a0_not_reach_mid': not_reach_mid(0, depth),
            'a1_not_reach_mid': not_reach_mid(1, depth),
            'a2_not_reach_mid': not_reach_mid(2, depth)
            }


# ------------------------------------------------------------------------------------------------------

# CLASS I: BASIC TESTS

# [1,1,1]
spec0 = [ev('a0_reach'),
         ev('a1_reach'),
         ev('a2_reach')]

# [1,1,1]
spec1 = [alw('a0_not_reach', ev('a1_reach')),
         ev('a1_reach'),
         ev('a2_reach')]

# [1,1,1]
spec2 = [seq(alw('a0_not_reach', ev('a1_reach')), ev('a0_reach')),
         ev('a1_reach'),
         ev('a2_reach')]

# ------------------------------------------------------------------------------------------------------

# CLASS II: COMPETITIVE : NOT ALL AGENTS CAN SATISFY GOAL


# 0 should reach final before 1
# 1 should reach final before 0
# (Both can't reach at the same time)

# At least one of Agent 0 and Agent 1 will have to sacrifice its goal.
# Nash equilibria are [1,0,1], or [0,1,1], or [0,0,1]

spec3 = [seq(alw('a0_not_reach', ev('a1_reach')), ev('a0_reach')),
         seq(alw('a1_not_reach', ev('a0_reach')), ev('a1_reach')),
         ev('a2_reach')]


spec4 = [alw('a1_not_reach', ev('a0_reach')),
         alw('a0_not_reach', ev('a1_reach')),
         ev('a2_reach')]

spec5 = [seq(alw('a1_not_reach', ev('a0_reach')), ev('a1_reach')),
         seq(alw('a0_not_reach', ev('a1_reach')), ev('a0_reach')),
         ev('a2_reach')]

# ------------------------------------------------------------------------------------------------------

# CLASS III: EXTREME COORDINATION. ALL AGENTS CAN SATISFY GOAL.

# 0 reaches mid before 1. Eventually 0 will reach the final state.
# 1 reaches final before 0
# All should satisfy their goal.
# If p = 1, then [1,1,1]. For other values of p, prob. of satisfiability will change
# since co-ordination is involved.

spec6 = [seq(alw('a1_not_reach_mid', ev('a0_reach_mid')), ev('a1_reach_mid')),
         seq(alw('a0_not_reach', ev('a1_reach')), ev('a0_reach')),
         ev('a2_reach')]

# 0 reaches mid before 1. Eventually 0 will reach the final state.
# 1 reaches final before 0
# 2 reaches final after 0
# All should satisfy their goal.
# If p = 1, then [1,1,1]. For other values of p, prob. of satisfiability will
# change since co-ordination is involved.

spec7 = [seq(alw('a1_not_reach_mid', ev('a0_reach_mid')), ev('a1_reach_mid')),
         seq(alw('a0_not_reach', ev('a1_reach')), ev('a0_reach')),
         seq(alw('a2_not_reach', ev('a0_reach')), ev('a2_reach'))]


# ------------------------------------------------------------------------------------------------------

# CLASS IV: EXTREME COORDINATION. SOME AGENTS CAN SATISFY GOAL. At p=1: Finds the largest set of
# agents that can satisfy their goals simulatenously.
# CLASS II is a special case of this.


specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6, spec7]
horizons = [20, 20, 20, 20, 20, 20, 20, 20]


if __name__ == '__main__':

    flags = parse_command_line_options()
    spec_num = flags['spec_num']
    itno = flags['itno']
    folder = flags['folder']
    render = flags['render']
    folder = os.path.join(folder, 'spec{}'.format(spec_num), 'nvi')
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.random.seed(SEEDS[itno])
    random.seed(SEEDS[itno])

    # Create environment
    env = SortEnv(len(specs[spec_num]), 0.95, depth)

    # Choose Hyperparameters
    params = HyperParams(horizon=horizons[spec_num], num_verify_samples=NUM_STEPS)

    # estimate mdp and perform nash value iteration
    start_time = time.time()
    joint_policy, steps = estimate_and_solve(env, specs[spec_num], pred_fns, params)
    time_taken = time.time() - start_time

    # render learnt policy
    if render:
        print_new_block('Rendering NE Policy')
        get_rollout(env, joint_policy, True, stateful_policy=True, max_timesteps=params.horizon)

    # Evaluate learned policy
    eval_dict = test_nash_solution(env, joint_policy, horizons[spec_num], steps)
    save_object('eval', eval_dict, itno, folder)
