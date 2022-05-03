from spectrl.envs.sortworld import SortEnv
from spectrl.rl.qrmlearning import MultiQRMAgent
from spectrl.multi_agent.ne_verification import test_nash_solution
from spectrl.main.spec_compiler import ev, alw, seq
from spectrl.util.rl import get_rollout, test_policy_mutli
from spectrl.util.io import print_new_block, parse_command_line_options, save_object, save_log_info

import os
import random
import numpy as np

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

# At least one of Agent 0 and Agent 1 will have to sacrifice its goal. Nash equilibria are [1,0,1],
# or [0,1,1], or [0,0,1]

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
# If p = 1, then [1,1,1]. For other values of p, prob. of satisfiability will change
# since co-ordination is involved.

spec7 = [seq(alw('a1_not_reach_mid', ev('a0_reach_mid')), ev('a1_reach_mid')),
         seq(alw('a0_not_reach', ev('a1_reach')), ev('a0_reach')),
         seq(alw('a2_not_reach', ev('a0_reach')), ev('a2_reach'))]

specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6, spec7]

num_steps = [4000000, 4000000, 4000000, 4000000, 4000000,
             4000000, 4000000, 4000000, 4000000, 4000000]

horizons = [20, 20, 20, 20, 20, 20, 20, 20]

if __name__ == '__main__':

    flags = parse_command_line_options()
    spec_num = flags['spec_num']
    itno = flags['itno']
    folder = flags['folder']
    folder = os.path.join(folder, 'spec{}'.format(spec_num), 'multi_qrm')
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.random.seed(SEEDS[itno])
    random.seed(SEEDS[itno])

    # Create environment
    env = SortEnv(len(specs[spec_num]), 0.95, depth, max_steps=horizons[spec_num])

    # Create Reward Machine from Specifications
    rm_list = [spec.get_rm(pred_fns) for spec in specs[spec_num]]

    # Create Multi-Agent QRM Environment
    model = MultiQRMAgent(len(specs[spec_num]))

    # Train Joint Policy
    joint_policy, log_info = model.learn(env, rm_list, num_steps[spec_num], itno, folder)

    save_log_info(log_info, itno, folder)

    # Evaluate Joint Policy
    _, av_prob = test_policy_mutli(
        env, joint_policy, 100, use_rm_reward=True, stateful_policy=True,
        max_timesteps=horizons[spec_num])

    print("Success Probability is {}".format(av_prob))

    # Render
    print_new_block('Rendering NE Policy')
    get_rollout(env, joint_policy, True, stateful_policy=True, max_timesteps=horizons[spec_num])

    # Evaluate learned policy
    eval_dict = test_nash_solution(env, joint_policy, horizons[spec_num], num_steps[spec_num])
    save_object('eval', eval_dict, itno, folder)
