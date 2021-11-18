from spectrl.envs.intersection import IntersectionEnv
# from spectrl.multi_agent.finite_state import HyperParams, MultiAbstractReachability
from spectrl.rl.qrmlearning import MultiQRMAgent
from spectrl.main.spec_compiler import ev, alw
from spectrl.multi_agent.ne_verification import test_nash_solution
from spectrl.util.rl import get_rollout, test_policy_mutli
from spectrl.util.io import print_new_block, parse_command_line_options, save_object

import os
import random
import numpy as np

SEEDS = [907, 651, 985,	221, 182, 53, 152, 142, 224, 386]
MAX_STEPS = 20


# Add predicates
# Safety predicate
def no_collision(num_ns):
    def predicate(state):
        ns_at_intersection = False
        for a in range(num_ns):
            if state[a] == 1:
                ns_at_intersection = True

        ew_at_intersection = False
        for a in range(num_ns, len(state)):
            if state[a] == 1:
                ew_at_intersection = True

        return not(ns_at_intersection and ew_at_intersection)
    return predicate


def ns_crosses_first(agent, num_ns):
    def predicate(state):
        ew_crossed = False
        for ew_a in range(num_ns, len(state)):
            if state[ew_a] < 1:
                ew_crossed = True
        return ((state[agent] < 1) and not(ew_crossed))
    return predicate


def ew_crosses_first(agent, num_ns):
    def predicate(state):
        ns_crossed = False
        for ns_a in range(num_ns):
            if state[ns_a] < 1:
                ns_crossed = True
        return (state[agent] < 1 and not(ns_crossed))
    return predicate


def crosses_before(agent, agent_list):
    def predicate(state):
        crossed = False
        for a in agent_list:
            if state[a] < 1:
                crossed = True
        return (state[agent] < 1 and not(crossed))
    return predicate


def maintain_lead(agent, agent_list):
    def predicate(state):
        if state[agent] < 1:
            return 1.
        retval = True
        for a in agent_list:
            retval = retval and (state[a] > state[agent]+1)
        return retval
    return predicate


init_locations = [[[3, 3], [2]], [[2], [2, 4, 4]], [[2], [2, 4, 4]], [[3, 3], [3]],
                  [[2, 3], [2, 3, 4]], [[3, 3], [2]], [[3, 3], [3]], [[2, 3], [2, 3, 4]]]
num_ns = [len(init_location[0]) for init_location in init_locations]
stay_probs = [0.05, 0.05, 0.05, 0.05, 0.05, 0.4, 0.4, 0.4]


# ------------------------------------------------------------------------------------------------------
spec0 = [alw('no_collision', ev('cross_0')),
         alw('no_collision', ev('cross_1')),
         alw('no_collision', ev('cross_2'))]

spec1 = [alw('no_collision', ev('0b23')),
         alw('no_collision', ev('1b0')),
         alw('no_collision', ev('2b0')),
         alw('no_collision', ev('3b0'))]

spec2 = [alw('no_collision', ev('0b23')),
         alw('1l23', alw('no_collision', ev('1b0'))),
         alw('no_collision', ev('2b0')),
         alw('no_collision', ev('3b0'))]

spec3 = [alw('no_collision', ev('0b12')),
         alw('no_collision', ev('1b')),
         alw('no_collision', ev('2b1'))]

spec4 = [alw('no_collision', ev('cross_0')),
         alw('no_collision', ev('cross_1')),
         alw('no_collision', ev('cross_2')),
         alw('no_collision', ev('cross_3')),
         alw('no_collision', ev('cross_4'))]

spec5 = spec0
spec6 = spec3
spec7 = spec4

specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6, spec7]
horizons = [20, 20, 20, 20, 20, 20, 20, 20]
num_steps = np.array([20, 150, 150, 22, 600, 20, 20, 600], dtype=np.int) * 100000


if __name__ == '__main__':

    flags = parse_command_line_options()
    spec_num = flags['spec_num']
    itno = flags['itno']
    folder = flags['folder']
    render = flags['render']
    folder = os.path.join(folder, 'spec{}'.format(spec_num), 'multi_qrm')
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.random.seed(SEEDS[itno])
    random.seed(SEEDS[itno])

    # Add specs
    pred_fns = {'no_collision': no_collision(num_ns[spec_num]),
                'cross_0': ns_crosses_first(0, num_ns[spec_num]),
                'cross_1': ns_crosses_first(1, num_ns[spec_num]),
                'cross_2': ew_crosses_first(2, num_ns[spec_num]),
                'cross_3': ew_crosses_first(3, num_ns[spec_num]),
                'cross_4': ew_crosses_first(4, num_ns[spec_num]),
                '0b23': crosses_before(0, [2, 3]),
                '0b12': crosses_before(0, [1, 2]),
                '1b0': crosses_before(1, [0]),
                '2b1': crosses_before(2, [1]),
                '2b0': crosses_before(2, [0]),
                '3b0': crosses_before(3, [0]),
                '1l23': maintain_lead(1, [2, 3]),
                '1b': crosses_before(1, [])}

    # Create environment
    env = IntersectionEnv(init_locations[spec_num], stay_prob=stay_probs[spec_num],
                          max_steps=horizons[spec_num])

    # Create Reward Machine from Specifications
    rm_list = [spec.get_rm(pred_fns) for spec in specs[spec_num]]

    # Create Multi-Agent QRM Environment

    model = MultiQRMAgent(len(specs[spec_num]))

    # Train Joint Policy
    joint_policy, _ = model.learn(env, rm_list, num_steps[spec_num], itno, folder)

    # Evaluate Joint Policy

    _, av_prob = test_policy_mutli(
        env, joint_policy, 500, use_rm_reward=True, stateful_policy=True,
        max_timesteps=horizons[spec_num])

    print("Success Probability is {}".format(av_prob))

    # Render
    print_new_block('Rendering NE Policy')
    get_rollout(env, joint_policy, True, stateful_policy=True, max_timesteps=horizons[spec_num])

    # Evaluate learned policy
    eval_dict = test_nash_solution(env, joint_policy, horizons[spec_num], num_steps[spec_num])
    save_object('eval', eval_dict, itno, folder)
