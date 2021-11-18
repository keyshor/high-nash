from spectrl.envs.gridworld import GridWorld
from spectrl.multi_agent.finite_state import HyperParams
from spectrl.multi_agent.nash_value_iteration import estimate_and_solve
from spectrl.multi_agent.ne_verification import test_nash_solution
from spectrl.main.spec_compiler import ev, alw, seq, choose
from spectrl.util.io import parse_command_line_options, print_new_block, save_object
from spectrl.util.rl import get_rollout

import os
import random
import numpy as np

MAX_STEPS = 20
NUM_STEPS = [3500, 5500, 6000, 9000, 14500, 20000, 9000]

SEEDS = [249, 801, 501, 759, 434, 949, 513, 573, 383, 204]


# Predicate for reachability
def reach(point, agent):
    def predicate(states):
        return -(abs(states[agent][0] - point[0]) + abs(states[agent][1] - point[1])) + 1
    return predicate


# Predicate for safety
def safe(agent):
    def predicate(states):
        retval = np.inf
        for a in range(len(states)):
            if a != agent:
                retval = min(retval, abs(states[agent][0] - states[a][0]) +
                             abs(states[agent][1] - states[a][1]))
        return retval
    return predicate


pred_fns = {'a1_safe': safe(0),
            'a2_safe': safe(1),
            'a1_reach_bottom_left': reach((0, 0), 0),
            'a1_reach_bottom_right': reach((0, 3), 0),
            'a1_reach_top_left': reach((3, 0), 0),
            'a1_reach_top_right': reach((3, 3), 0),
            'a2_reach_bottom_left': reach((0, 0), 1),
            'a2_reach_bottom_right': reach((0, 3), 1),
            'a2_reach_top_left': reach((3, 0), 1),
            'a2_reach_top_right': reach((3, 3), 1),
            'a1_reach_middle': reach((1, 1), 0),
            'a2_reach_middle': reach((1, 1), 1)}


############################################
# Swap positions (without choice)
############################################

spec0 = [alw('a1_safe', ev('a1_reach_top_right')),
         alw('a2_safe', ev('a2_reach_bottom_left'))]

spec1 = [seq(alw('a1_safe', ev('a1_reach_top_left')), alw('a1_safe', ev('a1_reach_top_right'))),
         seq(alw('a2_safe', ev('a2_reach_bottom_right')), alw('a2_safe', ev('a2_reach_top_right')))]

##########################################
# Choose locations to navigate safety
##########################################

# Both agents choose between the same set of locations.
spec2 = [alw('a1_safe', choose(ev('a1_reach_top_left'), ev('a1_reach_bottom_right'))),
         alw('a2_safe', choose(ev('a2_reach_top_left'), ev('a1_reach_bottom_right')))]

# Extend spec2 to swap positions
spec3 = [seq(spec2[0], alw('a1_safe', ev('a1_reach_top_right'))),
         seq(spec2[1], alw('a2_safe', ev('a2_reach_bottom_left')))]

# Extend spec3 to swap positions again (directly)
spec4 = [seq(spec3[0], alw('a1_safe', ev('a1_reach_bottom_left'))),
         seq(spec3[1], alw('a2_safe', ev('a2_reach_top_right')))]

# Extend spec4 to swap positions again (directly)
spec5 = [seq(spec4[0], alw('a1_safe', ev('a1_reach_top_right'))),
         seq(spec4[1], alw('a2_safe', ev('a2_reach_bottom_left')))]


# Extend spec2 to reach the same position (middle point) - agents must
# coordinate their arrival at the mid-point
spec6 = [seq(spec2[0], alw('a1_safe', ev('a1_reach_middle'))),
         seq(spec2[1], alw('a2_safe', ev('a2_reach_middle')))]


start_positions = [(0, 0), (3, 3)]

specs = [spec0, spec1, spec2, spec3, spec4, spec5, spec6]

horizons = [20, 20, 20, 20, 20, 20, 20]

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
    env = GridWorld((4, 4), start_positions)

    # Choose Hyperparameters
    params = HyperParams(horizon=horizons[spec_num], num_verify_samples=NUM_STEPS[spec_num])

    # Estimate MDP and perform nash value iteration
    joint_policy, steps = estimate_and_solve(env, specs[spec_num], pred_fns, params)

    # render learnt policy
    if render:
        print_new_block('Rendering NE Policy')
        get_rollout(env, joint_policy, True, stateful_policy=True, max_timesteps=params.horizon)

    # Evaluate learned policy
    eval_dict = test_nash_solution(env, joint_policy, horizons[spec_num], steps)
    save_object('eval', eval_dict, itno, folder)
