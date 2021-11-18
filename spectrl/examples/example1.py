from spectrl.envs.finite_mdp import FiniteMDP
from spectrl.multi_agent.finite_state import HyperParams, MultiAbstractReachability
from spectrl.main.spec_compiler import ev
from spectrl.util.rl import get_rollout
from spectrl.util.io import print_new_block

MAX_STEPS = 10

# Add predicates


def satisfy0():
    def predicate(state):
        return (state == 1) or (state == 3)
    return predicate


def satisfy1():
    def predicate(state):
        return (state == 2) or (state == 3)
    return predicate

# Add specs


pred_fns = {'satisfy0': satisfy0(),
            'satisfy1': satisfy1()}

spec0 = [ev('satisfy0'),
         ev('satisfy1')]

specs = [spec0]
horizons = [10]

if __name__ == '__main__':

    spec_num = 0

    # Create environment

    start_state = 0
    num_actions = [2, 2]
    transitions = {}

    # transitions[(joint action)][state_num] = Prob Distribution

    transitions[(0, 0)] = [[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]]
    transitions[(1, 0)] = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
    transitions[(0, 1)] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    transitions[(1, 1)] = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]

    env = FiniteMDP(transitions, start_state, num_actions)

    # Choose Hyperparameters

    params = HyperParams(max_steps=MAX_STEPS, num_training_steps=100000,
                         num_estimation_rollouts=500, horizon=horizons[spec_num],
                         render=True)

    # Create reachability instance
    model = MultiAbstractReachability(env, specs[spec_num], pred_fns, params)

    # Learn best nash policy
    joint_policy = model.learn_best_nash()

    # render learnt policy
    print_new_block('Rendering NE Policy')
    get_rollout(env, joint_policy, True, stateful_policy=True, max_timesteps=params.horizon)
