from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.multi_agent.reachability import (
    MultiAbstractReachability, HierarchicalPolicy, ConstrainedEnv)
from spectrl.main.spec_compiler import ev, alw, seq, choose
from spectrl.util.io import (parse_command_line_options, save_log_info,
                             save_object, load_object)
from spectrl.util.rl import get_rollout, test_policy_mutli
from spectrl.rl.ddpg import DDPGParams
from spectrl.rl.ddpg.multi_agent import MultiDDPGParams
from spectrl.rl.maddpg import MADDPGParams
from spectrl.envs.particles import MultiParticleEnv
from numpy import linalg as LA

import numpy as np
import gym
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

DDPG_ITERS = [2000]
MULTI_DDPG_ITERS = [5000]
MADDPG_ITERS = [10000]
OBS_SCALE = 0.01
AGENT_SIZE = 1
DIM = 2
MAX_STEPS = 20
TEST_STEPS = 100


# Starting locations
start_pos_low = [[-0.1, -0.1], [-0.1, 4.9], [-0.1, 9.9]]
start_pos_high = [[0.1, 0.1], [0.1, 5.1], [0.1, 10.1]]


# Wrapper to keep track of goals
class GoalLimitWrapper(MultiParticleEnv):

    def __init__(self, pred, *args, **kwargs):
        self.pred = pred
        super().__init__(*args, **kwargs)
        if self.pred is not None:
            self.observation_space = [gym.spaces.Box(low=-np.inf, high=np.inf,
                                                     shape=(space.shape[0] + self.n,))
                                      for space in self.observation_space]

    def reset(self):
        self.additional_state = np.array([0.] * self.n)
        obs = super().reset()
        if self.pred is not None:
            obs = self._augment_obs(obs)
        return obs

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        if self.pred is not None:
            for i in range(len(obs)):
                if self.pred(obs[i], None) > 0:
                    self.additional_state[i] = OBS_SCALE * 10.
            obs = self._augment_obs(obs)
        return obs, rew, done, info

    def get_sim_state(self):
        state = super().get_sim_state()
        if self.pred is not None:
            state = state, self.additional_state.copy()
        return state

    def set_sim_state(self, full_state):
        state = full_state
        if self.pred is not None:
            state, self.additional_state = full_state
        obs = super().set_sim_state(state)
        if self.pred is not None:
            obs = self._augment_obs(obs)
        return obs

    def _augment_obs(self, obs):
        new_obs = []
        for o in obs:
            new_obs.append(np.concatenate([o, self.additional_state]))
        return new_obs


# Relevant atomic predicates:
# a. Reach predicate
#    goal: np.array(2), err: float
def reach(goal, err=1):
    scaled_goal = OBS_SCALE*goal
    scaled_err = OBS_SCALE*err

    def predicate(sys_state, res_state):
        return -LA.norm(sys_state[:DIM] - scaled_goal) + scaled_err
    return predicate


# satisfy pred and be among the first K to reach it
def pred_first(pred, num_agents, first_K=2):

    def predicate(sys_state, res_state):
        val = pred(sys_state, res_state)
        num_reached = np.sum((sys_state[num_agents*DIM:] > 0.).astype(np.int))
        if num_reached <= first_K:
            return val
        else:
            return -1.
    return predicate


# b. Avoid other agents
def safe(num_agents):

    def predicate(sys_state, res_state):
        min_dist = np.inf
        for i in range(1, num_agents):
            min_dist = min(min_dist, LA.norm(sys_state[i*DIM:(i+1)*DIM]))
        return min_dist - (AGENT_SIZE*OBS_SCALE)
    return predicate


# c. Prevent another agent from reaching goal
def prevent_reach(agent, goal, err=1):
    scaled_goal = OBS_SCALE*goal
    scaled_err = OBS_SCALE*err
    a_pos = agent*DIM

    def predicate(sys_state, res_state):
        rel_goal = scaled_goal - sys_state[:DIM]
        return LA.norm(sys_state[a_pos:a_pos+DIM] - rel_goal) - scaled_err
    return predicate


# predicates for agents not making progress
def stay_pred(pred):
    def predicate(sys_state, res_state):
        return pred(sys_state, res_state) + OBS_SCALE*2
    return predicate


# initial stay pred
def init_stay_pred(agent):
    return stay_pred(reach(np.array(start_pos_low[agent])))


# Goals and obstacles
gtop = np.array([0.0, 5.0])
gbot = np.array([0.0, 0.0])
gright = np.array([5.0, 0.0])
gcorner = np.array([5.0, 5.0])
gcorner2 = np.array([-5.0, 5.0])
gmid = np.array([0.0, 2.5])
gdown = np.array([0.0, -2.5])

# Specifications
specs0 = [alw(safe(2), ev(reach(gtop, 1))), alw(safe(2), ev(reach(gcorner, 1)))]
specs1 = [seq(alw(safe(2), ev(reach(gtop, 1))), alw(safe(2), ev(reach(gbot, 1)))),
          seq(alw(safe(2), ev(reach(gcorner, 1))), alw(safe(2), ev(reach(gtop, 1))))]
specs2 = [alw(safe(2), ev(reach(gtop, 1))), alw(safe(2), ev(reach(gbot, 1)))]
specs3 = [alw(safe(2), ev(reach(gmid, 0.3))), alw(safe(2), ev(reach(gmid, 0.3)))]
specs4 = [choose(alw(safe(2), ev(reach(gdown, 0.3))), alw(safe(2), ev(reach(gright, 0.3)))),
          alw(safe(2), ev(reach(gdown, 0.3)))]
specs5 = [alw(prevent_reach(2, gtop, 1), ev(reach(gright, 1))),
          alw(prevent_reach(2, gtop, 1), ev(reach(gcorner, 1))),
          ev(reach(gtop, 1))]
specs6 = [alw(safe(3), ev(pred_first(reach(gright, 0.5), 3))),
          seq(alw(safe(3), ev(pred_first(reach(gright, 0.5), 3))),
              seq(alw(safe(3), ev(reach(gcorner, 0.4))),
                  alw(safe(3), alw(reach(gcorner, 0.4), ev(reach(gcorner, 0.4)))))),
          seq(alw(safe(3), ev(pred_first(reach(gright, 0.5), 3))),
              seq(alw(safe(3), ev(reach(gcorner, 0.4))),
                  alw(safe(3), alw(reach(gcorner, 0.4), ev(reach(gcorner, 0.4))))))]
specs = [specs0, specs1, specs2, specs3, specs4, specs5, specs6]
goal_wrapper_preds = [None, None, None, None, None, None, reach(gright, 0.5)]

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    use_gpu = flags['gpu_flag']
    render = flags['render']
    spec_num = flags['spec_num']
    folder = flags['folder']
    itno = flags['itno']
    algo = flags['alg']
    no_stay = flags['no_stay']
    test = flags['test']

    # setup the corrext number of agents
    n_agents = len(specs[spec_num])
    start_pos_low = start_pos_low[:n_agents]
    start_pos_high = start_pos_high[:n_agents]

    log_info = []
    logdir = os.path.join(folder, 'spec{}'.format(spec_num), 'nash')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    for i in range(len(DDPG_ITERS)):

        env = GoalLimitWrapper(goal_wrapper_preds[spec_num], start_pos_low,
                               start_pos_high, AGENT_SIZE,
                               max_timesteps=MAX_STEPS, obs_multiplier=OBS_SCALE)
        state_dims = [space.shape[0] for space in env.observation_space]
        action_dims = [space.shape[0] for space in env.action_space]
        action_bounds = [space.high for space in env.action_space]
        ddpg_params = DDPGParams(state_dims[0], action_dims[0], action_bounds[0],
                                 minibatch_size=100, num_episodes=DDPG_ITERS[i],
                                 discount=0.95, actor_hidden_dim=16,
                                 critic_hidden_dim=16, steps_per_update=1)
        multi_ddpg_params = MultiDDPGParams(state_dims, action_dims, action_bounds,
                                            minibatch_size=512, discount=0.95,
                                            warmup=5000, actor_hidden_dim=16,
                                            critic_hidden_dim=32,
                                            num_episodes=MULTI_DDPG_ITERS[i],
                                            gradients_per_update=1, steps_per_update=1,
                                            epsilon_min=0.5, epsilon_decay=2e-6, sigma=0.1,
                                            buffer_size=100000)
        maddpg_params = MADDPGParams(MAX_STEPS, MADDPG_ITERS[i], num_units=32, lr=3e-4)

        print('\n**** Learning Policy for Spec {} ****'.format(spec_num))

        abstract_graphs = []
        final_vertices = []
        for j in range(n_agents):
            _, abstract_reach = automaton_graph_from_spec(specs[spec_num][j])
            print('\n**** Abstract Graph for Agent {} ****'.format(j))
            abstract_reach.pretty_print()
            abstract_graphs.append(abstract_reach.abstract_graph)
            final_vertices.append(abstract_reach.final_vertices)

        # Initialize multi reachability
        null_actions = [np.zeros((d,)) for d in action_dims]
        abstract_reach = MultiAbstractReachability(
            abstract_graphs, final_vertices, env, ddpg_params, maddpg_params, multi_ddpg_params,
            max_steps=MAX_STEPS, safety_penalty=-OBS_SCALE, neg_inf=-1., use_gpu=use_gpu,
            render=render, null_actions=null_actions, no_stay=no_stay, stay_pred=stay_pred,
            init_stay_pred=init_stay_pred)

        if not test:

            # Step 5: Learn policy
            if algo == 'topo':
                abstract_policy, nn_policies, stats = abstract_reach.learn_nash_topo()
            elif algo == 'dijkstra':
                abstract_policy, nn_policies, stats = abstract_reach.learn_nash_dijkstra()
            else:
                raise ValueError('Invalid algorithm!')

            # Print statements
            print('\nTotal sample steps: {}'.format(stats[0]))
            print('Total time taken: {}'.format(stats[1]))
            print('Total edges learned: {}'.format(stats[2]))

        else:

            # Load policy from file
            abstract_policy = load_object('abstract_policy_{}'.format(i), itno, logdir)
            nn_policies = load_object('nn_policies_{}'.format(i), itno, logdir)

        # Visualize abstract policy
        print('\nAbstract policy: {}'.format(abstract_policy))

        # Evaluate the learned policy
        policy = HierarchicalPolicy(
            abstract_policy, nn_policies, abstract_reach, null_actions, no_stay=no_stay)
        test_env = ConstrainedEnv(env, abstract_reach, abstract_policy, max_steps=TEST_STEPS)
        avg_rewards, succ_rates = test_policy_mutli(test_env, policy, 200, stateful_policy=True)

        # More print statements
        print('\nAverage rewards: {}'.format(avg_rewards.tolist()))
        print('Success rates: {}'.format(succ_rates.tolist()))
        print('\nSimulating learned policy...')
        get_rollout(test_env, policy, True, stateful_policy=True)

        # Update log
        if not test:
            log_info.append([stats[0], stats[1]])
            save_object('abstract_policy_{}'.format(i), abstract_policy, itno, logdir)
            save_object('nn_policies_{}'.format(i), nn_policies, itno, logdir)

    if not test:
        save_log_info(log_info, itno, logdir)
