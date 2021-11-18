'''
Multi agent RL from specifications for finite state MDPs.
'''

import gym
import time
import itertools
import numpy as np

from spectrl.util.rl import get_rollout, test_policy_mutli
from spectrl.util.io import print_new_block
from spectrl.main.spec_compiler import land
from spectrl.util.dist import FiniteDistribution
from spectrl.rl.qlearning import QLearningAgent
from spectrl.hierarchy.reachability import AbstractEdge
from spectrl.multi_agent.ne_verification import (
    FiniteStatePolicy, estimate_mdp, get_product_mdp,
    nash_value_iteration_zero_sum, FullNashPolicy)
from spectrl.hierarchy.construction import true_pred, automaton_graph_from_spec


class HyperParams:
    '''
    Hyperparameters for learning edge policies.
    '''

    def __init__(self, max_steps=100, horizon=100, safety_penalty=-1, neg_inf=-10, alpha=0,
                 epsilon=0.15, ne_epsilon=0.05, lr=0.1, gamma=0.9, num_training_steps=10000,
                 num_estimation_rollouts=500, num_verify_samples=1000,
                 null_actions=None, render=False, verbose=False, use_lp=True):
        '''
        max_steps: int
        safety_penalty: float (min penalty for violating constraints)
        neg_inf: float (negative reward for failing to satisfy constraints)
        alpha: float (alpha * original_reward will be added to reward)
        epsilon: float (to be used in Q-learning for ep-greedy exploration)
        lr: float (learning rate for Q-learning)
        gamma: float (discount factor to be used in Q-learning)
        num_training_steps: int (number of sample steps for training edge policies)
        num_estimation_rollouts: int (number of sample rollouts to estimate edge
                                        satisfaction probabilities)
        null_actions: List[int] (list of actions that enable agents to stay)
        render: bool (whether to render rollouts after learning each edge)
        '''
        self.max_steps = max_steps
        self.horizon = horizon
        self.safety_penalty = safety_penalty
        self.neg_inf = neg_inf
        self.alpha = alpha
        self.epsilon = epsilon
        self.ne_epsilon = ne_epsilon
        self.lr = lr
        self.gamma = gamma
        self.num_training_steps = num_training_steps
        self.num_verify_samples = num_verify_samples
        self.num_estimation_rollouts = num_estimation_rollouts
        self.null_actions = null_actions
        self.render = render
        self.verbose = verbose
        self.use_lp = use_lp


class MultiReachabilityEnv(gym.Env):
    '''
    Environment for training a co-operative joint policy for an edge in product abstract graph.
    Terminates when all agents reach their goals.
    '''

    def __init__(self, multi_env, params, init_dist=None, final_preds=None, constraints=None,
                 after_reach_contraints=None, non_final_agents=[]):
        '''
        Parameters:
            multi_env: Multi-agent gym.Env (with set_state() method)
            init_dist: Distribution (initial state distribution)
            final_preds: List[state_i, resource_i -> float] (Goal of the reachability task,
                            can be none for adversarial agents)
            constraints: List[List[state_i, resource_i -> float]]
                (Constraints that need to be satisfied (defines reward function))
            params: HyperParams
            final_agents: agents that start in a final abstract state
            null_actions: list of actions that enable agents to stay
        '''
        self.wrapped_env = multi_env
        self.init_dist = init_dist
        self.final_preds = final_preds
        self.constraints = constraints
        self.params = params
        self.non_final_agents = non_final_agents
        self.after_reach_constraints = after_reach_contraints
        self.is_self_loop = np.all([pred is None for pred in final_preds])
        self.n = len(self.wrapped_env.action_space)

        # Handle undefined inputs
        if self.final_preds is None:
            self.final_preds = [None] * self.n
        if self.constraints is None:
            self.constraints = [[true_pred] for _ in range(self.n)]
        if self.after_reach_constraints is None:
            self.after_reach_constraints = [[true_pred] for _ in range(self.n)]

        # Set observation and action spaces
        self.observation_space = self.wrapped_env.observation_space
        self.action_space = self.wrapped_env.action_space

        # Reset the environment
        self.reset()

    def reset(self):
        # Reset wrapped env
        self.obs = self.wrapped_env.reset()

        # Set correct sys and res states based on init dist
        if self.init_dist is not None:
            sim_state = self.init_dist.sample()
            self.obs = self.wrapped_env.set_sim_state(sim_state)

        # Initialize additional bookeeping info
        self.violated_constraints = [0] * self.n
        self.prev_safety_rewards = [self.params.neg_inf] * self.n

        self.goal_reached = [False] * self.n
        for a in range(self.n):
            if self.final_preds[a] is None:
                self.goal_reached[a] = True

        # Reset time
        self.t = 0
        return self.obs

    def step(self, actions):
        # Set actions of agents already in goal to null action
        corrected_actions = list(actions)
        if self.params.null_actions is not None:
            for a in range(self.non_final_agents):
                if self.goal_reached[a]:
                    corrected_actions[a] = self.params.null_actions[a]

        # Take step in system and resource model
        self.obs, r, _, _ = self.wrapped_env.step(tuple(corrected_actions))
        self.t += 1

        # Total reward
        reward = sum([self.reward(a) + self.params.alpha * min(r, 0) for a in range(self.n)])

        # Update agents' done status
        all_reached = True
        for a in range(self.n):
            if not self.goal_reached[a]:
                self.goal_reached[a] = (self.final_preds[a](self.obs, None) > 0 and
                                        self.violated_constraints[a] < len(self.constraints[a]))
                all_reached = all_reached and self.goal_reached[a]

                # after reach constraints once agent reaches its goal
                if self.goal_reached[a]:
                    self.violated_constraints[a] = 0
            else:
                all_reached = all_reached and (self.violated_constraints[a] < len(
                    self.after_reach_constraints[a]))
        done = self.t >= self.params.max_steps
        if not self.is_self_loop:
            done = done or all_reached

        return self.obs, reward, done, {}

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)

    def get_state(self):
        return self.wrapped_env.get_sim_state()

    def reward(self, agent):
        # Compute reward for reaching goal states
        reach_reward = 0
        if not self.goal_reached[agent]:
            reach_reward = self.final_preds[agent](self.obs, None)

        # Compute reward for obeying (or violating) constraints
        safety_reward = self.prev_safety_rewards[agent]
        if self.goal_reached[agent]:
            constraints = self.after_reach_constraints[agent]
        else:
            constraints = self.constraints[agent]
        for i in range(self.violated_constraints[agent], len(constraints)):
            cur_constraint_val = constraints[i](self.obs, None)
            safety_reward = max(safety_reward, cur_constraint_val)
            if cur_constraint_val > 0:
                break
            self.violated_constraints[agent] += 1

        # Ignore safety reward if safe
        if safety_reward > 0:
            return reach_reward

        safety_reward = min(safety_reward, self.params.safety_penalty)
        self.prev_safety_rewards[agent] = safety_reward
        return reach_reward + safety_reward

    def satisfies_edge(self, obs_list, agent):
        violated_constraints = 0
        reached_goal = self.final_preds[agent] is None
        always_safe = True

        for obs in obs_list:

            # Safety
            if reached_goal:
                constraints = self.after_reach_constraints[agent]
            else:
                constraints = self.constraints[agent]
            for i in range(violated_constraints, len(constraints)):
                if constraints[i](obs, None) <= 0:
                    violated_constraints += 1
            always_safe = always_safe and violated_constraints < len(constraints)

            # Reachability
            if not reached_goal:
                reached_goal = (self.final_preds[agent](obs, None) > 0 and always_safe)
                if reached_goal:
                    violated_constraints = 0

        return reached_goal and always_safe

    def close(self):
        self.wrapped_env.close()


class MultiAbstractReachability:
    '''
    Class defining the multi-agent abstract reachability problem.
    '''

    def __init__(self, multi_env, specs, pred_fns, hyperparams):
        '''
        Parameters:
            specs: List[TaskSpec] (spec for each agent).
            multi_env: gym.Env (finite state multi-agnt environment)
                        action_space: gym.space.Tuple
            hyperparams: HyperParams (hyperparameters for learning)
        '''

        self.multi_env = multi_env
        self.reward_machines = [spec.get_rm(pred_fns) for spec in specs]

        specs = [spec.add_pred_fns(pred_fns) for spec in specs]

        # Specification compilation
        automata_graphs = [automaton_graph_from_spec(spec) for spec in specs]
        self.abstract_graphs = [r.abstract_graph for _, r in automata_graphs]
        self.final_vertices = [r.final_vertices for _, r in automata_graphs]
        self.n = len(specs)  # number of agents
        self.num_vertices = [len(g) for g in self.abstract_graphs]
        self.params = hyperparams
        self.P = None  # probability estimates used in verification

        # Precompute mappings from edges to predicates and constraints
        self.pred_maps = []
        self.constraint_maps = []
        for a in range(self.n):
            p_map, c_map = self.compute_edge_maps(a)
            self.pred_maps.append(p_map)
            self.constraint_maps.append(c_map)

    def learn_best_nash(self):
        '''
        Computes and returns the highest value path policy that can be
            verified to be a nash.
        '''
        steps_taken = 0
        start_time = time.time()

        all_paths = []
        edge_policy_map = {}

        # Iterate over all sets of agents
        for sac_int in range(2**self.n-1):
            sac_agents = self.int_to_set(sac_int)

            # Learn edge policies for the corresponding graph
            print_new_block('Learning Edge Policies for Sac Set: {}'.format(sac_agents))
            topo_list, edge_policies, edge_costs, stats = self.learn_edge_policies(sac_agents)
            print('\nTime taken for learning {} edges for set {}: {} secs'.format(
                stats[0], sac_agents, stats[-1]))
            steps_taken += stats[1]
            edge_policy_map[sac_int] = edge_policies

            # Compute and add all full paths
            all_paths.extend(self.compute_all_paths(
                topo_list, edge_policies, edge_costs, sac_agents))

        print_new_block('Computing Best Verifiable NE Policy')
        fs_policy, steps, probs = self.compute_best_nash(all_paths, edge_policy_map)
        steps_taken += steps

        print_new_block('Learning Complete')
        print('\nTotal time taken: {} secs'.format(time.time() - start_time))
        print('Total sample steps taken: {}'.format(steps_taken))
        if fs_policy is None:
            print('No nash policy found!!')

        return fs_policy, probs, steps_taken, (time.time() - start_time)

    def learn_edge_policies(self, sac_agents=set()):
        '''
        Learn one joint policy for each edge in the product abstract graph.

        Returns:
            edge_policies: Product edges -> Joint policy
            edge_costs: Product edges -> neg-log probability of satisfying the edge
            stats: [edges_learned, steps_taken, time_taken]
        '''
        # Get topo sort of product graph
        topo_list = list(self.topo_sort(sac_agents=sac_agents))

        # Initialize edge policies
        edge_policies = {}

        # Average reach distribution for each vertex
        reach_dist = {}
        _, init_vertex = topo_list[0]
        reach_dist[init_vertex] = None

        # Costs
        edge_costs = {}  # for each edge (neg-log probability)

        # Other stats
        num_edges_learned = 0
        total_steps = 0
        start_time = time.time()

        # Set of bad edges for which RL fails to learn a policy
        # useful for debugging and tuning RL algorithm
        bad_edges = []

        for vertex_list, vertex in topo_list:
            if vertex not in reach_dist:
                continue

            for edge in self.compute_outgoing_edges(vertex_list, sac_agents=sac_agents):
                target_list, target = self.compute_target(vertex_list, edge)

                print_new_block('Learning Joint Policy for Edge {} -> {}'.format(vertex, target))
                print_prefix = '({})[{} -> {}] '.format(self.set_to_int(sac_agents), vertex, target)

                # Learn edge policies
                joint_policy, reach_env, steps = self.learn_single_edge(
                    edge, vertex_list, target_list, reach_dist[vertex],
                    print_prefix=print_prefix, sac_agents=sac_agents)
                edge_policies[(vertex, target)] = joint_policy

                # Update stats
                num_edges_learned += 1
                total_steps += steps

                # Compute reach probability and collect visited states
                reach_prob, states_reached, steps_taken = self.compute_reach_prob(
                    reach_env, joint_policy)
                total_steps += steps_taken
                edge_cost = -np.log(reach_prob)

                # Print probability
                print('\n{}Reach probability: {}'.format(print_prefix, reach_prob))

                # Edge is valid if it can possibly be a part of an NE path
                if reach_prob > 0:
                    if target not in reach_dist:
                        reach_dist[target] = FiniteDistribution()
                    if target != vertex:
                        reach_dist[target].add_points(states_reached)
                    edge_costs[(vertex, target)] = edge_cost
                else:
                    bad_edges.append((vertex, target))

        total_time = time.time() - start_time
        return topo_list, edge_policies, edge_costs, [num_edges_learned, total_steps, total_time]

    def compute_all_paths(self, topo_list, edge_policies, edge_costs, sac_agents=set()):

        # store all paths and corresponding costs
        _, init_vertex = topo_list[0]
        paths = {init_vertex: [[init_vertex]]}
        path_costs = {init_vertex: [0]}

        # paths to final states along with their total costs
        full_paths = []

        for vertex_list, vertex in topo_list:
            if vertex not in paths or len(paths[vertex]) == 0:
                continue

            for edge in self.compute_outgoing_edges(vertex_list, sac_agents=sac_agents):
                target_list, target = self.compute_target(vertex_list, edge)
                if (vertex, target) not in edge_costs or vertex == target:
                    continue

                # first time a target is visited
                if target not in paths:
                    paths[target] = []
                    path_costs[target] = []

                # compute all paths to target using current edge
                for p in range(len(paths[vertex])):
                    path_cost = path_costs[vertex][p] + edge_costs[(vertex, target)]
                    if path_cost < np.inf:
                        path = paths[vertex][p] + [target]
                        paths[target].append(path)
                        path_costs[target].append(path_cost)

                        # if path ends in a final vertex of the product graph
                        if (target, target) in edge_costs and edge_costs[(target, target)] < np.inf:
                            num_final_agents = np.sum([target_list[a] in self.final_vertices[a]
                                                       for a in range(self.n)])
                            J_sum = num_final_agents * \
                                np.exp(-(path_cost + edge_costs[(target, target)]))
                            full_paths.append(
                                (path + [target], J_sum, self.set_to_int(sac_agents)))

        return full_paths

    def compute_best_nash(self, full_paths, edge_policies):

        total_steps = 0
        pcount = 0
        for path, _, sac_int in sorted(full_paths, key=lambda x: x[1], reverse=True):
            pcount += 1

            # Construct FS policy
            sac_set = self.int_to_set(sac_int)
            finite_state_policy = self.get_finite_state_policy(
                path, edge_policies[sac_int], sac_set)

            # Evaluate the policy to get individual success probabilities
            eval_env = EvalEnv(self.multi_env, self, path, self.params.horizon)
            _, probs = test_policy_mutli(eval_env, finite_state_policy,
                                         self.params.num_estimation_rollouts,
                                         stateful_policy=True, max_timesteps=self.params.horizon)

            print_new_block('Verifying policy {}'.format(pcount))
            print('Abstract Path: {}'.format(path))
            print('Success probabilities: {}'.format(probs.tolist()))

            # Verify policy
            valid_ne, steps_taken, adv_policies = self.verify(finite_state_policy, probs)
            total_steps += steps_taken

            # Return verified policy with maximum score
            if valid_ne:
                print_new_block('NE Path Policy Found')
                print('Abstract Path: {}'.format(path))
                print('Success probabilities: {}'.format(probs.tolist()))
                all_n = [sp.n for sp in self.multi_env.action_space.spaces]
                finite_state_policy = FullNashPolicy(finite_state_policy, adv_policies,
                                                     self.reward_machines, all_n,
                                                     self.params.horizon)
                return finite_state_policy, total_steps, probs

        return None, total_steps, np.zeros((self.n,))

    def verify(self, finite_state_policy, probs):
        # estimate the MDP is estimates are not available
        steps_taken = 0
        if self.P is None:
            print('\nEstimating MDP...')
            start_time = time.time()
            self.P, steps_taken = estimate_mdp(self.multi_env, self.params.num_verify_samples)
            print('Estimated MDP in {} secs'.format(time.time() - start_time))

        adv_policies = []
        for a in range(self.n):
            # Construct the product MDP
            print('\nConstructing Product MDP for agent {} ...'.format(a))
            start_time = time.time()
            prod_P, prod_R, start_state = get_product_mdp(
                self.multi_env, self.P, finite_state_policy, self.reward_machines[a], a)
            print('Construction completed in {} secs'.format(time.time() - start_time))

            # Compute the best deviation of the agent
            print('\nPerforming Nash Value Iteration for agent {} ...'.format(a))
            start_time = time.time()
            adv_policy = {}
            deviation = nash_value_iteration_zero_sum(prod_P, prod_R, start_state,
                                                      self.params.horizon, DP={},
                                                      adv_policy=adv_policy,
                                                      verbose=self.params.verbose,
                                                      use_lp=self.params.use_lp)
            print('Completed VI in {} secs'.format(time.time() - start_time))
            print('Best deviation of agent {}: {}'.format(a, deviation))

            # Deviation exceeds return
            if deviation >= probs[a] + self.params.ne_epsilon:
                return False, steps_taken, None

            # add punishment strategy
            adv_policies.append(adv_policy)

        return True, steps_taken, adv_policies

    def learn_single_edge(self, edge, vertex_list, target_list, start_dist=None,
                          print_prefix='', sac_agents=set()):
        '''
        Learn policies corresponding to given edge (cooperative).

        Parameters:
            egde: List[Optional[AbstractEdge]]
            vertex_list: List[int]
            target_list: List[int]
            start_dist: Optional[Distribution]
        '''

        # Compute predicates and constraints
        final_preds = []
        constraints = []
        for e in edge:
            if e is None:
                final_preds.append(None)
                constraints.append(None)
            else:
                final_preds.append(e.predicate)
                constraints.append(e.constraints)

        # Define non final agents
        non_final_agents = [a for a in range(self.n)
                            if target_list[a] not in self.final_vertices[a]
                            and a not in sac_agents]

        # Define constraints to follow after reaching goal
        after_reach_constraints = [[self.stay_constraint(a, target_list[a], sac_agents)]
                                   for a in range(self.n)]

        # Define reach environment
        reach_env = MultiReachabilityEnv(self.multi_env, self.params, start_dist, final_preds,
                                         constraints, after_reach_constraints, non_final_agents)

        # Use single agent Q-learning to learn joint policy
        agent = QLearningAgent(self.params.epsilon, self.params.lr, self.params.gamma)
        joint_policy = agent.learn(reach_env, self.params.num_training_steps)

        # Render learned joint policy
        if self.params.render:
            print('\n{}Simulating cooperative policy...'.format(print_prefix))
            get_rollout(reach_env, joint_policy, True)

        return joint_policy, reach_env, self.params.num_training_steps

    def compute_reach_prob(self, reach_env, joint_policy):
        '''
        Computes probability that all agents complete the edge corresponding to reach_env.
        '''
        steps_taken = 0
        states_reached = []
        reach_prob = 0
        for _ in range(self.params.num_estimation_rollouts):
            sarss = get_rollout(reach_env, joint_policy, False)
            states = [state for state, _, _, _ in sarss] + [sarss[-1][-1]]
            steps_taken += len(sarss)
            if np.all([reach_env.satisfies_edge(states, a) for a in range(self.n)]):
                reach_prob += 1
                states_reached.append(reach_env.get_state())
        reach_prob = (reach_prob / self.params.num_estimation_rollouts)
        return reach_prob, states_reached, steps_taken

    def get_finite_state_policy(self, path, edge_policies, sac_agents=set()):
        '''
        Computes a verification compatible finite state policy corresponding to path.
        '''
        edge_policy_list = [edge_policies[(path[i], path[i+1])] for i in range(len(path)-1)]
        path = [self.get_vertices(vertex) for vertex in path]
        preds = []
        null_actions = []
        if self.params.null_actions is None:
            def_actions = [None] * self.n
        else:
            def_actions = self.params.null_actions
        for i in range(len(path)-1):
            preds.append([])
            null_actions.append([])
            for a in range(self.n):
                if path[i][a] == path[i+1][a]:
                    preds[i].append(None)
                else:
                    preds[i].append(self.pred_maps[a][(path[i][a], path[i+1][a])])
                if path[i+1][a] not in self.final_vertices[a] and a not in sac_agents:
                    null_actions[i].append(def_actions[a])
                else:
                    null_actions[i].append(None)
        return PathPolicy(path, edge_policy_list, preds, null_actions)

    def get_product_vertex(self, vertices):
        '''
        Converts a list of vertices (vertex in product abstract graph) to a string.
        '''
        return '_'.join(map(str, vertices))

    def get_vertices(self, product_vertex):
        '''
        Converts string to list of int correspoding to the
            product vertex represented by the string.
        '''
        if product_vertex != '':
            return list(map(int, product_vertex.split('_')))
        else:
            return []

    def compute_outgoing_edges(self, vertices, sac_agents=set()):
        '''
        Each element of the returned list is an edge from 'vertices'
            in the product abstract graph, each such edge is a list
            of objects of type AbstractEdge (denoting individual edges).
        '''
        edges = []

        # Retreive graph and state of satisfying agents
        sat_agents = [a for a in range(self.n) if a not in sac_agents]
        sat_graphs = [self.abstract_graphs[a] for a in sat_agents]
        sat_final = [self.final_vertices[a] for a in sat_agents]
        sat_vertices = [vertices[a] for a in sat_agents]

        # Compute all relevant product indices
        indices = tuple([list(range(len(g[v])+1)) for g, v in zip(sat_graphs, sat_vertices)])
        biggest = np.array([len(ind)-1 for ind in indices], dtype=int)

        # Add all valid edges
        for i in itertools.product(*indices):

            # No agent makes progress and not a final product vertex
            if (np.all(np.array(i, dtype=int) == biggest) and
                    not np.all([v in f for v, f in zip(sat_vertices, sat_final)])):
                continue

            # Some agent makes progress or is a final product vertex
            edge = [None] * self.n
            bad_edge = False
            for j in range(len(sat_agents)):
                if i[j] != biggest[j]:
                    if sat_vertices[j] in sat_final[j]:
                        bad_edge = True
                        break
                    edge[sat_agents[j]] = sat_graphs[j][sat_vertices[j]][i[j]]
            if not bad_edge:
                edges.append(edge)
        return edges

    def compute_edge_list(self, vertex_list, target_list):
        '''
        Compute list of AbstractEdge objects.
        '''
        edge_list = []
        for a in range(self.n):
            if vertex_list[a] == target_list[a]:
                edge_list.append(None)
            else:
                pred = self.pred_maps[a][(vertex_list[a], target_list[a])]
                constraints = self.constraint_maps[a][(vertex_list[a], target_list[a])]
                edge = AbstractEdge(target_list[a], pred, constraints)
                edge_list.append(edge)
        return edge_list

    def topo_sort(self, sac_agents=set()):
        '''
        Topological sort of the product abstract graph.
        Returns sorted vertices in a list.
        '''
        visited = set()
        stack = []

        def dfs(v_list, v):
            for e in self.compute_outgoing_edges(v_list, sac_agents=sac_agents):
                t_list, t = self.compute_target(v_list, e)
                if t not in visited and v != t:
                    dfs(t_list, t)
            stack.append((v_list, v))
            visited.add(v)

        init_v_list = [0] * self.n
        init_v = self.get_product_vertex(init_v_list)
        dfs(init_v_list, init_v)
        return reversed(stack)

    def compute_target(self, vertices, abstract_edges):
        '''
        Computes targets for all agents given a list of abstract edges.
        If abstract edge for agent a is None, target is vertices[a].
        '''
        t = []
        for e, v in zip(abstract_edges, vertices):
            if e is None:
                t.append(v)
            else:
                t.append(e.target)
        return t, self.get_product_vertex(t)

    def compute_edge_maps(self, agent):
        '''
        Preprocessing to compute mappings from edges to predicates and constraints.
        '''
        p_map = {}
        c_map = {}
        graph = self.abstract_graphs[agent]
        for u in range(len(graph)):
            for j in range(len(graph[u])):
                v = graph[u][j].target
                p_map[(u, v)] = graph[u][j].predicate
                c_map[(u, v)] = graph[u][j].constraints
        return p_map, c_map

    def stay_constraint(self, agent, vertex, sac_agents=set()):
        if agent in sac_agents:
            return true_pred
        edges = self.abstract_graphs[agent][vertex]
        c = edges[0].constraints[0]
        for e in edges[1:]:
            c = land(c, e.constraints[0])
        return c

    def set_to_int(self, agents):
        retval = 0
        for a in agents:
            retval = retval | (1 << a)
        return retval

    def int_to_set(self, agents):
        return set([a for a in range(self.n) if agents & (1 << a) != 0])


class PathPolicy(FiniteStatePolicy):
    '''
    Finite state policy corresponding to a path.
    '''

    def __init__(self, path, edge_policies, preds, null_actions=None):
        self.path = path
        self.edge_policies = edge_policies
        self.preds = preds
        self.null_actions = null_actions
        self.n = len(self.preds[0])
        self.init_reach = [self._init_reach(index) for index in range(len(self.preds))]
        super().__init__()

    def init_state(self):
        return (0, self.init_reach[0])

    def step(self, policy_state, env_state, action):
        index, reach = policy_state
        if index == len(self.preds)-1:
            return policy_state

        # update the reach array
        for a in range(self.n):
            if (reach & (1 << a) == 0) and self.preds[index][a](env_state, None) > 0:
                reach = reach | (1 << a)

        # update current edge if all agents reached goal
        if reach == (1 << self.n) - 1:
            index += 1
            reach = self.init_reach[index]

        return (index, reach)

    def act(self, policy_state, env_state):
        index, reach = policy_state
        actions = list(self.edge_policies[index].get_action(env_state))
        for a in range(self.n):
            if (reach & (1 << a)) != 0 and self.null_actions[index][a] is not None:
                actions[a] = self.null_actions[index][a]
        return tuple(actions)

    def _init_reach(self, index):
        reach = 0
        vertex_list = self.path[index]
        target_list = self.path[index+1]
        for a in range(self.n):
            if vertex_list[a] == target_list[a]:
                reach = reach | (1 << a)
        return reach


class EvalEnv(gym.Env):
    '''
    Environment for evaluating joint policy.

    Parameters:
        multi_env: gym like multi-agent env
        multi_reach: MultiAbstractReachability
        path: List[str]
        max_steps: int
    '''

    def __init__(self, multi_env, multi_reach, path, max_steps=100):
        self.multi_env = multi_env
        self.multi_reach = multi_reach
        self.n = self.multi_reach.n
        self.compute_individual_plans(path)
        self.max_steps = max_steps
        self.action_space = self.multi_env.action_space
        self.observation_space = self.multi_env.observation_space

    def reset(self):
        self.state = self.multi_env.reset()
        self.t = 0

        # set initial targets
        self.vertex_list = [0] * self.n
        self.target_list = [None] * self.n
        for i in range(self.n):
            self.compute_target(i)

        # set violated constraints
        self.constraints = [None] * self.n
        self.blocked_constraints = [0] * self.n
        self.violated_constraints = [False] * self.n
        return self.state

    def step(self, action):
        self.state, _, done, info = self.multi_env.step(action)

        self.update_blocked_constraints()
        rewards = [0] * self.n

        for a in range(self.n):
            # check constraints on current edge
            if self.blocked_constraints[a] >= len(self.constraints[a]):
                self.violated_constraints[a] = True

            # check reachability of target
            if self.target_list[a] is not None and not self.violated_constraints[a]:
                reach_pred = self.multi_reach.pred_maps[a][
                    (self.vertex_list[a], self.target_list[a])]
                if reach_pred(self.state, None) > 0:
                    self.vertex_list[a] = self.target_list[a]
                    self.target_list[a] = None
                    self.compute_target(a)
                    self.blocked_constraints[a] = 0

            # compute reward
            if self.vertex_list[a] in self.multi_reach.final_vertices[a]:
                if not self.violated_constraints[a]:
                    rewards[a] = 1
                else:
                    rewards[a] = -(self.max_steps + 1)

        self.t += 1
        return self.state, rewards, done or (self.t >= self.max_steps), info

    def update_blocked_constraints(self):
        for a in range(self.n):
            u, v = self.vertex_list[a], self.target_list[a]
            if v is not None:
                self.constraints[a] = self.multi_reach.constraint_maps[a][(u, v)]
            elif u in self.multi_reach.final_vertices[a]:
                self.constraints[a] = self.multi_reach.constraint_maps[a][(u, u)]
            else:
                self.constraints[a] = [true_pred]
            for j in range(self.blocked_constraints[a], len(self.constraints[a])):
                if self.constraints[a][j](self.state, None) > 0:
                    break
                self.blocked_constraints[a] += 1

    def render(self):
        self.multi_env.render()
        print('Abstract states: {}'.format(self.vertex_list))

    def compute_target(self, agent):
        if self.vertex_list[agent] in self.plans[agent]:
            self.target_list[agent] = self.plans[agent][self.vertex_list[agent]]

    def compute_individual_plans(self, path):
        self.plans = [{} for _ in range(self.n)]
        vertex_list = [0] * self.n
        for j in range(len(path)-1):
            target = path[j+1]
            target_list = self.multi_reach.get_vertices(target)
            for a in range(self.n):
                if vertex_list[a] != target_list[a]:
                    self.plans[a][vertex_list[a]] = target_list[a]
            vertex_list = target_list
