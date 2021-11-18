import gym
import time
import itertools
import numpy as np

from spectrl.main.monitor import Resource_Model
from spectrl.main.spec_compiler import land
from spectrl.util.dist import FiniteDistribution
from spectrl.util.rl import get_rollout, MultiAgentPolicy, AgentEnv
from spectrl.hierarchy.construction import true_pred
from spectrl.hierarchy.reachability import AbstractEdge
from spectrl.rl.ddpg.multi_agent import MultiDDPG
from spectrl.rl.ddpg.ddpg import DDPG
from spectrl.rl.maddpg.train import MADDPG
from heapq import heappop, heappush
from copy import deepcopy


class MultiReachabilityEnv(gym.Env):
    '''
    Product of multi-agent system and resource models.
    Terminates when a given set of agents (reward_agents) reach their goals.
    Action and Observation spaces are lists.

    Parameters:
        multi_env: Multi-agent gym.Env (with set_state() method)
        init_dist: Distribution (initial state distribution)
        final_preds: List[state_i, resource_i -> float] (Goal of the reachability task,
                        can be none for adversarial agents)
        constraints: List[List[state_i, resource_i -> float]]
            (Constraints that need to be satisfied (defines reward function),
             can be None for adversarial agents)
        res_models: List[Resource_Model] (optional, can be None)
        max_steps: int
        safety_penalty: float (min penalty for violating constraints)
        neg_inf: float (negative reward for failing to satisfy constraints)
        alpha: float (alpha * original_reward will be added to reward)
        reward_agents: List[int] (Agents rewarded for reaching goal)
    '''

    def __init__(self, multi_env, init_dist=None, final_preds=None, constraints=None,
                 max_steps=100, res_models=None, safety_penalty=-1, neg_inf=-10,
                 alpha=0, reward_agents=None, cooperative=False, stay_agents=[],
                 null_actions=None, testing=False):
        self.wrapped_env = multi_env
        self.init_dist = init_dist
        self.final_preds = final_preds
        self.constraints = constraints
        self.res_models = res_models
        self.max_steps = max_steps
        self.safety_penalty = safety_penalty
        self.neg_inf = neg_inf
        self.alpha = alpha
        self.reward_agents = reward_agents
        self.stay_agents = stay_agents
        self.n = len(self.wrapped_env.action_space)
        self.cooperative = cooperative
        self.null_actions = null_actions
        self.testing = testing

        # Extract dimensions from env
        self.orig_state_dims = [space.shape[0]
                                for space in self.wrapped_env.observation_space]
        self.action_dims = [space.shape[0]
                            for space in self.wrapped_env.action_space]

        # Handle undefined inputs
        if self.final_preds is None:
            self.final_preds = [None] * self.n
        if self.constraints is None:
            self.constraints = [[true_pred] for _ in range(self.n)]
        if self.res_models is None:
            def delta(sys_state, res_state, sys_action):
                return np.array([])
            self.res_models = [Resource_Model(self.orig_state_dims[i], self.action_dims[i],
                                              0, np.array([]), delta)
                               for i in range(self.n)]
        if self.reward_agents is None:
            self.reward_agents = list(range(self.n))

        # Set new observation and action spaces
        obs_dims = [self.orig_state_dims[i] + self.res_models[i].res_init.shape[0]
                    for i in range(self.n)]
        self.observation_space = [gym.spaces.Box(-np.inf, np.inf, shape=(obs_dims[i],))
                                  for i in range(self.n)]
        self.action_space = self.wrapped_env.action_space

        # Reset the environment
        self.reset()

    def reset(self):
        # Reset wrapped env
        self.sys_states = self.wrapped_env.reset()

        # Set correct sys and res states based on init dist
        if self.init_dist is not None:
            sim_state, self.res_states = self.init_dist.sample()
            self.sys_states = self.wrapped_env.set_sim_state(sim_state)
        else:
            self.res_states = [self.res_models[i].res_init for i in range(self.n)]

        # Initialize additional bookeeping info
        self.violated_constraints = [0] * self.n
        self.prev_safety_rewards = [self.neg_inf] * self.n
        self.goal_reached = [False] * self.n
        for i in self.stay_agents:
            self.goal_reached[i] = True

        # Reset time
        self.t = 0
        return self.get_obs()

    def step(self, action):
        # Set actions of agents already in goal to null action
        corrected_action = [a for a in action]
        if self.null_actions is not None:
            for i in range(self.n):
                if self.goal_reached[i]:
                    corrected_action[i] = self.null_actions[i]

        # Take step in system and resource model
        self.res_states = [self.res_models[i].res_delta(
            self.sys_states[i], self.res_states[i], corrected_action[i])
            for i in range(self.n)]
        self.sys_states, r, _, _ = self.wrapped_env.step(corrected_action)
        self.t += 1

        # Total reward
        sum_reward = sum([self.reward(i) + self.alpha * min(r[i], 0) for i in self.reward_agents])

        # Assign negative rewards in the presence of adversaries
        if self.cooperative:
            rewards = sum_reward
        else:
            rewards = []
            for i in range(self.n):
                if i in self.reward_agents:
                    rewards.append(sum_reward)
                else:
                    rewards.append(-sum_reward)

        # Update agents' done status
        done = True
        for i in self.reward_agents:
            if (self.final_preds[i] is not None) and (
                    self.violated_constraints[i] < len(self.constraints[i])):
                self.goal_reached[i] = self.goal_reached[i] or \
                    self.final_preds[i](self.sys_states[i], self.res_states[i]) > 0
                done = done and self.goal_reached[i]
            else:
                done = done and (self.violated_constraints[i] < len(self.constraints[i]))
        done = (done and not self.testing) or self.t > self.max_steps

        return self.get_obs(), rewards, done, {}

    def render(self):
        self.wrapped_env.render()

    def get_obs(self):
        return [np.concatenate([s, r]) for s, r in zip(self.sys_states, self.res_states)]

    def get_state(self):
        return self.wrapped_env.get_sim_state(), self.res_states

    def reward(self, agent):
        # Compute reward for reaching goal states
        reach_reward = 0
        if self.final_preds[agent] is not None:
            reach_reward = self.final_preds[agent](
                self.sys_states[agent], self.res_states[agent])

        # Compute reward for obeying (or violating) constraints
        safety_reward = self.prev_safety_rewards[agent]
        for i in range(self.violated_constraints[agent], len(self.constraints[agent])):
            cur_constraint_val = self.constraints[agent][i](
                self.sys_states[agent], self.res_states[agent])
            safety_reward = max(safety_reward, cur_constraint_val)
            if cur_constraint_val > 0:
                break
            self.violated_constraints[agent] += 1
        safety_reward = min(safety_reward, 0)

        # Store safety reward at first violation
        if safety_reward < 0:
            safety_reward = min(safety_reward, self.safety_penalty)
            self.prev_safety_rewards[agent] = safety_reward

        return reach_reward + safety_reward

    def cum_reward(self, states, agent):
        states = [s[agent] for s in states]
        reach_reward = self.neg_inf
        safety_reward = -self.neg_inf
        violated_constraints = 0

        for s in states:
            # Reach reward
            sys_state = s[:self.orig_state_dims[agent]]
            res_state = s[self.orig_state_dims[agent]:]
            if self.final_preds[agent] is not None:
                reach_reward = max(reach_reward, self.final_preds[agent](sys_state, res_state))

            # Safety reward
            cur_safety_reward = self.neg_inf
            for i in range(violated_constraints, len(self.constraints[agent])):
                tmp_reward = self.constraints[agent][i](sys_state, res_state)
                if tmp_reward <= 0:
                    violated_constraints += 1
                else:
                    cur_safety_reward = tmp_reward
                    break
            safety_reward = min(safety_reward, cur_safety_reward)

        # Agents with no goals
        if self.final_preds[agent] is None:
            reach_reward = -self.neg_inf

        return min(reach_reward, safety_reward)

    def close(self):
        self.wrapped_env.close()


class MultiAbstractReachability:
    '''
    Class defining the multi-agent abstract reachability problem.

    Parameters:
        abstract_graphs: list of abstract graphs.
        final_vertices: list of sets of int (sets of final vertices).

    Initial vertex is assumed to be 0 for all agents.
    '''

    def __init__(self, abstract_graphs, final_vertices, multi_env, ddpg_params, maddpg_params,
                 multi_ddpg_params, res_models=None, max_steps=100, safety_penalty=-1.,
                 neg_inf=-10., alpha=0., num_samples=500, use_gpu=False,
                 render=False, null_actions=None, no_stay=False,
                 stay_pred=None, init_stay_pred=None):
        '''
        Parameters:
            multi_env: Gym-like multi-agent environment
            ddpg_params: DDPGParams
            maddpg_params: maddpg.HyperParams
            multi_ddpg_params: MultiDDPGParams
            res_models: List[Resource_Model] (optional)
            safety_penalty: float (min penalty for violating constraints)
            neg_inf: float (large negative constant)
            alpha: float (multiplier for original env rewards)
            num_samples: int (number of samples used to compute reach probabilities)
            no_stay: bool (indicates if inactive agents are helping)
        '''

        # Specifications
        self.abstract_graphs = abstract_graphs
        self.final_vertices = final_vertices
        self.n = len(self.abstract_graphs)  # number of agents
        self.num_vertices = [len(g) for g in self.abstract_graphs]

        # Precompute mappings from edges to predicates and constraints
        self.pred_maps = []
        self.constraint_maps = []
        for i in range(self.n):
            p_map, c_map = self.compute_edge_maps(i)
            self.pred_maps.append(p_map)
            self.constraint_maps.append(c_map)

        # Env and params for learning
        self.multi_env = multi_env
        self.ddpg_params = ddpg_params
        self.maddpg_params = maddpg_params
        self.multi_ddpg_params = multi_ddpg_params
        self.res_models = res_models
        self.max_steps = max_steps
        self.safety_penalty = safety_penalty
        self.neg_inf = neg_inf
        self.alpha = alpha
        self.num_samples = num_samples
        self.use_gpu = use_gpu
        self.render = render
        self.null_actions = null_actions
        self.no_stay = no_stay
        self.stay_pred = stay_pred
        self.init_stay_pred = init_stay_pred

    def learn_nash_dijkstra(self):
        '''
        Dijkstra's algorithm based learning for multi agent abstract reachability.

        Returns:
            abstract_policies: Map[str, str] (next vertex for each source vertex)
            nn_policies: Product edges -> list of NN policies (one for each agent)
        '''
        # Initialize abstract and NN policies.
        parent = {}
        abstract_policy = {}
        nn_policies = {}

        # Dijkstra initialization
        explored = set()
        min_neg_log_prob = {}
        queue = []

        # Inital vertex
        init_vertex = self.get_product_vertex([0] * self.n)
        heappush(queue, (0, init_vertex, ''))  # (distance, vertex, source) triples

        # Reach states for each vertex and source (represents induced distributions)
        reach_states = {}

        # All costs
        edge_deviations = {}  # for each edge and agent (neg-log)
        min_deviations = {}  # along best path to a vertex (for each vertex and agent)
        edge_costs = {}
        min_deviations[init_vertex] = np.array([np.inf] * self.n)

        # Other stats
        num_edges_learned = 0
        total_steps = 0
        start_time = time.time()

        # Set of bad edges for which RL fails to learn a policy
        bad_edges = []

        nash_states = []

        while len(queue) > 0:

            neg_log_prob, vertex, source = heappop(queue)

            if vertex not in explored:
                vertex_list = self.get_vertices(vertex)

                # Set minimum log probability of reaching the vertex and the last edge taken
                min_neg_log_prob[vertex] = neg_log_prob
                if vertex != init_vertex:
                    parent[vertex] = source

                # Set min deviations
                if source != '':
                    cur_deviation = neg_log_prob - edge_costs[(source, vertex)] +\
                        np.array(edge_deviations[(source, vertex)])
                    min_deviations[vertex] = np.minimum(min_deviations[source], cur_deviation)

                # Check if the state is a NE state w.r.t. best path
                scores = np.array([np.inf]*self.n)
                for i in range(self.n):
                    if vertex_list[i] in self.final_vertices[i]:
                        scores[i] = neg_log_prob
                if np.all(min_deviations[vertex] >= scores):
                    nash_states.append((vertex, scores))

                # Explore the vertex by learning policies for each outgoing edge
                edges = self.compute_outgoing_edges(vertex_list)
                for edge in edges:

                    target_list, target = self.compute_target(vertex_list, edge)
                    if target not in explored:

                        # Set the initial distribution for learning
                        if vertex == init_vertex:
                            start_dist = None
                        else:
                            start_dist = FiniteDistribution(reach_states[(source, vertex)])

                        print('\n***** Learning policies for edge {} -> {} ****'.format(
                            vertex, target))
                        print_prefix = '[{} -> {}] '.format(vertex, target)

                        # Learn edge policies
                        edge_policies, reach_env, deviations, steps = self.learn_edge_policies(
                            edge, vertex_list, target_list, start_dist, print_prefix=print_prefix)
                        nn_policies[(vertex, target)] = edge_policy = MultiAgentPolicy(
                            edge_policies)
                        edge_deviations[(vertex, target)] = deviations

                        # Update stats
                        num_edges_learned += 1
                        total_steps += steps

                        # Compute reach probability and collect visited states
                        reach_prob, states_reached, steps_taken = self.compute_reach_prob(
                            reach_env, edge_policy, list(range(self.n)))
                        total_steps += steps_taken
                        edge_cost = -np.log(reach_prob)

                        # Print all probabilities
                        print('\n{}Local deviating probabilities: {}'.format(
                            print_prefix, [np.exp(-c) for c in deviations]))
                        print('{}Reach probability: {}'.format(print_prefix, reach_prob))

                        # Edge is valid if it can possibly be a part of an NE path
                        if len(states_reached) > 0 and min(deviations) >= edge_cost:
                            reach_states[(vertex, target)] = states_reached
                            edge_costs[(vertex, target)] = edge_cost
                            target_neg_log_prob = edge_cost + min_neg_log_prob[vertex]
                            heappush(queue, (target_neg_log_prob, target, vertex))
                        else:
                            bad_edges.append((vertex, target))

                # Set the explored tag
                explored.add(vertex)

        # Print bad edges
        if len(bad_edges) > 0:
            print('\nBad Edges:')
            for s, t in bad_edges:
                print('{} -> {}'.format(s, t))

        # Pick the socially optimal NE
        best_score = 0.
        best_vertex = ''
        for vertex, scores in nash_states:
            score = np.sum(np.exp(-scores))
            if score > best_score:
                best_score = score
                best_vertex = vertex
        if best_vertex == '':
            print('No nash found!')
            exit(1)

        # Compute abstract policy
        v = best_vertex
        while v != init_vertex:
            abstract_policy[parent[v]] = v
            v = parent[v]

        total_time = time.time() - start_time
        return abstract_policy, nn_policies, [total_steps, total_time, num_edges_learned]

    def learn_nash_topo(self):
        '''
        BFS based search for a NE path.

        Returns:
            abstract_policies: Map[str, str] (next vertex for each source vertex)
            nn_policies: Product edges -> list of NN policies (one for each agent)
        '''
        # Initialize abstract and NN policies.
        nn_policies = {}

        # Topological sort of product graph
        topo_list = list(self.topo_sort())

        # Average reach distribution for each vertex
        reach_dist = {}
        _, init_vertex = topo_list[0]
        reach_dist[init_vertex] = None

        # All costs
        edge_deviations = {}  # for each edge and agent (neg-log)
        edge_costs = {}  # for each edge (neg-log)

        # Other stats
        num_edges_learned = 0
        total_steps = 0
        start_time = time.time()

        # Set of bad edges for which RL fails to learn a policy
        bad_edges = []

        for vertex_list, vertex in topo_list:
            if vertex not in reach_dist:
                continue

            for edge in self.compute_outgoing_edges(vertex_list):
                target_list, target = self.compute_target(vertex_list, edge)

                print('\n***** Learning policies for edge {} -> {} ****'.format(
                    vertex, target))
                print_prefix = '[{} -> {}] '.format(vertex, target)

                # Learn edge policies
                edge_policies, reach_env, deviations, steps = self.learn_edge_policies(
                    edge, vertex_list, target_list, reach_dist[vertex], print_prefix=print_prefix)
                nn_policies[(vertex, target)] = edge_policy = MultiAgentPolicy(edge_policies)
                edge_deviations[(vertex, target)] = deviations

                # Update stats
                num_edges_learned += 1
                total_steps += steps

                # Compute reach probability and collect visited states
                reach_prob, states_reached, steps_taken = self.compute_reach_prob(
                    reach_env, edge_policy, list(range(self.n)))
                total_steps += steps_taken
                edge_cost = -np.log(reach_prob)

                # Print probabilities
                print('\n{}Local deviating probabilities: {}'.format(
                    print_prefix, [np.exp(-c) for c in deviations]))
                print('{}Reach probability: {}'.format(print_prefix, reach_prob))

                # Edge is valid if it can possibly be a part of an NE path
                if len(states_reached) > 0 and min(deviations) >= edge_cost:
                    if target not in reach_dist:
                        reach_dist[target] = FiniteDistribution()
                    reach_dist[target].add_points(states_reached)
                    edge_costs[(vertex, target)] = edge_cost
                else:
                    bad_edges.append((vertex, target))

        abstract_policy, steps_taken = self.compute_best_nash(
            topo_list, edge_costs, edge_deviations, nn_policies)
        total_steps += steps_taken
        if abstract_policy is None:
            print('\nNo nash found!')
            exit(1)

        total_time = time.time() - start_time
        return abstract_policy, nn_policies, [total_steps, total_time, num_edges_learned]

    def compute_best_nash(self, topo_list, edge_costs, edge_deviations,
                          nn_policies=None, dev_policies=None):

        # store all paths and corresponding costs
        _, init_vertex = topo_list[0]
        paths = {init_vertex: [[init_vertex]]}
        min_devs = {init_vertex: [np.inf * np.ones((self.n,))]}
        path_costs = {init_vertex: [0]}

        # NE paths along with their total costs
        ne_paths = []

        for vertex_list, vertex in topo_list:
            if vertex not in paths or len(paths[vertex]) == 0:
                continue

            for edge in self.compute_outgoing_edges(vertex_list):
                target_list, target = self.compute_target(vertex_list, edge)
                if (vertex, target) not in edge_costs:
                    continue

                # first time a target is visited
                if target not in paths:
                    paths[target] = []
                    path_costs[target] = []
                    min_devs[target] = []

                # compute all paths to target using current edge
                for p in range(len(paths[vertex])):
                    min_deviations = np.minimum(min_devs[vertex][p], path_costs[vertex][p] +
                                                np.array(edge_deviations[(vertex, target)]))
                    path_cost = path_costs[vertex][p] + edge_costs[(vertex, target)]
                    if min(min_deviations) >= path_cost:
                        path = paths[vertex][p] + [target]
                        paths[target].append(path)
                        path_costs[target].append(path_cost)
                        min_devs[target].append(min_deviations)

                        scores = np.inf * np.ones((self.n,))
                        for i in range(self.n):
                            if target_list[i] in self.final_vertices[i]:
                                scores[i] = path_cost
                        if np.all(min_deviations >= scores):
                            ne_value = -sum([np.exp(-s) for s in scores])
                            ne_paths.append((path, ne_value))

        abstract_policy = None
        total_steps = 0
        for path, _ in sorted(ne_paths, key=lambda x: x[1]):
            new_policies, steps_taken = self.verify(path)
            total_steps += steps_taken
            if new_policies is not None:
                abstract_policy = {}
                for j in range(len(path)-1):
                    abstract_policy[path[j]] = path[j+1]
                    nn_policies[(path[j], path[j+1])] = new_policies[j]
                break
        return abstract_policy, total_steps

    def verify(self, path):
        print('\n**** Attempting to verify path {} ****'.format(path))
        total_steps = 0
        new_policies = []

        # devioations and costs for currently processed part of path
        min_deviations = np.inf * np.ones((self.n,))
        path_cost = 0.
        reach_dist = None

        for j in range(len(path)-1):
            v_list = self.get_vertices(path[j])
            t_list = self.get_vertices(path[j+1])
            edges = self.compute_edge_list(v_list, t_list)

            print('\n***** Learning policies for edge {} -> {} ****'.format(
                path[j], path[j+1]))
            print_prefix = '[{} -> {}] '.format(path[j], path[j+1])

            # Learn edge policies
            edge_policies, reach_env, deviations, steps = self.learn_edge_policies(
                edges, v_list, t_list, reach_dist, print_prefix=print_prefix)
            edge_policy = MultiAgentPolicy(edge_policies)
            new_policies.append(edge_policy)
            total_steps += steps

            # Compute reach probability and collect visited states
            reach_prob, states_reached, steps_taken = self.compute_reach_prob(
                reach_env, edge_policy, list(range(self.n)))
            total_steps += steps_taken

            # update costs and deviations
            min_deviations = np.minimum(min_deviations, path_cost + np.array(deviations))
            path_cost -= np.log(reach_prob)

            # infeasible edge or existense of good deviation strategies
            if len(states_reached) == 0 or path_cost > min(min_deviations):
                return None, total_steps

            # update reach distribution
            reach_dist = FiniteDistribution(states_reached)

        # take final states into account
        scores = np.inf * np.ones((self.n,))
        t_list = self.get_vertices(path[-1])
        for i in range(self.n):
            if t_list[i] in self.final_vertices[i]:
                scores[i] = path_cost
        if np.all(min_deviations >= scores):
            return new_policies, total_steps
        else:
            return None, total_steps

    def learn_edge_policies(self, edge, vertex_list, target_list, start_dist=None, print_prefix=''):
        '''
        Learn policies corresponding to given edge (cooperative and deviating).

        Parameters:
            egde: List[Optional[AbstractEdge]]
            vertex_list: List[int]
        '''

        # Compute predicates and constraints
        final_preds = []
        constraints = []
        stay_agents = []
        for i, e in zip(range(self.n), edge):
            if e is None:
                final_preds.append(None)
                if not self.no_stay:
                    stay_agents.append(i)
                stay_constraint = self.help_constraint(i, vertex_list[i])
                if self.stay_pred is not None:
                    if vertex_list[i] == 0:
                        stay_constraint = land(stay_constraint, self.init_stay_pred(i))
                    else:
                        for u in range(len(self.abstract_graphs[i])):
                            if (u, vertex_list[i]) in self.pred_maps[i]:
                                stay_constraint = land(stay_constraint, self.stay_pred(
                                    self.pred_maps[i][(u, vertex_list[i])]))
                                break
                constraints.append([stay_constraint])
            else:
                final_preds.append(e.predicate)
                constraints.append(e.constraints)

        # Learn cooperative policy
        print('\n' + print_prefix + 'Learning cooperative policy...')
        reach_env = MultiReachabilityEnv(self.multi_env, start_dist, final_preds, constraints,
                                         self.max_steps, self.res_models, self.safety_penalty,
                                         self.neg_inf, self.alpha, stay_agents=stay_agents,
                                         cooperative=True, null_actions=self.null_actions)
        multi_ddpg = MultiDDPG(self.multi_ddpg_params, use_gpu=self.use_gpu)
        multi_ddpg.train(reach_env)
        steps_taken = multi_ddpg.rewardgraph[-1][0]
        edge_policies = multi_ddpg.get_policies()
        if self.render:
            print('\n{}Simulating cooperative policy...'.format(print_prefix))
            get_rollout(reach_env, MultiAgentPolicy(edge_policies), True)

        # Learn deviation policies
        deviations = []
        for i in range(self.n):
            if self.ddpg_params.num_episodes > 0:
                print('\n' + print_prefix + 'Learning deviation policy for agent {}...'.format(i))
                dev, steps = self.learn_deviation_policy(
                    i, vertex_list[i], target_list[i], edge_policies, stay_agents, start_dist,
                    print_prefix=print_prefix + '[Agent {}] '.format(i))
                deviations.append(dev)
                steps_taken += steps
            else:
                deviations.append(np.inf)

        return edge_policies, reach_env, deviations, steps_taken

    def learn_deviation_policy(self, a, start_vertex, target_vertex, policies,
                               stay_agents, init_dist=None, print_prefix=''):
        '''
        Learn deviation policy for the given agent from the given vertex and
            initial policies of other agents.
        '''
        # Remove a from stay_agents
        if a in stay_agents:
            stay_agents = deepcopy(stay_agents)
            stay_agents.remove(a)

        # Initialize abstract policy and NN policies.
        parent = [-1] * self.num_vertices[a]
        nn_policies = [[] for _ in self.abstract_graphs[a]]

        # Dijkstra initialization
        explored = [False] * self.num_vertices[a]
        min_neg_log_prob = [np.inf] * self.num_vertices[a]
        queue = []
        heappush(queue, (0, start_vertex, -1))  # (distance, vertex, source) triples
        reached_final_vertex = False

        # Reach states for each vertex and source
        reach_states = {}
        success_measure = np.inf  # success probability
        num_edges_learned = 0
        total_steps = 0

        # Set of bad edges for which RL fails to learn a policy
        bad_edges = []

        while len(queue) > 0 and not reached_final_vertex:

            neg_log_prob, vertex, source = heappop(queue)

            if not explored[vertex]:

                # Set minimum log probability of reaching the vertex and the last edge taken
                min_neg_log_prob[vertex] = neg_log_prob
                if vertex != start_vertex:
                    parent[vertex] = source

                # Explore the vertex by learning policies for each outgoing edge
                for edge in self.abstract_graphs[a][vertex]:

                    if explored[edge.target]:
                        nn_policies[vertex].append(None)
                    elif edge.target != vertex:
                        # Set initial state distribution for edge
                        if vertex == start_vertex:
                            start_dist = init_dist
                            stay_as = stay_agents
                        else:
                            start_dist = FiniteDistribution(reach_states[(source, vertex)])
                            stay_agents = []

                        # Define env for learning edge
                        final_preds = [None] * self.n
                        final_preds[a] = edge.predicate
                        constraints = [[true_pred] for _ in range(self.n)]
                        constraints[a] = edge.constraints
                        reach_env = MultiReachabilityEnv(
                            self.multi_env, start_dist, final_preds, constraints, self.max_steps,
                            self.res_models, self.safety_penalty, self.neg_inf, self.alpha, [a],
                            stay_agents=stay_as, null_actions=self.null_actions)

                        # Learn policy
                        print('\n' + print_prefix +
                              'Learning deviation policy for edge {} -> {}'.format(
                                  vertex, edge.target))
                        if vertex == start_vertex:
                            print('DDPG starting...')
                            agent_reach_env = AgentEnv(reach_env, policies, a)
                            ddpg_agent = DDPG(self.ddpg_params, use_gpu=self.use_gpu)
                            ddpg_agent.train(agent_reach_env)
                            policy = ddpg_agent.get_policy()
                            all_policies = [p for p in policies]
                            all_policies[a] = policy
                            edge_policy = MultiAgentPolicy(all_policies)
                            total_steps += ddpg_agent.rewardgraph[-1][0]
                        else:
                            print('MADDPG starting...')
                            maddpg = MADDPG(reach_env, self.maddpg_params)
                            total_steps += maddpg.train(main_agent=a)
                            edge_policy = MultiAgentPolicy(maddpg.get_policies())

                        # Update stats
                        nn_policies[vertex].append(edge_policy)
                        num_edges_learned += 1
                        if self.render:
                            print('\nRollout for edge {} -> {}'.format(vertex, edge.target))
                            get_rollout(reach_env, edge_policy, True)

                        # Compute reach probability and collect visited states
                        reach_prob, states_reached, steps_taken = self.compute_reach_prob(
                            reach_env, edge_policy, [a])
                        total_steps += steps_taken

                        # Print reach probability
                        print('\n{} Reach probability: {}'.format(print_prefix, reach_prob))

                        # Add target to queue if reach_prob is positive
                        if len(states_reached) > 0:
                            reach_states[(vertex, edge.target)] = states_reached
                            target_neg_log_prob = -np.log(reach_prob) + min_neg_log_prob[vertex]
                            heappush(queue, (target_neg_log_prob, edge.target, vertex))
                        else:
                            bad_edges.append((vertex, edge.target))
                    else:
                        success_measure = min_neg_log_prob[vertex]
                        reached_final_vertex = True

                # Set the explored tag
                explored[vertex] = True

        # Print bad edges
        if len(bad_edges) > 0:
            print('\n' + print_prefix + 'Bad Edges:')
            for s, t in bad_edges:
                print('{} -> {}'.format(s, t))

        return success_measure, total_steps

    def compute_reach_prob(self, reach_env, multi_policy, reach_agents):
        '''
        Computes probability that all agents in reach_agents reach their respective goals.
        '''
        steps_taken = 0
        states_reached = []
        reach_prob = 0
        for _ in range(self.num_samples):
            sarss = get_rollout(reach_env, multi_policy, False)
            states = np.array([state for state, _, _, _ in sarss] + [sarss[-1][-1]])
            steps_taken += len(sarss)
            if np.all([reach_env.cum_reward(states, a) > 0 for a in reach_agents]):
                reach_prob += 1
                states_reached.append(reach_env.get_state())
        reach_prob = (reach_prob / self.num_samples)
        return reach_prob, states_reached, steps_taken

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

    def compute_outgoing_edges(self, vertices):
        '''
        Each element of the returned list is an edge from 'vertices'
            in the product abstract graph, each such edge is a list
            of objects of type AbstractEdge (denoting individual edges).
        '''
        edges = []
        indices = tuple([list(range(len(g[v])+1)) for g, v in zip(self.abstract_graphs, vertices)])
        biggest = np.array([len(ind)-1 for ind in indices], dtype=int)
        for i in itertools.product(*indices):
            if np.all(np.array(i, dtype=int) == biggest):
                continue
            edge = []
            bad_edge = False
            for j in range(self.n):
                if i[j] == biggest[j]:
                    edge.append(None)
                else:
                    if vertices[j] in self.final_vertices[j]:
                        bad_edge = True
                    edge.append(self.abstract_graphs[j][vertices[j]][i[j]])
            if not bad_edge:
                edges.append(edge)
        return edges

    def compute_edge_list(self, vertex_list, target_list):
        '''
        Compute list of AbstractEdge objects.
        '''
        edge_list = []
        for j in range(self.n):
            if vertex_list[j] == target_list[j]:
                edge_list.append(None)
            else:
                pred = self.pred_maps[j][(vertex_list[j], target_list[j])]
                constraints = self.constraint_maps[j][(vertex_list[j], target_list[j])]
                edge = AbstractEdge(target_list[j], pred, constraints)
                edge_list.append(edge)
        return edge_list

    def topo_sort(self):
        '''
        Topological sort of the product abstract graph.
        Returns sorted vertices in a list.
        '''
        visited = set()
        stack = []

        def dfs(v_list, v):
            for e in self.compute_outgoing_edges(v_list):
                t_list, t = self.compute_target(v_list, e)
                if t not in visited:
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

    def help_constraint(self, agent, vertex):
        edges = self.abstract_graphs[agent][vertex]
        c = edges[0].constraints[0]
        for e in edges[1:]:
            c = land(c, e.constraints[0])
        return c


class HierarchicalPolicy:

    def __init__(self, abstract_policy, nn_policies, multi_reach, null_actions,
                 res_models=None, no_stay=False):
        '''
        Computes hierarchical policy from an abstract policy and a set of NN policies.

        Parameters:
            multi_reach: MultiAbstractReachability
            null_actions: List of actions (null/stay action for each agent)
        '''
        self.abstract_policy = abstract_policy
        self.nn_policies = nn_policies
        self.multi_reach = multi_reach
        self.n = self.multi_reach.n
        self.null_actions = null_actions
        self.res_dims = [0] * self.n
        self.no_stay = no_stay
        if res_models is not None:
            self.res_dims = [rm.res_init.shape[0] for rm in res_models]

    def reset(self):
        self.vertex_list = [0] * self.n
        self.vertex = self.multi_reach.get_product_vertex(self.vertex_list)
        self.update_edge()

    def get_action(self, states):
        # Check if target is reached
        for i in range(self.n):
            if not self.reached[i]:
                split_idx = len(states[i])-self.res_dims[i]
                sys_state = states[i][:split_idx]
                res_state = states[i][split_idx:]
                if self.reach_preds[i](sys_state, res_state) > 0:
                    self.reached[i] = True

        # Update the current edge
        if np.all(self.reached):
            self.vertex = self.target
            self.vertex_list = self.target_list
            self.update_edge()

        # If NN policy does not exist
        if self.cur_policy is None:
            return self.null_actions

        # Compute NN actions
        actions = self.cur_policy.get_action(states)

        # Null actions for all agents that are staying
        for i in range(self.n):
            if self.reached[i]:
                if self.vertex_list[i] != self.target_list[i] or not self.no_stay:
                    actions[i] = self.null_actions[i]

        return actions

    def update_edge(self):
        self.set_next_target()
        self.compute_predicates()
        self.cur_policy = None
        if (self.vertex, self.target) in self.nn_policies:
            self.cur_policy = self.nn_policies[(self.vertex, self.target)]
        self.reached = (np.array(self.vertex_list) == np.array(self.target_list))

    def set_next_target(self):
        if self.vertex in self.abstract_policy:
            self.target = self.abstract_policy[self.vertex]
        else:
            self.target = self.vertex
        self.target_list = self.multi_reach.get_vertices(self.target)

    def compute_predicates(self):
        self.reach_preds = [None] * self.n
        for i in range(self.n):
            if self.target_list[i] != self.vertex_list[i]:
                self.reach_preds[i] = self.multi_reach.pred_maps[i][
                    (self.vertex_list[i], self.target_list[i])]


class ConstrainedEnv(MultiReachabilityEnv):
    '''
    Environment for the full task enforcing constraints on the chosen abstract path.

    Parameters:
        multi_env: gym like multi-agent env
        init_dist: Distribution (initial state distribution)
        multi_reach: MultiAbstractReachability
        abstract_policy: Dict[str, str] (edge to choose in each product abstract state)
        res_models: List[Resource_Model] (optional, can be None)
        max_steps: int
    '''

    def __init__(self, multi_env, multi_reach, abstract_policy,
                 res_models=None, max_steps=100):
        self.multi_reach = multi_reach
        self.n = self.multi_reach.n
        self.compute_individual_plans(abstract_policy)
        super(ConstrainedEnv, self).__init__(multi_env, max_steps=max_steps,
                                             res_models=res_models, testing=True)

    def reset(self):
        obs = super(ConstrainedEnv, self).reset()

        # set initial targets
        self.vertex_list = [0] * self.n
        self.target_list = [None] * self.n
        for i in range(self.n):
            self.compute_target(i)

        # set violated constraints
        self.blocked_constraints = [0] * self.n
        self.update_blocked_constraints()
        self.violated_constraints = [False] * self.n
        return obs

    def step(self, action):
        obs, _, done, info = super(ConstrainedEnv, self).step(action)

        self.update_blocked_constraints()
        rewards = [0] * self.n

        for i in range(self.n):
            # check constraints on current edge
            if self.blocked_constraints[i] >= len(self.constraints[i]):
                self.violated_constraints[i] = True

            # check reachability of target
            if self.target_list[i] is not None and not self.violated_constraints[i]:
                if self.vertex_list[i] != self.target_list[i]:
                    reach_pred = self.multi_reach.pred_maps[i][
                        (self.vertex_list[i], self.target_list[i])]
                    if reach_pred(self.sys_states[i], self.res_states[i]) > 0:
                        self.vertex_list[i] = self.target_list[i]
                        self.compute_target(i)
                        self.blocked_constraints[i] = 0

            # compute reward
            if self.vertex_list[i] in self.multi_reach.final_vertices[i]:
                if not self.violated_constraints[i]:
                    rewards[i] = 1
                else:
                    rewards[i] = -(self.max_steps + 1)

        return obs, rewards, done, info

    def update_blocked_constraints(self):
        for i in range(self.n):
            u, v = self.vertex_list[i], self.target_list[i]
            if v is not None and u != v:
                constraints = self.multi_reach.constraint_maps[i][(u, v)]
            elif u == v and u in self.multi_reach.final_vertices[i]:
                constraints = self.multi_reach.abstract_graphs[i][u][0].constraints
            else:
                constraints = [true_pred]
            for j in range(self.blocked_constraints[i], len(constraints)):
                if constraints[j](self.sys_states[i], self.res_states[i]) > 0:
                    break
                self.blocked_constraints[i] += 1

    def render(self):
        super().render()
        print('Abstract states: {}'.format(self.vertex_list))

    def compute_target(self, agent):
        if self.vertex_list[agent] in self.plans[agent]:
            self.target_list[agent] = self.plans[agent][self.vertex_list[agent]]

    def compute_individual_plans(self, abstract_policy):
        self.plans = [{} for _ in range(self.n)]
        vertex_list = [0] * self.n
        vertex = self.multi_reach.get_product_vertex(vertex_list)
        while vertex in abstract_policy:
            target = abstract_policy[vertex]
            target_list = self.multi_reach.get_vertices(target)
            for i in range(self.n):
                if vertex_list[i] != target_list[i]:
                    self.plans[i][vertex_list[i]] = target_list[i]
            vertex = target
            vertex_list = target_list
