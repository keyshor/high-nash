'''
Tabular QRM Learning for Finite-State Environments

Single and Multi-Agent Environments
'''

import time
import numpy as np

from spectrl.util.rl import test_policy_mutli
from spectrl.util.io import save_log_info
from spectrl.multi_agent.ne_verification import FiniteStatePolicy


class QRMAgent:

    '''
    Single Agent Q-Learning Algorithm from Reward Machines
    '''

    def __init__(self, epsilon=0.15, lr=0.1, gamma=0.9, use_crm=True):

        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.use_crm = use_crm
        self.q_table = {}
        self.best_action = {}

    def learn(self, env, rm, num_steps):
        # set start state
        env_state = env.reset()
        rm_state = rm.get_initial()

        # training loop

        for _ in range(num_steps):

            # get epsilon greedy action
            action = self.ep_greedy_action(env_state, rm_state, env.action_space)

            # take step in rm and env
            next_rm_state, reward = rm.step(rm_state, env_state)
            next_env_state, _, done, _ = env.step(action)

            experience = [(env_state, rm_state, action, reward,
                           next_env_state, next_rm_state, done)]

            # Use CRM:
            if self.use_crm:
                # generate more experiences.
                for rm_state_prime in range(rm.num_states):
                    if rm_state_prime != rm_state:
                        next_rm_state_prime, reward_prime = rm.step(rm_state_prime, env_state)
                        experience.append((env_state, rm_state_prime, action, reward_prime,
                                           next_env_state, next_rm_state_prime, done))

            # update q table
            self.update(experience)

            # update state and reset env and rm if done
            rm_state = next_rm_state
            env_state = next_env_state

            if done:
                env_state = env.reset()
                rm_state = rm.get_initial()

        return QRMPolicy(self.best_table, env.action_space, rm)

    def ep_greedy_action(self, env_state, rm_state, action_space):

        if np.random.rand() <= self.epsilon or (env_state, rm_state) not in self.best_action:
            return action_space.sample()
        else:
            return self.best_action[(env_state, rm_state)]

    def update(self, experiences):

        for (env_state, rm_state, action, reward,
             next_env_state, next_rm_state, done) in experiences:
            # Compute value of the next state
            if ((next_env_state, next_rm_state) not in self.best_action) or done:
                V = 0
            else:
                V = self.q_table[(next_env_state, next_rm_state,
                                  self.best_action[(next_env_state, next_rm_state)])]

            # Compute current Q value of state
            if (env_state, rm_state, action) in self.q_table:
                Q_curr = self.q_table[(env_state, rm_state, action)]
            else:
                Q_curr = 0

            # Update Q value of state

            self.q_table[(env_state, rm_state, action)] = (1-self.lr) * \
                Q_curr + self.lr * (reward + self.gamma * V)

            # Update best action at state
            if ((env_state, rm_state) not in self.best_action) or \
               (self.q_table[(env_state, rm_state, action)] > self.q_table[
                   (env_state, rm_state, self.best_action[(env_state, rm_state)])]):
                self.best_action[(env_state, rm_state)] = action


class QRMPolicy(FiniteStatePolicy):

    def __init__(self, best_action, action_space, rm):

        self.best_action = best_action
        self.action_space = action_space
        self.rm = rm
        super().__init__(update_first=False, joint_policy=False)

    def init_state(self):
        return self.rm.get_initial()

    def step(self, policy_state, env_state, action):
        return self.rm.step(policy_state, env_state)[0]

    def act(self, policy_state, env_state):
        if (env_state, policy_state) in self.best_action:
            return self.best_action[(env_state, policy_state)]
        else:
            return self.action_space.sample()


class MultiQRMAgent:

    def __init__(self, num_agents, use_crm=True):

        self.n = num_agents
        self.tables = [QRMAgent() for _ in range(self.n)]
        self.use_crm = use_crm

    def learn(self, env, rm_list, num_steps, itno, folder):

        log_info = []
        start_time = time.time()
        env_state = env.reset()
        rm_state = tuple([rm_list[a].get_initial() for a in range(self.n)])

        if self.use_crm:
            all_rm_states = [[]]
            for agent in range(self.n):
                all_rm_states_temp = [
                    ell+[elem] for ell in all_rm_states
                    for elem in range(rm_list[agent].num_states)]
                # print(all_rm_states_temp)
                all_rm_states = all_rm_states_temp

            all_rm_states = [tuple(elem) for elem in all_rm_states]

        for stps in range(num_steps):

            actions = tuple([self.tables[a].ep_greedy_action(env_state, rm_state[a],
                                                             env.action_space.spaces[a])
                             for a in range(self.n)])

            next_rm_list_temp = [rm_list[a].step(rm_state[a], env_state)
                                 for a in range(self.n)]
            next_rm_state = tuple([temp[0] for temp in next_rm_list_temp])
            rewards = tuple([temp[1] for temp in next_rm_list_temp])
            next_env_state, _, done, _ = env.step(actions)

            experience = [(env_state, rm_state, actions, rewards,
                           next_env_state, next_rm_state, done)]

            if self.use_crm:
                for rm_state_prime in all_rm_states:
                    if rm_state_prime != rm_state:
                        next_rm_list_temp_prime = [rm_list[a].step(rm_state_prime[a], env_state)
                                                   for a in range(self.n)]
                        next_rm_state_prime = tuple([t[0] for t in next_rm_list_temp_prime])
                        rewards_prime = tuple([t[1] for t in next_rm_list_temp_prime])

                        experience.append((env_state, rm_state_prime, actions, rewards_prime,
                                           next_env_state, next_rm_state_prime, done))

            for a in range(self.n):
                self.tables[a].update([(e, rs, act[a], r[a], next_e, next_rs, d) for
                                       (e, rs, act, r, next_e, next_rs, d) in experience])

            env_state = next_env_state
            rm_state = next_rm_state

            if done:
                env_state = env.reset()
                rm_state = tuple([rm_list[a].get_initial() for a in range(self.n)])

            if stps % 10000 == 0:
                policies = [QRMPolicy(self.tables[a].best_action,
                                      env.action_space.spaces[a], rm_list[a])
                            for a in range(self.n)]
                joint_policy = MultiQRMAgentPolicy(policies)
                _, av_prob = test_policy_mutli(
                    env, joint_policy, 100, use_rm_reward=True, stateful_policy=True)
                time_taken = time.time() - start_time
                log_info.append(np.concatenate([[stps], [time_taken], av_prob, [np.mean(av_prob)]]))

            if stps % 100000 == 0:
                save_log_info(log_info, itno, folder)
                # save_object('policy', joint_policy, itno, folder)

        policies = [QRMPolicy(self.tables[a].best_action,
                              env.action_space.spaces[a], rm_list[a])
                    for a in range(self.n)]
        return MultiQRMAgentPolicy(policies), log_info


class MultiQRMAgentPolicy(FiniteStatePolicy):

    def __init__(self, policies):
        self.policies = policies
        self.reward_machines = [policies[a].rm for a in range(len(policies))]
        super().__init__(update_first=False)

    def init_state(self):
        return tuple([policy.init_state() for policy in self.policies])

    def step(self, policy_state, env_state, action):
        return tuple([self.policies[a].step(policy_state[a], env_state, action[a])
                      for a in range(len(self.policies))])

    def act(self, policy_state, env_state):
        return tuple([self.policies[a].act(policy_state[a], env_state)
                      for a in range(len(self.policies))])
