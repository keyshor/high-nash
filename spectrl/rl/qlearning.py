'''
Tabular Q-learning for finite state systems.
'''

import numpy as np


class QLearningAgent:
    '''
    Q-learning agent.

    Parameters:
        epsilon: epsilon for epsilon-greedy exploration
        action_list: state -> list of actions available in the state
    '''

    def __init__(self, epsilon=0.15, lr=0.1, gamma=0.9):
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.q_table = {}
        self.best_action = {}

    def learn(self, env, num_steps):
        # set start state
        state = env.reset()

        # training loop
        for _ in range(num_steps):

            # get epsilon greedy action
            action = self.ep_greedy_action(state, env.action_space)

            # take a step in environment
            next_state, reward, done, _ = env.step(action)

            # update q table
            self.update(state, action, reward, next_state, done)

            # reset environment if done
            state = next_state
            if done:
                state = env.reset()

        return QPolicy(self.best_action, env.action_space)

    def ep_greedy_action(self, state, action_space):
        if np.random.rand() <= self.epsilon or state not in self.best_action:
            return action_space.sample()
        else:
            return self.best_action[state]

    def update(self, state, action, reward, next_state, done):
        # Compute value of next state
        if next_state not in self.best_action or done:
            V = 0
        else:
            V = self.q_table[(next_state, self.best_action[next_state])]

        # Compute current Q value of state
        if (state, action) in self.q_table:
            Q_curr = self.q_table[(state, action)]
        else:
            Q_curr = 0

        # Update Q value of state
        self.q_table[(state, action)] = (1-self.lr) * Q_curr + self.lr * (reward + self.gamma * V)

        # Update best action at state
        if state not in self.best_action or \
                self.q_table[(state, action)] > self.q_table[(state, self.best_action[state])]:
            self.best_action[state] = action


class QPolicy:
    '''
    Policy learned using Q-learning
    '''

    def __init__(self, best_action, action_space):
        self.best_action = best_action
        self.action_space = action_space

    def get_action(self, state):
        if state in self.best_action:
            return self.best_action[state]
        else:
            return self.action_space.sample()


class MultiQLearningAgent:

    def __init__(self, num_agents):

        self.n = num_agents

        self.tables = [QLearningAgent() for _ in range(self.n)]

        def learn(self, env, num_steps):

            state = env.reset()

            for _ in range(num_steps):

                actions = tuple([self.tables[a].ep_greedy_action(
                    state, env.action_space.spaces[a]) for a in range(self.n)])

                next_state, rewards, done, _ = env.step(actions)

                for a in range(self.n):
                    self.tables[a].update(state, actions[a], rewards[a], next_state, done)

                state = next_state

                if done:
                    state = env.reset()
