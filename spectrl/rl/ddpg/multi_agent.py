"""
Cooperative multi-agent DDPG
Adapted from:
    Deep Deterministic Policy Gradient agent
    Author: Sameera Lanka
    Website: https://sameera-lanka.com
"""

# Torch
import torch
import torch.nn as nn
import torch.optim as optim


# Lib
from copy import deepcopy
import numpy as np
import time

# Files
from spectrl.rl.ddpg.noise import OrnsteinUhlenbeckActionNoise as OUNoise
from spectrl.rl.ddpg.noise import NormalActionNoise
from spectrl.rl.ddpg.replaybuffer import MultiBuffer
from spectrl.rl.ddpg.actorcritic import MultiActor, Critic
from spectrl.util.rl import test_policy


# HyperParams
class MultiDDPGParams:

    def __init__(self,
                 state_dims,
                 action_dims,
                 action_bounds,
                 actor_lr=0.0001,
                 critic_lr=0.001,
                 minibatch_size=100,
                 num_episodes=10000,
                 mu=0,
                 sigma=0.1,
                 target_noise=0.002,
                 target_clip=0.005,
                 buffer_size=200000,
                 discount=0.99,
                 tau=0.005,
                 warmup=200,
                 epsilon_decay=1e-6,
                 epsilon_min=0.5,
                 decay_function='linear',
                 steps_per_update=1,
                 gradients_per_update=1,
                 actor_hidden_dim=100,
                 critic_hidden_dim=100,
                 noise='normal',
                 max_timesteps=1000,
                 test_max_timesteps=1000):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.action_bounds = action_bounds
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.minibatch_size = minibatch_size
        self.num_episodes = num_episodes
        self.mu = mu
        self.sigma = sigma
        self.target_noise = target_noise
        self.target_clip = target_clip
        self.buffer_size = buffer_size
        self.discount = discount
        self.tau = tau
        self.warmup = warmup
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.decay_function = decay_function
        self.steps_per_update = steps_per_update
        self.gradients_per_update = gradients_per_update
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.noise = noise
        self.max_timesteps = max_timesteps
        self.test_max_timesteps = test_max_timesteps


class MultiDDPG:
    def __init__(self, params, use_gpu=False):
        self.params = params
        self.n = len(self.params.state_dims)
        self.epsilon = 1.0
        self.use_gpu = use_gpu
        self.total_state_dim = sum(self.params.state_dims)
        self.total_action_dim = sum(self.params.action_dims)
        self.actor = MultiActor(self.params.state_dims, self.params.action_dims,
                                self.params.actor_hidden_dim, self.params.action_bounds)
        self.critic = Critic(self.total_state_dim, self.total_action_dim,
                             self.params.critic_hidden_dim)
        self.critic2 = Critic(self.total_state_dim, self.total_action_dim,
                              self.params.critic_hidden_dim)
        self.targetActor = deepcopy(self.actor)
        self.targetCritic = deepcopy(self.critic)
        self.targetCritic2 = deepcopy(self.critic2)
        if self.use_gpu:
            self.actor = self.actor.cuda()
            self.actor.set_use_gpu()
            self.critic = self.critic.cuda()
            self.critic2 = self.critic2.cuda()
            self.targetActor = self.targetActor.cuda()
            self.targetActor.set_use_gpu()
            self.targetCritic = self.targetCritic.cuda()
            self.targetCritic2 = self.targetCritic2.cuda()
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=self.params.actor_lr)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=self.params.critic_lr)
        self.criticOptim2 = optim.Adam(self.critic2.parameters(), lr=self.params.critic_lr)
        self.criticLoss = nn.MSELoss()
        if self.params.noise == 'ou':
            self.noise = [OUNoise(mu=np.zeros(d), sigma=self.params.sigma * np.ones(d))
                          for d in self.params.action_dims]
        elif self.params.noise == 'normal':
            self.noise = [NormalActionNoise(mu=np.zeros(d),
                                            sigma=self.params.sigma*np.ones(d))
                          for d in self.params.action_dims]
        self.replayBuffer = MultiBuffer(self.params.buffer_size, self.n)
        self.rewardgraph = []

    def getQTarget(self, nextStateBatch, rewardBatch, terminalBatch):
        """Inputs: Batch of next states, rewards and terminal flags of size self.batchSize
            Calculates the target Q-value from reward and bootstraped Q-value of next state
            using the target actor and target critic
           Outputs: Batch of Q-value targets"""

        with torch.no_grad():

            # create required tensors
            targetBatch = torch.Tensor(rewardBatch)
            nonFinalMask = torch.Tensor(list(map(lambda s: s is not True, terminalBatch)))

            # move tensors to gpu
            if self.use_gpu:
                nonFinalMask = nonFinalMask.cuda()
                targetBatch = targetBatch.cuda()

            # get actions from targetActor
            nextActionBatch = self.targetActor.forward(nextStateBatch)
            target_noise = [self.params.target_noise *
                            np.random.randn(self.params.minibatch_size, self.params.action_dims[i])
                            for i in range(self.n)]
            target_noise = [np.clip(t, -self.params.target_clip, self.params.target_clip)
                            for t in target_noise]
            target_noise = [torch.Tensor(t) for t in target_noise]
            if self.use_gpu:
                target_noise = [t.cuda() for t in target_noise]
            nextActionBatch = [torch.clamp(nextActionBatch[i] + target_noise[i],
                                           -self.params.action_bounds[i][0],
                                           self.params.action_bounds[i][0])
                               for i in range(self.n)]

            # get Q-values for nest states
            fullStates = torch.cat(nextStateBatch, dim=1)
            fullActions = torch.cat(nextActionBatch, dim=1)
            qNext = torch.squeeze(self.targetCritic(fullStates, fullActions), dim=1)
            qNext2 = torch.squeeze(self.targetCritic2(fullStates, fullActions), dim=1)
            qNext = torch.min(qNext, qNext2)
            nonFinalMask = self.params.discount * nonFinalMask
            targetBatch += nonFinalMask * qNext

        return targetBatch

    def updateTargets(self, target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""

        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_(((1 - self.params.tau) * targetParam.data) +
                                   (self.params.tau * orgParam.data))

    def getMaxAction(self, curState):
        """Inputs: Current state of the episode
            Returns the action which maximizes the Q-value of the current state-action pair"""

        if len(self.replayBuffer) < self.params.warmup:
            action = [self.params.action_bounds[i] *
                      np.random.uniform(low=-1.0, high=1.0, size=self.params.action_dims[i])
                      for i in range(self.n)]
            action = [torch.Tensor(a) for a in action]
            if self.use_gpu:
                action = [a.cuda() for a in action]
            return action

        noise = [torch.Tensor(self.epsilon * self.params.action_bounds[i] * self.noise[i]())
                 for i in range(self.n)]
        if self.use_gpu:
            noise = [ns.cuda() for ns in noise]
        action = self.actor.get_action(curState, using_tensors=True)
        actionNoise = [torch.clamp(action[i] + noise[i],
                                   -self.params.action_bounds[i][0],
                                   self.params.action_bounds[i][0])
                       for i in range(self.n)]

        return actionNoise

    def train(self, env):

        best_reward = -1e9
        best_actor = deepcopy(self.actor)
        num_steps = 0
        start_time = time.time()
        g_per_u = self.params.gradients_per_update

        for i in range(self.params.num_episodes):
            state = env.reset()
            done = False
            ep_reward = 0
            ep_steps = 0

            while (not done) and (ep_steps < self.params.max_timesteps):
                # Get maximizing action
                curState = [torch.Tensor(s) for s in state]
                if self.use_gpu:
                    curState = [s.cuda() for s in curState]
                action = self.getMaxAction(curState)
                numpy_action = [a.cpu().numpy() for a in action]

                # Step episode
                state, reward, terminal, _ = env.step(numpy_action)
                nextState = [torch.Tensor(s) for s in state]
                if self.use_gpu:
                    nextState = [s.cuda() for s in nextState]
                ep_reward += reward
                done = terminal

                # Update replay bufer
                self.replayBuffer.append((curState, action, nextState, reward, terminal))

                # Training loop
                if (len(self.replayBuffer) >= self.params.warmup) and \
                        (num_steps % self.params.steps_per_update == 0):

                    total_critic_loss = 0.
                    total_critic_loss2 = 0.
                    total_actor_loss = 0.

                    for g in range(g_per_u):

                        (curStateBatch, actionBatch, nextStateBatch, rewardBatch,
                         terminalBatch) = self.replayBuffer.sample_batch(self.params.minibatch_size)

                        fullStateBatch = torch.cat(curStateBatch, dim=1)
                        fullActionBatch = torch.cat(actionBatch, dim=1)
                        qPredBatch = torch.squeeze(self.critic(
                            fullStateBatch, fullActionBatch), dim=1)
                        qPredBatch2 = torch.squeeze(self.critic2(
                            fullStateBatch, fullActionBatch), dim=1)
                        qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)

                        # Critic update
                        self.criticOptim.zero_grad()
                        criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                        total_critic_loss += float(criticLoss)
                        criticLoss.backward()
                        self.criticOptim.step()

                        # Critic update 2
                        self.criticOptim2.zero_grad()
                        criticLoss2 = self.criticLoss(qPredBatch2, qTargetBatch)
                        total_critic_loss2 += float(criticLoss2)
                        criticLoss2.backward()
                        self.criticOptim2.step()

                        # Actor update
                        if (g % 2 == 0) or (g_per_u == 1 and
                                            num_steps % 2 * self.params.steps_per_update == 0):
                            self.actorOptim.zero_grad()
                            combinedAction = torch.cat(self.actor.forward(curStateBatch), dim=1)
                            losses = -torch.squeeze(self.critic(fullStateBatch,
                                                                combinedAction), dim=1)
                            actorLoss = torch.mean(losses)
                            total_actor_loss += float(actorLoss)
                            actorLoss.backward()
                            self.actorOptim.step()

                            # Update Targets
                            self.updateTargets(self.targetActor, self.actor)
                        self.updateTargets(self.targetCritic, self.critic)
                        self.updateTargets(self.targetCritic2, self.critic2)

                    if num_steps % 1000 == 0:
                        print('\nCritic Loss 1: {}'.format(total_critic_loss / g_per_u))
                        print('Critic Loss 2: {}'.format(total_critic_loss2 / g_per_u))
                        print('Actor Loss: {}'.format(2 * total_actor_loss / g_per_u))
                        print('Exploration Noise: {}\n'.format(
                            self.epsilon * self.params.sigma))

                # Decay exploration noise
                if self.params.decay_function == 'linear':
                    self.epsilon = max(self.params.epsilon_min,
                                       self.epsilon - self.params.epsilon_decay)
                elif self.params.decay_function == 'exp':
                    self.epsilon = max(self.params.epsilon_min,
                                       0.8 ** (num_steps / self.params.epsilon_decay))

                # update num_steps
                num_steps += 1
                ep_steps += 1

            print('Reward at episode {}: {}'.format(i, ep_reward))
            if i % 10 == 4:
                avg_reward, _ = test_policy(
                    env, self.actor, 20, max_timesteps=self.params.test_max_timesteps)
                print('Expected reward after {} episodes: {}'.format(i, avg_reward))
                time_spent = (time.time() - start_time) / 60
                self.rewardgraph.append([num_steps, time_spent, avg_reward])
                if i > (self.params.num_episodes/2) and best_reward < avg_reward:
                    best_reward = avg_reward
                    best_actor = deepcopy(self.actor)

        self.actor = best_actor

    def get_policy(self):
        return self.actor

    def get_policies(self):
        if self.use_gpu:
            return self.actor.set_use_cpu()
        else:
            return self.actor.actors


if __name__ == '__main__':
    from spectrl.envs.particles import MultiParticleEnv
    from spectrl.util.rl import MultiAgentPolicy, get_rollout
    start_pos_low = [[-0.1, -0.1], [-0.1, 9.9]]
    start_pos_high = [[0.1, 0.1], [0.1, 10.1]]
    goals = [[10., 0.], [10., 10.]]
    env = MultiParticleEnv(start_pos_low, start_pos_high, goals=goals,
                           sum_rewards=True, max_timesteps=20, cooperative=True,
                           collision_penalty=0.5, terminate_on_reach=True)
    state_dims = [o.shape[0] for o in env.observation_space]
    action_dims = [a.shape[0] for a in env.action_space]
    action_bounds = [a.high for a in env.action_space]
    agent = MultiDDPG(MultiDDPGParams(state_dims, action_dims, action_bounds,
                                      minibatch_size=512, discount=0.95,
                                      warmup=5000, actor_hidden_dim=16,
                                      critic_hidden_dim=64, num_episodes=5000,
                                      gradients_per_update=1, steps_per_update=1,
                                      epsilon_min=0.5, epsilon_decay=2e-6, sigma=0.1,
                                      buffer_size=100000))
    agent.train(env)
    get_rollout(env, MultiAgentPolicy(agent.get_policies()), True)
