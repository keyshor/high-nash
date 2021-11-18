import numpy as np
import gym


class MultiParticleEnv(gym.Env):
    '''
    Generic multi agent environment, each agent is a sphere/disc in R^n.
    start_pos_low : array-like (lower bounds on starting positions)
    start_pos_high : array-like (upper bounds on starting positions)
    agent_size : float (diameter of the agent disc)
    '''

    def __init__(self, start_pos_low, start_pos_high, agent_size=1, max_timesteps=50,
                 noise=0., noise_clip=0., goals=None, terminate_on_collision=False,
                 terminate_on_reach=False, collision_penalty=1., sum_rewards=False,
                 cooperative=False, reward_multiplier=0.02, obs_multiplier=0.01):
        self.n = len(start_pos_low)
        self.start_low = np.array(start_pos_low)
        self.start_high = np.array(start_pos_high)
        self.agent_size = agent_size
        self.max_timesteps = max_timesteps
        self.noise = noise
        self.noise_clip = noise_clip
        self.goals = goals
        self.terminate_on_collision = terminate_on_collision
        self.terminate_on_reach = terminate_on_reach
        self.collision_penalty = collision_penalty
        self.sum_rewards = sum_rewards
        self.cooperative = cooperative
        self.reward_multiplier = reward_multiplier
        self.obs_multiplier = obs_multiplier

        # action and observation spaces are arrays
        pos_dim = len(self.start_low[0])
        obs_dim = pos_dim * self.n
        self.action_space = [gym.spaces.Box(low=-1, high=1, shape=(pos_dim,))
                             for _ in range(self.n)]
        self.observation_space = [gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
                                  for _ in range(self.n)]
        self.reset()

    def reset(self):
        self.t = 0
        self.positions = np.random.uniform(self.start_low, self.start_high)
        self.collisions = np.zeros((self.n,))
        self.reached = [False] * self.n
        return self.get_obs()

    def step(self, actions):
        actions = np.clip(actions, -1, 1)
        actions += np.clip(self.noise * np.random.randn(*actions.shape),
                           -self.noise_clip, self.noise_clip)
        self.positions = self.positions + actions
        self.check_collisions()
        goal_dist = self.goal_distances()

        self.t += 1
        return (self.get_obs(), self.reward_multiplier * self.rewards(goal_dist),
                self.check_termination(goal_dist), {})

    def render(self):
        print([list(np.around(pos, 2)) for pos in self.positions])

    def get_sim_state(self):
        return self.positions.copy()

    def set_sim_state(self, state):
        self.positions = state
        return self.get_obs()

    def get_obs(self):
        obs = []
        for i in range(self.n):
            obs_list = [self.positions[i]]
            for j in range(self.n):
                if not j == i:
                    obs_list.append(self.positions[j]-self.positions[i])
            obs.append(self.obs_multiplier * np.concatenate(obs_list))
        return obs

    def check_collisions(self):
        for i in range(self.n):
            for j in range(i+1, self.n):
                dist = np.linalg.norm(self.positions[i] - self.positions[j]) - \
                    (4. * self.agent_size)
                self.collisions[i] = min(dist, self.collisions[i])
                self.collisions[j] = min(dist, self.collisions[j])

    def goal_distances(self):
        if self.goals is None:
            gd = np.ones((self.n,)) + 1e-6
        else:
            gd = [np.linalg.norm(self.positions[i] - self.goals[i]) for i in range(self.n)]
        return np.array(gd)

    def check_termination(self, goal_dist):
        done = False
        if self.terminate_on_collision:
            done = np.any(self.collisions < -2. * self.agent_size)
        for i in range(self.n):
            self.reached[i] = self.reached[i] or (goal_dist[i] < 1. and
                                                  self.collisions[i] >= -2. * self.agent_size)
        if self.terminate_on_reach:
            done = done or np.all(self.reached)
        return done or self.t > self.max_timesteps

    def rewards(self, goal_dist):
        r = (-goal_dist * (1-np.array(self.reached, dtype=float))) + \
            self.collision_penalty * self.collisions
        if self.sum_rewards:
            r = np.sum(r)
            if not self.cooperative:
                r = r * np.ones((self.n,))
        return r

    def close(self):
        pass
