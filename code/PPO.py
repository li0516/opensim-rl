import os
import torch as T
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from osim.env import L2M2019Env
import xlwt
import matplotlib.pyplot as plt


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.rtgs = []

    def store_memory(self, state, action, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)

    def store_ep_reward(self, ep_reward):
        self.rewards.append(ep_reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.rtgs = []

    def get_observations(self):
        return self.states, self.actions, self.rewards, self.log_probs


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, layer1_dims, layer2_dims):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(input_dims, layer1_dims)
        self.layer2 = nn.Linear(layer1_dims, layer2_dims)
        self.output = nn.Linear(layer2_dims, output_dims)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = T.tensor(state, dtype=T.float)

        layer1 = F.relu(self.layer1(state))
        layer2 = F.relu(self.layer2(layer1))
        out = self.output(layer2)

        return out


class PPOAgent:
    def __init__(self, env):
        self.input_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.gamma = 0.99
        self.n_iter = 8
        self.clip = 0.2
        self.lr = 0.0001

        self.memory = PPOMemory()
        self.actor = FeedForwardNetwork(self.input_dim, self.n_actions, layer1_dims=256, layer2_dims=256)
        self.critic = FeedForwardNetwork(self.input_dim, 1, layer1_dims=256, layer2_dims=256)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = T.full(size=(self.n_actions,), fill_value=0.5)
        self.cov_mat = T.diag(self.cov_var)

    def evaluate(self, batch_obs, batch_acts):
        v = self.critic(batch_obs).squeeze()
        mu = self.actor(batch_obs)
        std = self.cov_mat
        dist = MultivariateNormal(mu, std)
        log_prob = dist.log_prob(batch_acts)

        return v, log_prob

    def choose_action(self, state):
        state = T.FloatTensor(state)
        mu = self.actor(state)
        dist = MultivariateNormal(mu, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob

    def learn(self):
        states, actions, rewards, old_log_probs = self.memory.get_observations()
        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        old_log_probs = T.tensor(old_log_probs, dtype=T.float)
        rtgs = self.compute_rtgs(rewards)

        old_value, _ = self.evaluate(states, actions)
        # print(rtgs.size())
        # print(old_value.detach().size())
        advantages = rtgs - old_value.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(self.n_iter):
            current_v, current_log_probs = self.evaluate(states, actions)

            ratios = T.exp(current_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = T.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            actor_loss = (-T.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(current_v, rtgs)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

        self.memory.clear_memory()

    def compute_rtgs(self, batch_rewards):
        rtgs = []
        for ep_rw in batch_rewards:
            for rw in range(len(ep_rw)):
                discount = 1
                r = 0
                for k in range(rw, len(ep_rw)):
                    r += discount * ep_rw[k]
                    discount *= self.gamma
                rtgs.append(r)
        rtgs = T.tensor(rtgs, dtype=T.float)
        return rtgs

    def remember(self, state, action, log_prob):
        self.memory.store_memory(state, action, log_prob)

    def remember_reward(self, ep_reward):
        self.memory.store_ep_reward(ep_reward)


if __name__ == '__main__':
    # env = gym.make('Pendulum-v0')
    book = xlwt.Workbook()
    sheets = book.add_sheet('sheet1')

    env = L2M2019Env(visualize=False)
    agent = PPOAgent(env)
    max_time_steps = 5000000
    batch_size = 4000
    t = 0
    # total_learning = 0
    score_history = []
    score_history_avg = []
    episode = 0
    row = 0

    while t < max_time_steps:  # max time steps for learning
        e_t = 0
        while e_t < batch_size:  # time steps store for traingin in a batch
            observation = env.reset()
            observation = env.get_observation()
            done = False
            ep_rwd = []
            score = 0
            while not done:
                # env.render()
                action, log_prob = agent.choose_action(observation)
                observation_, reward, done, _ = env.step(action)
                observation_ = env.get_observation()
                t += 1
                e_t += 1
                score += reward
                ep_rwd.append(reward)
                agent.remember(observation, action, log_prob)
                observation = observation_
            agent.remember_reward(ep_rwd)  # store episodic rewards
            episode += 1
            # store values for debugging
            # score_history.append(score)
            # avg_score = np.mean(score_history[-100:])
            # score_history_avg.append(avg_score)

            print('time_steps = {} episode = {} score = {} '.format(t, episode, score))

            sheets.write(row, 0, str(t))
            sheets.write(row, 1, str(episode))
            sheets.write(row, 2, str(score))
            sheets.col(0).width = 5000
            sheets.col(1).width = 6000
            sheets.col(2).width = 5000
            book.save("data_ppo_2.xls")
            
            row = row + 1
        # start learining for a batch
        agent.learn()
        # total_learning += 1

    # plt.plot(score_history_avg)
    # plt.show()
