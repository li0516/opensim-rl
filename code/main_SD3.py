import gc
import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import math
from osim.env import L2M2019Env
import memory as memory
import models as models
import utilities as utilities
import xlwt

def learn(update_critic_1):
    states, actions, rewards, next_states, d = ram.sample_exp(128)

    states = Variable(torch.from_numpy(states))
    actions = Variable(torch.from_numpy(actions))
    rewards = Variable(torch.from_numpy(rewards))
    next_states = Variable(torch.from_numpy(next_states))
    d = Variable(torch.from_numpy(d))

    # print('p', states)
    with torch.no_grad():
        if update_critic_1:
            predicted_action = target_actor_1.forward(next_states).detach()
        else:
            predicted_action = target_actor_2.forward(next_states).detach()
        # print('p', predicted_action)
        min_q = torch.min(target_critic_1.forward(next_states, predicted_action),
                          target_critic_2.forward(next_states, predicted_action))
        softmax_q_s_a = softmax_operator(min_q)
        y_expected = rewards + 0.99 * (1 - d) * softmax_q_s_a

    if update_critic_1:
        y_predicted = torch.squeeze(critic_1.forward(states, actions))

        critic_loss = F.smooth_l1_loss(y_predicted, y_expected)
        critic_optimizer_1.zero_grad()
        critic_loss.backward()
        critic_optimizer_1.step()

        predicted_action = actor_1.forward(states)
        actor_loss = -1 * torch.sum(critic_1.forward(states, predicted_action))
        actor_optimizer_1.zero_grad()
        actor_loss.backward()
        actor_optimizer_1.step()

        for target_param, param in zip(target_actor_1.parameters(), actor_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - 0.005) + param.data * 0.005)

        for target_param, param in zip(target_critic_1.parameters(), critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - 0.005) + param.data * 0.005)
    else:
        y_predicted = torch.squeeze(critic_2.forward(states, actions))

        critic_loss = F.smooth_l1_loss(y_predicted, y_expected)
        critic_optimizer_2.zero_grad()
        critic_loss.backward()
        critic_optimizer_2.step()

        predicted_action = actor_2.forward(states)
        actor_loss = -1 * torch.sum(critic_2.forward(states, predicted_action))
        actor_optimizer_2.zero_grad()
        actor_loss.backward()
        actor_optimizer_2.step()

        for target_param, param in zip(target_actor_2.parameters(), actor_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - 0.005) + param.data * 0.005)

        for target_param, param in zip(target_critic_2.parameters(), critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - 0.005) + param.data * 0.005)


def softmax_operator(q_eval):
    q_max = torch.max(q_eval, dim=1, keepdim=True).values
    q_norm = q_eval - q_max
    q = torch.exp(0.05 * q_norm)
    q_s_a = q * q_eval

    # if self.important_sampling:
    #     q_s_a /= prob_noise
    #     q /= prob_noise

    softmax_q = torch.sum(q_s_a, 1) / torch.sum(q, 1)

    return softmax_q

if __name__ == "__main__":

    book = xlwt.Workbook()
    sheets = book.add_sheet('sheet1')

    # env = gym.make('BipedalWalker-v3')
    env = L2M2019Env(visualize=False)

    env.set_difficulty(2)

    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]
    action_max = env.action_space.high[0]

    print("State dimension: {}".format(state_dimension))
    print("Action dimension: {}".format(action_dimension))
    print("Action max: {}".format(action_max))

    load_models = False

    # Actor network, critic network

    actor_1 = models.Actor(state_dimension, action_dimension, action_max)
    target_actor_1 = models.Actor(state_dimension, action_dimension, action_max)
    actor_optimizer_1 = torch.optim.Adam(actor_1.parameters(), lr=0.00001)

    actor_2 = models.Actor(state_dimension, action_dimension, action_max)
    target_actor_2 = models.Actor(state_dimension, action_dimension, action_max)
    actor_optimizer_2 = torch.optim.Adam(actor_2.parameters(), lr=0.00001)

    critic_1 = models.Critic(state_dimension, action_dimension)
    target_critic_1 = models.Critic(state_dimension, action_dimension)
    critic_optimizer_1 = torch.optim.Adam(critic_1.parameters(), lr=0.00001)

    critic_2 = models.Critic(state_dimension, action_dimension)
    target_critic_2 = models.Critic(state_dimension, action_dimension)
    critic_optimizer_2 = torch.optim.Adam(critic_2.parameters(), lr=0.00001)

    # Parameter noise-d zoriulsan actor

    actor_copy_1 = models.Actor(state_dimension, action_dimension, action_max)
    actor_copy_2 = models.Actor(state_dimension, action_dimension, action_max)

    # 初始化 Target network参数
    for target_param, param in zip(target_actor_1.parameters(), actor_1.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_critic_1.parameters(), critic_1.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_actor_2.parameters(), actor_2.parameters()):
        target_param.data.copy_(param.data)

    for target_param, param in zip(target_critic_2.parameters(), critic_2.parameters()):
        target_param.data.copy_(param.data)

    # Replay buffer
    ram = memory.ReplayBuffer(1000000)

    episode_step = 0
    # Reward list
    reward_list = []
    average_reward_list = []

    # Parameter noise
    parameter_noise1 = utilities.AdaptiveParamNoiseSpec(initial_stddev=1.55, desired_action_stddev=0.001,
                                                       adaptation_coefficient=1.05)

    parameter_noise2 = utilities.AdaptiveParamNoiseSpec(initial_stddev=1.05, desired_action_stddev=0.001,
                                                       adaptation_coefficient=1.05)

    noise = utilities.OrnsteinUhlenbeckActionNoise(action_dimension)

    observation = env.reset()
    observation = env.get_observation()

    row = 0
    ep_reward = 0
    ep = 0
    for t in range(1000000):
        episode_step += 1
        # step_cntr = 0

        # env.render()
        state = np.float32(observation)

        noise.reset()

        initial_state = Variable(torch.from_numpy(state))
        # print('1', initial_state)
        if t < 100:
            action_with_parameter_noise = env.action_space.sample()
            # action_with_parameter_ou_noise = action_with_parameter_noise + (noise.sample() * action_max)
            action_with_parameter_ou_noise = np.clip(action_with_parameter_noise + np.random.normal(0, 0.2, size=action_dimension), 0, 1)
        else:
            initial_state = Variable(torch.from_numpy(state)).reshape(1, -1)
            print(initial_state.shape)
            action_with_parameter_noise_1 = actor_copy_1.forward(initial_state).detach()
            # action_1 = actor_1.forward(initial_state).detach()
            action_with_parameter_noise_2 = actor_copy_2.forward(initial_state).detach()
            # action_2 = actor_2.forward(initial_state).detach()

            q_1 = critic_1.forward(initial_state, action_with_parameter_noise_1)
            q_2 = critic_2.forward(initial_state, action_with_parameter_noise_2)

            action_with_parameter_noise_1 = action_with_parameter_noise_1[0]
            action_with_parameter_noise_2 = action_with_parameter_noise_2[0]

            action = action_with_parameter_noise_1 if q_1 >= q_2 else action_with_parameter_noise_2
            # action = action_1 if q_1 >= q_2 else action_2
            # print('1', action_with_parameter_noise_1)
            # print('2', action_with_parameter_noise_2)
            action_with_parameter_ou_noise = action.numpy() + (noise.sample() * action_max)
            action_with_parameter_ou_noise = np.clip(action_with_parameter_ou_noise, 0, 1)

            # print(action_with_parameter_noise)
        new_observation, reward, done, info = env.step(action_with_parameter_ou_noise)
        new_observation = env.get_observation()

        new_state = np.float32(new_observation)

        # Replay buffer-d state, action, reward, new_state
        d = float(done) if episode_step < env.time_limit else 0
        ram.add(state, action_with_parameter_ou_noise, reward, new_state, d)
        ep_reward += reward

        observation = new_observation

        # 采样进行学习
        learn(update_critic_1=True)
        learn(update_critic_1=False)

        # step_cntr += 1

        if done:
            print("timeStep: {} Episode: {} episode_step: {} Reward: {}".format(t + 1, ep + 1, episode_step, ep_reward))

            sheets.write(row, 0, str(t + 1))
            sheets.write(row, 1, str(ep + 1))
            sheets.write(row, 2, str(episode_step))
            sheets.write(row, 3, str(ep_reward))
            sheets.col(0).width = 5000
            sheets.col(1).width = 6000
            sheets.col(2).width = 5000
            sheets.col(3).width = 6000
            # 保存数据到.xlsx文件
            book.save("data_SD3.xls")

            noise_data_list = list(ram.buffer)
            noise_data_list = np.array(noise_data_list[-episode_step:])
            actor_copy_state, actor_copy_action, _, _, _ = zip(*noise_data_list)
            actor_copy_actions = np.array(actor_copy_action)

            actor_1_actions = []
            actor_2_actions = []

            for state in np.array(actor_copy_state):
                state = Variable(torch.from_numpy(state))
                action_1 = actor_1.forward(state).detach().numpy()
                action_2 = actor_2.forward(state).detach().numpy()
                actor_1_actions.append(action_1)
                actor_2_actions.append(action_2)

            diff_actions_1 = actor_copy_actions - actor_1_actions
            mean_diff_actions_1 = np.mean(np.square(diff_actions_1), axis=0)
            distance_1 = math.sqrt(np.mean(mean_diff_actions_1))

            diff_actions_2 = actor_copy_actions - actor_2_actions
            mean_diff_actions_2 = np.mean(np.square(diff_actions_2), axis=0)
            distance_2 = math.sqrt(np.mean(mean_diff_actions_2))

            # Sigma-g update 更新scale
            parameter_noise1.adapt(distance_1)
            parameter_noise2.adapt(distance_2)
            # if ep % 100 == 0:
            #     torch.save(target_actor.state_dict(), './Models/' + str(ep) + '_actor.pt')
            #     torch.save(target_critic.state_dict(), './Models/' + str(ep) + '_critic.pt')
            #     print("Target actor, critic models saved")

            # reward_list.append(ep_reward)
            # average_reward = np.mean(reward_list[-40:])

            # average_reward_list.append(average_reward)

            # Actor-g actor_copy-d 参数赋值
            for target_param, param in zip(actor_copy_1.parameters(), actor_1.parameters()):
                target_param.data.copy_(param.data)

            for target_param, param in zip(actor_copy_2.parameters(), actor_2.parameters()):
                target_param.data.copy_(param.data)

            # Parameter noise 神经网络更新
            parameters_1 = actor_copy_1.state_dict()
            for name in parameters_1:
                parameter_1 = parameters_1[name]
                rand_number = torch.randn(parameter_1.shape)
                parameter_1 = parameter_1 + rand_number * parameter_noise1.current_stddev

            parameters_2 = actor_copy_2.state_dict()
            for name in parameters_2:
                parameter_2 = parameters_2[name]
                rand_number = torch.randn(parameter_2.shape)
                parameter_2 = parameter_2 + rand_number * parameter_noise2.current_stddev

            ep = ep + 1
            episode_step = 0
            ep_reward = 0
            observation = env.reset()
            observation = env.get_observation()

            row = row + 1
        # gc.collect()

    # print("Reward max: ", max(average_reward_list))

    # plt.plot(average_reward_list)
    # plt.legend()
    # plt.xlabel("Episode")
    # plt.ylabel("Average Episode Reward")
    # plt.show()
