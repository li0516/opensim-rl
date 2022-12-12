import torch
import numpy as np
import TD3_agent
import matplotlib.pyplot as plt
from osim.env import L2M2019Env
import os
import xlwt

env = L2M2019Env(visualize=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
max_step = 1e6
start_step = 1e4


def eval_policy(policy, env1, seed, eval_episodes=10, eval_cnt=None):
    eval_env = env1
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = eval_env.get_observation()
        while not done:
            action = policy.choose_action(state)
            next_state, reward, done, _ = eval_env.step(action)
            next_state = eval_env.get_observation()
            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


def main():
    book = xlwt.Workbook()
    sheets = book.add_sheet('sheet1')

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    s = env.reset()
    env.set_difficulty(2)
    s = env.get_observation()
    q_re = 0
    episode_step = 0
    return_g = 0
    episode_num = 0
    eval_cnt = 0
    Value_list = []
    episode_value_list = []
    TD3 = TD3_agent.TD3_agent(state_counters=state_dim,
                              action_counters=action_dim,
                              device=device,
                              batch_size=256,
                              max_size=int(5e6),
                              LR_A=0.0001,
                              LR_C=0.0001,
                              gamma=0.99,
                              TAU=0.005)

    # return_aver = eval_policy(TD3, env, 0, eval_episodes=10, eval_cnt=eval_cnt)
    row = 0
    log_dir = './model_TD3/saved_model.pth'
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        TD3.Actor_Net_eval.load_state_dict(checkpoint['model1'])
        TD3.Actor_Net_target.load_state_dict(checkpoint['model2'])
        TD3.Critic_Net_eval_1.load_state_dict(checkpoint['model3'])
        TD3.Critic_Net_eval_2.load_state_dict(checkpoint['model4'])
        TD3.Critic_Net_target_1.load_state_dict(checkpoint['model5'])
        TD3.Critic_Net_target_2.load_state_dict(checkpoint['model6'])
        start_epoch = checkpoint['epoch']
        # start_epoch = start_epoch + 1
        print('加载模型成功')
    else:
        start_epoch = 0
        print('从0开始')
    for t in range(start_epoch + 1, int(max_step)):

        episode_step += 1
        if t < int(start_step):

            action = env.action_space.sample()
        else:
            action = TD3.choose_action(s)
            action = np.clip(action + np.random.normal(0, 0.2, size=action_dim), 0, 1)
        # if t % 10000 == 0:
        #     print(action)
        s_, r, done, _ = env.step(action)
        s_ = env.get_observation()
        d = float(done) if episode_step < env.time_limit else 0

        TD3.store_memory(s, action, r, s_, d)

        s = s_
        q_re += r

        if t > int(start_step):
            TD3.learn(t)

        if done:
            print("time: {} episode: {} episode_step: {} reword: {}"
                  .format(t + 1, episode_num + 1, episode_step, q_re))

            sheets.write(row, 0, str(t + 1))
            sheets.write(row, 1, str(episode_num + 1))
            sheets.write(row, 2, str(episode_step))
            sheets.write(row, 3, str(q_re))
            sheets.col(0).width = 5000
            sheets.col(1).width = 6000
            sheets.col(2).width = 5000
            sheets.col(3).width = 6000

            # 保存数据到.xlsx文件
            book.save("data_TD3.xls")

            row = row + 1

            TD3.save(t, log_dir)

            episode_num += 1
            return_g += q_re
            aver_episode = return_g / episode_num
            episode_value_list.append((aver_episode, episode_num))
            aver_return = return_g / (t + 1)
            Value_list.append((aver_return, t))
            s = env.reset()
            s = env.get_observation()
            episode_step = 0
            q_re = 0

        # if (t + 1) % 5000 == 0:
        #
        #     return_aver = eval_policy(SD3, env, 0, eval_episodes=10, eval_cnt=eval_cnt)
        #     eval_cnt += 1

    x = []
    y = []
    x1 = []
    y1 = []
    for i, j in Value_list:
        y.append(i)
        x.append(j)
    for m, n in episode_value_list:
        y1.append(m)
        x1.append(n)
    plt.figure(1)
    plt.plot(x, y)
    plt.xlabel("time")
    plt.ylabel("Average time value")
    plt.figure(2)
    plt.plot(x1, y1)
    plt.xlabel("episode")
    plt.ylabel("Average episode value")
    plt.show()


if __name__ == "__main__":
    main()
