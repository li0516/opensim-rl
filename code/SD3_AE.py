import torch
import numpy as np
import SD3_AE_Agent
import matplotlib.pyplot as plt
from osim.env import L2M2019Env
import xlwt
import os
env = L2M2019Env(visualize=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
max_step = 5000000
start_step = 10000


def main():
    book = xlwt.Workbook()
    sheets = book.add_sheet('sheet1')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # env.seed(0)
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
    SD3 = SD3_AE_Agent.SD3_Agent(state_dim,
                                 action_dim,
                                 batch_size=256,
                                 LR_A=1e-4,
                                 LR_C=1e-4,
                                 memory_size=500000,
                                 gamma=0.99,
                                 TAU=0.005,
                                 policy_noise=0.2,
                                 sample_size=50,
                                 noise_clip=0.5,
                                 beta=0.05,
                                 important_sampling=0,
                                 device=device)
    print(SD3.memory_size)
    # return_aver = eval_policy(SD3, env, 0, eval_episodes=10, eval_cnt=eval_cnt)

    row = 0
    log_dir = 'model_SD3_AE/saved_model.pth'
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        SD3.Actor_net_eval_1.load_state_dict(checkpoint['model1'])
        SD3.Actor_net_eval_2.load_state_dict(checkpoint['model2'])
        SD3.Actor_net_Target1.load_state_dict(checkpoint['model3'])
        SD3.Actor_net_Target2.load_state_dict(checkpoint['model4'])
        SD3.Critic_net_eval_1.load_state_dict(checkpoint['model5'])
        SD3.Critic_net_eval_2.load_state_dict(checkpoint['model6'])
        SD3.Critic_net_Target1.load_state_dict(checkpoint['model7'])
        SD3.Critic_net_Target2.load_state_dict(checkpoint['model8'])
        start_epoch = checkpoint['epoch']
        SD3.load_mem("./SD3_AE_logs/rpm.pickle")
        print(SD3.memory.shape[0])
        if SD3.memory.shape[0] < SD3.memory_size:
            mem_new = np.zeros((SD3.memory_size - SD3.memory.shape[0], SD3.state_dim * 2 + SD3.action_dim + 2),
                               dtype='float32')
            print(mem_new.shape[0])
            SD3.memory = np.vstack((SD3.memory, mem_new))
            print(SD3.memory.shape[0])
        # start_epoch = start_epoch + 1
        print('加载模型成功')
        print(SD3.memory_size)
    else:
        start_epoch = 0
        print('从0开始')
    for t in range(start_epoch + 1, int(max_step)):
        episode_step += 1
        if t < int(start_step):
            action = env.action_space.sample()
        else:
            action = SD3.choose_action(s)
            action = np.clip(action + np.random.normal(0, 0.2, size=action_dim), 0, 1)
        # print("hip_addt_angle {} hip_extent_angle {} "
        #       "knee_angle {} ankle_angle {} ground_force_reaction {}"
        #       .format(env.hip_addt_angle, env.hip_extent_angle, env.knee_angle, env.ankle_angle, env.ground_force_reaction))
        s_, r, done, _ = env.step(action)
        s_ = env.get_observation()
        d = float(done) if episode_step < env.time_limit else 0
        SD3.store_memory(s, action, r, s_, d)

        if env.index == 51:
            continue

        s = s_
        q_re += r

        if t > int(start_step):
            SD3.train()

        if done:
            print("time: {} episode: {} episode_step: {} reward: {}".format(t + 1, episode_num + 1, episode_step, q_re))
            # print("hip {} knee {} ankle {}".format(env.reward_hip_extension + env.reward_hip_abdunction, env.reward_knee, env.reward_ankle))
            sheets.write(row, 0, str(t + 1))
            sheets.write(row, 1, str(episode_num + 1))
            sheets.write(row, 2, str(episode_step))
            sheets.write(row, 3, str(q_re))
            sheets.col(0).width = 5000
            sheets.col(1).width = 6000
            sheets.col(2).width = 5000
            sheets.col(3).width = 6000
            # 保存数据到.xlsx文件
            book.save("data_SD3_AE.xls")
            if (episode_num + 1) % 10 == 0:
                SD3.save(t, log_dir)
                SD3.save_mem("./SD3_AE_logs/rpm.pickle")

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

            row = row + 1




if __name__ == "__main__":
    main()
