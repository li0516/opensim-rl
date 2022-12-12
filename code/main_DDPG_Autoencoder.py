import torch
import numpy as np
import DDPG_Agent
import matplotlib.pyplot as plt
from osim.env import L2M2019Env
import xlwt
import os

env = L2M2019Env(visualize=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_max = env.action_space.high[0]
max_step = 3e6
start_step = 10000
mem_size = 10000


def main():
    book = xlwt.Workbook()
    sheets = book.add_sheet('sheet1')

    torch.manual_seed(0)
    np.random.seed(0)

    env.set_difficulty(2)

    episode_step = 0

    DDPG = DDPG_Agent.DDPG_agent(state_counters=state_dim,
                                 action_counters=action_dim,
                                 memory_size=mem_size,
                                 batch_size=256,
                                 LR_A=0.01,
                                 LR_C=0.01,
                                 gamma=0.99,
                                 TAU=0.01)

    row = 0
    log_dir = './model_ddpg/saved_model.pth'
    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        DDPG.Actor_Net_eval.load_state_dict(checkpoint['model1'])
        DDPG.Actor_Net_target.load_state_dict(checkpoint['model2'])
        DDPG.Critic_Net_eval.load_state_dict(checkpoint['model3'])
        DDPG.Critic_Net_target.load_state_dict(checkpoint['model4'])
        start_epoch = checkpoint['epoch']
        DDPG.load_mem("./DDPG_logs/rpm.pickle")
        print(DDPG.memory.shape[0])
        if DDPG.memory.shape[0] < DDPG.memory_size:
            mem_new = np.zeros((DDPG.memory_size - DDPG.memory.shape[0], state_dim * 2 + action_dim + 2))
            print(mem_new.shape[0])
            DDPG.memory = np.vstack((DDPG.memory, mem_new))
            print(DDPG.memory.shape[0])
        # start_epoch = start_epoch + 1
        print('加载模型成功')
        print(DDPG.memory_size)
    else:
        start_epoch = 0
        print('从0开始')
    for epoch in range(start_epoch, 20000):
        s = env.reset()
        s = env.get_observation()
        ep_reward = 0
        for t in range(1000):
            if epoch > 5:
                action = DDPG.choose_action(s)
                action = np.clip(action + np.random.normal(0, 0.2, size=action_dim), 0, 1)
            else:
                action = env.action_space.sample()

            s_, r, done, _ = env.step(action)
            s_ = env.get_observation()
            done = float(done) if episode_step < env.time_limit else 0
            DDPG.store_memory(s, action, r, s_, done)

            if DDPG.index_memory > mem_size:
                # print('start learn')
                DDPG.learn()

            s = s_
            ep_reward += r

            if t == 999 or done:
                # print('一个情节结束')
                print('epoch', epoch + 1, 'time_step', t + 1, 'reward', ep_reward)

                sheets.write(row, 0, str(epoch + 1))
                sheets.write(row, 1, str(t + 1))
                sheets.write(row, 2, str(ep_reward))
                sheets.col(0).width = 5000
                sheets.col(1).width = 6000
                sheets.col(2).width = 5000
                # 保存数据到.xlsx文件
                book.save("data_ddpg.xls")
                if (epoch + 1) % 10 ==0:
                    DDPG.save(epoch + 1, log_dir)
                    DDPG.save_mem("./DDPG_logs/rpm.pickle")

                row = row + 1
                break


if __name__ == '__main__':
    main()
