import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Actor_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)
        self.fc4 = nn.Linear(64, action_dim)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.normal_(0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = torch.tanh(out)

        return out


class Critic_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)

        self.fc4 = nn.Linear(action_dim, 256)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.normal_(0.1)
        self.fc5 = nn.Linear(256, 128)
        self.fc5.weight.data.normal_(0, 0.1)
        self.fc5.bias.data.normal_(0.1)
        self.fc6 = nn.Linear(128, 64)
        self.fc6.weight.data.normal_(0, 0.1)
        self.fc6.bias.data.normal_(0.1)

        self.fc7 = nn.Linear(64, 1)
        self.fc7.weight.data.normal_(0, 0.1)
        self.fc7.bias.data.normal_(0.1)

    def forward(self, s, a):
        x = self.fc1(s)
        x = self.fc2(x)
        x = self.fc3(x)

        y = self.fc4(a)
        y = self.fc5(y)
        y = self.fc6(y)

        out = F.relu(x + y)
        out = self.fc7(out)

        return out



class TD3_agent(object):

    def __init__(self, state_counters, action_counters, device, batch_size, max_size, LR_A=0.001, LR_C=0.001,
                 gamma=0.99, TAU=0.005):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.device = device
        self.max_size = max_size
        self.batch_size = batch_size
        self.LR_A = LR_A
        self.LR_C = LR_C
        self.gamma = gamma
        self.TAU = TAU

        self.index_memory = 0
        self.memory = np.zeros((self.max_size, self.state_counters * 2 + self.action_counters + 2))
        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net(self.state_counters, self.action_counters).to(self.device), \
                                                     Actor_Net(self.state_counters, self.action_counters).to(self.device)

        self.Critic_Net_eval_1, self.Critic_Net_target_1 = Critic_Net(self.state_counters, self.action_counters).to(self.device), \
                                                       Critic_Net(self.state_counters, self.action_counters).to(self.device)

        self.Critic_Net_eval_2, self.Critic_Net_target_2 = Critic_Net(self.state_counters, self.action_counters).to(self.device), \
                                                           Critic_Net(self.state_counters, self.action_counters).to(self.device)

        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=self.LR_A)


        self.optimizer_C_1 = torch.optim.Adam(self.Critic_Net_eval_1.parameters(), lr=self.LR_C)
        self.optimizer_C_2 = torch.optim.Adam(self.Critic_Net_eval_2.parameters(), lr=self.LR_C)

        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)
        observation = torch.unsqueeze(observation, 0).to(self.device)
        action = self.Actor_Net_eval(observation)[0].detach()


        return action.cpu().data.numpy()

    def store_memory(self, s, a, r, s_, d):

        memory = np.hstack((s, a, [r], s_, [d]))
        index = self.index_memory % self.max_size
        self.memory[index, :] = memory
        self.index_memory += 1
        self.index_memory = min(self.index_memory, self.max_size)

    def learn(self, step):


        sample_memory_index = np.random.randint(0, self.index_memory, size=self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters].to(self.device)
        sample_memory_s_ = sample_memory[:, -self.state_counters - 1: -1].to(self.device)
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters].to(self.device)
        sample_memory_r = sample_memory[:, - self.state_counters - 2: - self.state_counters - 1].to(self.device)
        sample_memory_d = sample_memory[:, -1].reshape(256, 1).to(self.device)

        with torch.no_grad():

            # target_actor采取的动作
            a = self.Actor_Net_target(sample_memory_s_)
            # 在动作中加入噪声
            a_ = torch.clamp(a + torch.clamp(torch.randn(a.shape[0], a.shape[1], dtype=a.dtype,
                                layout=a.layout, device=a.device), -0.5, 0.5), -2, 2)
            # eval_actor采取的动作
            a_s = self.Actor_Net_eval(sample_memory_s)

            # 取target网络中较小的值，防止过估计
            q1 = self.Critic_Net_target_1(sample_memory_s_, a_)
            q2 = self.Critic_Net_target_2(sample_memory_s_, a_)

            min_q = torch.min(q1, q2)
            q_target = sample_memory_r + (1 - sample_memory_d) * self.gamma * min_q

        a_s = self.Actor_Net_eval(sample_memory_s)
        q_eval_1 = self.Critic_Net_eval_1(sample_memory_s, sample_memory_a)
        q_eval_2 = self.Critic_Net_eval_2(sample_memory_s, sample_memory_a)
        loss_c_2 = self.loss(q_target, q_eval_2)
        self.optimizer_C_2.zero_grad()
        loss_c_2.backward()
        self.optimizer_C_2.step()

        loss_c_1 = self.loss(q_target, q_eval_1)
        self.optimizer_C_1.zero_grad()
        loss_c_1.backward()
        self.optimizer_C_1.step()

        if step % 2 == 0:
            loss_a = - torch.mean(self.Critic_Net_eval_1(sample_memory_s, a_s))
            self.optimizer_A.zero_grad()
            loss_a.backward()
            self.optimizer_A.step()


            for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Critic_Net_eval_1.parameters(), self.Critic_Net_target_1.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Critic_Net_eval_2.parameters(), self.Critic_Net_target_2.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)


    def save(self, epoch, log_dir):

        state = {'model1': self.Actor_Net_eval.state_dict(),
                 'model2': self.Actor_Net_target.state_dict(),
                 'model3': self.Critic_Net_eval_1.state_dict(),
                 'model4': self.Critic_Net_eval_2.state_dict(),
                 'model5': self.Critic_Net_target_1.state_dict(),
                 'model6': self.Critic_Net_target_2.state_dict(),
                 'epoch': epoch}
        # torch.save(state, './model/saved_model'+str(epoch)+'.pth')
        torch.save(state, log_dir)



