import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(64, action_dim)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = torch.tanh(out)

        return out


class Critic_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(action_dim, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)

    def forward(self, s, a):
        x = self.fc1(s)
        y = self.fc2(a)

        out = F.relu(x + y)
        out = self.fc3(out)

        return out


class DDPG_agent(object):

    def __init__(self, state_counters, action_counters, batch_size, memory_size, LR_A=0.001, LR_C=0.001,
                 gamma=0.99, TAU=0.005):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.index_memory = 0
        self.TAU = TAU
        self.gamma = gamma
        # self.LR_A = LR_A
        # self.LR_C = LR_C
        self.memory = np.zeros((self.memory_size, self.state_counters * 2 + self.action_counters + 2))

        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net(self.state_counters, self.action_counters).to(device), \
                                                     Actor_Net(self.state_counters, self.action_counters).to(device)

        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters).to(device), \
                                                       Critic_Net(self.state_counters, self.action_counters).to(device)

        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)

        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)

        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation).to(device)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()

        return action.cpu().data.numpy()

    def store_memory(self, s, a, r, s_, d):

        memory = np.hstack((s, a, [r], s_, [d]))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    def learn(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)

        sample_memory_s = sample_memory[:, : self.state_counters].to(device)
        sample_memory_s_ = sample_memory[:, - self.state_counters:].to(device)
        sample_memory_a = sample_memory[:, self.state_counters: self.state_counters + self.action_counters].to(device)
        sample_memory_r = sample_memory[:, -self.state_counters - 1:- self.state_counters].to(device)

        a = self.Actor_Net_target(sample_memory_s_)
        a_s = self.Actor_Net_eval(sample_memory_s)
        q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
        q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
        loss_c = self.loss(q_target, q_eval)
        self.optimizer_C.zero_grad()
        loss_c.backward()
        self.optimizer_C.step()

        loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
        self.optimizer_A.zero_grad()
        loss_a.backward()
        self.optimizer_A.step()

        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)

    def save(self, epoch, log_dir):

        state = {'model1': self.Actor_Net_eval.state_dict(),
                 'model2': self.Actor_Net_target.state_dict(),
                 'model3': self.Critic_Net_eval.state_dict(),
                 'model4': self.Critic_Net_target.state_dict(),
                 'epoch': epoch}
        # torch.save(state, './model/saved_model'+str(epoch)+'.pth')
        torch.save(state, log_dir)

    def save_mem(self, mem_dir):
        pickle.dump([self.memory, self.index_memory], open(mem_dir, 'wb'), protocol=4)
        print('memory dump into', mem_dir)

    def load_mem(self, mem_dir):
        [self.memory, self.index_memory] = pickle.load(open(mem_dir, 'rb'))
        print('self.index_memory', self.index_memory)
        print('memory loaded from', mem_dir)
