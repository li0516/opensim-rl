import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 100),
            nn.Sigmoid()
        )

        self.fc2 = nn.Linear(100, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(64, 32)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)
        self.fc4 = nn.Linear(32, action_dim)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.normal_(0.1)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        out = F.relu(self.fc2(decoded))
        out = F.relu(self.fc3(out))
        out = torch.tanh(self.fc4(out))

        return out


class Critic_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc4 = nn.Linear(256, 128)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.normal_(0.1)
        self.fc5 = nn.Linear(128, 64)
        self.fc5.weight.data.normal_(0, 0.1)
        self.fc5.bias.data.normal_(0.1)

        self.fc2 = nn.Linear(action_dim, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc6 = nn.Linear(256, 128)
        self.fc6.weight.data.normal_(0, 0.1)
        self.fc6.bias.data.normal_(0.1)
        self.fc7 = nn.Linear(128, 64)
        self.fc7.weight.data.normal_(0, 0.1)
        self.fc7.bias.data.normal_(0.1)

        self.fc3 = nn.Linear(64, 1)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)

    def forward(self, s, a):
        x = self.fc1(s)
        x = self.fc4(x)
        x = self.fc5(x)

        y = self.fc2(a)
        y = self.fc6(y)
        y = self.fc7(y)

        out = F.relu(x + y)
        out = self.fc3(out)

        return out


class SD3_Agent(object):

    def __init__(self, state_dim, action_dim, batch_size, LR_A, LR_C, memory_size, gamma, TAU,
                 policy_noise, sample_size, noise_clip, beta, important_sampling, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.TAU = TAU
        self.policy_noise = policy_noise
        self.sample_size = sample_size
        self.noise_clip = noise_clip
        self.beta = beta
        self.important_sampling = important_sampling
        self.memory = np.zeros((self.memory_size, self.state_dim * 2 + self.action_dim + 2), dtype='float32')
        self.index_memory = 0
        self.device = device

        # 建立actor_network(2个eval, 2个target)
        self.Actor_net_eval_1 = Actor_Net(self.state_dim, self.action_dim).to(device)
        self.Actor_net_eval_2 = Actor_Net(self.state_dim, self.action_dim).to(device)
        self.Actor_net_Target1, self.Actor_net_Target2 = Actor_Net(self.state_dim, self.action_dim).to(device), \
                                                         Actor_Net(self.state_dim, self.action_dim).to(device)

        # 建立critic_network(2个eval, 2个target)
        self.Critic_net_eval_1 = Critic_Net(self.state_dim, self.action_dim).to(device)
        self.Critic_net_eval_2 = Critic_Net(self.state_dim, self.action_dim).to(device)
        self.Critic_net_Target1, self.Critic_net_Target2 = Critic_Net(self.state_dim, self.action_dim).to(device), \
                                                           Critic_Net(self.state_dim, self.action_dim).to(device)

        # 建立optimizer
        self.optimizer_A_1 = torch.optim.Adam(self.Actor_net_eval_1.parameters(), lr=LR_A)
        self.optimizer_A_2 = torch.optim.Adam(self.Actor_net_eval_2.parameters(), lr=LR_A)

        self.optimizer_C_1 = torch.optim.Adam(self.Critic_net_eval_1.parameters(), lr=LR_C)
        self.optimizer_C_2 = torch.optim.Adam(self.Critic_net_eval_2.parameters(), lr=LR_C)

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation).reshape(1, -1).to(self.device)
        a_1 = self.Actor_net_eval_1(observation)[0].detach()
        a_2 = self.Actor_net_eval_2(observation)[0].detach()

        q_1 = self.Critic_net_eval_1(observation, a_1)
        q_2 = self.Critic_net_eval_2(observation, a_2)
        action = a_1 if q_1 >= q_2 else a_2

        return action.cpu().data.numpy()

    def store_memory(self, s, a, r, s_, d):

        memory = np.hstack((s, a, [r], s_, [d]))
        index = self.index_memory % self.memory_size
        # print('index', index)
        self.memory[index, :] = memory
        self.index_memory += 1
        self.index_memory = min(self.index_memory, self.memory_size)

    def sampling_memory(self):

        # 在经验池里随机采样
        sample_memory_index = np.random.choice(self.index_memory, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, :self.state_dim]
        sample_memory_a = sample_memory[:, self.state_dim:self.state_dim + self.action_dim]
        sample_memory_r = sample_memory[:, self.state_dim + self.action_dim]
        sample_memory_r = torch.unsqueeze(sample_memory_r, 1)
        sample_memory_s_ = sample_memory[:, - self.state_dim - 1: -1]
        sample_memory_d = sample_memory[:, -1]
        sample_memory_d = torch.unsqueeze(sample_memory_d, 1)

        return sample_memory_s, sample_memory_a, sample_memory_r, sample_memory_s_, sample_memory_d

    def softmax_operator(self, q_eval, prob_noise=None):

        q_max = torch.max(q_eval, dim=1, keepdim=True).values
        q_norm = q_eval - q_max
        q = torch.exp(self.beta * q_norm)
        q_s_a = q * q_eval

        if self.important_sampling:
            q_s_a /= prob_noise
            q /= prob_noise

        softmax_q = torch.sum(q_s_a, 1) / torch.sum(q, 1)

        return softmax_q

    def Gaussian_distribution(self, sample, expectation=0):

        prob = (1 / (self.policy_noise * torch.sqrt(2 * np.pi))) * \
               torch.exp(- (sample - expectation) ** 2 / (2 * self.policy_noise ** 2))

        prob = torch.prod(prob, dim=2)

        return prob

    def train(self):

        self.learn(update_critic1=True)
        self.learn(update_critic1=False)

    def learn(self, update_critic1):

        state, action, r, state_, d = self.sampling_memory()

        state = state.to(self.device)
        action = action.to(self.device)
        r = r.to(self.device)
        state_ = state_.to(self.device)
        d = d.to(self.device)

        with torch.no_grad():
            if update_critic1:
                a = self.Actor_net_Target1(state_)
            else:
                a = self.Actor_net_Target2(state_)

            noise = torch.randn((a.shape[0], self.sample_size, a.shape[1]), dtype=a.dtype,
                                layout=a.layout, device=a.device)
            noise = noise * self.policy_noise
            if self.important_sampling:
                prob_noise = self.Gaussian_distribution(noise, expectation=0)
            else:
                prob_noise = None
            # noise = self.Gaussian_distribution(noise)
            noise = torch.clamp(noise, - self.noise_clip, self.noise_clip)

            a = torch.unsqueeze(a.cuda(), 1)
            a_ = torch.clamp(a + noise.cuda(), 0, 1)

            a_ = torch.squeeze(a_, 1)
            state_ = torch.unsqueeze(state_, 1)
            state_ = state_.repeat((1, self.sample_size, 1))
            min_q = torch.min(self.Critic_net_Target1(state_, a_), self.Critic_net_Target2(state_, a_))
            # 计算softmax的q值
            softmax_q_s_a = self.softmax_operator(min_q, prob_noise)
            # 计算target的值
            target_q = r + self.gamma * (1 - d) * softmax_q_s_a
        if update_critic1:
            # 计算critic1的q值
            q_ = self.Critic_net_eval_1(state, action)
            # 梯度下降
            loss = torch.nn.functional.mse_loss(target_q, q_)
            self.optimizer_C_1.zero_grad()
            loss.backward()
            self.optimizer_C_1.step()

            # 更新Actor_network
            loss_a = - torch.mean(self.Critic_net_eval_1(state, self.Actor_net_eval_1(state)))
            self.optimizer_A_1.zero_grad()
            loss_a.backward()
            self.optimizer_A_1.step()

            for parm, target_parm in zip(self.Critic_net_eval_1.parameters(), self.Critic_net_Target1.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Actor_net_eval_1.parameters(), self.Actor_net_Target1.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        else:
            # 计算critic2的q值
            q_ = self.Critic_net_eval_2(state, action)
            # 梯度下降
            loss = torch.nn.functional.mse_loss(target_q, q_)
            self.optimizer_C_2.zero_grad()
            loss.backward()
            self.optimizer_C_2.step()

            # 更新Actor_network
            loss_a = - torch.mean(self.Critic_net_eval_2(state, self.Actor_net_eval_2(state)))
            self.optimizer_A_2.zero_grad()
            loss_a.backward()
            self.optimizer_A_2.step()

            for parm, target_parm in zip(self.Critic_net_eval_2.parameters(), self.Critic_net_Target2.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
            for parm, target_parm in zip(self.Actor_net_eval_2.parameters(), self.Actor_net_Target2.parameters()):
                target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)

    def save(self, epoch, log_dir):
        state = {'model1': self.Actor_net_eval_1.state_dict(),
                 'model2': self.Actor_net_eval_2.state_dict(),
                 'model3': self.Actor_net_Target1.state_dict(),
                 'model4': self.Actor_net_Target2.state_dict(),
                 'model5': self.Critic_net_eval_1.state_dict(),
                 'model6': self.Critic_net_eval_2.state_dict(),
                 'model7': self.Critic_net_Target1.state_dict(),
                 'model8': self.Critic_net_Target2.state_dict(),
                 'epoch': epoch}
        # torch.save(state, './model/saved_model'+str(epoch)+'.pth')
        torch.save(state, log_dir)

    def load(self, epoch, log_dir):

        checkpoint = torch.load(log_dir)
        self.Actor_net_eval_1.load_state_dict(checkpoint['model1'])
        self.Actor_net_eval_2.load_state_dict(checkpoint['model2'])
        self.Actor_net_Target1.load_state_dict(checkpoint['model3'])
        self.Actor_net_Target2.load_state_dict(checkpoint['model4'])
        self.Critic_net_eval_1.load_state_dict(checkpoint['model5'])
        self.Critic_net_eval_2.load_state_dict(checkpoint['model6'])
        self.Critic_net_Target1.load_state_dict(checkpoint['model7'])
        self.Critic_net_Target2.load_state_dict(checkpoint['model8'])
        epoch.load_state_dict(checkpoint['epoch'])

    def save_mem(self, mem_dir):
        pickle.dump([self.memory, self.index_memory], open(mem_dir, 'wb'), protocol=4)
        print('memory dump into', mem_dir)

    def load_mem(self, mem_dir):
        [self.memory, self.index_memory] = pickle.load(open(mem_dir, 'rb'))
        print('self.index_memory', self.index_memory)
        print('memory loaded from', mem_dir)
