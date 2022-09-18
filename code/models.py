import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Actor(nn.Module):
    def __init__(self, state_dimension, action_dimension, action_max):
        super(Actor, self).__init__()

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.action_max = action_max

        # self.fc1 = nn.Linear(state_dimension, 256)
        # self.fc1.weight.data.normal_(0, 0.1)
        # self.fc1.bias.data.normal_(0.1)

        self.fc2 = nn.Linear(state_dimension, 128)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)
        self.fc4 = nn.Linear(64, action_dimension)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc4.bias.data.normal_(0.1)

    def forward(self, state):

        # output = self.fc1(state)
        # # print(output.shape[0])
        # self.ln1 = nn.LayerNorm(output.shape[-1], elementwise_affine=False)
        # ln1_out = self.ln1(output)
        # output = F.relu(ln1_out)

        output = self.fc2(state)
        self.ln2 = nn.LayerNorm(output.shape[-1], elementwise_affine=False)
        ln2_out = self.ln2(output)
        output = F.relu(ln2_out)

        output = self.fc3(output)
        self.ln3 = nn.LayerNorm(output.shape[-1], elementwise_affine=False)
        ln3_out = self.ln3(output)
        output = F.relu(ln3_out)

        action = self.fc4(output)
        action = F.tanh(action)

        # action = action * self.action_max

        return action


class Critic(nn.Module):
    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension

        self.fcs1 = nn.Linear(state_dimension, 128)
        self.fcs1.weight.data.normal_(0, 0.1)
        self.fcs1.bias.data.normal_(0.1)
        self.fcs2 = nn.Linear(128, 64)
        self.fcs2.weight.data.normal_(0, 0.1)
        self.fcs2.bias.data.normal_(0.1)

        self.fca1 = nn.Linear(action_dimension, 128)
        self.fca1.weight.data.normal_(0, 0.1)
        self.fca1.bias.data.normal_(0.1)
        self.fca2 = nn.Linear(128, 64)
        self.fca2.weight.data.normal_(0, 0.1)
        self.fca2.bias.data.normal_(0.1)

        self.fc2 = nn.Linear(64, 1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)

    def forward(self, state, action):
        s1 = self.fcs1(state)
        # s1 = F.relu(s1)
        s2 = self.fcs2(s1)
        # s2 = F.relu(s2)

        a1 = self.fca1(action)
        # a1 = F.relu(a1)
        a2 = self.fca2(a1)

        # output = torch.cat((s2, a2), dim=1)
        output = F.relu(s2 + a2)
        output = self.fc2(output)
        # output = F.relu(output)
        # q_value = self.fc3(output)

        return output
