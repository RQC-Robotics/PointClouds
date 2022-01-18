import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.nn.modules.linear import Linear 

LOG_STD_MAX = 2
LOG_STD_MIN = -20




#TODO CRITIC TAM PIZDEC




class MLPEncoder(nn.Module):
    def __init__(self, state_dim, hid_size):
        super(MLPEncoder, self).__init__()
        self.state_dim = state_dim
        self.hid_size = hid_size

        self.net = nn.Sequential(
            nn.Linear(state_dim, hid_size), nn.ReLU(),
            nn.Linear(hid_size, hid_size), nn.ReLU(),
        )

    def forward(self, state):
        return self.net(state)


class ConvEncoder(nn.Module):
    def __init__(self, state_dim, num_filters, hid_size):
        super(ConvEncoder, self).__init__()
        self.state_dim = state_dim
        self.num_filters = num_filters
        self.hid_size = hid_size

        self.net = nn.Sequential(
            nn.Conv2d(state_dim[0], num_filters, 3, 2), nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1), nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1), nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 3, 1), nn.ReLU(),
            
        )


        x = torch.rand(((1, state_dim[0], state_dim[1], state_dim[2])))
        self.after_conv_size = self.enc(x).size()[1]  


        self.mlp = nn.Sequential(
            nn.Linear(self.after_conv_size, hid_size), nn.ReLU(),
            nn.Linear(hid_size, hid_size), nn.ReLU(),
        )


    def forward(self, image):
        image = image/255
        x = self.net(image)
        x.view(x.shape[0], -1)
        x = self.mlp(x)
        return x

    def conv(self, image):
        image = image/255
        x = self.net(image)
        return x.view(x.shape[0], -1)


class PointCloudEncoder(nn.Module):
    def __init__(self, state_dim, num_filters, hid_size):
        super(PointCloudEncoder, self).__init__()
        self.state_dim = state_dim
        self.num_filters = num_filters
        self.hid_size = hid_size

        self.net = nn.Sequential(
            nn.Conv1d(3, num_filters, 1), nn.BatchNorm1d(num_filters),
            nn.Conv1d(num_filters, 2*num_filters, 1), nn.BatchNorm1d(2*num_filters),
            nn.Conv1d(2*num_filters, hid_size, 1), nn.BatchNorm1d(hid_size),
        )

        self.after_conv_size = hid_size

        self.mlp = nn.Sequential(
            nn.Linear(hid_size, hid_size), nn.ReLU(),
        )
        

    def forward(self, state):
        x = self.net(state)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.hid_size)
        x = self.mlp(x)
        return x

    def conv(self, state):
        x = self.net(state)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.hid_size)
        return x










class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, hid_size, num_filters, enc_type):
        super(Actor, self).__init__()
        self.act_dim = act_dim
        self.hid_size = hid_size
        self.enc_type = enc_type

        if enc_type == 'Linear':
            self.enc = MLPEncoder(state_dim, hid_size)
        elif enc_type == 'Conv':
            if num_filters == None:
                raise "missed number of filters"
            self.enc = ConvEncoder(state_dim, num_filters, hid_size)
        elif enc_type == 'PointCloud':
            if num_filters == None:
                raise "missed number of filters"
            self.enc = PointCloudEncoder(state_dim, num_filters, hid_size)
        else:
            raise "ERROR: valid Encoder types: ['Linear', 'Conv', 'PointCloud']"


        self.mu = nn.Linear(hid_size, act_dim)
        self.log_std = nn.Linear(hid_size, act_dim)


    def forward(self, state, determ=False):
        x = self.enc(state)
        mu = self.mu(x)
        log_std = self.log_std(x)

        #Эвристика от OpenAI
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        std = torch.exp(log_std)

        distrib = Normal(mu, std)

        if determ:
            action = mu
        else:
            action = distrib.rsample()

        log_prob = distrib.log_prob(action).sum(axis=-1)

        #Эвристика от OpenAI
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=1)

        action = torch.tanh(action)

        return action, log_prob



class Critic(nn.Module):
    def __init__(self, state_dim, act_dim, hid_size, num_filters, enc_type):
        super(Critic, self).__init__()

        self.hid_size = hid_size
        self.act_dim = act_dim
        self.enc_type = enc_type

        if enc_type == 'Linear':
            # Для критика нам важно подавать не только скрытое представление, но и действия
            self.enc = MLPEncoder(state_dim + act_dim, hid_size)
            
        elif enc_type == 'Conv':
            if num_filters == None:
                raise "missed number of filters"
            # Переопределение mlp части энкодера, так как для критика нам важно подавать не только скрытое представление, но и действия 
            self.enc = ConvEncoder(state_dim, num_filters, hid_size)
            self.enc.mlp = nn.Sequential(
                nn.Linear(self.enc.after_conv_size + act_dim, hid_size), nn.ReLU(),
                nn.Linear(hid_size, hid_size), nn.ReLU(),
            )

        elif enc_type == 'PointCloud':
            if num_filters == None:
                raise "missed number of filters"

            self.enc = PointCloudEncoder(state_dim, num_filters, hid_size)
            self.enc.mlp = nn.Sequential(
                nn.Linear(self.enc.after_conv_size + act_dim, hid_size), nn.ReLU(),
            )

        else:
            raise "ERROR: valid Encoder types: ['Linear', 'Conv', 'PointCloud']"

        self.q = nn.Linear(hid_size, 1)

    def forward(self, state, action):
        if self.enc_type == 'Conv':
            state = self.enc.conv(state)
            Q = self.enc.mlp(torch.cat([state, action], dim=-1))
            Q = self.q(Q)

        elif self.enc_type == 'PointCloud':
            state = self.enc.conv(state)
            Q = self.enc.mlp(torch.cat([state, action], dim=-1))
            Q = self.q(Q)

        elif self.enc_type == 'Linear':
            Q = self.enc(torch.cat([state, action], dim=-1))
            Q = self.q(Q)

        return torch.squeeze(Q, dim=-1)



class SACAgent(nn.Module):
    def __init__(self, state_dim, act_dim, device, enc_type = 'Linear', hid_size=256, num_filters=None, alpha=0.2, polyak=0.995, lr=1e-3, batch_size=100, gamma=0.99):
        super(SACAgent, self).__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.device = device
        self.hid_size = hid_size
        self.num_filters = num_filters
        self.alpha = alpha
        self.polyak = polyak
        self.batch_size = batch_size
        self.gamma = gamma
        self.enc_type = enc_type


        self.policy = Actor(state_dim, act_dim, hid_size, num_filters, enc_type).to(device)
        self.Q1 = Critic(state_dim, act_dim, hid_size, num_filters, enc_type).to(device)
        self.Q2 = Critic(state_dim, act_dim, hid_size, num_filters, enc_type).to(device)

        self.policy_target = Actor(state_dim, act_dim, hid_size, num_filters, enc_type).to(device)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.Q1_target = Critic(state_dim, act_dim, hid_size, num_filters, enc_type).to(device)
        self.Q1_target.load_state_dict(self.Q1.state_dict())

        self.Q2_target = Critic(state_dim, act_dim, hid_size, num_filters, enc_type).to(device)
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.optim_Q1 = torch.optim.Adam(self.Q1.parameters(), lr=lr)
        self.optim_Q2 = torch.optim.Adam(self.Q2.parameters(), lr=lr)

    def _soft_update_net(self, net, net_target):
        for net_param, net_target_param in zip(net.parameters(), net_target.parameters()):
            net_target_param.data.copy_(net_param.data * (1 - self.polyak) + net_target_param.data * self.polyak)

    def compute_loss_Q(self, s, a, r, s_next, d):
        q1 = self.Q1(s, a)
        q2 = self.Q2(s, a)

        with torch.no_grad():
            a_next, log_prob_a_next = self.policy(s_next)
            q1_next_target = self.Q1_target(s_next, a_next)
            q2_next_target = self.Q2_target(s_next, a_next)
            q_next_target = torch.min(q1_next_target, q2_next_target)
            y = r + self.gamma * (1 - d) * (q_next_target - self.alpha * log_prob_a_next)

        loss_q1 = ((q1 - y)**2).mean()
        loss_q2 = ((q2 - y)**2).mean()

        return loss_q1, loss_q2
    

    def compute_loss_policy(self, s):
        a, log_prob_a = self.policy(s)
        q1 = self.Q1(s, a)
        q2 = self.Q2(s, a)

        q = torch.min(q1, q2)

        loss_policy = (self.alpha * log_prob_a - q).mean()  # ПОЧЕМУ НЕТ НИГДЕ DETACH()???
        return loss_policy

    def update(self, s, a, r, s_next, d):
        loss_q1, loss_q2 = self.compute_loss_Q(s, a, r, s_next, d)

        self.optim_Q1.zero_grad()
        loss_q1.backward()
        # МЕСТО ПОД GRAD_NORM
        self.optim_Q1.step()

        self.optim_Q2.zero_grad()
        loss_q2.backward()
        # МЕСТО ПОД GRAD_NORM
        self.optim_Q2.step()       
        
        for p1, p2 in zip(self.Q1.parameters(), self.Q2.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

        loss_policy = self.compute_loss_policy(s)
        
        self.optim_policy.zero_grad()
        loss_policy.backward()
        # МЕСТО ПОД GRAD_NORM
        self.optim_policy.step()

        for p1, p2 in zip(self.Q1.parameters(), self.Q2.parameters()):
            p1.requires_grad = True
            p2.requires_grad = True

        with torch.no_grad():
            self._soft_update_net(self.Q1, self.Q1_target)
            self._soft_update_net(self.Q2, self.Q2_target)
            self._soft_update_net(self.policy, self.policy_target)

        self.policy.zero_grad()
        self.Q1.zero_grad()
        self.Q2.zero_grad()
    def get_action(self, s, determ=False):
        with torch.no_grad():
            a, _ = self.policy(torch.as_tensor(s, dtype=torch.float32, device=self.device), determ)
        return a

    # def eval - НУ ВДРУГ НАДО 




# TODO - доделать logger



