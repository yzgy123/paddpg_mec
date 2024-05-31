import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,args):
        super(Actor, self).__init__()
        self.actor_encoder=nn.Identity()
        #输出b动作
        self.fc1_b = nn.Linear(args.obs_shape, 256)
        self.fc2_b = nn.Linear(256, 512)
        self.fc3_b = nn.Linear(512, 256)
        self.fc4_b = nn.Linear(256, 256)
        self.fc5_b = nn.Linear(256, 64)
        self.action_out_b= nn.Linear(64, int(args.action_shape/2))
        #输出f动作
        # self.fc1_f = nn.Linear(args.obs_shape, 128)
        # self.fc2_f = nn.Linear(128, 256)
        # self.fc3_f = nn.Linear(256, 256)
        # self.fc4_f = nn.Linear(256, 64)
        self.action_out_f= nn.Linear(64, int(args.action_shape/2))


    def forward(self, obs):
        # obs=F.normalize(obs,p=2,dim=-1)
        #b动作取值范围
        obs = self.actor_encoder(obs)
        x = F.relu(self.fc1_b(obs))
        x = F.relu(self.fc2_b(x))
        x = F.relu(self.fc3_b(x))
        x = F.relu(self.fc4_b(x))
        x = F.relu(self.fc5_b(x))
        action_b = self.action_out_b(x)
        #f动作取值范围[0,2]
        # x = F.relu(self.fc1_f(obs))
        # x = F.relu(self.fc2_f(x))
        # x = F.relu(self.fc3_f(x))
        # x = F.relu(self.fc4_f(x))
        action_f = F.relu(self.action_out_f(x))
        return torch.cat([action_b, action_f],dim=1)


class Critic(nn.Module):    
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.obs_shape + args.action_shape, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        # x=F.normalize(x,p=2,dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        q_value = self.q_out(x)
        return q_value
