import numpy as np
import torch
import torch.nn.functional as F
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, args):
        self.args = args
        self.policy = MADDPG(args)

    def select_action(self, o, epsilon, is_evaluate):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        action = self.policy.actor_network(inputs).squeeze(0).numpy()
        if(is_evaluate):
            return  action  
        else:
            if np.random.uniform() > epsilon:
                return action
            else:
                u1= np.random.uniform(self.args.b_low_action, self.args.b_high_action, int(self.args.action_shape/2))
                u2 = np.random.uniform(self.args.f_low_action, self.args.f_high_action, int(self.args.action_shape/2))
                # u2=F.softmax(torch.tensor(u2, dtype=torch.float32),dim=0).numpy()
                # noise = noise_rate  * np.random.randn(*u1.shape)  # gaussian noise
                # u2 += noise
                # u1 = np.clip(u1, self.args.low_action, self.args.high_action)
                # u2 = np.clip(u2, self.args.low_action+1, self.args.high_action+1)
                return np.concatenate((u1,u2))

    def learn(self, transitions):
        self.policy.train(transitions)

