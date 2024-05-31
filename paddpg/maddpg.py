import torch
import os
from paddpg.actor_critic import Actor, Critic
import torch.nn as nn
import numpy as np


class MADDPG:
    def __init__(self, args): 
        self.args = args
        self.train_step = 0
        self.train_episodes = 0

        # create the network
        self.actor_network = Actor(args)
        self.critic_network = Critic(args)

        # build up the target network
        self.actor_target_network = Actor(args)
        self.critic_target_network = Critic(args)

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir

        if os.path.exists(self.model_path + '/'+str(self.args.service_num)+'_'+str(self.args.total_capacity)+'_'+str(self.args.total_cycle)+'_actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/'+str(self.args.service_num)+'_'+str(self.args.total_capacity)+'_'+str(self.args.total_cycle)+'_actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/'+str(self.args.service_num)+'_'+str(self.args.total_capacity)+'_'+str(self.args.total_cycle)+'_critic_params.pkl'))
            print('Successfully loaded actor_network: {}'.format(self.model_path + '/'+str(self.args.service_num)+'_'+str(self.args.total_capacity)+'_'+str(self.args.total_cycle)+'_actor_params.pkl'))
            print('Successfully loaded critic_network: {}'.format(self.model_path + '/'+str(self.args.service_num)+'_'+str(self.args.total_capacity)+'_'+str(self.args.total_cycle)+'_critic_params.pkl'))

    # soft update the target networks
    def _soft_update_target_networks(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    

    # update the network
    def train(self, transitions):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        # 用来装经验中的各项
        o=transitions['o']
        action=transitions['action']
        r=transitions['r']
        o_next=transitions['o_next']

        # calculate the target Q value function
        with torch.no_grad():
            # 得到下一个状态对应的动作
            action_next=self.actor_target_network(o_next)
            q_next = self.critic_target_network(o_next, action_next).detach()
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # the q loss
        q_value = self.critic_network(o, action)
        critic_loss = (target_q - q_value).pow(2).mean()
        # print(critic_loss.data)
        # the actor loss
        u = self.actor_network(o)
        actor_loss = -torch.mean(self.critic_network(o, u))
        # print(actor_loss.data)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=0.5, norm_type=2)
        # for param in self.actor_network.parameters():
        #    print(np.array(param.grad[1]))
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=0.5, norm_type=2)
        self.critic_optim.step()

        self._soft_update_target_networks()
        #保存模型
        self.train_episodes += 1
        if self.train_episodes==(self.args.episodes*self.args.time_steps):
            self.save_model()

    def save_model(self):
        torch.save(self.actor_network.state_dict(), self.model_path + '/200_'+str(self.args.service_num)+'_'+str(self.args.total_capacity)+'_'+str(self.args.total_cycle)+'_50_2_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), self.model_path + '/200_'+str(self.args.service_num)+'_'+str(self.args.total_capacity)+'_'+str(self.args.total_cycle)+'_50_2_critic_params.pkl')


