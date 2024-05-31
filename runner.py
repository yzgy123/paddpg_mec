from agent import Agent
from common.replay_buffer import Buffer
import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.env = env
        self.agent = self._init_agent()
        self.buffer = Buffer(args)

    def _init_agent(self):
        agent = Agent(self.args)
        return agent

    def run(self):
        #首先给buffer填充一个batch_size的经验
        o = self.env.reset()
        for i in range(self.args.batch_size):
            if(i%self.args.time_steps==0):
                    o = self.env.reset()
            with torch.no_grad():
                action = self.agent.select_action(o, self.epsilon,False)
                o_next, r= self.env.step(o,action,False)
                self.buffer.store_episode(o, action, r, o_next)
                o=o_next
        #开始训练
        returns = []
        for episode in range(self.args.episodes):            
            # reset the environment
            o = self.env.reset(episode)
            for time_step in range(self.args.time_steps):    
                #与环境交互，获得奖励，下一个状态值
                with torch.no_grad():
                    action = self.agent.select_action(o, self.epsilon,False)
                o_next, r= self.env.step(o,action,False)
                self.buffer.store_episode(o, action, r, o_next)
                o = o_next
                #在buff中进行采样
                transitions = self.buffer.sample(self.args.batch_size)
                self.agent.learn(transitions)    
            returns.append(self.evaluate(episode))
            self.epsilon = max(0.005, self.epsilon - 0.001)
        sio.savemat('rewards.mat', {'lr': returns})
        # plt.plot(range(0,len(returns)),returns)
        # plt.show()

    def evaluate(self,current_episode):
        # o=o_pre
        # o[:20]=torch.zeros(20)
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            o = self.env.reset()
            # o=o_next
            reward = 0
            for time_step in range(self.args.evaluate_time_steps):
                with torch.no_grad():
                    action = self.agent.select_action(o, 0,True)
                o_next, r= self.env.step(o,action,True)
                reward += r
                o = o_next
        # print(actions)
        print('episode:'+str(current_episode)+',rewards:'+str(reward/ (self.args.evaluate_episodes*self.args.evaluate_time_steps)))
        return reward / (self.args.evaluate_episodes*self.args.evaluate_time_steps)
