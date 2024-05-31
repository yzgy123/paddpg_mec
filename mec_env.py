import numpy as np
from scipy.stats import zipfian
class mec_env:
    def __init__(self):
        # np.random.seed(1)
        #环境中service的数量
        self.service_num=10
        #每个edge_server的内存容量G
        self.total_capacity=20
        #每个edge_server的CPU容量GHz
        self.total_cycle=10
        #t-1时刻服务配置+当前时刻每种类型任务到达数量+边缘服务器之间的带宽+到云服务器的带宽
        self.o_shape=self.service_num*2+1
        #t时各agent动作
        self.action_shape=self.service_num*2
        #记录环境当前状态
        self.o=np.zeros(self.o_shape)
        #每个节点不同任务类型在单位时间内到达的数量
        self.arr_task_num=np.zeros(self.service_num)
        #边缘节点与云节点之间带宽MHz
        self.b_cloud=50
        #服务l所需的内存资源G
        self.service_m=2+np.random.rand(self.service_num)
        #服务l所需的CPU资源 GHz
        self.service_c=0.05+0.05*np.random.rand(self.service_num)
        #服务l的平均任务大小 MB
        self.arr_task_size=50+50*np.random.rand(self.service_num)
        #服务类型为l的任务传输至其他边缘服务器的延迟
        self.task_cloud_delay=np.zeros(self.service_num)
        #服务l的startup time
        self.startup=np.random.rand(self.service_num)*0.3+0.2
        #系统时隙间隔
        self.delta_t=1

    def get_task_num(self):
        total_tasks=200
        zipf_para, service_num = 2, self.service_num
        x = np.arange(1, service_num+1)
        y = zipfian.pmf(x, zipf_para, service_num)*total_tasks
        maxy=np.around(y*1.25)
        miny=np.around(y*0.75)
        task_num=np.zeros(self.service_num)
        for i in range (self.service_num):
            task_num[i]= np.random.randint(miny[i],maxy[i])
        return task_num

    def get_network_conditions(self):
        #控制任务类型l的生成数量
        self.arr_task_num=self.get_task_num()   
        #同时更新传输延迟  
        self.task_cloud_delay=self.arr_task_size/self.b_cloud

    def reset(self,episode=0):
        # if episode==10:
        #     self.b_cloud=40
        self.get_network_conditions()
        self.o=np.zeros(self.o_shape)
        self.o[self.service_num:self.service_num*2]=self.arr_task_num
        self.o[self.service_num*2:]=self.b_cloud
        o=np.copy(self.o)
        return o
    
    def step(self,pre_o,action,is_evaluate):
        pre_o=np.copy(pre_o)
        action_b=np.copy(action[:self.service_num])
        action_b_real=np.zeros(self.service_num)
        action_f=np.copy(action[self.service_num:])
        #按action_b的从大到小值部署service直到RAM不够
        order=np.flip(np.argsort(action_b))
        RAM_capacity=0
        for index in range (self.service_num):
            if(RAM_capacity+self.service_m[order[index]]<=self.total_capacity):
                action_b_real[order[index]]=1
                RAM_capacity=RAM_capacity+self.service_m[order[index]]
            else:
                break
        action_f=np.multiply(action_b_real,action_f)
        action_f=action_f/np.sum(action_f)
        #某个agent由于所有服务部署变量都为0，可能会导致分母为0，将nan转换为0
        action_f[np.isnan(action_f)]=0
        service_cpu_allocation=action_f*self.total_cycle*self.delta_t
        #时隙内可处理任务类型为l的数量
        can_process_task_num=np.floor(service_cpu_allocation/self.service_c)
        #本地队列长度
        local_queue_len=np.zeros(self.service_num)
        for i in range (self.service_num):
            local_queue_len[i]=np.min([can_process_task_num[i],self.arr_task_num[i]])
        #转发至cloud的任务数量
        cloud_process_num=self.arr_task_num-local_queue_len
        
        #计算每个agent任务l的平均逗留时间
        avg_task_delay=1/(can_process_task_num-local_queue_len+1) #防止出现分母为0
        #计算联合动作所获取的奖励
        # if(is_evaluate):
        #     a=1

        reward=0
        startup_time=0
        for service_j in range(self.service_num):
            if(action_b_real[service_j]-pre_o[service_j]==1):
                startup_time=self.startup[service_j]+startup_time

        reward=np.sum(avg_task_delay*local_queue_len+cloud_process_num*self.task_cloud_delay)+startup_time

        #更新网络环境
        self.get_network_conditions()

        self.o[0:self.service_num]=action_b_real
        self.o[self.service_num:self.service_num*2]=self.arr_task_num
        self.o[self.service_num*2:]=self.b_cloud
        next_o=np.copy(self.o)
        if(is_evaluate):
            x=3
        return next_o, -reward

# mec=mec_env()
# # # mec.reset()
# pre_o=mec.s
# actions=np.random.rand(mec.edge_num,mec.action_shape)
# #print(mec.reset())
# mec.step(pre_o,actions)