import numpy as np
from mec_env2 import mec_env
def comput_latency(service_num,total_capacity,CPU_frequency,task_num,bandwidth,zipf):
    env=mec_env(service_num,total_capacity,CPU_frequency,task_num,bandwidth,zipf)
    o=env.reset()
    reward=0
    for k in range(20):
        total_RAM=0
        service_RAM=np.copy(env.service_m)
        arr_task_num=env.arr_task_num
        order=np.flip(np.argsort(arr_task_num))
        action_b=np.zeros(env.service_num)
        action_f=np.zeros(env.service_num)
        for i in range(len(order)):
            if(total_RAM+service_RAM[order[i]]<=env.total_capacity):
                action_b[order[i]]=1
                total_RAM=total_RAM+service_RAM[order[i]]

        # edge_process=action_b*arr_task_num
        avg_f=env.total_cycle/np.sum(action_b)
        for i in range(env.service_num):
            if(action_b[i]==1):
                action_f[i]=avg_f
        service_cpu_allocation=action_f
        # #时隙内可处理任务类型为l的数量
        can_process_task_num=np.floor(service_cpu_allocation/env.service_c)
        #本地队列长度
        local_queue_len=np.zeros(env.service_num)
        for i in range (env.service_num):
            local_queue_len[i]=np.min([can_process_task_num[i],env.arr_task_num[i]])
            #转发至cloud的任务数量
            cloud_process_num=env.arr_task_num-local_queue_len
            #计算每个agent任务l的平均逗留时间
            avg_task_delay=1/(can_process_task_num-local_queue_len+1) #防止出现分母为0
        #计算动作所获取的奖励
        startup_time=0
        for service_j in range(env.service_num):
            if(action_b[service_j]-o[service_j]==1):
                startup_time=env.startup[service_j]+startup_time
        reward=np.sum(avg_task_delay*local_queue_len+cloud_process_num*env.task_cloud_delay)+startup_time+reward
        o[0:env.service_num]=action_b
        env.get_network_conditions()
    return reward/20

total=np.zeros(7)
for num in range(10):
    for i in range(0,7):
        total[i]=comput_latency(7+i,20,10,200,50,0.5)+total[i]
print(total/10)