import numpy as np
from mec_env2 import mec_env

def comput_latency(service_num,total_capacity,CPU_frequency,task_num,bandwidth,zipf):
    env=mec_env(service_num,total_capacity,CPU_frequency,task_num,bandwidth,zipf)
    o=env.reset()
    select_service=[]
    total_RAM=0
    s_RAM=np.copy(env.service_m)
    while(True):
        min_RAM=min(s_RAM)
        if len(select_service)==env.service_num:
            break
        if env.total_capacity-total_RAM<min_RAM:
            break
        s_index=np.random.randint(0,env.service_num)
        if(s_index in select_service):
            continue
        else:
            if(total_RAM+env.service_m[s_index]>env.total_capacity):
                continue
            else:
                select_service.append(s_index)
                s_RAM[s_index]=1000
                total_RAM=total_RAM+env.service_m[s_index]
    action_b=np.zeros(env.service_num)
    action_f=np.zeros(env.service_num)
    avg_f=env.total_cycle/len(select_service)
    for s in select_service:
        action_b[s]=1
        action_f[s]=avg_f

    service_cpu_allocation=action_f
    # #时隙内可处理任务类型为l的数量
    can_process_task_num=np.floor(service_cpu_allocation/env.service_c)
    reward=0
    for k in range(20):
        local_queue_len=np.zeros(env.service_num)
        for i in range (env.service_num):
            #本地队列长度
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
        o[0:env.service_num]=action_b
        reward=np.sum(avg_task_delay*local_queue_len+cloud_process_num*env.task_cloud_delay)+startup_time+reward
        env.get_network_conditions()
    return reward/20

total=np.zeros(7)
for num in range(10):
    for i in range(0,7):
        total[i]=comput_latency(7+i,20,10,200,50,0.5)+total[i]
print(total/10)