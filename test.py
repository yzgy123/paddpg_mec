import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.random import zipf
from scipy.stats import zipfian
# a=torch.tensor([1],dtype=float)
# b=torch.tensor([2],dtype=float)
# c=torch.cat([a,b],dim=0)
# d=torch.mean(c)
# # d.backward()
# f=d.numpy()
# e=f
# e=e*2
# print(d)

# a = np.array([1, 3, 2, 4, 5])
# print(np.flip(np.argsort(a)))
np.random.seed(1)
total_tasks=200
zipf_para, service_num = 2, 20
x = np.arange(1, service_num+1)
y = zipfian.pmf(x, zipf_para, service_num)*total_tasks
maxy=np.around(y*1.25)
miny=np.around(y*0.75)
plt.stem(x, maxy)
plt.title('zipfian pmf')
plt.show()
