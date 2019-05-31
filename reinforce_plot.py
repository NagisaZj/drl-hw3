import numpy as np
import matplotlib.pyplot as plt
logpath = './reinforce_data/'

mean = np.array(np.load(logpath+'mean.npy'),dtype=np.float32)
min = np.array(np.load(logpath+'min.npy'),dtype=np.float32)
max = np.array(np.load(logpath+'max.npy'),dtype=np.float32)


errors=np.array([mean-min,max-mean])

x=np.arange(len(mean))*10
plt.figure()
plt.errorbar(x,mean,errors)
plt.xlabel('train episodes')
plt.ylabel('accumulated reward')
plt.title('CartPole-v0')
plt.show()