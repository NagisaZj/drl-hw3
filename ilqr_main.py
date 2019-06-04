import gym
import ilqr
import numpy as np
import matplotlib.pyplot as plt
env_name = "TwoLinkArm-v0"
tN = 100

env = gym.make(env_name)
sim_env = gym.make(env_name)
done = False
state = env.reset()
sim_env.reset()
tcost = 0
q = []
q_dot = []

u_t = ilqr.calc_ilqr_input(env,sim_env,tN)
for i in range(len(u_t)):
    env.render()
    q.append(env.position)
    q_dot.append(env.velocity)
    env.step(u_t[i])

q = np.array(q)
q_dot = np.array(q_dot)
u_t = np.array(u_t)
plt.subplot(421)
plt.plot(q[:,0])
plt.title("q[0]")
plt.subplot(422)
plt.plot(q[:,1])
plt.title("q[1]")
plt.subplot(423)
plt.plot(q_dot[:,0])
plt.title("q_dot[0]")
plt.subplot(424)
plt.plot(q_dot[:,1])
plt.title("q_dot[1]")
plt.subplot(425)
plt.plot(u_t[:,0])
plt.title("u[0]")
plt.subplot(426)
plt.plot(u_t[:,1])
plt.title("u[1]")
plt.show()
env.close()
sim_env.close()
    