import gym
import controllers
import numpy as np
import matplotlib.pyplot as plt
env_name = "TwoLinkArm-v0"

env = gym.make(env_name)
sim_env = gym.make(env_name)
done = False
state = env.reset()
sim_env.reset()
treward = 0
q = []
q_dot = []
u_t = []
u_limit = []
step_num = 0
tN = 100
#env._step(np.random.uniform(-100.0,100.0,(2,)),0.001)
# env.state = np.hstack([env.goal_q,env.goal_dq])
# print (env.state)
while not done and step_num<tN:
    env.render()
    q.append(env.position)
    q_dot.append(env.velocity)
    u = controllers.calc_lqr_input(env,sim_env)
    state,reward,done,_ = env.step(u)
    treward+=reward
    step_num+=1
    u_t.append(u)
    u_limit.append(np.clip(u,env.action_space.low,env.action_space.high))
    if np.max(env.velocity)>1000:
        break
    
q = np.array(q)
q_dot = np.array(q_dot)
u_t = np.array(u_t)
u_limit = np.array(u_limit)
print("total reward:",treward)
print("total steps:",step_num)
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
plt.subplot(427)
plt.plot(u_limit[:,0])
plt.title("u_limit[0]")
plt.subplot(428)
plt.plot(u_limit[:,1])
plt.title("u_limit[1]")
plt.show()
env.close()
sim_env.close()
