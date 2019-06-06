"""LQR, iLQR and MPC."""

#from controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def simulate_dynamics_next(env, x, u):
    """Step simulator to see how state changes.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    next_x: np.array
    """
    env.state = x.copy()
    next_x,_,_,_ = env.step(u)
    return next_x.copy()


def cost_inter(env, x, u,lamb = 1.0):
    """intermediate cost function

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.

    Returns
    -------
    l, l_x, l_xx, l_u, l_uu, l_ux. The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables, ex: (1) l_x is the first order derivative d l/d x (2) l_xx is the second order derivative
    d^2 l/d x^2
    """
    l = np.dot(u,u)*lamb
    l_x = np.zeros_like(x)*lamb
    l_xx = np.zeros([x.shape[0],x.shape[0]])*lamb
    l_u = 2*u*lamb
    l_uu = 2*np.eye(u.shape[0])*lamb
    l_ux = np.zeros([u.shape[0],x.shape[0]])*lamb
    return l,l_x.copy(),l_xx.copy(),l_u.copy(),l_uu.copy(),l_ux.copy()


def cost_final(env, x,lamb = 1.0):
    """cost function of the last step

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.

    Returns
    -------
    l, l_x, l_xx The first term is the loss, where the remaining terms are derivatives respect to the
    corresponding variables
    """
    l = 10000*np.square(np.linalg.norm(x-env.goal,2))*lamb
    l_x = 20000*(x-env.goal)*lamb
    l_xx = 20000*np.eye(x.shape[0])*lamb
    return l,l_x.copy(),l_xx.copy()


def approximate_F(env,x,u,delta = 1e-5):
    """
    x_t+1 = F(x_t,u_t)^T
    return F
    """
    #env.state = x.copy()
    con = np.hstack([x,u])
    x_next = simulate_dynamics_next(env,x,u)
    F = np.zeros([x.shape[0],con.shape[0]])
    for j in range(con.shape[0]):
      con_pert = con.copy()
      con_pert[j] += delta
      x_next_pert = simulate_dynamics_next(env,con_pert[:x.shape[0]],con_pert[x.shape[0]:])
      x_next_delta = x_next_pert-x_next
      for i in range(x.shape[0]):
        F[i,j] = x_next_delta[i]/delta
    return F.copy()

def simulate(env, x0, U):
    return None


def calc_ilqr_input(env, sim_env, tN=100, max_iter=1e5):
    """Calculate the optimal control input for the given state.


    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.
    tN: number of control steps you are going to execute
    max_itr: max iterations for optmization

    Returns
    -------
    U: np.array
      The SEQUENCE of commands to execute. The size should be (tN, #parameters)
    """
    
    x_0 = env.state.copy()
    adim = env.action_space.shape[0]
    xdim = x_0.shape[0]
    x_hat = np.zeros([tN+1,xdim])
    u_hat = np.zeros([tN,adim])
    # if init_u is not None:
    #   u_hat = init_u.copy()
    #   sim_env.state = x_0.copy()
    #   for i in range(tN):
    #     x_hat[i] = sim_env.state.copy()
    #     sim_env.step(u_hat[i])
    
    
    K_t = np.zeros([tN,adim,xdim])
    k_t = np.zeros([tN,adim])
    # Q_t = np.zeros([tN,x_dim+adim,x_dim+adim])
    # q_t = np.zeros([tN,x_dim+adim])
    V_t = np.zeros([tN+1,xdim,xdim])
    v_t = np.zeros([tN+1,xdim])
    # v = np.zeros([tN+1,])
    # v_x = np.zeros([tN+1,xdim])
    # v_xx = np.zeros([tN+1,xdim,xdim])
    
    total_cost = []
    total_reward = []

    for i in range(int(max_iter)):
      print("iteration: %d"%(i+1))
      rtotal = 0
      tcost = 0
      u_hat_old = u_hat.copy()

      #backward
      _,v_t[-1],V_t[-1] = cost_final(sim_env,x_hat[-1],300.0)
      for t in reversed(range(tN)):
        F_t = approximate_F(sim_env,x_hat[t],u_hat[t])
        #print(F_t)
        # if t == tN-1:#final
        #   l,l_x,l_xx = cost_final(sim_env,x_hat[t])
        #   c_t = np.hstack([l_x,np.zeros([adim,])])
        #   C_t = np.concatenate([l_xx,np.zeros([xdim,adim])],axis=1)
        #   C_t = np.concatenate([C_t,np.zeros([adim,xdim+adim])],axis = 0)
        # else:
        _,l_x,l_xx,l_u,l_uu,l_ux = cost_inter(sim_env,x_hat[t],u_hat[t])
        c_t = np.hstack([l_x,l_u])
        C_t = np.concatenate([l_xx,np.transpose(l_ux)],axis = 1)
        temp = np.concatenate([l_ux,l_uu],axis = 1)
        C_t = np.concatenate([C_t,temp],axis = 0)
        #print(C_t)
        Q_t = C_t+np.matmul(np.matmul(np.transpose(F_t),V_t[t+1]),F_t)#Q_t = C_t+F_t^T*V_{t+1}*F_t
        q_t = c_t+np.matmul(np.transpose(F_t),v_t[t+1])#q_t = c_t+F_t^T*v_{t+1}
        K_t[t] = -np.matmul(np.linalg.pinv(Q_t[xdim:,xdim:]),Q_t[xdim:,:xdim])
        k_t[t] = -np.matmul(np.linalg.pinv(Q_t[xdim:,xdim:]),q_t[xdim:])
        V_t[t] = Q_t[:xdim,:xdim]+np.matmul(Q_t[:xdim,xdim:],K_t[t])+np.matmul(np.transpose(K_t[t]),Q_t[xdim:,:xdim])+np.matmul(np.matmul(np.transpose(K_t[t]),Q_t[xdim:,xdim:]),K_t[t])
        v_t[t] = q_t[:xdim]+np.matmul(Q_t[:xdim,xdim:],k_t[t])+np.matmul(np.transpose(K_t[t]),q_t[xdim:])+np.matmul(np.matmul(np.transpose(K_t[t]),Q_t[xdim:,xdim:]),k_t[t])
      
    # x_0 = env.state.copy()
    # adim = env.action_space.shape[0]
    # xdim = x_0.shape[0]
    # x_hat = np.zeros([tN+1,xdim])
    # u_hat = np.zeros([tN,adim])
    # kk_seq = np.zeros([tN,adim,xdim])
    # k_seq = np.zeros([tN,adim])
    # v = np.zeros([tN+1,])
    # v_x = np.zeros([tN+1,xdim])
    # v_xx = np.zeros([tN+1,xdim,xdim])

    # for i in range(int(max_iter)):
    #   print("iteration: %d"%(i+1))
    #   rtotal = 0
    #   tcost = 0
    #   u_hat_old = u_hat.copy()

    #   #backward
    #   v[-1],v_x[-1],v_xx[-1] = cost_final(sim_env,x_hat[-1])
    #   for t in reversed(range(tN)):
    #     F_t = approximate_F(sim_env,x_hat[t],u_hat[t])
    #     f_x_t = F_t[:,:xdim]
    #     f_u_t = F_t[:,xdim:]
    #     _,l_x,l_xx,l_u,l_uu,l_ux = cost_inter(sim_env,x_hat[t],u_hat[t])
    #     q_x = l_x + np.matmul(f_x_t.T,v_x[t+1])
    #     q_u = l_u + np.matmul(f_u_t.T, v_x[t + 1])
    #     q_xx = l_xx + np.matmul(np.matmul(f_x_t.T,v_xx[t+1]),f_x_t)
    #     tmp = np.matmul(f_u_t.T, v_xx[t + 1])
    #     q_uu = l_uu + np.matmul(tmp, f_u_t)
    #     q_ux = l_ux + np.matmul(tmp, f_x_t)
    #     inv_q_uu = np.linalg.pinv(q_uu)
    #     k = -np.matmul(inv_q_uu, q_u)
    #     kk = -np.matmul(inv_q_uu, q_ux)
    #     dv = 0.5 * np.matmul(q_u, k)
    #     v[t] += dv
    #     v_x[t] = q_x - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
    #     v_xx[t] = q_xx + np.matmul(q_ux.T, kk)
    #     kk_seq[t] = kk.copy()
    #     k_seq[t] = k.copy()
      #forward
      
      sim_env.state = x_0.copy()
      x = x_0.copy()
      for t in range(tN):
        #sim_env.render()
        delta_x = x-x_hat[t]
        x_hat[t] = x.copy()
        u_hat[t] = np.matmul(K_t[t],delta_x)+k_t[t]+u_hat[t]
        x,r,done,_ = sim_env.step(u_hat[t])
        # if t != tN-1:
        cost,_,_,_,_,_=cost_inter(sim_env,x,u_hat[t])
        # else:
        #   cost,_,_ = cost_final(sim_env,x)
        rtotal+=r
        tcost+=cost
      x_hat[tN] = x.copy()
      tcost+=cost_final(sim_env,x)[0]
      print("total reward:",rtotal)
      print("total cost:",tcost)
      total_cost.append(tcost)
      total_reward.append(rtotal)
      if np.abs(u_hat-u_hat_old).max()<1e-3 :
        break
      
      # sim_env.state = x_0.copy()
      # x_seq = x_hat.copy()
      # u_seq = u_hat.copy()
      # for t in range(tN):
      #   #sim_env.render()
      #   delta_x = x_hat[t]-x_seq[t]
      #   u_hat[t] = np.matmul(kk_seq[t],delta_x)+k_seq[t]+u_seq[t]
      #   x_hat[t+1],r,done,_ = sim_env.step(u_hat[t])
      #   # if t != tN-1:
      #   cost,_,_,_,_,_=cost_inter(sim_env,x_hat[t],u_hat[t])
      #   # else:
      #   #   cost,_,_ = cost_final(sim_env,x)
      #   rtotal+=r
      #   tcost+=cost
      # tcost+=cost_final(sim_env,x_hat[tN])[0]
      # print("total reward:",rtotal)
      # print("total cost:",tcost)
      # if np.abs(u_hat-u_hat_old).max()<1e-4 :
      #   break
    plt.figure()
    plt.subplot(121)
    plt.plot(total_cost)
    plt.title("cost")
    plt.subplot(122)
    plt.plot(total_reward)
    plt.title("reward")
    return u_hat.copy()
