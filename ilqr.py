"""LQR, iLQR and MPC."""

#from controllers import approximate_A, approximate_B
import numpy as np
import scipy.linalg


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


def cost_inter(env, x, u):
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
    l = np.dot(u,u)
    l_x = np.zeros_like(x)
    l_xx = np.zeros([x.shape[0],x.shape[0]])
    l_u = u
    l_uu = np.eye(u.shape[0])
    l_ux = np.zeros([u.shape[0],x.shape[0]])
    return l,l_x.copy(),l_xx.copy(),l_u.copy(),l_uu.copy(),l_ux.copy()


def cost_final(env, x):
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
    l = 10000*np.square(np.linalg.norm(x-env.goal,2))
    l_x = 20000*(x-env.goal)
    l_xx = 20000*np.eye(x.shape[0])
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
    x_hat = np.zeros([tN,xdim])
    for i in range(tN):
      x_hat[i]=x_0.copy()
    u_hat = np.zeros([tN,adim])
    K_t = np.zeros([tN,adim,xdim])
    k_t = np.zeros([tN,adim])
    # Q_t = np.zeros([tN,x_dim+adim,x_dim+adim])
    # q_t = np.zeros([tN,x_dim+adim])
    V_t = np.zeros([tN+1,xdim,xdim])
    v_t = np.zeros([tN+1,xdim])
    for i in range(int(max_iter)):
      print("iteration: %d"%(i+1))
      rtotal = 0
      tcost = 0
      u_hat_old = u_hat.copy()
      #backward
      for t in reversed(range(tN)):
        F_t = approximate_F(sim_env,x_hat[t],u_hat[t])
        #print(F_t)
        if t == tN-1:#final
          l,l_x,l_xx = cost_final(sim_env,x_hat[t])
          c_t = np.hstack([l_x,np.zeros([adim,])])
          C_t = np.concatenate([l_xx,np.zeros([xdim,adim])],axis=1)
          C_t = np.concatenate([C_t,np.zeros([adim,xdim+adim])],axis = 0)
        else:
          l,l_x,l_xx,l_u,l_uu,l_ux = cost_inter(sim_env,x_hat[t],u_hat[t])
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
      
      #forward
      sim_env.state = x_0.copy()
      x = x_0.copy()
      for t in range(tN):
        #sim_env.render()
        delta_x = x-x_hat[t]
        x_hat[t] = x.copy()
        u_hat[t] = np.matmul(K_t[t],delta_x)+k_t[t]+u_hat[t]
        x,r,done,_ = sim_env.step(u_hat[t])
        if t != tN-1:
          cost,_,_,_,_,_=cost_inter(sim_env,x,u_hat[t])
        else:
          cost,_,_ = cost_final(sim_env,x)
        rtotal+=r
        tcost+=cost
      print("total reward:",rtotal)
      print("total cost:",tcost)
      if np.abs(u_hat-u_hat_old).max()<1e-4:
        break

    return u_hat.copy()
