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
    l_xx = np.zeros(x.shape[0],x.shape[0])
    l_u = u
    l_uu = np.eye(u.shape[0])
    l_ux = np.zeros(u.shape[0],x.shape[0])
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
    env.state = x
    con = np.hstack([x,u])
    x_next = simulate_dynamics_next(env,x,u)
    F = np.zeros(x.shape[0],con.shape[0])
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


def calc_ilqr_input(env, sim_env, tN=50, max_iter=1e6):
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
    x_0 = env.state
    adim = env.action_space.shape[0]
    xdim = x_0.shape[0]
    x_hat = np.zeors([tN,x_dim])
    u_hat = np.zeros([tN,adim])
    K_t = np.zeros([tN,adim,x_dim])
    k_t = np.zeros([tN,adim])
    # Q_t = np.zeros([tN,x_dim+adim,x_dim+adim])
    # q_t = np.zeros([tN,x_dim+adim])
    V_t = np.zeros([tN+1,x_dim,x_dim])
    v_t = np.zeros([tN+1,x_dim])
    for i in range(max_iter):
      #backward
      for t in reversed(range(tN)):
        F_t = approximate_F(sim_env,x_hat[t],u_hat[t])
        if t == tN-1:#final
          l,l_x,l_xx = cost_final(sim_env,x_hat[t])
          c_t = np.hstack([l_x,np.zeros([adim,])])
          C_t = np.concatenate([l_xx,np.zeros([x_dim,adim])],axis=1)
          C_t = np.concatenate([C_t,np.zeros([adim,x_dim+adim])],axis = 0)
        else:
          l,l_x,l_xx,l_u,l_uu,l_ux = cost_inter(sim_env,x_hat[t],u_hat[t])
          c_t = np.hstack([l_x,l_u])
          C_t = np.concatenate([l_xx,np.transpose(l_ux)],axis = 1)
          temp = np.concatenate([l_ux,l_uu],axis = 1)
          C_t = np.concatenate([C_t,temp],axis = 0)
        Q_t = C_t+np.matmul(np.matmul(np.transpose(F_t),V_t[t+1]),F_t)
        q_t = c_t+np.matmul(np.matmul(np.transpose(F_t),v_t[t+1]))
        K_t = -np.matmul(np.linalg.inv(Q_t[xdim:,xdim:]),Q_t[xdim:,:xdim])
        k_t = -np.matmul(np.linalg.inv(Q_t[xdim:,xdim:]),q_t[xdim:])
        V_t[t] = Q_t[:xdim,:xdim]+np.matmul(Q_t[:xdim,xdim:],K_t)+
                np.matmul(np.transpose(K_t),Q_t[xdim:,:xdim])+
                np.matmul(np.matmul(np.transpose(K_t),Q_t[xdim:,xdim:]),K_t)
        v_t[t] = q_t[:xdim]+np.matmul(Q_t[:xdim,xdim:],k_t)+np.matmul(np.transpose(K_t),q[xdim:])
                  +np.matmul(np.matmul(np.transpose(K_t),Q_t[xdim:,xdim:]),k_t)

    return np.zeros((50, 2))
