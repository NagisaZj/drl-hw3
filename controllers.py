"""LQR, iLQR and MPC."""

import numpy as np
from scipy.linalg import solve_continuous_are

def simulate_dynamics(env, x, u, dt=1e-5):
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
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    env.state = x.copy()
    x_next,_,_,_ = env._step(u,dt)
    # print("state:",x)
    # print("next_state",x_next)
    xdot = (x_next-x)/dt
    return xdot.copy()


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    # env.state(x)
    # x_next = env._step(u,dt)
    # print("A_x:",x)
    x_dot = simulate_dynamics(env,x,u,dt)
    A = np.zeros([x.shape[0],x.shape[0]])
    for j in range(x.shape[0]):
      x_pert = x.copy()
      x_pert[j]+=delta
      # print("A_x_pert:",x_pert)
      # print("_x",x)
      # env.state(x_pert)
      # x_pert_next = env._step(u,dt)
      x_pert_dot = simulate_dynamics(env,x_pert,u,dt)
      delta_x_dot = x_pert_dot-x_dot
      for i in range(x.shape[0]):
        A[i,j] = delta_x_dot[i]/delta

    return A.copy()


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    # env.state(x)
    # x_next = env._step(u,dt)
    x_dot = simulate_dynamics(env,x,u,dt)
    B = np.zeros([x.shape[0],u.shape[0]])
    for j in range(u.shape[0]):
      u_pert = u.copy()
      u_pert[j]+=delta
      
      # env.state(x)
      # x_pert_next = env._step(u_pert,dt)
      x_pert_dot = simulate_dynamics(env,x,u_pert,dt)
      delta_x_dot = x_pert_dot-x_dot
      for i in range(x.shape[0]):
        B[i,j] = delta_x_dot[i]/delta
    return B.copy()


def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    If you are following the API and simulate dynamics is returning
      function to solve the Ricatti equations.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    x = env.state
    A_sample = []
    B_sample = []
    for _ in range(10):
      u = np.random.uniform(low = -1.0,high = 1.0,size = sim_env.action_space.shape)
      #print("u",u)
      A_s = approximate_A(sim_env,x,u,1e-5,1e-5)
      B_s = approximate_B(sim_env,x,u,1e-5,1e-5)
      A_sample.append(A_s)
      B_sample.append(B_s)
    A = np.mean(A_sample,axis=0)
    B = np.mean(B_sample,axis=0)
    # print(A)
    # print(B)
    # while True:
    #   pass
    P = solve_continuous_are(A,B,env.Q,env.R)
    K = np.matmul(np.matmul(np.linalg.inv(env.R),np.transpose(B)),P)#K = R^{-1}*B^T*P
    #print(K)
    # while True:
    #   pass
    #u = np.clip(-np.matmul(K,(x-env.goal)),-1000,1000)#u = -Kx
    u = -np.matmul(K,(x-env.goal))
    return u.copy()
