import importlib
import numpy as np
import plot_cartpole_trajectory
importlib.reload(plot_cartpole_trajectory)

from math import sin, cos, pi
from scipy.integrate import solve_ivp
from plot_cartpole_trajectory import plot_cartpole_trajectory



from cartpole import Cartpole


def simulate_cartpole(cartpole, x0, tf, plotting=False):
  n_frames = 30

  g = 9.81
  mc = 1
  mp = 1
  L = 1

  def f(t, x):
    M = np.array([[mc + mp, mp*L*cos(x[1])],
                  [mp*L*cos(x[1]), mp*L**2]])
    C = np.array([-mp*L*sin(x[1])*x[3]**2,
                  mp*g*L*sin(x[1])])
    B = np.array([1, 0])

    u = cartpole.compute_efforts(t, x)

    x_dot = np.hstack((x[-2:],
                      np.linalg.solve(M, B * u  - C)))
    return x_dot


  sol = solve_ivp(f, (0, tf), x0, max_step=1e-3)
  if plotting:
    anim, fig = plot_cartpole_trajectory(sol.t, sol.y, n_frames, L)
    return anim, fig
  else:
    return sol.y

if __name__ == '__main__':
  x0 = np.zeros(4)
  x0[1] = pi/6
  tf = 10
  cartpole = Cartpole()
  simulate_cartpole(cartpole, x0, tf, True)