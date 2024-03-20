import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Dynamics for the quadrotor
def _f(x, u):
  g = 9.81
  m = 1
  a = 0.25
  I = 0.0625

  theta = x[2]
  ydot = x[3]
  zdot = x[4]
  thetadot = x[5]
  u0 = u[0]
  u1 = u[1]

  xdot = np.array([ydot,
                   zdot,
                   thetadot,
                   -sin(theta) * (u0 + u1) / m,
                   -g + cos(theta) * (u0 + u1) / m,
                   a * (u0 - u1) / I])

  return xdot


def F(xc, uc, dt):
  # Simulate the open loop quadrotor for one step
  def f(_, x):
    return _f(x, uc)
  sol = solve_ivp(f, (0, dt), xc, first_step=dt)
  return sol.y[:, -1].ravel()