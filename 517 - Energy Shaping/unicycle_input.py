import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def unicycle_input(t : float, y_spline : CubicSpline, z_spline: CubicSpline) -> np.ndarray:
  #UNICYCLE_INPUT returns input to the unicycle
  #   @param t - current time
  #   @param y_spline - spline object for desired y trajectory
  #   @param z_spline - spline object for desired z trajectory
  #   
  #   @return u - input u(t) to the unicycle system

  # TODO: modify u to return the correct input for time t.
  u = np.zeros(2);

  y_dot = y_spline(t, nu=1)
  z_dot = z_spline(t, nu=1)
  y_ddot = y_spline(t, nu=2)
  z_ddot = z_spline(t, nu=2)

  # print(z_dot)
  
  # u1
  # u[0] = y_ddot * z_ddot * (y_ddot * z_dot - y_dot * z_ddot) / (z_dot**2 + y_dot**2)
  u[0] = -(y_ddot * z_dot - y_dot * z_ddot) / (z_dot**2 + y_dot**2)
  # u2
  u[1] = z_dot / np.sin(np.arctan(z_dot / y_dot))

  return u
