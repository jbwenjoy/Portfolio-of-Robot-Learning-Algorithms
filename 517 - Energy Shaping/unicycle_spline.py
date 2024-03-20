import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from random import uniform

def unicycle_spline(t0, tf, obs):
  # UNICYCLE_SPLINE returns a spline object representing a path from
  # (y(t0),z(t0)) = (0,0) to (y(t0),z(t0)) = (10,0) that avoids a circular
  # obstacle, such that d\dt y(t) > 0
  #   @param t0 - initial time
  #   @param tf - final time
  #
  #   @return y_spline - spline object for desired y trajectory
  #   @return z_spline - spline object for desired z trajectory
  
  # obs is the obstacle class
  # obs.radius:     uniform(2, 4)
  # obs.y:          uniform(0.5 + radius, 9.5 - radius)
  # obs.z:          uniform(-radius, radius)
  
  y0 = 0;
  z0 = 0;

  yf = 10;
  zf = 0;

  # TODO: design the spline here
  # add three mid points whose croods are defined with respect to the obstacle so the traj never hits it
  if obs.z <= 0:
      midp_1 = np.array([obs.y - obs.radius * 0.7, obs.z + obs.radius - 0.1])
      midp_2 = np.array([obs.y, obs.z + obs.radius + 0.3])
      midp_3 = np.array([obs.y + obs.radius * 0.7, obs.z + obs.radius - 0.1])
  else:
      midp_1 = np.array([obs.y - obs.radius * 0.7, obs.z - obs.radius + 0.1])
      midp_2 = np.array([obs.y, obs.z - obs.radius - 0.3])
      midp_3 = np.array([obs.y + obs.radius * 0.7, obs.z - obs.radius + 0.1])

  length_1_apprx = np.sqrt( (midp_1[0] - y0)**2 + (midp_1[1] - z0)**2 ) * 1.4
  length_2_apprx = np.sqrt( (midp_1[0] - midp_2[0])**2 + (midp_1[1] - midp_2[1])**2 )
  length_3_apprx = np.sqrt( (midp_3[0] - midp_2[0])**2 + (midp_3[1] - midp_2[1])**2 ) 
  length_4_apprx = np.sqrt( (midp_3[0] - yf)**2 + (midp_3[1] - zf)**2 ) * 1.4

  full_length_apprx = length_1_apprx + length_2_apprx + length_3_apprx + length_4_apprx
  midt_1 = (tf - t0) * length_1_apprx / full_length_apprx
  midt_2 = (tf - t0) * length_2_apprx / full_length_apprx + midt_1
  midt_3 = (tf - t0) * length_3_apprx / full_length_apprx + midt_2

  t = np.array([t0, midt_1, midt_2, midt_3, tf])
  y = np.array([y0, midp_1[0], midp_2[0], midp_3[0], yf])
  z = np.array([z0, midp_1[1], midp_2[1], midp_3[1], zf])

  print("t", t)
  print("y", y)
  print("z", z)

  y_spline = CubicSpline(t, y);
  z_spline = CubicSpline(t, z);


  return y_spline, z_spline
