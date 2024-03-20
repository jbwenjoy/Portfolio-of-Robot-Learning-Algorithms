from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

def create_animation(x, x_d, tf, n_frame):
  # Sample desired trajectory
  n_samples = 1000
  t_samples = np.linspace(0.0, tf, n_samples)
  x_des = np.zeros((n_samples, 6))
  for i in range(t_samples.shape[0]):
    x_des[i] = x_d(t_samples[i])

  from matplotlib import rc
  rc('animation', html='jshtml')

  fig = plt.figure(figsize=(8,6))
  ax = plt.axes()

  x_max = max(np.max(x_des[:, 0]), np.max(x[:, 0]))
  x_min = min(np.min(x_des[:, 0]), np.min(x[:, 0]))
  y_max = max(np.max(x_des[:, 1]), np.max(x[:, 1]))
  y_min = min(np.min(x_des[:, 1]), np.min(x[:, 1]))

  frame_idx = [round(x) for x in np.linspace(0, x.shape[0]-1, n_frame).tolist()]
  x_anim = np.zeros((n_frame, 6))
  for i in range(n_frame):
    x_anim[i, :] = x[frame_idx[i], :]

  a = 0.25
  y = x_anim[:, 0]
  z = x_anim[:, 1]
  theta = x_anim[:, 2]

  def frame(i):
    ax.clear()

    ax.plot(x_des[:, 0], x_des[:, 1], label='desired trajectory')
    ax.plot(x_anim[:i+1, 0], x_anim[:i+1, 1], '--', label='actual trajectory')
    # plot=ax.scatter(x_anim[i, 0], x_anim[i, 1], c='r', label='quadrotor position')
    plot = ax.plot([y[i] + a*cos(theta[i]), y[i] - a*cos(theta[i])],
                   [z[i] + a*sin(theta[i]), z[i] - a*sin(theta[i])] , color='g', linewidth=3)
    
    #ax.set_xlim(x_min,x_max)
    #ax.set_ylim(y_min,y_max)
    ax.set_xlabel('y (m)')
    ax.set_ylabel('z (m)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left')

    return plot

  return animation.FuncAnimation(fig, frame, frames=n_frame, blit=False, repeat=False), fig
  
