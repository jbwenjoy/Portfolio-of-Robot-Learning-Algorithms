import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Tuple
from IPython import get_ipython
from mpl_toolkits import mplot3d
from matplotlib.patches import Rectangle, Circle


def plot_cartpole_trajectory(t: np.ndarray, x: np.ndarray, n_frames: int, L: float) -> Tuple[matplotlib.animation.FuncAnimation, plt.figure]:
  theta = x[1, :]
  x = x[0, :]

  px = x + L * np.sin(theta);
  py = - L * np.cos(theta);

  # cartpole geometry parameters
  h = .2;
  w = .4;

  x_range = np.array([-2, 2])
  y_range = np.array([-1, 3])

  pennblue = np.array([1,37,110]) / 256;
  pennred = np.array([149,0,26]) / 256;

  fig = plt.figure(figsize=(8,6))
  ax = plt.axes()

  def frame(i):
    ax.clear()

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    i = int(i / n_frames * t.shape[0])

    ax.plot(4 * x_range, [0,  0], 'k', linewidth=3)
    cartpole_base = Rectangle([x[i] -w/2, -h/2], w, h, facecolor=pennblue, edgecolor='k', linewidth=3)
    cartpole_mass = Circle((px[i], py[i]), 0.02, facecolor=pennred, edgecolor=pennred, linewidth=3)
    ax.add_patch(cartpole_base)
    ax.add_patch(cartpole_mass)

    ax.plot([x[i], x[i] + L*np.sin(theta[i])], [0, -L*np.cos(theta[i])], 'k', linewidth=3);
    plot = ax.plot(px[:i], py[:i], 'g', linewidth=3);

    ax.set_xlabel('q1 (m)')

    if get_ipython() is None:
      plt.draw()
      plt.pause(t[-1]/n_frames)
    return plot

  if get_ipython() is None:
    plt.ion()
    plt.show()
    for i in range(n_frames):
      frame(i)

  anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False, repeat=False)
  plt.close()
  return anim, fig