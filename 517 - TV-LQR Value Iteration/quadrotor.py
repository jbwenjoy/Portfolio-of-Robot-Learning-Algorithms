import numpy as np
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp

from trajectories import *

class Quadrotor(object):
  '''
  Constructor. Compute function S(t) using S(t) = L(t) L(t)^t, by integrating backwards
  from S(tf) = Qf. We will then use S(t) to compute the optimal controller efforts in 
  the compute_feedback() function
  '''
  def __init__(self, Q, R, Qf, tf):
    self.m = 1
    self.a = 0.25
    self.I = 0.0625
    self.Q = Q
    self.R = R

    ''' 
    We are integrating backwards from Qf
    '''

    # Get L(tf) L(tf).T = S(tf) by decomposing S(tf) using Cholesky decomposition
    L0 = cholesky(Qf).transpose()

    # We need to reshape L0 from a square matrix into a row vector to pass into solve_ivp()
    l0 = np.reshape(L0, (36))
    # L must be integrated backwards, so we integrate L(tf - t) from 0 to tf
    initial_condition = [0, tf]
    sol = solve_ivp(self.dldt_minus, [0, tf], l0, dense_output=True)
    t = sol.t
    l = sol.y

    # Reverse time to get L(t) back in forwards time
    t = tf - t
    t = np.flip(t)
    l = np.flip(l, axis=1) # flip in time
    self.l_spline = interp1d(t, l)


  def f(self, y, z, theta, dy, dz, dtheta, u1, u2): 
    '''
    # original nonlinear state function f(x, u)
    '''
    g = 9.81
    return np.array([dy, dz, dtheta, 
      -np.sin(theta) / self.m * (u1 + u2), 
      -g + np.cos(theta) / self.m * (u1 + u2)], 
      self.a / self.I * (u1 - u2))
  
  
  def Ldot(self, t, L):

    x = x_d(t)
    u = u_d(t)
    Q = self.Q
    R = self.R
    # m = self.m

    dLdt = np.zeros((6,6))
    
    # STUDENT CODE: compute d/dt L(t) ###

    ### STUDENT CODE START ###

    A = np.zeros((6,6))
    B = np.zeros((6,2))
    
    # A = partial f(x, u) with x at point (x_d, u_d)
    A[0][3] = 1
    A[1][4] = 1
    A[2][5] = 1
    A[3][2] = -(u[0] + u[1]) / self.m * np.cos(x[2])
    A[4][2] = -(u[0] + u[1]) / self.m * np.sin(x[2])

    # B = partial f(x, u) with u at point (x_d, u_d)
    B[3][0] = -np.sin(x[2]) / self.m
    B[3][1] = -np.sin(x[2]) / self.m
    B[4][0] = np.cos(x[2]) / self.m
    B[4][1] = np.cos(x[2]) / self.m
    B[5][0] = self.a / self.I
    B[5][1] = -self.a / self.I

    L_inv = np.linalg.inv(L)
    L_tra = np.transpose(L)
    L_tra_inv = np.transpose(L_inv)
    R_inv = np.linalg.inv(self.R)
    B_tra = np.transpose(B)
    A_tra = np.transpose(A)

    dLdt = -1/2 * np.dot(Q, L_tra_inv) - np.dot(A_tra, L) + 1/2 * np.dot(L, np.dot(L_tra, np.dot(B, np.dot(R_inv, np.dot(B_tra, L)))))

    ### DEBUG ONLY ###
    # print("A =\n")
    # print(A)
    # print("B =\n")
    # print(B)
    # print("dLdt =\n")
    # print(dLdt)
    # print('\n')

    ### STUDENT CODE END ###

    return dLdt


  def dldt_minus(self, t, l):
    # reshape l to a square matrix
    L = np.reshape(l, (6, 6))

    # compute Ldot
    dLdt_minus = -self.Ldot(t, L)

    # reshape back into a vector
    dldt_minus = np.reshape(dLdt_minus, (36))
    return dldt_minus


  def compute_feedback(self, t, x):
    # Retrieve L(t)
    L = np.reshape(self.l_spline(t), (6, 6))

    u_fb = np.zeros((2,))
    # STUDENT CODE: Compute optimal feedback inputs u_fb using LQR

    ### STUDENT CODE START ###
    # x = x_e + x_d
    x_e = x - x_d(t)
    
    # B = partial f(x, u) with u at point (x_d, u_d)
    B = np.zeros((6,2))
    B[3][0] = -np.sin(x_d(t)[2]) / self.m
    B[3][1] = -np.sin(x_d(t)[2]) / self.m
    B[4][0] = np.cos(x_d(t)[2]) / self.m
    B[4][1] = np.cos(x_d(t)[2]) / self.m
    B[5][0] = self.a / self.I
    B[5][1] = -self.a / self.I
    
    # S = np.dot(L, np.transpose(L))
    S = L @ np.transpose(L)

    # u_fb = -np.dot(np.linalg.inv(self.R), np.dot(np.transpose(B), np.dot(S, x_e)))
    u_fb = -np.linalg.inv(self.R) @ np.transpose(B) @ S @ x_e
    ### STUDENT CODE END ###

    # Add u_fb to u_d(t), the feedforward term. 
    # u = u_fb + u_d
    u = u_d(t) + u_fb;
    return u