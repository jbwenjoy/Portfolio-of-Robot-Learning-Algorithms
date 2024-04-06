import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))


class map_t:
    """
    This will maintain the occupancy grid and log_odds.
    You do not need to change anything in the initialization
    """

    def __init__(self, resolution=0.05):
        self.resolution = resolution
        self.xmin, self.xmax = -20, 20  # range of x in the world
        self.ymin, self.ymax = -20, 20  # range of y in the world
        self.szx = int(np.ceil((self.xmax - self.xmin) / self.resolution + 1))  # number of cells in x direction
        self.szy = int(np.ceil((self.ymax - self.ymin) / self.resolution + 1))  # number of cells in y direction

        # binarized map and log-odds
        self.cells = np.zeros((self.szx, self.szy), dtype=np.int8)  # initialize the map as empty
        self.log_odds = np.zeros(self.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        self.log_odds_max = 5e6
        # number of observations received yet for each cell
        self.num_obs_per_cell = np.zeros(self.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        self.occupied_prob_thresh = 0.6
        self.log_odds_thresh = np.log(self.occupied_prob_thresh / (1 - self.occupied_prob_thresh))

    def grid_cell_from_xy(self, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding to these (x,y) locations.
        You should return an array of shape (2 x len(x)).
        Be careful to handle instances when x/y go outside the map bounds,
        you can use np.clip to handle these situations.
        """
        #### DONE: XXXXXXXXXXX

        # Convert x, y to grid cell indices
        grid_x = np.floor((x - self.xmin) / self.resolution).astype(int)
        grid_y = np.floor((y - self.ymin) / self.resolution).astype(int)

        # Make sure the indices are within the map
        grid_x = np.clip(grid_x, 0, self.szx - 1).astype(int)
        grid_y = np.clip(grid_y, 0, self.szy - 1).astype(int)

        return grid_x, grid_y


class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """

    def __init__(self, resolution=0.05, Q=1e-3 * np.eye(3), resampling_threshold=0.3):
        self.n = None
        self.p = None
        self.w = None
        self.lidar_log_odds_free = None
        self.lidar_log_odds_occ = None
        self.lidar_angles = None
        self.lidar_angular_resolution = None
        self.lidar_dmin = None
        self.lidar_dmax = None
        self.head_height = None
        self.lidar_height = None
        self.find_joint_t_idx_from_lidar = None
        self.joint = None
        self.lidar = None
        self.idx = None

        self.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        self.Q = Q
        # self.Q = 1e-8 * np.eye(3)

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        self.resampling_threshold = resampling_threshold

        # initialize the map
        self.map = map_t(resolution)

    def read_data(self, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        self.idx = idx
        self.lidar = load_lidar_data(os.path.join(src_dir, 'data/%s/%s_lidar%d' % (split, split, idx)))
        self.joint = load_joint_data(os.path.join(src_dir, 'data/%s/%s_joint%d' % (split, split, idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        self.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(self.joint['t'] - t))

    def init_sensor_model(self):
        # lidar height from the ground in meters
        self.head_height = 0.93 + 0.33
        self.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        self.lidar_dmin = 1e-3
        self.lidar_dmax = 30
        self.lidar_angular_resolution = 0.25  # degrees
        # these are the angles of the rays of the Hokuyo, [-135, 135] degrees, with 0.25 degrees resolution
        self.lidar_angles = np.arange(-135, 135 + self.lidar_angular_resolution,
                                      self.lidar_angular_resolution) * np.pi / 180.0  # in radians

        # Sensor model
        # lidar_log_odds_occ: value by which we would increase the log_odds for occupied cells.
        # lidar_log_odds_free: value by which we should decrease the log_odds for free cells
        # (which are all cells that are not occupied)
        self.lidar_log_odds_occ = np.log(9)
        self.lidar_log_odds_free = -np.log(9)

    def init_particles(self, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        self.n = n
        self.p = deepcopy(p) if p is not None else np.zeros((3, self.n), dtype=np.float64)
        self.w = deepcopy(w) if w is not None else np.ones(n) / float(self.n)  # 1/n

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = (3,n) array of
        particles with w = (n,) array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """
        #### DONE: XXXXXXXXXXX

        # If we enter this function, we need to resample the particles as it falls below the threshold
        # print('Resampling')

        # Get the number of particles
        n = w.shape[0]

        # Get the cumulative sum of the weights
        cum_sum = np.cumsum(w)

        # Generate the random numbers
        # r is uniformly distributed random numbers between 0 and 1/n
        r = (np.random.rand() + np.arange(n)) / n

        # Initialize the new particles and weights
        new_p = np.zeros(p.shape)
        new_w = np.zeros(n)

        # Resample the particles
        i, j = 0, 0
        while i < n:
            if r[i] < cum_sum[j]:
                new_p[:, i] = p[:, j]
                new_w[i] = 1 / n
                i += 1
            else:
                j += 1

        return new_p, new_w


    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w - w.max()).sum())

    def rays2world(self, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p: p is the pose of the particle (x,y,yaw), a 3x1 array describing the robot position and orientation
        d: an array that stores the distance along the ray of the lidar for each ray
           the length of d has to be equal to that of angles, this is s.lidar[t]['scan']
        head_angle: the angle of the head in the body frame, usually 0, need to be in radians
        neck_angle: the angle of the neck in the body frame, usually 0, need to be in radians
        angles: angle of each ray in the body frame in radians
                (usually be simply self.lidar_angles for the different lidar rays)

        Return an array (2 x num_rays) which are the (x,y) locations of the end point of each ray in world coordinates
        """
        #### Done and Checked: XXXXXXXXXXX

        # Make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading the data
        # We will just drop out-of-range distances
        valid_indices = np.logical_and(d >= self.lidar_dmin, d <= self.lidar_dmax)
        d = d[valid_indices]
        angles = angles[valid_indices]

        # 1. from lidar distances to points in the LiDAR frame
        lidar_frame_points_3d = np.vstack((d * np.cos(angles), d * np.sin(angles), np.zeros(len(d))))

        # 2. from LiDAR frame to the body frame
        # We need to consider the head and neck angles, so here we have three euler angles with zero roll
        # We also need to consider the height of the LiDAR in the body frame, i.e. self.lidar_height
        ### WARNING! Make sure: (a) radians/degrees; (b) positive/negative
        lidar_to_body = euler_to_se3(0, head_angle, neck_angle, np.array([0, 0, self.lidar_height]))
        body_frame_points_4d = lidar_to_body @ make_homogeneous_coords_3d(lidar_frame_points_3d)  # 4xN

        # 3. from body frame to world frame
        # Using p as the pose of the particle, we can transform the points from the body frame to the world frame
        # We also need to consider the height of the head in the world frame, i.e. self.head_height
        ### WARNING! Make sure: (a) radians/degrees; (b) positive/negative
        # print(p.shape)
        body_to_world = euler_to_se3(0, 0, p[2, 0], np.array([p[0, 0], p[1, 0], self.head_height]))
        world_frame_points_4d = body_to_world @ body_frame_points_4d  # 4xN

        # Normalize the points to 3xN, using the homogeneous coordinates
        world_frame_points_3d = world_frame_points_4d[:3] / world_frame_points_4d[3]

        # Return the 2D points
        # result = world_frame_points_3d[:2]
        return world_frame_points_3d[:2]

    def get_control(self, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
          at time t-1 at state (x,y,th)_{t-1} to come tomax_weight_particle state (x,y,th)_t.
        We will assume that this is the same control that
          the robot will take in the function dynamics_step below at time t to go to time t-1.
        Need to use the smart_minus_2d function to get the difference of the two poses,
          and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        #### Done and Checked: XXXXXXXXXXX

        # Get the pose at time t-1 andmax_weight_particle_pose = self.lidar[t]['xyth']
        previous_pose = self.lidar[t - 1]['xyth']
        current_pose = self.lidar[t]['xyth']

        # Get the control, i.e. the difference between the two poses
        control = smart_minus_2d(current_pose, previous_pose)

        return control

    def dynamics_step(self, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of
          the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """
        #### DONE: XXXXXXXXXXX

        # Get the control
        control = self.get_control(t)

        # Perform control on each particle
        for i in range(self.n):
            # Add noise to the control
            noisy_control = control + np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)

            # Update the particle
            self.p[:, i] = smart_plus_2d(self.p[:, i].copy(), noisy_control)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
          new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        #### Done and Checked: XXXXXXXXXXX

        # Parse the observation log-probability
        w = obs_logp + np.log(w)
        w -= slam_t.log_sum_exp(w)
        w = np.exp(w)
        return w

    def observation_step(self, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                    calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight,
                and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration
        (it is simply the binarized version of log_odds).
        map.log_odds is of course maintained across iterations.
        """
        #### DONE: XXXXXXXXXXX

        # (a) Find the head, neck angle at t
        #     This is the same for every particle
        neck_angle = self.joint['head_angles'][0, self.find_joint_t_idx_from_lidar(self.lidar[t]['t'])]
        head_angle = self.joint['head_angles'][1, self.find_joint_t_idx_from_lidar(self.lidar[t]['t'])]
        # neck_angle, head_angle = self.joint['head_angles'][:, self.find_joint_t_idx_from_lidar(self.lidar[t]['t'])]

        obs_logp = np.zeros(self.n)  # Observation log-probability for each particle
        for i in range(self.n):
            # (b) Project lidar scan into the world frame, assuming the particle is the true pose of the robot lidar
            # self.p stores the particles used for estimating the robot pose with PF
            # self.lidar[t]['scan'] is the distance measurements from the lidar at time t
            p = self.p[:, i].reshape((3, 1))
            world_frame_points = self.rays2world(p, self.lidar[t]['scan'], head_angle, neck_angle, self.lidar_angles)

            # (c) Calculate the grid cell indices of the occupied cells
            occupied_cells = self.map.grid_cell_from_xy(world_frame_points[0], world_frame_points[1])

            # (c) Calculate the observation log-probability
            obs_logp[i] = np.sum(self.map.log_odds[occupied_cells[0], occupied_cells[1]])

        # Update the particle weights using observation log-probability
        self.w = self.update_weights(self.w, obs_logp)

        # Find the particle with the largest weight
        self.estimated_pose = self.p[:, np.argmax(self.w)]  # The pose of the particle with the largest weight

        # Transform to world frame coordinates
        best_particle_world_points_coord = self.rays2world(self.estimated_pose.reshape((3, 1)),
                                                           self.lidar[t]['scan'], head_angle,
                                                           neck_angle, self.lidar_angles)

        # Compute the grid cell indices of the occupied cells
        occupied_cells_idx_x, occupied_cells_idx_y = self.map.grid_cell_from_xy(best_particle_world_points_coord[0], best_particle_world_points_coord[1])

        # Note that I learned the below method from: https://github.com/KailajeAnirudh/Robotics-Learning
        # Update the map.cells using the occupied cells of the particle with the largest weight
        # Find free cells
        lim_x_lower = self.estimated_pose[0] - self.lidar_dmax / 2
        lim_x_upper = self.estimated_pose[0] + self.lidar_dmax / 2
        lim_y_lower = self.estimated_pose[1] - self.lidar_dmax / 2
        lim_y_upper = self.estimated_pose[1] + self.lidar_dmax / 2
        limit_x_coord = np.array([lim_x_lower, lim_x_upper, self.estimated_pose[0]])
        limit_y_coord = np.array([lim_y_lower, lim_y_upper, self.estimated_pose[1]])
        limit_grid_x, limit_grid_y = self.map.grid_cell_from_xy(limit_x_coord, limit_y_coord)

        free_cells_x = np.linspace([limit_grid_x[2]] * len(occupied_cells_idx_x), occupied_cells_idx_x, endpoint=False).astype(int)
        free_cells_y = np.linspace([limit_grid_y[2]] * len(occupied_cells_idx_y), occupied_cells_idx_y, endpoint=False).astype(int)
        free_cells_x, free_cells_y = free_cells_x.flatten(), free_cells_y.flatten()

        # Update map.log_odds using the largest weight particle
        # i.e. add log_odds_occ (>0) to the occupied cells
        # and add log_odds_free (<0) to the free cells
        self.map.log_odds[occupied_cells_idx_x, occupied_cells_idx_y] += self.lidar_log_odds_occ
        self.map.log_odds[free_cells_x, free_cells_y] += self.lidar_log_odds_free
        self.map.log_odds = np.clip(self.map.log_odds, -self.map.log_odds_max, self.map.log_odds_max)

        # Update map.cells using the updated map.log_odds (binarize the map)
        ### WARNING! I'm not sure if this is the same operation as the one several lines above
        self.map.cells = (self.map.log_odds > self.map.log_odds_thresh).astype(int)

        # Resample the particles if necessary
        self.resample_particles()

    def resample_particles(self):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance in the particles.
        We should resample only if the effective number of particles falls below
          a certain threshold (resampling_threshold).
        A good heuristic to calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
          of the particles, if this number of close to n, then all particles have about equal weights,
          and we do not need to resample
        """
        e = 1 / np.sum(self.w ** 2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e / self.n < self.resampling_threshold:
            self.p, self.w = self.stratified_resampling(self.p, self.w)
            logging.debug('> Resampling')
