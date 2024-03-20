import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        N is the number of rows in the grid, and M is the number of columns in the grid.
        :param cmap: The binary NxM colormap known to the robot, here 20x20.
        :param belief: An NxM numpy ndarray representing the prior belief, here 20x20.
        :param action: The action as a numpy ndarray. [R(1, 0), L(-1, 0), U(0, 1), D(0, -1)].
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution. NxM, here 20x20.
        '''

        ### Your Algorithm goes Below.

        n, m = np.array(cmap).shape  # 20, 20

        num_states = n * m  # 20 * 20 = 400
        num_actions = 4  # L, U, R, D
        num_colors = 2  # 0, 1
        # K = observation.shape[0]  # 30

        # cmap_flat = np.array(cmap).flatten()  # 400

        # Update belief after taking action, before sensing
        new_belief = np.zeros_like(belief)
        for i in range(n):
            for j in range(m):
                if action[0] == 0 and action[1] == 1:  # up
                    if i == 0:  # top
                        new_belief[i][j] = belief[i][j] + 0.9 * belief[i + 1][j]
                    elif i == n - 1:  # bottom
                        new_belief[i][j] = 0.1 * belief[i][j]
                    else:  # other
                        new_belief[i][j] = 0.1 * belief[i][j] + 0.9 * belief[i + 1][j]
                elif action[0] == 0 and action[1] == -1:  # down
                    if i == n - 1:  # bottom
                        new_belief[i][j] = belief[i][j] + 0.9 * belief[i - 1][j]
                    elif i == 0:  # top
                        new_belief[i][j] = 0.1 * belief[i][j]
                    else:  # other
                        new_belief[i][j] = 0.1 * belief[i][j] + 0.9 * belief[i - 1][j]
                elif action[0] == 1 and action[1] == 0:  # right
                    if j == m - 1:  # right edge
                        new_belief[i][j] = belief[i][j] + 0.9 * belief[i][j - 1]
                    elif j == 0:  # left edge
                        new_belief[i][j] = 0.1 * belief[i][j]
                    else:  # other
                        new_belief[i][j] = 0.1 * belief[i][j] + 0.9 * belief[i][j - 1]
                # if action[0] == -1 and action[1] == 0:  # left
                else:
                    if j == 0:  # left edge
                        new_belief[i][j] = belief[i][j] + 0.9 * belief[i][j + 1]
                    elif j == m - 1:  # right edge
                        new_belief[i][j] = 0.1 * belief[i][j]
                    else:  # other
                        new_belief[i][j] = 0.1 * belief[i][j] + 0.9 * belief[i][j + 1]

        # Update belief after sensing
        # P(color = observation | state = i) = 0.9 if the true is the same as sensed, and 0.1 otherwise.
        obs = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if observation == cmap[i][j]:
                    obs[i][j] = 0.9
                else:
                    obs[i][j] = 0.1
        new_belief = obs * new_belief

        # Normalize the belief
        sum_belief = np.sum(new_belief)
        new_belief = new_belief / sum_belief


        '''
        The HMM is defined by (T, M, pi), 
        where T is a current-state-dependent matrix (1/4 for each direction), 400x400.
        M is a state-dependent matrix (1/2 for each color 0 and 1), 400x2.
        pi is the initial state distribution and every state has a 1/400 probability.

        The Markov chain has 400 states, with 2 outputs (0 and 1) and 4 actions (up, down, left, right).
        When not at the boundary, the transition probability to the next grid is like (not T)
        [[...                                                       ],
         [..., 0.0,     0.0,        0.0,        0.0,        0.0, ...],
         [..., 0.0,     0.0,        0.9*0.25,   0.0,        0.0, ...],
         [..., 0.0,     0.9*0.25,   0.1,        0.9*0.25,   0.0, ...],
         [..., 0.0,     0.0,        0.9*0.25,   0.0,        0.0, ...],
         [..., 0.0,     0.0,        0.0,        0.0,        0.0, ...],
         [...                                                       ]].
        When at the boundaries, the transition probability to the next grid is like (not M)
        [[...                                   ],
         [..., 0.0,     0.0,        0.0         ],
         [..., 0.0,     0.0,        0.9*0.25    ],
         [..., 0.0,     0.9*0.25,   0.1+0.9*0.25],
         [..., 0.0,     0.0,        0.9*0.25    ],
         [..., 0.0,     0.0,        0.0         ],
         [...                                   ]].
        When at the corners, the transition probability to the next grid is like (not M)
        [[...                                       ],
         [..., 0.0,     0.0,        0.0             ],
         [..., 0.0,     0.0,        0.9*0.25        ],
         [..., 0.0,     0.9*0.25,   0.1+2*0.9*0.25  ]].

        What we know in this Bayes filter problem is the given Y_i and actions. 
        We want to find the current X, i.e. the probability for all the cells in the grid in current step.
        It's interesting that we not only know Y, but also the action, which the Bayes filter doesn't need.
        The return value of the filter is the probability of all the cells in the grid in current step k, given Y_{1:k}, 
        so it should be a 400x1 vector.
        '''

        # Compute observation matrix M
        # P(sensed color given the true color) = 0.9 if the true is the same as sensed, and 0.1 otherwise.
        # M = np.zeros((num_states, num_colors))
        # for i in range(num_states):  # i = 0, 1, ..., 399 (400)
        #     if cmap_flat[i] == 0:
        #         M[i][0] = 0.9
        #         M[i][1] = 0.1
        #     else:
        #         M[i][0] = 0.1
        #         M[i][1] = 0.9

        # Compute transition matrix T
        # T = np.zeros((num_states, num_states))  # 400x400, initialized all with 0
        # for i in range(num_states):  # Here i, j are the indices of the flattened cells of the cmap, 0-399
        #     for j in range(num_states):  # T[i][j]: probability of state i -> j
        #         # Stay (need to be careful about the boundaries and corners)
        #         if i == j:
        #             # Special case: 4 corners, transit to itself
        #             if i == 0 or i == m - 1 or i == n * m - m or i == n * m - 1:
        #                 T[i][j] = 2 * 0.9 * 0.25 + 0.1
        #             # Special case: 4 boundaries, transit to itself
        #             elif i < m or i % m == 0 or i % m == m - 1 or i >= n * m - m:
        #                 T[i][j] = 0.9 * 0.25 + 0.1
        #             # Normal case: internal cells, transit to itself
        #             else:
        #                 T[i][j] = 0.1
        #         # Transit to neighbor (need to be careful about the boundaries and corners)
        #         elif abs(i - j) == 1 or abs(i - j) == n:
        #             # Normal case: internal cells, transit to neighbor
        #             T[i][j] = 0.9 * 0.25
        #             # Special case: at left/right boundary, cannot transit to the other side
        #             if (i % m == 0 and j % m == m - 1) or (i % m == m - 1 and j % m == 0):
        #                 T[i][j] = 0  # Set back to 0

        # Compute alpha_k using forward algorithm for every k in (1, 2, ..., 30)
        # alpha = np.zeros(num_states)  # 400, will be updated every iteration
        # Initialize alpha_0 (note that the index starts at 1 in our textbook
        # for i in range(K):
        #
        #
        #
        # eta = 1 / sum_alpha
        #
        # alpha_k =
        #
        # result_k = eta * alpha_k

        return new_belief

    def histogram_filter2(self, cmap, belief, action, observation):
        cmap = np.array(cmap)
        action = np.array(action)
        num_rows, num_cols = cmap.shape
        go = 0.9
        stay = 0.1
        next = np.zeros_like(belief)
        for i in range(num_rows):
            for j in range(num_cols):
                if (action == np.array([0, 1])).all():  # move up
                    if i == 0:  # top
                        next[i, j] = belief[i, j] + go * belief[i + 1, j]
                    elif i == num_rows - 1:  # bottom
                        next[i, j] = stay * belief[i, j]
                    else:  # other
                        next[i, j] = stay * belief[i, j] + go * belief[i + 1, j]
                elif (action == np.array([0, -1])).all():
                    if i == num_rows - 1:
                        next[i, j] = belief[i, j] + go * belief[i - 1, j]
                    elif i == 0:
                        next[i, j] = stay * belief[i, j]
                    else:
                        next[i, j] = stay * belief[i, j] + go * belief[i - 1, j]
                elif (action == np.array([1, 0])).all():
                    if j == num_cols - 1:
                        next[i, j] = belief[i, j] + go * belief[i, j - 1]
                    elif j == 0:
                        next[i, j] = stay * belief[i, j]
                    else:
                        next[i, j] = stay * belief[i, j] + go * belief[i, j - 1]
                else:
                    if j == 0:
                        next[i, j] = belief[i, j] + go * belief[i, j + 1]
                    elif j == num_cols - 1:
                        next[i, j] = stay * belief[i, j]
                    else:
                        next[i, j] = stay * belief[i, j] + go * belief[i, j + 1]
        o = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                if observation == cmap[i, j]:
                    o[i, j] = go
                else:
                    o[i, j] = stay
        next *= next
        sum_belief = np.sum(next)
        next /= sum_belief
        return next