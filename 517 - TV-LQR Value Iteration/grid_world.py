import numpy as np


class GridWorld():


  ''' 
  Gridworld dynamics very loosely borrowed from Sutton and Barto:
  Reinforcement Learning

  The dynamics have been simplified to be deterministic
  '''
  def __init__(self, slip_prob = 0.1):
    # check input
    if(slip_prob < 0 or slip_prob > 1):
        raise ValueError('Not a valid slip prob')
        return
    
    # actions legend 
    # 0: N 
    # 1: E 
    # 2: S 
    # 3: W 
    self.rows = 10
    self.cols = 10

    base_cost = 1
    large_cost = 5
    wall_cost = 2
    
    ## Because row and col indexes start from 0, 
    ## "a * self.rows + b" means (row_index=a, col_index=b)

    ## slow_states is a 1D array storing the indices of slow grids
    slow_states = [1 * self.rows + 2, 2 * self.rows + 2, 
                3 * self.rows + 2, 4 * self.rows + 2,
                5 * self.rows + 4, 5 * self.rows + 5, 
                5 * self.rows + 6, 5 * self.rows + 7, 
                5 * self.rows + 8]  # Actually these should be self.cols, but since cols==rows it is also OK
    goal_state = 2 * self.rows + 4 
                
    ## P is a 2D array with shape (rows*cols) * 4
    ## (rows*cols) is for the grids (states) and 4 is for the 4 actions
    ## each element of this 2D array is a list []
    ## that is to be appended with probability, next_state, and cost

    P = np.zeros((self.rows * self.cols,4), dtype=(list))

    # TODO: Fill out the transition matrix P
    # P is a n x m matrix whose elements are a list of 2 tuples, the contents of 
    # each tuple is (probability, next_state, cost)
    #
    # The entry for the goal state has already been completed for you. You may 
    # find the convenience functions map_row_col_to_state() and 
    # map_state_to_row_col() helpful but are not required to use them.

    ## Filling the P matrix
    # goal state

    # Prepare for E and W wall-hit detection
    westend = []
    eastend = []
    for i in range(self.rows):
        westend.append(i * self.rows + 0)
        eastend.append(i * self.rows + 9)
    # print(westend)
    # print('\n')
    # print(eastend)
    # print('\n')

    for s in range(self.rows * self.cols):
        if(s == goal_state):
            # Taking any action at the goal state will stay at the goal state
            P[s][0] = [(1.0, s, 0),
                    (0.0, 0, 0)]
            P[s][1] = [(1.0, s, 0),
                    (0.0, 0, 0)]
            P[s][2] = [(1.0, s, 0),
                    (0.0, 0, 0)]
            P[s][3] = [(1.0, s, 0),
                    (0.0, 0, 0)]
            continue
        # STUDENT CODE HERE
        
        # from grey areas, next step will never hit the walls
        # if(s in slow_states):  
        #     P[s][0] = [(slip_prob, s, large_cost + base_cost),
        #             (1.0 - slip_prob, s - self.cols, large_cost + base_cost)]
        #     P[s][1] = [(slip_prob, s, large_cost + base_cost),
        #             (1.0 - slip_prob, s + 1, large_cost + base_cost)]
        #     P[s][2] = [(slip_prob, s, large_cost + base_cost),
        #             (1.0 - slip_prob, s + self.cols, large_cost + base_cost)]
        #     P[s][3] = [(slip_prob, s, large_cost + base_cost),
        #             (1.0 - slip_prob, s - 1, large_cost + base_cost)]
        #     continue
        if(s in slow_states):  
            P[s][0] = [(slip_prob, s, large_cost),
                    (1.0 - slip_prob, s - self.cols, large_cost)]
            P[s][1] = [(slip_prob, s, large_cost),
                    (1.0 - slip_prob, s + 1, large_cost)]
            P[s][2] = [(slip_prob, s, large_cost),
                    (1.0 - slip_prob, s + self.cols, large_cost)]
            P[s][3] = [(slip_prob, s, large_cost),
                    (1.0 - slip_prob, s - 1, large_cost)]
            continue

        # normal conditions
        P[s][0] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s - self.cols, base_cost)]
        P[s][1] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s + 1, base_cost)]
        P[s][2] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s + self.cols, base_cost)]
        P[s][3] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s - 1, base_cost)]

        # N and S wall-hit decetion
        # if(s <= self.cols - 1):                                     # hit upper/north wall
        #     P[s][0] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s, base_cost + wall_cost)]
        # if(s >= (self.cols - 1) * self.rows):                       # hit lower/south wall
        #     P[s][2] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s, base_cost + wall_cost)]
        # # E and W wall-hit detection
        # if(s in eastend):                                           # hit right/east wall
        #     P[s][1] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s, base_cost + wall_cost)]
        # if(s in westend):                                           # hit left/west wall
        #     P[s][3] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s, base_cost + wall_cost)]
        if(s <= self.cols - 1):                                     # hit upper/north wall
            P[s][0] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s, wall_cost)]
        if(s >= (self.cols - 1) * self.rows):                       # hit lower/south wall
            P[s][2] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s, wall_cost)]
        # E and W wall-hit detection
        if(s in eastend):                                           # hit right/east wall
            P[s][1] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s, wall_cost)]
        if(s in westend):                                           # hit left/west wall
            P[s][3] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s, wall_cost)]

    ### For debug only ###
    # COST = np.zeros((self.rows, self.cols))
    # for ACTION in range(4):
    #     print('\n')
    #     for i in range(self.rows):
    #         for j in range(self.cols):
    #             COST[i][j] = P[i * self.rows + j][ACTION][1][2]
    #     print(COST)
    #     print('\n')

    ## Finish filling matrix P

    self.P = P


  def map_row_col_to_state(self, row, col):
    return row * self.rows + col


  def map_state_to_row_col(self, state):
    return state // self.cols, np.mod(state, self.cols)


  def eval_action(self, state, action):
    row, col = self.map_state_to_row_col(state)
    if action < 0 or action > 3:
        raise ValueError('Not a valid action')
    if row < 0 or row >= self.rows:
        raise ValueError('Row out of bounds')
    if col < 0 or col >= self.cols:
        raise ValueError('Col out of bounds')
    return self.P[state, action]
