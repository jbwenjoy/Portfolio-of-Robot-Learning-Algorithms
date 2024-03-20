import numpy as np
from math import *
from grid_world import *
import sys

import matplotlib.pyplot as plt


def plot_value_function_and_optimal_policy(world, V, u_opt):
    plt.clf()
    v_plot = plt.imshow(V, interpolation='nearest')
    colorbar = plt.colorbar()
    colorbar.set_label("Value function")
    plt.xlabel("Column")
    plt.ylabel("Row")
    arrow_length = 0.25
    for row in range(world.rows):
        for col in range(world.cols):
            if u_opt[row, col] == 0: #N
                plt.arrow(col, row, 0, -arrow_length, head_width=0.1)
            elif u_opt[row, col] == 1: #E
                plt.arrow(col, row, arrow_length, 0, head_width=0.1)
            elif u_opt[row, col] == 2: #S
                plt.arrow(col, row, 0, arrow_length, head_width=0.1)
            elif u_opt[row, col] == 3: #W
                plt.arrow(col, row, -arrow_length, 0, head_width=0.1)
            else:
                raise ValueError("Invalid action")
    plt.savefig('value_function.png', dpi=240)
    plt.show()


def cal_exp_cost(slip_prob, go_prob, slip_cost, go_cost):
    # Example for P: P[s][0] = [(slip_prob, s, base_cost), (1.0 - slip_prob, s - self.cols, base_cost)]
    if slip_prob == 0:
        return go_prob * go_cost
    if slip_prob == 1:
        return slip_prob * slip_cost
    return slip_cost * go_prob * slip_prob / (1 - slip_prob * slip_prob) + go_cost * go_prob / (1 - slip_prob)


def value_iteration(world, threshold, gamma, plotting=True):
    V = np.zeros((world.rows, world.cols))
    u_opt = np.zeros((world.rows, world.cols))
    grid_x, grid_y = np.meshgrid(np.arange(0, world.rows, 1), 
                                np.arange(0, world.cols, 1))
    delta = 10.0
    # gamma = 1.0
    fig = plt.figure("Gridworld")
    
    # STUDENT CODE: calculate V and the optimal policy u using value iteration

    ### CODE START ###
    
    itercount = 0
    goal_row = 2
    goal_col = 4
    
    while (itercount < 10000):
        itercount += 1
        max_delta = 0
        V_next = np.zeros((world.rows, world.cols))
        max_delta_grid = np.array([10, 10])

        for i in range(world.rows):
            for j in range(world.cols):
                
                s = world.map_row_col_to_state(i,j)

                minV = 65535
                for act in range(4):
                    # get the current state and next state and their coords
                    next_s = world.P[s][act][1][1]
                    curr_s = world.P[s][act][0][1]
                    curr_i, curr_j = world.map_state_to_row_col(curr_s)
                    next_i, next_j = world.map_state_to_row_col(next_s)
                    
                    # Expected w_e for moving to the next state
                    w_e = world.P[s][act][1][0] * world.P[s][act][1][2] + world.P[s][act][0][0] * world.P[s][act][0][2]  

                    # Expected V, note that V itself also need to be expected, adding the slip and move cases multiplied by weight
                    V_e = gamma * (world.P[s][act][1][0] * V[next_i][next_j] + world.P[s][act][0][0] * V[curr_i][curr_j]) + w_e
                    
                    # get the min V
                    if V_e < minV:
                        u_opt[i][j] = act
                        minV = V_e

                V_next[i][j] = minV

                # print("max_delta is ")
                # print(max_delta)
                # print(abs(V_next[i][j] - V[i][j]))
                # print("-----")
                # get the maximum error between this time and last time
                if max_delta < abs(V_next[i][j] - V[i][j]):
                    max_delta = abs(V_next[i][j] - V[i][j])
                    max_delta_grid[0] = i
                    max_delta_grid[1] = j
        print("Max delta grid =", max_delta_grid, " Delta =", max_delta)

        V = V_next
        
        # quit condition
        if max_delta < threshold and itercount > 5:
            print(itercount)
            break
    
    # print(itercount)

    ### CODE END ###

    if plotting:
        plot_value_function_and_optimal_policy(world, V, u_opt)

    ### FOR DEBUG ONLY ###
    print(V)

    return V, u_opt


if __name__=="__main__":
    world = GridWorld() 
    threshold = 0.0001
    gamma = 0.9
    # value_iteration(world, threshold, gamma, False)
    value_iteration(world, threshold, gamma, True)
