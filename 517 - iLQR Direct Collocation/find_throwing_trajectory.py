import matplotlib.pyplot as plt
import numpy as np
import importlib

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, 
    MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve
)

import kinematic_constraints
import dynamics_constraints
importlib.reload(kinematic_constraints)
importlib.reload(dynamics_constraints)
from kinematic_constraints import (
    AddFinalLandingPositionConstraint
)
from dynamics_constraints import (
    AddCollocationConstraints,
    EvaluateDynamics
)


def find_throwing_trajectory(N, initial_state, final_configuration, distance, tf):
    '''
    Parameters:
        N - number of knot points
        initial_state - starting configuration
        distance - target distance to throw the ball

    '''

    builder = DiagramBuilder()
    plant = builder.AddSystem(MultibodyPlant(0.0))
    file_name = "planar_arm.urdf"
    Parser(plant=plant).AddModels(file_name)
    plant.Finalize()
    planar_arm = plant.ToAutoDiffXd()

    plant_context = plant.CreateDefaultContext()
    context = planar_arm.CreateDefaultContext()

    # Dimensions specific to the planar_arm
    n_q = planar_arm.num_positions()
    n_v = planar_arm.num_velocities()
    n_x = n_q + n_v
    n_u = planar_arm.num_actuators()

    # Store the actuator limits here
    effort_limits = np.zeros(n_u)
    for act_idx in range(n_u):
        effort_limits[act_idx] = \
            planar_arm.get_joint_actuator(JointActuatorIndex(act_idx)).effort_limit()
    joint_limits = np.pi * np.ones(n_q)
    vel_limits = 15 * np.ones(n_v)

    # Create the mathematical program
    prog = MathematicalProgram()
    x = np.zeros((N, n_x), dtype="object")
    u = np.zeros((N, n_u), dtype="object")
    for i in range(N):
        x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
        u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

    t_land = prog.NewContinuousVariables(1, "t_land")

    t0 = 0.0
    timesteps = np.linspace(t0, tf, N)
    x0 = x[0]
    xf = x[-1]

    # print("N = ", N)
    # print("t0 = ", t0)
    # print("tf = ", tf)
    # print("t1 = ", timesteps[1])
    
    # DO NOT MODIFY THE LINES ABOVE


    # Add the kinematic constraints (initial state, final state)
    # TODO: 3(a) Add constraints on the initial state
    # var initial_state is passed into this func
    prog.AddLinearEqualityConstraint(x0 - initial_state, np.zeros(4))

    # Add the kinematic constraint on the final state
    AddFinalLandingPositionConstraint(prog, xf, distance, t_land)

    # Add the collocation aka dynamics constraints
    AddCollocationConstraints(prog, planar_arm, context, N, x, u, timesteps)


    # TODO: Add the cost function here
    dt = (tf - t0) / (N - 1)
    # print("dt =", dt)
    cost = 0.0
    for i in range(N - 1):
        cost += (np.dot(u[i].T, u[i]) + np.dot(u[i + 1].T, u[i + 1]))
        # print(i, cost)
    cost *= dt / 2
    # print(cost)
    prog.AddQuadraticCost(cost)

    
    # TODO: Add bounding box constraints on the inputs and qdot
    # print(x)
    # print(N)
    # print(n_x)
    # print(u)
    # print(effort_limits)
    e_limits = effort_limits
    for i in range(N - 1):
        e_limits = np.vstack((e_limits, effort_limits))
    prog.AddBoundingBoxConstraint(-1 * e_limits, e_limits, u)
    
    x_limits_single = np.hstack((joint_limits, vel_limits))
    x_limits = x_limits_single
    for i in range(N - 1):
        x_limits = np.vstack((x_limits, x_limits_single))
    # print(x_limits.shape)
    # print(x_limits)
    prog.AddBoundingBoxConstraint(-x_limits, x_limits, x)


    # TODO: give the solver an initial guess for x and u using prog.SetInitialGuess(var, value)

    # u_guess_single = [0, 0]
    # u_guess = u_guess_single
    # for i in range(N - 1):
    #     u_guess = np.vstack((u_guess, u_guess_single))
    u_guess = [[ 4.50261684, -8.40260512],
        [-2.68785316, -6.4507012 ],
        [-9.46519596, -2.42914562],
        [ 3.87352065,  6.16395681],
        [ 8.6897844,  -6.67714803]]
    prog.SetInitialGuess(u, u_guess)

    # x_guess_single = [0, 0, 0, 0]
    # x_guess_last = x_limits_single
    # delta_x_guess = (x_guess_last - x_guess_single) / (N - 1)
    # x_guess = x_guess_single
    # for i in range(N - 1):
    #     x_guess_single += delta_x_guess
    #     x_guess = np.vstack((x_guess, x_guess_single))
    x_guess = [[ 1.34511467e-15,  0.00000000e+00, -6.25366352e-15, -1.37549614e-14],
               [-1.67794564e+00, -2.63666327e+00, -3.56141814e+00,  3.77791464e+00],
               [-3.14159265e+00,  1.68355568e+00,  4.05540190e-01,  4.00902619e+00],
               [ 1.61485382e-02,  3.05002646e+00, -2.13949327e-02, -1.69572125e+00],
               [ 3.14156133e+00,  3.30819523e-05,  1.27312149e+01, -1.45007651e+01]]
    prog.SetInitialGuess(x, x_guess)


    #DO NOT MODIFY THE LINES BELOW 
    # Set up solver
    result = Solve(prog)
    
    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    t_land_sol = result.GetSolution(t_land)

    print('optimal cost: ', result.get_optimal_cost())
    print('x_sol: ', x_sol)
    print('u_sol: ', u_sol)
    print('t_land: ', t_land_sol)

    print(result.get_solution_result())

    # Reconstruct the trajectory
    xdot_sol = np.zeros(x_sol.shape)
    for i in range(N):
        xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i])
    
    x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
    u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

    return x_traj, u_traj, prog, prog.GetInitialGuess(x), prog.GetInitialGuess(u)


if __name__ == '__main__':
    N = 5
    initial_state = np.zeros(4)
    final_configuration = np.array([np.pi, 0])
    tf = 3.0
    distance = 15.0
    x_traj, u_traj, prog, _, _ = find_throwing_trajectory(N, initial_state, final_configuration, distance, tf)
    