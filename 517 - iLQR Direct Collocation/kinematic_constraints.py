import numpy as np
from pydrake.autodiffutils import AutoDiffXd
# from scipy import integrate


def cos(theta):
    return AutoDiffXd.cos(theta)


def sin(theta):
    return AutoDiffXd.sin(theta)


def landing_constraint(var):
    '''
    Impose a constraint such that if the ball is released at final state xf, 
    it will land a distance d from the base of the robot 
    '''
    l = 1
    g = 9.81
    constraint_eval = np.zeros((3,), dtype=AutoDiffXd)
    q = var[:2]
    qdot = var[2:4]
    t_land = var[-1]
    pos = np.array([
        -l * sin(q[0]) - l * sin(q[0] + q[1]),
        -l * cos(q[0]) - l * cos(q[0] + q[1])
    ])
    vel = np.array([
        -l * qdot[1] * cos(q[0] + q[1]) + qdot[0] * (-l * cos(q[0]) - l * cos(q[0] + q[1])), 
         l * qdot[1] * sin(q[0] + q[1]) + qdot[0] * ( l * sin(q[0]) + l * sin(q[0] + q[1]))
    ])

    # TODO: 3(b) Express the landing constraint as a function of q, qdot, and t_land
       
    # constraint_eval[0]: Eq (23)
    constraint_eval[0] = pos[1] + vel[1] * t_land - 0.5 * g * t_land * t_land

    # constraint_eval[1]: Eq (24)
    constraint_eval[1] = pos[0] + vel[0] * t_land

    # constraint_eval[2]: Eq (26)
    constraint_eval[2] = pos[1]

    return constraint_eval


def AddFinalLandingPositionConstraint(prog, xf, d, t_land):

    # TODO: Add the landing distance equality constraint as a system of inequality constraints 
    # using prog.AddConstraint(landing_constraint, lb, ub, var) 
    
    lb = np.zeros(3)
    lb[0] = 0.0
    lb[1] = d
    lb[2] = 2.0

    ub = np.zeros(3)
    ub[0] = 0.0
    ub[1] = d + 100
    ub[2] = 2.0

    var = np.concatenate((xf, t_land))

    prog.AddConstraint(landing_constraint, lb, ub, var)

    # TODO: Add a constraint that t_land is positive
    prog.AddConstraint(t_land[0] >= 0.1)
    ...
