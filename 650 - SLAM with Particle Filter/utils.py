import numpy as np

cos, sin = np.cos, np.sin

# some lambda functions for typical calculations
get_so2 = lambda a: np.array([[cos(a), -sin(a)],
                              [sin(a), cos(a)]])

get_se2 = lambda a, v: np.array([[cos(a), -sin(a), v[0]],
                                 [sin(a), cos(a), v[1]],
                                 [0, 0, 1]])


def euler_to_so3(r, p, y):
    rx = np.array([[1, 0, 0], [0, cos(r), -sin(r)], [0, sin(r), cos(r)]])
    ry = np.array([[cos(p), 0, sin(p)], [0, 1, 0], [-sin(p), 0, cos(p)]])
    rz = np.array([[cos(y), -sin(y), 0], [sin(y), cos(y), 0], [0, 0, 1]])
    so3 = rz @ ry @ rx
    return so3


def euler_to_se3(r, p, y, v):
    so3 = euler_to_so3(r, p, y)
    se3 = np.vstack((np.hstack((so3, v.reshape(-1, 1))), np.array([0, 0, 0, 1])))
    return se3


make_homogeneous_coords_2d = lambda xy: np.vstack((xy, np.ones(xy.shape[1])))
make_homogeneous_coords_3d = lambda xyz: np.vstack((xyz, np.ones(xyz.shape[1])))


def smart_plus_2d(p1, p2):
    """
    See guidance.pdf
    p1, p2 are two poses (x1, y1, yaw1) and (x2, y2, yaw2)
    """
    R = get_so2(p1[2])
    t = p1[:2] + (R @ p2[:2])
    return np.array([t[0], t[1], p1[2] + p2[2]])


def smart_minus_2d(p2, p1):
    """
    See guidance.pdf
    p2, p1 (note the order) are two poses (x2, y2, yaw2) and (x1, y1, yaw1)
    """
    R = get_so2(p1[2])
    t = R.T @ (p2[:2] - p1[:2])
    return np.array([t[0], t[1], p2[2] - p1[2]])
