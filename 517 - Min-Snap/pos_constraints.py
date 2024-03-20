import numpy as np

def Ab_i1(i, n, d, dt_i, w_i, w_ip1):
    '''
    Ab_i1(i, n, d, dt_i, w_i, w_ip1) computes the linear equality constraint
    constants that require the ith polynomial to meet waypoints w_i and w_{i+1}
    at it's endpoints.
    Parameters:
        i - index of the polynomial.
        n - total number of polynomials.
        d - the number of terms in each polynomial.
        dt_i - Delta t_i, duration of the ith polynomial.
        w_i - waypoint at the start of the ith polynomial.
        w_ip1 - w_{i+1}, waypoint at the end of the ith polynomial.
    Outputs
        A_i1 - A matrix from linear equality constraint A_i1 v = b_i1
        b_i1 - b vector from linear equality constraint A_i1 v = b_i1
    '''

    # A * v = B
    # V = [sigma_0_0; sigma_0_1; ...; sigma_0_(d-1);   sigma_1_0; sigma_1_1; ...; sigma_1_(d-1);   ...;   sigma_n_(d-1)]
    # Note that because this is 2D, each sigma_i_j is a 2D vector
    # therefore A_i1 has 2*d*n cols
    # similarly, there are 4 rows instead of 2 rows
    # also w_i are 2D vectors as they are the coords of the control points

    A_i1 = np.zeros((4, 2*d*n))
    b_i1 = np.zeros((4, 1))

    # TODO: fill in values for A_i1 and b_i1

    # ploynomial index i start from 0, up to n-1

    b_i1[0][0] = w_i[0]
    b_i1[1][0] = w_i[1]
    b_i1[2][0] = w_ip1[0]
    b_i1[3][0] = w_ip1[1]
    
    # w_i = sigma_i_0
    A_i1[0][2*i*d + 0] = 1
    A_i1[1][2*i*d + 1] = 1
    
    # w_(i+1) = sigma_i_0 + sigma_i_1 * delta_t_i + sigma_i_2 * delta_t_i ^ 2 + ... + sigma_i_(d-1) * delta_t_i ^ (d-1)
    # A_i1[2][2*i*d + 1] = pow(dt_i, 0)
    # A_i1[3][2*i*d + 2] = pow(dt_i, 0)
    # A_i1[2][2*i*d + 3] = pow(dt_i, 1)
    # A_i1[3][2*i*d + 4] = pow(dt_i, 1)
    # ...
    # A_i1[2][2*i*d + 2*a+1] = pow(d_ti, a)   # a = 0, 1, ..., d-1  a represents sigma_i_a
    # A_i1[3][2*i*d + 2*a+2] = pow(d_ti, a)   # a = 0, 1, ..., d-1
    for a in range(d):
        A_i1[2][2*i*d + 2*a+0] = pow(dt_i, a)
        A_i1[3][2*i*d + 2*a+1] = pow(dt_i, a)

    return A_i1, b_i1