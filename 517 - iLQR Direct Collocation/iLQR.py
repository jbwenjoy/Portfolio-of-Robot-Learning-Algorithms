import numpy as np
from scipy.signal import cont2discrete
from typing import List, Tuple
import quad_sim


class iLQR(object):

    def __init__(self, x_goal: np.ndarray, N: int, dt: float, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray):
        """
        Constructor for the iLQR solver
        :param N: iLQR horizon
        :param dt: timestep
        :param Q: weights for running cost on state
        :param R: weights for running cost on input
        :param Qf: weights for terminal cost on input
        """

        # Quadrotor dynamics parameters
        self.m = 1
        self.a = 0.25
        self.I = 0.0625
        self.nx = 6
        self.nu = 2

        # iLQR constants
        self.N = N
        self.dt = dt

        # Solver parameters
        self.alpha = 1        
        self.max_iter = 1e3
        self.tol = 1e-4

        # target state
        self.x_goal = x_goal
        self.u_goal = 0.5 * 9.81 * np.ones((2,))

        # Cost terms
        self.Q = Q
        self.R = R
        self.Qf = Qf


    def total_cost(self, xx, uu):
        J = sum([self.running_cost(xx[k], uu[k]) for k in range(self.N - 1)])
        return J + self.terminal_cost(xx[-1])


    def get_linearized_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: quadrotor state
        :param u: input
        :return: A and B, the linearized continuous quadrotor dynamics about some state x
        """
        m = self.m
        a = self.a
        I = self.I
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, -np.cos(x[2]) * (u[0] + u[1]) / m, 0, 0, 0],
                      [0, 0, -np.sin(x[2]) * (u[0] + u[1]) / m, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                      [0, 0],
                      [0, 0],
                      [-np.sin(x[2]) / m, -np.sin(x[2]) / m],
                      [np.cos(x[2]) / m, np.cos(x[2]) / m],
                      [a / I, -a / I]])

        return A, B


    def get_linearized_discrete_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: state
        :param u: input
        :return: the discrete linearized dynamics matrices, A, B as a tuple
        """
        A, B = self.get_linearized_dynamics(x, u)
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0],))
        [Ad, Bd, _, _, _] = cont2discrete((A, B, C, D), self.dt)
        return Ad, Bd


    def running_cost(self, xk: np.ndarray, uk: np.ndarray) -> float:
        """
        running_cost is the "l(x_k, u_k)" in homework
        :param xk: state
        :param uk: input
        :return: l(xk, uk), the running cost incurred by xk, uk
        """

        # Standard LQR cost on the goal state, @ is matrix multiplication
        # xk - self.x_goal is \delta_x, u the same
        lqr_cost = 0.5 * ((xk - self.x_goal).T @ self.Q @ (xk - self.x_goal) +
                          (uk - self.u_goal).T @ self.R @ (uk - self.u_goal))

        return lqr_cost


    def grad_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: [∂l/∂xᵀ, ∂l/∂uᵀ]ᵀ, evaluated at xk, uk
        """
        grad = np.zeros((8,))

        #TODO: Compute the gradient
        # grad x = 0.5 * (Q + Q.T) @ x, col vec with 6 elements
        # grad u = 0.5 * (R + R.T) @ u, col vec with 2 elements
        # grad is col vec with 8 elements
        grad_x = 0.5 * (self.Q + self.Q.T) @ (xk - self.x_goal)  # col vec with 6 elements
        grad_u = 0.5 * (self.R + self.R.T) @ (uk - self.u_goal)  # col vec with 2 elements
        grad = np.concatenate((grad_x, grad_u))
        
        return grad


    def hess_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: The hessian of the running cost
        [[∂²l/∂x², ∂²l/∂x∂u],
         [∂²l/∂u∂x, ∂²l/∂u²]], evaluated at xk, uk
        """
        H = np.zeros((self.nx + self.nu, self.nx + self.nu))

        # TODO: Compute the hessian

        H_x = 0.5 * (self.Q + self.Q.T)
        H_u = 0.5 * (self.R + self.R.T)

        lenx = len(H_x)
        lenu = len(H_u)
        H_lft = np.vstack( ( H_x, np.zeros((lenu, lenx)) ) )
        H_rgt = np.vstack( ( np.zeros((lenx, lenu)), H_u ) )
        H = np.hstack((H_lft, H_rgt))

        return H


    def terminal_cost(self, xf: np.ndarray) -> float:
        """
        :param xf: state
        :return: Lf(xf), the running cost incurred by xf
        """
        return 0.5*(xf - self.x_goal).T @ self.Qf @ (xf - self.x_goal)


    def grad_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂Lf/∂xf
        """

        grad = np.zeros((self.nx))

        # TODO: Compute the gradient

        grad = 0.5 * (self.Qf + self.Qf.T) @ (xf - self.x_goal)  # col vec with 6 elements

        return grad


    def hess_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂²Lf/∂xf²
        """ 

        H = np.zeros((self.nx, self.nx))

        # TODO: Compute H

        H = 0.5 * (self.Qf + self.Qf.T)

        return H


    def forward_pass(self, xx: List[np.ndarray], uu: List[np.ndarray], dd: List[np.ndarray], KK: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: list of states, should be length N, (every element contains 6 states)
        :param uu: list of inputs, should be length N-1, (every element contains 2 inputs)
        :param dd: list of "feed-forward" components of iLQR update, should be length N-1
        :param KK: list of "Feedback" LQR gain components of iLQR update, should be length N-1
        :return: A tuple (xx, uu) containing the updated state and input
                 trajectories after applying the iLQR forward pass
        """

        xtraj = [np.zeros((self.nx,))] * self.N
        utraj = [np.zeros((self.nu,))] * (self.N - 1)
        xtraj[0] = xx[0]

        # TODO: compute forward pass

        # self.alpha is 1
        for k in range(0, self.N - 1):
            delta_x = xtraj[k] - xx[k]
            utraj[k] = uu[k] + KK[k] @ delta_x + self.alpha * dd[k]
            xtraj[k + 1] = quad_sim.F(xtraj[k], utraj[k], self.dt)
        
        return xtraj, utraj


    def backward_pass(self,  xx: List[np.ndarray], uu: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: state trajectory guess, should be length N, (every element contains 6 states, the last element is xf)
        :param uu: input trajectory guess, should be length N-1, (every element contains 2 inputs)
        :return: KK and dd, the feedback and feedforward components of the iLQR update
        """
        N = self.N
        dd = [np.zeros((self.nu,))] * (self.N - 1)
        KK = [np.zeros((self.nu, self.nx))] * (self.N - 1)

        # TODO: compute backward pass

        # Ad, Bd = get_linearized_discrete_dynamics(self, xx[0], uu[0])

        # calculate termin
        xf = xx[self.N - 1]
        # print(xf.size)
        term_cost = self.terminal_cost(xf)
        term_grad = self.grad_terminal_cost(xf)
        term_hess = self.hess_terminal_cost(xf)

        H_kp1 = term_hess
        g_kp1 = term_grad
        for k in range(self.N - 2, -1, -1):  # N-2 to 0, N-1 steps, excluding xf

            xk = xx[k]
            uk = uu[k]
            A_k, B_k = self.get_linearized_discrete_dynamics(xk, uk)
            
            # we should call func grad_running_cost(), but I don't want to divide the matrices
            # so I move the code in the function to here
            # same for the func hess_running_cost()
            # l_ = self.grad_running_cost(xk, uk)
            l_x = 0.5 * (self.Q + self.Q.T) @ (xk - self.x_goal)
            l_u = 0.5 * (self.R + self.R.T) @ (uk - self.u_goal)
            
            # l__ = self.hess_running_cost(xk, uk)
            l_xx = 0.5 * (self.Q + self.Q.T)
            l_uu = 0.5 * (self.R + self.R.T)
            l_ux = np.zeros((len(l_uu), len(l_xx)))
            l_xu = l_ux.T

            # DEBUG ONLY
            # if k >= self.N - 3:
            #     print("\n grad 1")
            #     print(l_)
            #     print("\n grad 2")
            #     print(l_x)
            #     print(l_u)

            #     print("\n hess 1")
            #     print(l__)
            #     print("\n hess 2")
            #     print(l_xx)
            #     print(l_xu)
            #     print(l_ux)
            #     print(l_uu)

            KK[k] = -1 * np.linalg.inv(l_uu + B_k.T @ H_kp1 @ B_k) @ (l_ux + B_k.T @ H_kp1 @ A_k)
            dd[k] = -1 * np.linalg.inv(l_uu + B_k.T @ H_kp1 @ B_k) @ (l_u + B_k.T @ g_kp1)
            
            Q_x = l_x + A_k.T @ g_kp1
            Q_xx = l_xx + A_k.T @ H_kp1 @ A_k
            Q_uu = l_uu + B_k.T @ H_kp1 @ B_k

            H_kp1 = Q_xx - KK[k].T @ Q_uu @ KK[k]
            g_kp1 = Q_x - KK[k].T @ Q_uu @ dd[k]
        
        # print(KK[0])
        return dd, KK


    def calculate_optimal_trajectory(self, x: np.ndarray, uu_guess: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        """
        Calculate the optimal trajectory using iLQR from a given initial condition x,
        with an initial input sequence guess uu
        :param x: initial state
        :param uu_guess: initial guess at input trajectory
        :return: xx, uu, KK, the input and state trajectory and associated sequence of LQR gains
        """
        assert (len(uu_guess) == self.N - 1)

        # Get an initial, dynamically consistent guess for xx by simulating the quadrotor
        # xx is the x sequence calculated from uu_guess
        xx = [x]
        for k in range(self.N-1):
            xx.append(quad_sim.F(xx[k], uu_guess[k], self.dt))

        Jprev = np.inf
        Jnext = self.total_cost(xx, uu_guess)
        uu = uu_guess
        KK = None

        i = 0
        print(f'cost: {Jnext}')
        # self.tol = 1e-4 is solver parameter, stop solving when error is below this threshold
        # self.max_iter = 1e3
        while np.abs(Jprev - Jnext) > self.tol and i < self.max_iter:
            dd, KK = self.backward_pass(xx, uu)
            xx, uu = self.forward_pass(xx, uu, dd, KK)

            Jprev = Jnext
            Jnext = self.total_cost(xx, uu)
            print(f'cost: {Jnext}')
            i += 1
        print(f'Converged to cost {Jnext}')
        return xx, uu, KK
