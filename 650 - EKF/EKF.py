import numpy as np
import matplotlib.pyplot as plt


class system:
    def __init__(self, a):
        self.a = a  # dynamics parameter, to be estimated by EKF

    def dynamics(self, xk):
        epsilon_k = np.random.normal(0, 1)
        x_kp1 = self.a * xk + epsilon_k
        return x_kp1

    def observation(self, xk):
        nu_k = np.random.normal(0, np.sqrt(0.5))
        yk = np.sqrt(xk ** 2 + 1) + nu_k
        return yk

    def simulate(self, x0, N=100):
        x = np.zeros(N)
        y = np.zeros(N)
        x[0] = x0
        y[0] = self.observation(x0)
        for k in range(1, N):
            x[k] = self.dynamics(x[k - 1])
            y[k] = self.observation(x[k])
        return x, y

    def EKF(self, obs):
        Q = 0.5
        R = np.array([[1, 0], [0, 0.01]])

        N = len(obs)

        # Initial guesses
        x_k_k = np.ones(N+1)
        a_k_k = -10 * np.ones(N+1)
        # s_k_k = np.array([[x_k_k[0]], [a_k_k[0]]])
        Sigma_k_k = np.array([[2, 0], [0, 1]])
        cov_x = []
        cov_a = []

        for k in range(N):
            # Propagate nonlinear dynamics
            x_kp1_k = a_k_k[k] * x_k_k[k]  # mean x
            a_kp1_k = a_k_k[k]  # mean a
            s_kp1_k = np.array([[x_kp1_k], [a_kp1_k]])

            # Linearize dynamics
            J_k = np.array([[a_k_k[k], x_k_k[k]], [0, 1]])

            # Compute covariance
            Sigma_kp1_k = J_k @ Sigma_k_k @ J_k.T + R

            # Linearize observation
            C_k1 = x_kp1_k / np.sqrt(x_kp1_k ** 2 + 1)
            C_k = np.array([[C_k1, 0]])

            # Compute Kalman gain
            to_be_inv = C_k @ Sigma_kp1_k @ C_k.T + Q
            inv = np.linalg.inv(to_be_inv)
            K_k = Sigma_kp1_k @ C_k.T @ inv

            # Incorporate the observation
            obs_kp1 = obs[k]
            obs_hat_kp1 = np.sqrt(x_kp1_k ** 2 + 1)
            s_kp1_kp1 = s_kp1_k + K_k.reshape((2, 1)) @ ((obs_kp1 - obs_hat_kp1).reshape((-1, 1)))
            Sigma_kp1_kp1 = (np.eye(2) - K_k @ C_k) @ Sigma_kp1_k

            # Update iteration
            Sigma_k_k = Sigma_kp1_kp1
            s_k_k = s_kp1_kp1
            x_k_k[k+1] = s_k_k[0, 0]
            a_k_k[k+1] = s_k_k[1, 0]
            cov_x.append(Sigma_k_k[0,0])
            cov_a.append(Sigma_k_k[1,1])

        return x_k_k[1:], cov_x, cov_a, a_k_k[1:]


if __name__ == "__main__":

    a = -1  # true value of a
    sys = system(a)
    x0 = np.random.normal(1, np.sqrt(2))  # x0 N(1,2)
    x, D = sys.simulate(x0)  # we only know the observation

    # Plot x
    plt.figure()
    plt.plot(x)
    plt.xlabel("Iterations")
    plt.ylabel("$x_k$")
    plt.title("Real state $x_k$")

    # Plot D
    plt.figure()
    plt.plot(D)
    plt.xlabel("Iterations")
    plt.ylabel("$y_k$")
    plt.title("Observation $y_k$")

    # Use EKF
    x_arr, cov_x, cov_a, a_arr = sys.EKF(D)

    # Plot estimated x
    plt.figure()
    plt.plot(x_arr)
    plt.xlabel("Iterations")
    plt.ylabel("$\hat{x}_k$")
    plt.title("Estimated state $\hat{x}_k$")

    # Plot estimated cov
    plt.figure()
    plt.plot(cov_a)
    plt.xlabel("Iterations")
    plt.ylabel("$\Sigma$")
    plt.title("Covariance of $a$")

    plt.figure()
    plt.plot(cov_x)
    plt.xlabel("Iterations")
    plt.ylabel("$\Sigma$")
    plt.title("Covariance of $x$")

    # Plot estimated a
    plt.figure()
    plt.plot(a_arr)
    plt.plot(a_arr + cov_a)
    plt.plot(a_arr - cov_a)
    plt.plot(-np.ones(100))
    plt.legend(["$\hat{a}_k$", "$\mu+\sigma$", "$\mu-\sigma$", "$-1$"])
    plt.xlabel("Iterations")
    plt.ylabel("$\hat{a}_k$")
    plt.title("Estimated system parameter $\hat{a}_k$")
    plt.show()
