import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):
        '''
        Compute alpha_k(x_i) for all k and i
        :return: alpha, a matrix of size num_obs x num_states
        '''
        num_states = len(self.Transition)
        num_obs = len(self.Observations)
        # print("num_states: ", num_states)
        # print("num_obs: ", num_obs)
        # num_states = np.array(self.Transition).shape[0]
        # num_obs = np.array(self.Observations).shape[0]
        # print("num_states: ", num_states)
        # print("num_obs: ", num_obs)
        alpha = np.zeros((num_obs, num_states))  # alpha_k are row vectors in alpha

        # Compute alpha_1
        alpha[0, :] = self.Initial_distribution * self.Emission[:, self.Observations[0]]
        for i in range(1, num_obs):
            alpha[i, :] = self.Emission[:, self.Observations[i]] * (self.Transition.T @ alpha[i-1, :])

        # for i in range(num_obs):
        #     if i == 0:  # alpha_k, k = 1
        #         alpha = self.Initial_distribution * self.Emission[:, self.Observations[i]]  # alpha_1 is a col vector
        #     else:  # alpha_k, k = 2, 3, ..., T
        #         # Compute alpha_kp1
        #         # Sum alpha_k(x_j) * T(x_j, x) for all j
        #         alpha_kp1 = np.zeros((num_states, 1))
        #         sum_alpha_k_T = 0
        #         for j in range(num_states):  # alpha_kp1(x_j)
        #             for q in range(num_states):  # alpha_k(x_q), x_q -> x_j
        #                 sum_alpha_k_T += alpha[q, i-1] * self.Transition[q, j]
        #             # x_j -> y_i
        #             alpha_kp1[j][0] = self.Emission[j, self.Observations[i]] * sum_alpha_k_T
        #         # Concatenate as [alpha, alpha_k]
        #         alpha = np.concatenate((alpha, alpha_kp1), axis=1)

        # To pass the autograder, alpha need to be np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        return alpha
        # return np.zeros((np.array(self.Observations).shape[0], np.array(self.Transition).shape[0]))

    def backward(self):
        num_states = len(self.Transition)
        num_obs = len(self.Observations)
        beta = np.zeros((num_obs, num_states))  # beta_k are col vectors in beta

        # Compute beta_T
        beta[-1, :] = 1  # Initialize beta_T(x_i) = 1 for all i
        for i in range(num_obs-2, -1, -1):
            beta[i, :] = self.Transition @ (self.Emission[:, self.Observations[i+1]] * beta[i+1, :])

        # To pass the autograder, beta need to be np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        return beta
        # return np.zeros((np.array(self.Observations).shape[0], np.array(self.Transition).shape[0]))

    def gamma_comp(self, alpha, beta):

        '''
        Compute gamma_k(x_i) for all k and i
        gamma is the probability of each state given all the observations
        gamma_k(x) = alpha_k(x) * beta_k(x) / sum(alpha_t(x)), for all x
        sum(alpha_t(x)) = P(y_1, y_2, ..., y_t)
        :param alpha: num_states x num_obs
        :param beta: num_states x num_obs
        :return: gamma, a matrix of size num_states x num_obs
        '''
        # # As initially my code is computing alpha and beta as col vectors, we need to transpose them
        # alpha = alpha.T
        # beta = beta.T

        num_states = len(self.Transition)
        num_obs = len(self.Observations)
        gamma = np.zeros((num_obs, num_states))
        for i in range(num_obs):
            gamma[i, :] = alpha[i, :] * beta[i, :] / np.sum(alpha[i, :] * beta[i, :])

        # To pass the autograder, gamma need to be np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
        return gamma
        # return np.zeros((np.array(self.Observations).shape[0], np.array(self.Transition).shape[0]))

    def xi_comp(self, alpha, beta, gamma):
        # To pass the autograder, xi need to be
        # np.zeros((self.Observations.shape[0]-1, self.Transition.shape[0], self.Transition.shape[0]))
        xi = np.zeros((len(self.Observations)-1, len(self.Transition), len(self.Transition)))
        for k in range(len(self.Observations)-1):  # k = 0, 1, ..., T-2
            for i in range(len(self.Transition)):  # x_k = x
                for j in range(len(self.Transition)):  # x_k+1 = x'
                    xi[k, i, j] = alpha[k, i] * self.Transition[i, j] * self.Emission[j, self.Observations[k+1]] * beta[k+1, j]

        # Normalize xi
        sum_xi = np.sum(xi, axis=(1, 2))
        for k in range(len(self.Observations)-1):
            xi[k, :, :] = xi[k, :, :] / sum_xi[k]

        return xi


    def update(self, alpha, beta, gamma, xi):

        new_init_state = np.zeros_like(self.Initial_distribution)
        T_prime = np.zeros_like(self.Transition)
        M_prime = np.zeros_like(self.Emission)

        # Update initial state using pi' = gamma_1(x)
        for i in range(len(self.Initial_distribution)):
            new_init_state[i] = gamma[0, i]

        # Update transition matrix using xi and gamma
        # T_{x,x'}' = E{num of transitions from x to x'} / E{num of times the Markov chain is in state x}
        #           = sum_{k=1}^{t-1} xi_k(x,x') / sum_{k=1}^{t-1} gamma_k(x)
        # Note that index starts from 0 in the code
        for i in range(len(self.Transition)):
            for j in range(len(self.Transition)):
                T_prime[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])

        # Update emission matrix using gamma
        # M_{x,y}' = E{num of times in state x and emits y} / E{num of times in state x}
        #          = sum_{k=1}^{t} 1_{y_k = y} * gamma_k(x) / sum_{k=1}^{t} gamma_k(x)
        # Note that index starts from 0 in the code
        for i in range(len(self.Emission)):
            for j in range(len(self.Emission[0])):
                sum_nom = 0
                sum_den = 0
                for k in range(len(self.Observations)):
                    if self.Observations[k] == j:
                        sum_nom += gamma[k, i]
                    sum_den += gamma[k, i]
                M_prime[i, j] = sum_nom / sum_den

        return T_prime, M_prime, new_init_state


    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        # Compute P_original
        P_original = np.sum(alpha[-1, :])

        # Compute P_prime
        # Create a new HMM with the updated parameters
        hmm_ = HMM(self.Observations, T_prime, M_prime, new_init_state)
        alpha_ = hmm_.forward()
        P_prime = np.sum(alpha_[-1, :])

        return P_original, P_prime


if __name__ == "__main__":
    # Define HMM
    LA = 0
    NY = 1
    null = 2
    Observations = [null, LA, LA, null, NY, null, NY, NY, NY, null, NY, NY, NY, NY, NY, null, null, LA, LA, NY]  # (20,)
    Transition = np.array([[0.5, 0.5], [0.5, 0.5]])  # (2, 2)
    Emission = np.array([[0.4, 0.1, 0.5], [0.1, 0.5, 0.4]])  # (2, 3)
    Initial_distribution = np.array([0.5, 0.5])  # (2,)

    # Create HMM
    hmm = HMM(Observations, Transition, Emission, Initial_distribution)

    # Test forward
    alpha = hmm.forward()
    print("\nalpha: \n", alpha)

    # Test backward
    beta = hmm.backward()
    print("\nbeta: \n", beta)

    # Test gamma
    gamma = hmm.gamma_comp(alpha, beta)
    print("\ngamma: \n", gamma)
    for i in range(gamma.shape[0]):
        print("sum(gamma[:, ", i, "]): ", np.sum(gamma[i, :]))

    # Smooth sequence
    print("\nSmoothed sequence:")
    for i in range(gamma.shape[0]):
        if gamma[i, 0] > gamma[i, 1]:
            print("LA", end=" ")
        else:
            print("NY", end=" ")
    print("\n")

    # Test xi
    xi = hmm.xi_comp(alpha, beta, gamma)
    print("\nxi: \n", xi)

    # Compute updated HMM
    T_prime, M_prime, new_init_state = hmm.update(alpha, beta, gamma, xi)
    print("\nnew_init_state: \n", new_init_state)
    print("\nnew_transition: \n", T_prime)
    print("\nnew_emission: \n", M_prime)

    # Compute traj probability
    P_original, P_prime = hmm.trajectory_probability(alpha, beta, T_prime, M_prime, new_init_state)
    print("\nP_original: ", P_original)
    print("P_prime: ", P_prime)


