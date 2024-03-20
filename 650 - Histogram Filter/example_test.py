import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    print("belief_states: \n", belief_states)
    print(belief_states.shape)


    ### Test your code here

    n, m = np.array(cmap).shape  # 20, 20

    # belief is initialized as a uniform distribution over all the states
    belief = np.ones((n, m)) / (m * n)

    # create an instance of the HistogramFilter class
    hf = HistogramFilter()

    # run the histogram filter
    for i in range(len(actions)):
        # the histogram filter is run for each action-observation pair
        # it only accepts one action and one observation at a time
        # it also uses belief from the previous step
        belief = hf.histogram_filter(cmap, belief, actions[i], observations[i])
        # print("belief: \n", belief)

        # find the most likely state
        most_likely_state = np.unravel_index(np.argmax(belief, axis=None), belief.shape)
        print("most_likely_state: ", most_likely_state)

