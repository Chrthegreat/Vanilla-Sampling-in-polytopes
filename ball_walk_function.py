import numpy as np
from utils import *

# ----------------------------------------
# Ball walk sampler
# ----------------------------------------
def ball_walk(A, b, n_steps=10000, r=0.1):
    """
    Perform a ball walk inside the polytope {x | A x <= b}.

    Parameters:
        A, b : define the polytope A x <= b
        n_steps : number of steps to perform
        r : radius of the proposal ball

    Returns:
        samples : array of shape (n_steps, d)
        acceptance_rate : fraction of accepted proposals
    """
    # Dimensions of input polytope
    d = A.shape[1]

    # Find strictly interior starting point using LP
    current = find_interior_point(A, b)

    samples = [current.copy()]
    n_accepted = 0

    for i in range(1, n_steps):
        step = sample_ball(d, r)[0] # sample in current ball
        proposal = current + step # find new coordinates

        # check whether new point satisfy the inequalities
        if np.all(A @ proposal <= b + 1e-12):
            current = proposal # accept
            n_accepted += 1 # count how many got accepted
        samples.append(current.copy()) # add sample to list

    samples = np.array(samples)
    acceptance_rate = n_accepted / n_steps

    return samples, acceptance_rate
