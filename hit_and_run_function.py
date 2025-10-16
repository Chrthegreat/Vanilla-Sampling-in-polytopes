import numpy as np
from utils import sample_ball, find_interior_point  # import from utils.py

# ----------------------------------------
# Hit-and-Run sampler
# ----------------------------------------
def hit_and_run(A, b, n_steps=10000):
    """
    Perform Hit-and-Run sampling inside the polytope {x | A x <= b}.

    Parameters:
        A, b : define the polytope A x <= b
        n_steps : number of steps to perform

    Returns:
        samples : array of shape (n_steps, d)
    """
    # Dimension of input polytope
    d = A.shape[1]
    # Find strictly interior starting point
    current = find_interior_point(A, b)
    # This matrix will save the samples. 
    samples = [current.copy()]

    for _ in range(1, n_steps):
        # Random direction on unit circle
        direction = sample_ball(d=d, r=1.0, n=1)[0]
        direction /= np.linalg.norm(direction)

        #  Find intersection of line {current + t*direction} with Ax <= b
        #  For each constraint: a_i·(current + t u) <= b_i
        #  => (a_i·u) t <= b_i - a_i·current
        t_min, t_max = -np.inf, np.inf

        #zip is a nice function that pairs each row of A with its row in b.
    	#Thus, this loop iterates over each inequality. 
        for a_i, b_i in zip(A, b):
            a_dot_u = np.dot(a_i, direction) # Find the product a^T*u. 
            rhs = b_i - np.dot(a_i, current) # Find the right hand side (rhs)

            # We consider cases.
            if abs(a_dot_u) < 1e-12:
                # This is the zero case, direction is almost parallel to this constraint
                if rhs < 0:
                    # Entire line is infeasible
                    t_min, t_max = 0, -1  # empty interval
                    break
                else:
                    continue # no bound from this constraint

            bound = rhs / a_dot_u # The result of the inequality

            if a_dot_u > 0:
                # t <= bound
                t_max = min(t_max, bound) # Update max bound
            else:
                # t >= bound
                t_min = max(t_min, bound) # Update min bounds

        if t_min > t_max:
            continue  # skip if no feasible move

        # Sample t uniformly in [t_min, t_max]
        t = np.random.uniform(t_min, t_max)

        # Move to new point
        current = current + t * direction
        samples.append(current.copy())

    samples = np.array(samples)
    return samples



