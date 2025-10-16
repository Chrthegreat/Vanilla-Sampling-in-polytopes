import numpy as np
from utils import find_interior_point  

# -----------------------------
# Utility functions
# -----------------------------
# def grad_f(x, x_start):
#     diff = x - x_start
#     return (2 * diff * np.exp(diff**2))
#     #return np.zeros_like(x)
    

def unit_normal(a):
    return a / np.linalg.norm(a)


def intersection_candidates(A, b, x, u, tol=1e-12):
    # Remember t = (b_i - aT_i* x0) / (aT_i * u)
    a_dot_u = A.dot(u)  # denominator 
    s = b - A.dot(x)    # numerator 
    t_candidates = np.full(A.shape[0], np.inf)  # Start with infinite t. No wall hit yet
    mask = a_dot_u > tol  # Create a mask that will ignore negative t. We move away from these borders.
    t_candidates[mask] = s[mask] / a_dot_u[mask]  # Apply the mask
    valid = (t_candidates > tol) & (t_candidates <= 1.0 + tol)  # We only care about solutions during our finite step from x to x+u i.e. for t:(0,1]
    if not np.any(valid):  # If no valid t, we hit no border. Return nothing.
        return None, None
    t_hit = np.min(t_candidates[valid])  # Otherwise, we hit the border for the lowest t.
    hit_idx = np.nonzero(np.abs(t_candidates - t_hit) <= max(tol, 1e-14))[0]  # Special case where we hit a corner. We return all indices.
    return float(t_hit), hit_idx


def reflect_vector(current_u, wall_n):
    n = unit_normal(wall_n)
    return current_u - 2.0 * np.dot(current_u, n) * n


def reflect_segment_with_velocity(x, u, v_hat, A, b, tol=1e-12, max_reflections=200):
    """
    Move from x along u, reflecting both remaining u and v_hat at each hit.
    Returns x_final, v_hat_final, n_reflections.
    """
    x_curr = x.copy()  # current particle position
    u_curr = u.copy()  # remaining displacement vector for this leapfrog step
    v_curr = v_hat.copy()  # current momentum (velocity) to be reflected
    reflections = 0  # reflection counter, steps are small but we might be close to a corner

    # Move until there is displacement left to travel. 
    # We also set a maximum number of reflections in order to avoid corner traps. 
    while np.linalg.norm(u_curr) > tol and reflections < max_reflections:
        # Find the candidate facet we are about to hit
        t_hit, hit_idxs = intersection_candidates(A, b, x_curr, u_curr, tol=tol)
        # If t_hit is too far, we are safe to move the rest of the distance. End while loop.
        if t_hit is None:
            x_curr = x_curr + u_curr
            u_curr = np.zeros_like(u_curr)
            break
        move = t_hit * u_curr  # Movement until first hit
        x_curr = x_curr + move  # New position at the wall
        u_curr = u_curr - move  # Movement left after reflection
        normals = A[hit_idxs] 
        # If multiple walls are hit simultaneously, get a combined reflection direction
        combined = np.sum(normals, axis=0) 
        # If the sum is near zero (numerical issue), just use the first value.
        if np.linalg.norm(combined) < 1e-16: 
            combined = normals[0]
        # reflect remaining displacement and velocity vectors
        u_curr = reflect_vector(u_curr, combined)
        v_curr = reflect_vector(v_curr, combined)
        reflections += 1
        # Tiny nudge away from the wall to avoid hitting the same wall again due to numerical rounding.
        x_curr = x_curr + 1e-14 * (combined / np.linalg.norm(combined))
    if reflections >= max_reflections:
        # Keep track to avoid infinite loops if stuck in a corner.
        raise RuntimeError("Too many reflections in one position update.")

    """
    ***Return variables***
    x_curr: final position after all reflections in this step
    v_curr: updated momentum after reflections
    reflections: number of reflections occurred
    """
    return x_curr, v_curr, reflections


def leapfrog_reflect(x, v, eta, A, b, x0, grad_f):
    """
    v: current momentum
    v_hat: momentum at half step
    grad_f: The gradient of f
    u: distance moved this step η * p_(n+1/2)
    x_new: new position after considering reflections, also update momentum
    v_new: Second momentum update
    """
    # half momentum update
    v_hat = v - 0.5 * eta * grad_f(x, x0)
    # position update (with reflections) and reflect v_hat accordingly
    u = eta * v_hat
    x_new, v_hat_new, n_refl = reflect_segment_with_velocity(x, u, v_hat, A, b)
    # final half momentum update at new position
    v_new = v_hat_new - 0.5 * eta * grad_f(x_new, x0)
    return x_new, v_new, n_refl


# -----------------------------
# Reflective Hamiltonian sampler
# -----------------------------
def reflective_hamiltonian_dynamics(A, b, x0=None, n_steps=200, w=20, eta=0.1, grad_f=None):
    """
    Lets simulate a single chain: at every iteration, sample momentum ~ N(0,I) and run w leapfrog steps.
    """
    d = A.shape[1]

    if grad_f is None:
        grad_f = lambda x, x_start: np.zeros_like(x)  # default

    # If x0 not provided, find strictly interior feasible starting point
    if x0 is None:
        x0 = find_interior_point(A, b)

    x = x0.copy()
    traj = [x.copy()]

    for it in range(n_steps):
        # sample momentum
        v = np.random.normal(size=x.shape)
        # integrate w leapfrog steps
        for _ in range(w):
            x, v, _ = leapfrog_reflect(x, v, eta, A, b, x0, grad_f)
        traj.append(x.copy())

    traj = np.array(traj)
    return traj

