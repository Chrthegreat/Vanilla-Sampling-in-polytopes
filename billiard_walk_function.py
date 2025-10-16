import numpy as np
import matplotlib.pyplot as plt
from utils import * 

# ----------------------------------------
# Billiard Walk Sampler
# ----------------------------------------
def billiard_walk(A, b, n_steps=1000, L_scale=1.0, tol_scale=1e-12, renorm_every=10):
    """
    Dimension-proof billiard walk inside polytope {x : A x <= b}.

    A: (m, d) array
    b: (m,) array
    start: (d,) interior point
    n_samples: number of samples to generate
    L_scale: if exponential, mean scale; if using uniform replace sampling accordingly
    tol_scale: relative tolerance factor for numerical comparisons
    renorm_every: re-normalize direction after this many reflections to avoid drift

    Returns: (n_samples, d) array of samples
    """
    m, d = A.shape
    samples = []
    current = find_interior_point(A, b)
    rng = np.random.default_rng()

    for _ in range(n_steps):
        # random unit direction
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)

        # sample total travel length L (exponential here; change if desired)
        remaining = rng.exponential(scale=L_scale)

        reflection_count = 0
        # local tolerance for this step (scale with norm of current)
        tol = tol_scale * max(1.0, np.linalg.norm(current))

        while remaining > tol:
            # Find forward intersections t_i = (b_i - a_i^T current) / (a_i^T direction)
            a_dot_u = A @ direction   # shape (m,)
            rhs = b - (A @ current)   # slacks, shape (m,)

            # Consider only denominators sufficiently > tol (pointing toward faces)
            mask_forward = a_dot_u > tol
            if not np.any(mask_forward):
                # numeric fallback: treat as no hit
                current = current + remaining * direction
                remaining = 0.0
                break

            t_candidates = np.full(m, np.inf)
            t_candidates[mask_forward] = rhs[mask_forward] / a_dot_u[mask_forward]

            # Keep only positive forward intersections
            mask_positive = t_candidates > tol
            if not np.any(mask_positive):
                # no positive intersections -> travel remaining
                current = current + remaining * direction
                remaining = 0.0
                break

            # nearest forward hit
            t_hit = np.min(t_candidates[mask_positive])
            if t_hit > remaining:
                # can travel full remaining distance
                current = current + remaining * direction
                remaining = 0.0
                break

            # else we hit a wall at distance t_hit
            # find all faces whose t is within tolerance of t_hit (simultaneous hit)
            near = np.isfinite(t_candidates) & (np.abs(t_candidates - t_hit) <= max(tol, 1e-15 * max(1.0, abs(t_hit))))
            hit_indices = np.nonzero(near)[0]

            # move to the hit point
            current = current + t_hit * direction
            remaining -= t_hit

            # build a reflection normal. If multiple faces hit, combine their normals.
            normals = A[hit_indices]                      # shape (k, d)
            # Option 1: sum normals (simple), then normalize
            combined = np.sum(normals, axis=0)
            n_norm = np.linalg.norm(combined)
            if n_norm < 1e-16:
                # fallback: use the first face normal (degenerate)
                combined = normals[0]
                n_norm = np.linalg.norm(combined)
                if n_norm < 1e-16:
                    # give up and nudge point slightly inward, then continue
                    current = current - 1e-12 * direction
                    continue

            n = combined / n_norm

            # reflect direction
            direction = direction - 2.0 * np.dot(direction, n) * n

            reflection_count += 1
            if (reflection_count % renorm_every) == 0:
                direction /= np.linalg.norm(direction)

            # small inward push to avoid sticking exactly on the boundary due to numerics
            current = current + 1e-15 * n

        samples.append(current.copy())

    return np.array(samples)
