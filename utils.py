import numpy as np
from scipy.optimize import linprog


# ----------------------------------------
# Sample uniformly from a d-dimensional ball
# ----------------------------------------
def sample_ball(d, r=1.0, n=1):
    """
    This function samples n points uniformly from a d-dimensional ball of radius r.

    Args:
        d : int
            Dimension of the ball
        r : float
            Radius of the ball
        n : int
            Number of points to sample
    Returns:
        numpy array of shape (n, d)
    """
    #Sample Gaussian vector for direction
    z = np.random.normal(size=(n, d))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    u = z / norms
     #Scale samples using R * U^(1/d)
    u_radius = np.random.uniform(0, 1, size=(n, 1))
    rho = r * (u_radius ** (1 / d))
    return u * rho


# ----------------------------------------
# Find strictly interior point via LP
# ----------------------------------------
def find_interior_point(A, b):
    """
    Find a strictly interior feasible point x such that
    A x + δ * 1 <= b, δ > 0 (maximize δ)
    """
    m, d = A.shape
    c = np.zeros(d + 1)
    c[-1] = -1  # maximize δ <=> minimize -δ

    A_int = np.hstack([A, np.ones((m, 1))])
    b_int = b.copy()
    bounds = [(None, None)] * d + [(0, None)]  # δ >= 0

    res = linprog(c, A_ub=A_int, b_ub=b_int, bounds=bounds, method="highs")

    if not res.success:
        raise ValueError("No strictly interior point found!")

    x0 = res.x[:-1]  # drop δ
    return x0


def split_chain(chain, n_splits=4):
    """
    Splits a single chain into n_splits contiguous chains with equal length.
    Any leftover steps at the end are discarded.

    Parameters:
        chain: np.ndarray of shape (n_steps, dims)
        n_splits: int, number of subchains

    Returns:
        np.ndarray of shape (n_splits, n_steps_per_split, dims)
    """
    n_steps = chain.shape[0]
    split_size = n_steps // n_splits  # integer division
    chains = []

    for i in range(n_splits):
        start = i * split_size
        end = (i + 1) * split_size
        chains.append(chain[start:end])

    return np.stack(chains, axis=0)


def get_user_gradient_choice():
    print("\nSelect gradient for Reflective Hamiltonian Dynamics:")
    print("1. Zero gradient (uniform sampling)")
    print("2. Linear gradient: |x|")
    print("3. Gaussian-like potential: 2 * |x| * exp(|x|**2)")
    
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "1":
        def grad_f(x, x_start):
            return np.zeros_like(x)
    elif choice == "2":
        def grad_f(x, x_start):
            return x - x_start
    elif choice == "3":
        def grad_f(x, x_start):
            diff = x - x_start
            return (2 * diff * np.exp(diff**2))

    else:
        print("Invalid choice. Defaulting to zero gradient.")
        def grad_f(x, x_start):
            return np.zeros_like(x)

    print(f"Gradient option {choice} selected.\n")
    return grad_f



def load_polytopes_from_file(filename):
    """
    Load polytopes from a text file with the following format:

    Polytope Name
    A:
    ...rows of A matrix, comma-separated...
    b:
    ...rows of b vector, one number per line...
    ****************************************
    """
    with open(filename, "r") as f:
        content = f.read()

    blocks = content.split("*" * 10)
    polytopes = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        name = lines[0]
        A_lines = []
        b_lines = []
        reading_A = False
        reading_b = False

        for line in lines[1:]:
            if line.upper() == "A:":
                reading_A = True
                reading_b = False
                continue
            elif line.upper() == "B:":
                reading_A = False
                reading_b = True
                continue

            if reading_A:
                A_lines.append(line)
            elif reading_b:
                b_lines.append(line)

        if not A_lines or not b_lines:
            print(f"Warning: Polytope '{name}' missing A or b")
            continue

        # Convert to numpy arrays
        try:
            A = np.array([list(map(float, l.split(","))) for l in A_lines])
            b = np.array([float(l) for l in b_lines])
            polytopes.append({"name": name, "A": A, "b": b})
        except Exception as e:
            print(f"Error parsing polytope '{name}': {e}")
            continue

    return polytopes

def choose_polytope(polytopes):
    """
    Prompt the user to select one polytope from a list of loaded polytopes.

    Parameters
    ----------
    polytopes : list of dict
        Each dict should have keys: 'name', 'A', 'b'

    Returns
    -------
    tuple: (A, b, name)
        The matrices A, b and the polytope name selected by the user.
    """
    if not polytopes:
        raise ValueError("No polytopes available to choose from!")

    print("\nAvailable polytopes:")
    for i, p in enumerate(polytopes):
        print(f"{i+1}. {p['name']}")

    # Ask user until a valid choice is made
    while True:
        try:
            choice = int(input(f"Select a polytope (1-{len(polytopes)}): ").strip())
            if 1 <= choice <= len(polytopes):
                selected = polytopes[choice - 1]
                print(f"\nSelected polytope: {selected['name']}")
                return selected["A"], selected["b"], selected["name"]
            else:
                print(f"Invalid choice. Enter a number between 1 and {len(polytopes)}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")