import numpy as np
import arviz as az
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from ball_walk_function import ball_walk
from Hamilton_function import reflective_hamiltonian_dynamics
from hit_and_run_function import hit_and_run
from billiard_walk_function import billiard_walk
from utils import split_chain, get_user_gradient_choice

# ----------------------------------------
# Helper: run a single sampler
# ----------------------------------------
def run_sampling_method(name, func, A, b, **kwargs):
    """Runs a sampler, prints summary, returns samples."""
    start = time.time()
    print("="*60)
    print(f"Starting {name}...", flush=True)
    result = func(A, b, **kwargs)
    
    if isinstance(result, tuple):
        samples = result[0]
        acc_rate = result[1] if len(result) > 1 else None
    else:
        samples = result
        acc_rate = None

    print(f"{name} finished.\n", flush=True)
    d = samples.shape[1]
    print(f"Dimensions of polytope: {d}", flush=True)
    elapsed = time.time() - start
    print(f"{name} finished in {elapsed:.2f} seconds.\n", flush=True)

    if acc_rate is not None:
        print(f"Acceptance rate: {acc_rate:.3f}")

    # -----------------------------
    # Compute PSRF and ESS using split chains
    # -----------------------------
    all_chains = split_chain(samples, n_splits=4)
    idata = az.from_dict(posterior={"x": all_chains})

    rhat = az.rhat(idata)['x'].values
    ess = az.ess(idata)['x'].values

    print("\nPSRF per dimension:")
    for i, val in enumerate(rhat):
        print(f"  x{i+1}: {val:.4f}")

    print("\nESS per dimension:")
    for i, val in enumerate(ess):
        print(f"  x{i+1}: {val:.1f}")
    print("")

    return samples

# ----------------------------------------
# Main program
# ----------------------------------------
if __name__ == "__main__":
    # 1) Read A and b from CSV files
    A = np.loadtxt("A.csv", delimiter=",")
    b = np.loadtxt("b.csv")

    print("Polytope matrices loaded successfully.")
    print("A =\n", A)
    print("b =\n", b)

    # 2) Ask user which sampler to run
    print("\nWhich sampler do you want to run?")
    print("1 - Ball Walk")
    print("2 - Hit-and-Run")
    print("3 - Billiard Walk")
    print("4 - Reflective Hamiltonian Dynamics")
    
    try:
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice not in ["1", "2", "3", "4"]:
            print("Invalid choice. Defaulting to 2 (Hit-and-Run).")
            choice = "2"
    except Exception as e:
        print(f"Input error: {e}. Defaulting to 2 (Hit-and-Run).")
        choice = "2"
    
    # 3) Ask user for number of samples
    try:
        n_samples = int(input("\nEnter number of samples (default 40000, min 400): ").strip())
        if n_samples < 400:
            print("Sample count too low. Using default value 40000.")
            n_samples = 40000
    except ValueError:
        print("Invalid input. Defaulting to 40000.")
        n_samples = 40000

    print(f"\nSampler {choice} selected with {n_samples} samples.\n")


    # If Ball Walk, ask for radius
    r = 1  # default
    if choice == "1":
        try:
            r_input = float(input("Enter ball radius r: ").strip())
            if r_input <= 0:
                print("Non-positive radius; using default 1")
            else:
                r = r_input
        except Exception:
            print("Invalid input. Using default radius 1")

    # 3) Run the selected sampler
    if choice == "1":
        samples = run_sampling_method("Ball Walk", ball_walk, A, b, n_steps = n_samples, r=r)
    elif choice == "2":
        samples = run_sampling_method("Hit-and-Run", hit_and_run, A, b, n_steps = n_samples)
    elif choice == "3":
        samples = run_sampling_method("Billiard Walk", billiard_walk, A, b, n_steps = n_samples, L_scale=1)
    elif choice == "4":
        grad_f = get_user_gradient_choice()
        samples = run_sampling_method("Reflective Hamiltonian Dynamics", 
                        reflective_hamiltonian_dynamics, 
                        A, b, n_steps = n_samples, grad_f = grad_f)


    # d = dimension of polytope
    d = samples.shape[1]

    if (d > 1):
        # 2D scatter
        plt.figure(figsize=(6, 6))
        plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
        plt.title("Sampler 2D Projection (x1 vs x2)")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.axis("equal")
        plt.grid(True, ls="--", alpha=0.3)
        plt.show()
    
    if (d > 2):
        # 3D scatter for first three dimensions
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=2, alpha=0.5, color='red')
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("x₃")
        ax.set_title("Sampler 3D Projection (x1, x2, x3)")
        ax.set_box_aspect([1,1,1])
        plt.show()

