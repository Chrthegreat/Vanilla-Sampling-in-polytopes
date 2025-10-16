# Vanilla-Sampling-in-polytopes.

This repository contains basic vanilla methods for sampling in polytopes. It's what you would get if you tried to code a method you read in a textbook. No advanced techniques for rounding polytopes, adaptive step sizes, or variance reduction are included—just the core algorithms in their simplest form for educational purposes.

The code is in Python because of all the useful packages that saves us a lot of time. Each file that contains one of the methods is named after it. All lines are fully commented so that you may understand the logic of each method. In addition, there is a polytopes.txt file that contains example polytopes you can use to run the code and test it yourself. Lastly, the main file is the one you run to start the code. 

** How to run ** 

Running the code is simple. Download all files, create a Python virtual enviroment and install requirements.txt. Then, just run main.py and follow the menu. 

** How to test your custom polytopes **

If you check the polytopes.txt file, you will see that each polytope is defined in a specific way. Each polytope is represented by its inequalities in matrix form. First, the name of the polytope, then matrix A (the columns of A define the dimensions), then matrix b and lastly exactly 10 stars *. To define your own polytopes follow this format exactly. Add 10 * after the last polytope, then a name, then A: followded by the matrix elements split with , and then b: followed by the elemebts. If done right, the polytope will appear in the main menu when you run main. 

** Notes **

You will be promted to select parameters for some of the methods. For ball walk especially, if you choose a bad value for the ball radium or too few steps, the code might return nothing. Try again with different parameters.  

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

** References ** 

The sampling methods implemented in this repository are based on the following foundational papers:

 - Ball Walk

Lovász, L., & Simonovits, M. (1993). Random walks in a convex body and an improved volume algorithm. Random Structures & Algorithms, 4(4), 359–412.
Lovász, L. (1999). Hit-and-run mixes fast. Mathematical Programming, 86(3), 443–461.

- Hit-and-Run

Smith, R. L. (1984). Efficient Monte Carlo procedures for generating points uniformly distributed over bounded regions. Operations Research, 32(6), 1296–1308.

- Billiard Walk

Lyubich, Y., & Packard, N. (2014). Billiard walk: a fast algorithm for uniform sampling in convex bodies.
De Loera, J. A., & Onn, S. (2006). Random walks on polytopes and networks. SIAM Journal on Discrete Mathematics, 20(2), 482–499.

- Reflective Hamiltonian Dynamics

Apostolos Chalkis, Vissarion Fisikopoulos, Marios Papachristou, Elias Tsigaridas (2021). Truncated Log-concave Sampling with Reflective Hamiltonian Monte Carlo.
