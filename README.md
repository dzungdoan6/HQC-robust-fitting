About
============

The Python code for the paper "A Hybrid Quantum-Classical Algorithm for Robust Fitting"

The demo program was only tested under Conda in a standard computer with Ubuntu 20.04 

Installation
================

+ Python 3.7 
+ numpy 1.21 (pip install numpy)
+ matplotlib 3.4 (pip install matplotlib)
+ pickle5 (pip install pickle5)
+ D-Wave Ocean (pip install dwave-ocean-sdk)
+ qubovert (pip install qubovert)
+ Gurobi (python -m pip install gurobipy==9.1.2)
+ opencv (pip install opencv-python)

Usage
================

+ Synthetic data (script `main_synthetic.py`)
    1. (Optional) Register an account on D-Wave Leap (https://cloud.dwavesys.com/leap/login/?next=/leap/), obtain TOKEN of D-Wave Leap, and assign its value to the variable `TOKEN` within the script.
    2.  Run script
    Note: if TOKEN of D-Wave Leap is provided (step 1), QUBO will be solved by quantum annealing, otherwise QUBO will be solved by simulated annealing

+ Real data (script `main_fund.py`)
    - Simply run script (It may take around 30 minutes)
