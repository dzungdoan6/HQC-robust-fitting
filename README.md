About
============

The Python code for the paper [A Hybrid Quantum-Classical Algorithm for Robust Fitting](https://arxiv.org/abs/2201.10110)

The demo program was only tested under Conda in a standard computer with Ubuntu 20.04.

If you find our code is useful, please cite:
```
@article{Dzung22hqc,
  title = {A Hybrid Quantum-Classical Algorithm for Robust Fitting},
  author = {Anh-Dzung Doan and Michele Sasdelli and David Suter and Tat-Jun Chin},
  booktitle =  {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}
```

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
    - Simply run script (It will likely take around 30 minutes)
    - Note: To normalise image coordiates, we used `normalise2dpts.m` of Peter Kovesi's [toolbox](https://www.peterkovesi.com/matlabfns/index.html) 
 
