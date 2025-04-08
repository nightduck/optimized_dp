import imp
import numpy as np
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
# Plot options
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
from boucwen import BoucWen

import math
import os

""" USER INTERFACES
- Define grid
- Generate initial values for grid using shape functions
- Time length for computations
- System dynamics for computation
- Initialize plotting option
- Call HJSolver function

Note: If run on the server, please save the result and use the plot function on your local machine
"""

if os.path.exists("plots") == False:
    os.mkdir("plots")

MAX_ERROR = 0.5

# STEP 1: Define grid
grid_min = np.array([-2.0, -2.0, -3.0, -2.0, -2.0, -3.0])
grid_max = np.array([2.0, 2.0, 3.0, 2.0, 2.0, 3.0])
dims = 6
N = np.array([15, 15, 15, 15, 15, 15])
g = Grid(grid_min, grid_max, dims, N)

# TODO: Replace this with error function ( eg h(x) )
# STEP 2: Generate initial values for grid using shape functions
center = np.zeros(dims)
Initial_value_f = np.full(g.pts_each_dim, MAX_ERROR)
Initial_value_f = Initial_value_f - np.sqrt((g.vs[3] - g.vs[0])**2 + (g.vs[4] - g.vs[1])**2 + (g.vs[5] - g.vs[2])**2)

# STEP 3: Time length for computations
Lookback_length = 0.1
t_step = 0.01

small_number = 1e-5
tau = np.arange(start=0, stop=Lookback_length + small_number, step=t_step)

# TODO: Create system dynamics for augmented model
# STEP 4: System dynamics for computation
sys6D = BoucWen(1, 0.1, 0.5, 0.5, 1, 0.1, 1, 1, uMin=[0], uMax=[0], dMin=[-2], dMax=[2])

# TODO: Only plot linear dimensions
# STEP 5: Initialize plotting option
po = PlotOptions(do_plot=True, plot_type="set", plotDims=[3,4,5], slicesCut=[8,7,9], colorscale="Bluered", save_fig=False, filename="plots/bouc-wen_linear_error_reachability", interactive_html=True)

# TODO: Whatever this is
# STEP 6: Call HJSolver function
compMethod = { "TargetSetMode": "None"}
result_3 = HJSolver(sys6D, g, Initial_value_f, tau, compMethod, po, saveAllTimeSteps=True)

# Project 6D onto observable linear space by broadcasting minimums
projected_3d_result = np.min(np.min(np.min(result_3, axis=0), axis=0), axis=0)