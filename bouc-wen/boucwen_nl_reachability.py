# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure # For marching_cubes
# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import *

# Specify the  file that includes dynamic systems
# Plot options
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
from odp.dynamics import Plane2D, DubinsCar
from boucwen_nl import BoucWenNL
import math
import os

# %%
if os.path.exists("plots") == False:
    os.mkdir("plots")

MAX_ERROR = 1

# STEP 1: Define grid
grid_min = np.array([-5.0, -5.0, -5.0])
grid_max = np.array([5.0, 5.0, 5.0])
dims = 3
samples = np.array([100, 100, 100])
g = Grid(grid_min, grid_max, dims, samples)

# %%
# TODO: Replace this with error function ( eg h(x) )
# STEP 2: Generate initial values for grid using shape functions
center = np.zeros(dims)
Initial_value_f = np.full(g.pts_each_dim, MAX_ERROR)
Initial_value_f = np.sqrt(g.vs[0]**2 + g.vs[1]**2 + g.vs[2]**2) - 0.2

print(Initial_value_f.shape)

# %%
# STEP 3: Time length for computations
Lookback_length = 10
t_step = 0.01

small_number = 1e-6
tau = np.arange(start=0, stop=Lookback_length + small_number, step=t_step)

# %%
# STEP 4: System dynamics for computation
sys3D = BoucWenNL(1, 0.25, 0.5, 0.5, 1, 1, 1, 1, uMin=[-1], uMax=[1], dMin=[0], dMax=[0], uMode="max", dMode="min")

# %%
# STEP 5: Initialize plotting option
po = PlotOptions(do_plot=False, plot_type="set", plotDims=[0,1,2], slicesCut=[12,12,12], colorscale="Bluered",
                 save_fig=False, interactive_html=False, showlegend=True,
                 axis_labels=["Displacement", "Velocity", "Hysteresisic Displacement"])

# %%
# STEP 6: Call HJSolver function
compMethod = { "TargetSetMode": "None"}
value_fn = HJSolver(sys3D, g, Initial_value_f, tau, compMethod, po, saveAllTimeSteps=True)
print(value_fn.shape)

# %%
np.save("boucwen_nl_forward_reachability.npy", value_fn[:,:,:,::20])

# %%
# STEP 7: Plotting
# Run marching cubes
try:
    # verts: (N, 3) array of vertex coordinates
    # faces: (M, 3) array of triangles, indexing into verts
    # normals: (N, 3) array of normal vectors at each vertex
    # values: (N,) array of function values at each vertex
    verts, faces, normals, values = measure.marching_cubes(
        volume=value_fn[:,:,:,0],
        level=0,
        spacing=(grid_max - grid_min) / samples,
        step_size=5
    )
except ValueError as e:
    print(f"Marching cubes failed: {e}")
    print("This might happen if the level set value {target_level} is outside the range of data,")
    print("or if the data is constant.")
    exit() # Or handle appropriately

# --- 3. Adjust Vertex Coordinates ---
# Marching cubes returns vertices scaled by spacing, starting from the origin (0,0,0)
# We need to shift them to match the actual grid coordinates defined by x, y, z vectors.
verts += np.array(grid_min) # Add the starting point of the grid

# --- 4. Plot the 3D Surface ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a Poly3DCollection object using the vertices and faces
mesh = Poly3DCollection(verts[faces])

# Set visual properties (optional)
mesh.set_facecolor('cyan') # Color of the faces
mesh.set_edgecolor('k')    # Color of the edges (try 'none' for smooth look)
mesh.set_alpha(0.99)        # Transparency

# Add the mesh to the axes
ax.add_collection3d(mesh)

# Set plot limits to match the data extent
ax.set_xlim(grid_min[0], grid_max[0])
ax.set_ylim(grid_min[1], grid_max[1])
ax.set_zlim(grid_min[2], grid_max[2])

# Set labels and title
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_zlabel("Z coordinate")
ax.set_title(f'Isosurface at level 0')

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig("plots/boucwen_nl_forward_reachability.png", dpi=300)


# %%
