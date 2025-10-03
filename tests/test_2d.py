import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from tooppy import solve, mirrow_first_axis

def get_fixed(resolution, ndof, coordinates):  # Constrains defined here
    x, y = coordinates

    # Define which DOFs are fixed
    fixed = np.zeros((len(x), len(coordinates)), dtype=bool)
    fixed[x == 0, 0] = True  # We apply the mirror boundary conditions by fix the first DOF of each elements on the mirror plane
    fixed[np.logical_and(x == resolution[0], y == resolution[1]), 1] = True  # Fix the last DOF of the lase element, which means the structure is supported by the ground here, while horizontal movement is allowed
    fixed = fixed.flatten()
    fixed = np.arange(ndof, dtype=int)[fixed]

    return fixed

def get_load(resolution, ndof, coordinates):  # Load (force) defined here
    x, y = coordinates
    f = np.zeros((len(x), len(coordinates)))
    f[np.logical_and(x == 0, y == 0), 1] = -1  # Load on Y direction on the first vertex
    return  f.flatten()

# Default input parameters
resolution = [90, 30]
volume_fraction = 0.4  # Volume fraction (target volume / solution domain volume)

result = solve(
    get_fixed,
    get_load,
    resolution,
    volume_fraction,
    penal=3.0,  # Punish density between 0 and 1
    rmin=3,  # Larger values for more smooth results
    ft=1,  # 0: sens, 1: dens
    E=1,
    nu=1/3,
    iterations=50
)

# Save result
result_saving_path = './output/'
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
np.save(result_saving_path + 'result_2d.npy', result)

plt.imshow(-mirrow_first_axis(result).T, cmap='gray', interpolation ='none', norm=colors.Normalize(vmin=-1, vmax=0))
plt.axis('off')
plt.show()
