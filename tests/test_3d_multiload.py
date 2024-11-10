# In this script, we try to find a structure which is rugged under 3 different directions of load.

import numpy as np
import os
from tooppy import solve, get_indices_on_boundary_elements, plot_3d_array

def get_fixed(resolution, ndof, coordinates):  # Constrains
    # dofs = np.arange(ndof)

    # Fix the 4 corners completely
    indices = get_indices_on_boundary_elements([e + 1 for e in resolution],
                                               [[True, True],
                                                [True, True],
                                                [True, False]])
    fixed = indices * 3
    fixed = np.union1d(fixed, indices * 3 + 1)
    fixed = np.union1d(fixed, indices * 3 + 2)

    return fixed

def get_load(resolution, ndof, coordinates):  # Load
    f = np.zeros([ndof, 3])
    i, j, k = resolution[0] // 2, resolution[1] // 2, resolution[2]
    index = (i * (resolution[1] + 1) + j) * (resolution[2] + 1) + k

    # direction = np.random.randn(3)
    # direction /= np.sqrt(np.dot(direction, direction))
    # print(direction)

    # Multiple load cases. The algorithm will try to make the structure hard for each case.
    f[index * 3 + 0, 0] = 1  # First load case: force on X direction
    f[index * 3 + 1, 1] = 1  # Second load case: force on Y direction
    f[index * 3 + 2, 2] = 1  # Third load case: force on Z direction

    return f

# Default input parameters
resolution = [20, 20, 10]  # [30, 30, 10]
volfrac = 0.05  # volume fraction
rmin = 1.5
penal = 3.0
ft = 1  # 0: sens, 1: dens

result = solve(get_fixed,
               get_load,
               resolution,
               volfrac,
               penal,
               rmin,
               ft, 
               iterations=50)  # 50

# Save result
result_saving_path = './output/'
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
np.save(result_saving_path + 'result_3d.npy', result)

plot_3d_array(result)