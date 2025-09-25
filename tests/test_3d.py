import numpy as np
import os
from tooppy import solve, get_indices_on_face, mirror, plot_3d_array

def get_fixed(resolution, ndof, coordinates):  # Constrains defined here
    fixed = [ndof - 1]  # Fix the 4 corners on Z direction
    # fixed = [ndof - 3, ndof - 2, ndof - 1]  # Fix the 4 corners completely

    # Mirror boundary condition
    fixed = np.union1d(fixed, get_indices_on_face([e + 1 for e in resolution], 0, start=True) * 3)
    fixed = np.union1d(fixed, get_indices_on_face([e + 1 for e in resolution], 1, start=True) * 3 + 1)
    return fixed

def get_load(resolution, ndof, coordinates):  # Load (force) defined here
    f = np.zeros(ndof)
    f[2] = -1  # Load on Z direction on the last vertex
    return f

# Default input parameters
resolution = [20, 20, 10]  # [30, 30, 10]
volfrac = 0.05  # volume fraction
rmin = 1.5  # 1.5 4.5
penal = 3.0
ft = 1  # 0: sens, 1: dens

result = solve(
    get_fixed,
    get_load,
    resolution,
    volfrac,
    penal,
    rmin,
    ft,
    iterations=50,
    #    intermediate_results_saving_path='./intermediates/'
)

# Save result
result_saving_path = './output/'
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
np.save(result_saving_path + 'result_3d.npy', mirror(result, mirror_x=True, mirror_y=True))

plot_3d_array(result, mirror_x=True, mirror_y=True)
