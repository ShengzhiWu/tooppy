import numpy as np
import os
import pyvista
from tooppy import solve, get_indices_on_face, plot_3d_array

def get_fixed(resolution, ndof, coordinates):  # Constrains
    fixed = [ndof - 1]  # Fix the 4 corners on Z direction
    # fixed = [ndof - 3, ndof - 2, ndof - 1]  # Fix the 4 corners completely

    # Mirror boundary condition
    fixed = np.union1d(fixed, get_indices_on_face([e + 1 for e in resolution], 0, start=True) * 3)
    fixed = np.union1d(fixed, get_indices_on_face([e + 1 for e in resolution], 1, start=True) * 3 + 1)

    return fixed

def get_load(resolution, ndof, coordinates):  # Load
    f = np.zeros(ndof)
    f[2] = -1  # Load on Z direction on the last vertex

    return f

def get_mask(resolution, number_of_cells, coordinates):  # Mask, a boolean array with shape same to resolution, assigning where material can be located.
    mask = np.ones(number_of_cells, dtype=bool)
    x, y, z = coordinates

    mask[(x - 10) ** 2 + (y - 10) ** 2 < 4 ** 2] = False

    return mask

# Default input parameters
resolution = [20, 20, 10]  # [30, 30, 10]
volfrac = 0.05  # volume fraction
rmin = 1.5
penal = 3.0
ft = 1  # 0: sens, 1: dens

result = solve(get_fixed, get_load, resolution, volfrac, penal, rmin, ft, get_mask=get_mask, iterations=50)  # 50

# Save result
result_saving_path = './output/'
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
np.save(result_saving_path + 'result_3d.npy', result)

additional_meshes = [
    pyvista.Cylinder(center=np.array([10, 10, 5]),
                                      direction=[0, 0, 1],
                                      radius=4,
                                      height=10),
    pyvista.Cylinder(center=np.array([10, 30, 5]),
                                      direction=[0, 0, 1],
                                      radius=4,
                                      height=10),
    pyvista.Cylinder(center=np.array([30, 10, 5]),
                                      direction=[0, 0, 1],
                                      radius=4,
                                      height=10),
    pyvista.Cylinder(center=np.array([30, 30, 5]),
                                      direction=[0, 0, 1],
                                      radius=4,
                                      height=10)
]

plot_3d_array(result, mirror_x=True, mirror_y=True, additional_meshes=additional_meshes)