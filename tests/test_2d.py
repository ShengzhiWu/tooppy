import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from tooppy import solve, get_indices_on_boundary_elements, mirrow_first_axis

def get_fixed(resolution, ndof, coordinates):  # Constrains defined here
    elements_on_mirror_plane = get_indices_on_boundary_elements(  # Indices of elements on the mirror plane (x = 0)
        [e + 1 for e in resolution],
        [[True, False], None]
    )

    # Define which DOFs are fixed
    fixed = np.union1d(
        [ndof - 1],  # Fix the last DOF of the lase element, which means the structure is supported by the ground here, while horizontal movement is allowed
        elements_on_mirror_plane * 2  # We apply the mirror boundary conditions by fix the first DOF of each elements on the mirror plane
    )
    return fixed

def get_load(resolution, ndof, coordinates):  # Load (force) defined here
    f = np.zeros(ndof)
    f[1] = -1  # Load on Z direction on the first vertex
    return f

# Default input parameters
resolution = [90, 30]
volume_fraction = 0.4  # Volume fraction (target volume / solution domain volume)
rmin = 3  # Larger values for more smooth results.
penal = 3.0  # Punish density between 0 and 1
ft = 1  # 0: sens, 1: dens

result = solve(
    get_fixed,
    get_load,
    resolution,
    volume_fraction,
    penal,
    rmin,
    ft,
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
