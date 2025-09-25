import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from tooppy import solve, get_indices_on_boundary_elements

def get_fixed(resolution, ndof, coordinates):  # Constrains
    indices = get_indices_on_boundary_elements(np.array(resolution) + 1, [None, [False, True]])
    indices = indices
    fixed = np.concatenate([indices * 2, indices * 2 + 1])  # fixed
    return fixed

def get_load(resolution, ndof, coordinates):  # Load
    f = np.zeros(ndof)
    indices = get_indices_on_boundary_elements(np.array(resolution) + 1, [None, [True, False]])
    f[indices * 2 + 0] = 1
    return f

# Default input parameters
resolution = [80, 100]
volume_fraction = 0.2  # Volume fraction
rmin = 1.5  # Larger values for more smooth results.
penal = [1.5, 3.5]
ft = 1  # 0: sens, 1: dens

result = solve(get_fixed,
               get_load,
               resolution,
               volume_fraction,
               penal,
               rmin,
               ft,
               E=1,
               nu=1/3,
               iterations=100)

# Save result
result_saving_path = './output/'
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
np.save(result_saving_path + 'result_2d.npy', result)

plt.imshow(-result.T, cmap='gray', interpolation ='none', norm=colors.Normalize(vmin=-1, vmax=0))
plt.axis('off')
plt.show()