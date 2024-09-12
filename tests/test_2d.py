import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from tooppy import solve, get_indices_on_boundary_elements, mirrow_first_axis

def get_fixed(resolution, ndof, coordinates):  # Constrains
    fixed = [ndof - 1]  # fixed
    fixed = np.union1d(fixed,
                       get_indices_on_boundary_elements([e + 1 for e in resolution],
                                                        [[True, False],
                                                         None]) * 2)  # Mirror boundary conditions
    return fixed

def get_load(resolution, ndof, coordinates):  # Load
    f = np.zeros(ndof)
    f[1] = -1  # Load on Z direction on the last vertex
    return f

# Default input parameters
resolution = [90, 30]
volfrac = 0.4  # Volume fraction
rmin = 3  # Larger values for more smooth results.
penal = 3.0
ft = 1  # 0: sens, 1: dens

result = solve(get_fixed,
               get_load,
               resolution,
               volfrac,
               penal,
               rmin,
               ft,
               E=1,
               nu=1/3,
               iterations=50)

# Save result
result_saving_path = './output/'
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
np.save(result_saving_path + 'result_2d.npy', result)

plt.imshow(-mirrow_first_axis(result).T, cmap='gray', interpolation ='none', norm=colors.Normalize(vmin=-1, vmax=0))
plt.axis('off')
plt.show()