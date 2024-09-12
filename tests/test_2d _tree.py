import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from tooppy import solve

def get_fixed(resolution, ndof, coordinates):  # Constrains
    fixed = [(resolution[0] // 2 * (resolution[1] + 1) + resolution[1]) * 2 + 1]  # fixed
    return fixed

def get_load(resolution, ndof, coordinates):  # Load
    f = np.zeros(ndof)
    for i in range(resolution[1] + 1):
        f[(i * (resolution[1] + 1)) * 2 + 1] = 1
    return f

# Default input parameters
resolution = [200, 200]
volfrac = 0.2  # Volume fraction
rmin = 1.5  # Larger values for more smooth results.
penal = [1.5, 3.5]
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
               iterations=100)

# Save result
result_saving_path = './output/'
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
np.save(result_saving_path + 'result_2d.npy', result)

plt.imshow(-result.T, cmap='gray', interpolation ='none', norm=colors.Normalize(vmin=-1, vmax=0))
plt.axis('off')
plt.show()