import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
from PIL import Image
from tooppy import solve, get_indices_on_boundary_elements, mirrow_first_axis

img = Image.open('./tests/input/logo.png')
img = 1 - np.array(img, dtype=np.float32)[:, :, 0].T / 255
load = (np.pad(img, [[1, 0], [1, 0]]) + np.pad(img, [[1, 0], [0, 1]]) + np.pad(img, [[0, 1], [1, 0]]) + np.pad(img, [[0, 1], [0, 1]])) / 4

def get_fixed(resolution, ndof, coordinates):  # Constrains
    indices = get_indices_on_boundary_elements(
        [e + 1 for e in resolution],
        [None, [False, True]]
    )
    # fixed = indices * 2 + 1  # Fixed vertically
    fixed = np.union1d(indices * 2, indices * 2 + 1)  # Fixed completely
    return fixed

def get_load(resolution, ndof, coordinates):  # Load
    # return np.array([np.zeros_like(load.flatten()), load.flatten()]).T.flatten()  # Vertical force
    # return np.array([load.flatten(), np.zeros_like(load.flatten())]).T.flatten()  # Horizontal force
    relative_importance_of_horizontal_force = 0.7
    return np.array([np.array([np.zeros_like(load.flatten()), load.flatten()]).T.flatten(),
                     np.array([load.flatten(), np.zeros_like(load.flatten())]).T.flatten() * relative_importance_of_horizontal_force]).T  # Horizontal force

# Default input parameters
resolution = img.shape
volume_fraction = 0.2  # Volume fraction

result = solve(
    get_fixed,
    get_load,
    resolution,
    volume_fraction,
    penal=[1.5, 4],  # penal (increasing)
    rmin=1.5,  # Larger values for more smooth results
    ft=1,  # 0: sens, 1: dens
    E=1,
    nu=1/3,
    iterations=40
)

# Save result
result_saving_path = './output/'
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
np.save(result_saving_path + 'result_2d.npy', result)

result_color = [1, 0.5, 0]
result_img = np.array([(1 + result * (e - 1)) * (1 - img) for e in result_color])
result_img = np.transpose(result_img, [2, 1, 0])

plt.imsave("./output/logo.png", result_img)

plt.imshow(result_img, cmap='gray', interpolation ='none', norm=colors.Normalize(vmin=-1, vmax=0))
plt.axis('off')
plt.show()