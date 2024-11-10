import numpy as np
import pyvista as pv
from tooppy.mesh import construct_mesh, get_geometric_information

box = construct_mesh([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                      [[3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]])  # 正方体
information = get_geometric_information(box)
print('Volume of box =', information['volume'])
print('Ground truth =', 8)
print('Inertia of box =', np.diag(information['inertia']))
print('Ground truth =', 1 / 3 * 8 * (1 ** 2 + 1 ** 2))

print()

sphere = pv.Sphere(radius=1.0, center=(0, 0, 0), theta_resolution=100, phi_resolution=200)
information = get_geometric_information(sphere)
print('Volume of ball =', information['volume'])
print('Ground truth =', 4 / 3 * np.pi * 1 ** 3)
print('Inertia of ball =', np.diag(information['inertia']))
print('Ground truth =', 2 / 5 * 4 / 3 * np.pi * 1 ** 5)

# You can also use tooppy.mesh.load_mesh load mesh from files and analyze them.
