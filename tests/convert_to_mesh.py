import numpy as np
from tooppy.mesh import convert_to_mesh, plot_mesh, save_mesh

# Assume you runed test_3d.py firsly, resulting in a NumPy array file result_3d.npy at ./output/.
# Now we load it, convert it to a mesh and then save it as an STL file.

a = np.load("./output/result_3d.npy")
mesh = convert_to_mesh(a, iso=0.37)
plot_mesh(mesh)
save_mesh(mesh, './output/result.stl')
