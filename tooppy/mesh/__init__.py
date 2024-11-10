import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import pyvista

# Convert an 3d volume data, containing values in 0~1, to a pyvista mesh. 0 means empty and 1 means material there.
# The result is approximatelly in the box [0, array.shape[0]] × [0, array.shape[1]] × [0, array.shape[2]] in 3d space.
# If the input array has big values on boundary, the output mesh may slightly over the box.
def convert_to_mesh(array,
                    pad: int=2,  # Padding the input volume data. Unit: pixel.
                    upsample_factor: float=2,  # Typical value: 1~3. Higher value for more detailed meshes.
                    upsample_order: int=3,  # Order of interpolation. Typical value: 1~3.
                    sharpen_strength: float=3,  # Typical value: 0~10. Higher value for more uniform structures.The results of topology optimization may contain thin structures, which are usually impractical for manufacturing. Increasing this parameter will thicken originally thin structures and thin out originally thick ones, making the overall structure more uniform. Too high value causes deviation from the original optimized structure thereby reduces the structural performance.
                    sharpen_radius: float=4,  # Unit: pixel.
                    smoothen_radius: float=1.2,  # Unit: pixel. Higher value for 
                    iso=0.37,  # Value of the isosurface. Typical value: 0~1. Higher value for thicker results.
                    method='flying_edges'  # Method for calculating isosurface. Can be 'flying_edges' or 'marching_cubes'.
                    ):
    assert len(array.shape) == 3

    if pad > 0:
        array = np.pad(array, pad_width=pad, mode='constant', constant_values=0)
    
    # Upsample
    if upsample_factor != 1:
        array = zoom(array, zoom=[upsample_factor] * 3, order=upsample_order)
    
    # Sharpen
    if sharpen_strength != 0:
        array = array * (1 + sharpen_strength) - gaussian_filter(array, sigma=sharpen_radius, mode='constant', cval=0) * sharpen_strength
    
    # Smoothen
    assert smoothen_radius >= 0
    if smoothen_radius > 0:
        array = gaussian_filter(array, sigma=smoothen_radius, mode='constant', cval=0)

    grid = pyvista.ImageData()
    grid.dimensions = np.array(array.shape)
    # grid.origin = (0, 0, 0)  # The bottom left corner of the data set

    mesh = grid.contour([iso], array.flatten(order="F"), method=method)

    # Scale and move the mesh to match the input array.
    mesh.points /= upsample_factor
    mesh.points -= pad

    return mesh

def get_points(mesh):
    return mesh.points

def get_faces(mesh):  # Get faces. The result is a list of lists of integers. This function is expensive if the mesh contains faces of different numbers of sides.
    # If all faces are triangles, can get faces efficiently.
    if len(mesh.faces) % 4 == 0:
        faces = np.array(mesh.faces).reshape([-1, 4])
        if np.min(faces[:, 0] == 3) == 1:
            return faces[:, 1:]
    
    # If all faces are quadrangles, can get faces efficiently.
    if len(mesh.faces) % 5 == 0:
        faces = np.array(mesh.faces).reshape([-1, 5])
        if np.min(faces[:, 0] == 4) == 1:
            return faces[:, 1:]

    # General case. Slow.
    faces = []
    i = 0
    while i < len(mesh.faces):
        number_of_vertices = mesh.faces[i]
        faces.append(mesh.faces[i + 1: i + 1 + number_of_vertices])
        i += 1 + number_of_vertices
    
    return faces

def construct_mesh(points, faces):
    faces_for_pyvista = []
    for face in faces:
        faces_for_pyvista.append(len(face))
        faces_for_pyvista.extend(face)

    return pyvista.PolyData(np.array(points, dtype=float), faces_for_pyvista)

# Calculate integration of 1, x, y, z, x ** 2, y ** 2, z ** 2, x * y, x * z, y * z over the region enclosed by the mesh.
def integrate(mesh):
    m = np.array([[1 / 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1 / 24, 0, 0, 1 / 24, 0, 0, 1 / 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1 / 24, 0, 0, 1 / 24, 0, 0, 1 / 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1 / 24, 0, 0, 1 / 24, 0, 0, 1 / 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 60, 0, 0, 0, 0, 0, 1 / 60],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 120, 0, 0, 1 / 120, 0, 0, 0, 1 / 120, 0, 0, 1 / 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 120, 0, 0, 0, 1 / 120, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 120, 0, 0, 1 / 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 120, 0, 0, 1 / 120, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 120, 0, 0, 0, 0, 0, 0, 1 / 120, 0, 0, 0, 0, 1 / 60, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 120, 0, 0, 1 / 120, 0, 0, 1 / 120, 0, 0, 1 / 120, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 60, 0, 0, 1 / 120, 0, 0, 1 / 120, 0, 0, 0, 0, 0, 1 / 60, 0]])  # 系数矩阵
    points = get_points(mesh)
    faces = get_faces(mesh)

    if hasattr(faces, 'shape'):  # All faces have same number of sides.
        triangles = np.concatenate([faces[:, [0] + [i + 1, i + 2]] for i in range(faces.shape[1] - 2)])
    else:  # The mesh contains faces of different numbers of sides. Slow.
        triangles = []
        for face in faces:
            for i in range(len(face) - 2):
                triangles.append(face[[0] + [i + 1, i + 2]])
        triangles = np.array(triangles)
    
    a = points[triangles]
    det = np.linalg.det(a)
    a = a.reshape([a.shape[0], -1])
    a = a.T
    a = np.concatenate([np.ones_like(a[[0]]), a])
    b = []
    for i in range(a.shape[0]):
        for j in range(i, a.shape[0]):
            b.append(a[i] * a[j])
    b = np.array(b)
    b = m @ b
    b *= det
    b = np.sum(b, axis=-1)

    return b

def _move_m_matrix(stiffness, move):
    stiffness = np.array(stiffness)
    move_matrix = np.array([[0, move[2], -move[1]],
                            [-move[2], 0, move[0]],
                            [move[1], -move[0], 0]], dtype=np.float32)
    stiffness[3:, 3:] += move_matrix.T@stiffness[:3, :3]@move_matrix+move_matrix.T@stiffness[:3, 3:]+stiffness[3:, :3]@move_matrix
    stiffness[:3, 3:] += stiffness[:3, :3]@move_matrix
    stiffness[3:, :3] += move_matrix.T@stiffness[:3, :3]
    return stiffness

def _get_inertia_of_point(location, dtype=np.float32):
    m = np.zeros([6, 6], dtype=dtype)
    m[:3, :3] = np.eye(3)
    return _move_m_matrix(m, location)

# Calculate volume, centroid and inertia tensor of a mesh.
def get_geometric_information(mesh):
    integration = integrate(mesh)
    volume, x_i, y_i, z_i, xx_i, yy_i, zz_i, xy_i, xz_i, yz_i = integration
    centroid = integration[1:4] / volume
    I_absolute = np.array([
            [yy_i + zz_i, -xy_i, -xz_i],
            [-xy_i, xx_i + zz_i, -yz_i],
            [-xz_i, -yz_i, xx_i + yy_i]
        ])
    m_point = _get_inertia_of_point(centroid) * volume
    I = I_absolute - m_point[3:, 3:]
    m = np.zeros([6, 6], dtype=float)
    m[:3, :3] = np.eye(3) * volume
    m[3:, 3:] = I
    m = _move_m_matrix(m, centroid)
    return {
        'volume': volume,
        'centroid': centroid,
        'inertia': I,  # Inertia relative to its centroid
        'inertia_absolute': I_absolute,  # Inertia relative to origin
        'm_matrix': m  # The 6 * 6 mass matrix M. Let v = [v_x, v_y, v_z, w_x, w_y, w_z] be the generalized velocity of the rigid body. Then its kinetic energy equals to v.T @ M @ v / 2.
        }

def plot_mesh(mesh,
              colored_through_axis=0,
              smooth_shading=True,
              specular=1,  # 0 ~ 1
              cmap="plasma",
              notebook=False):
    if colored_through_axis is None:
        mesh.plot(smooth_shading=smooth_shading, specular=specular, cmap=cmap, show_scalar_bar=False, notebook=notebook)
    else:
        mesh.plot(scalars=mesh.points[:, colored_through_axis], smooth_shading=smooth_shading, specular=specular, cmap=cmap, show_scalar_bar=False, notebook=notebook)

def load_mesh(file):  # Load mesh from file. STL, OBJ, PLY, VTK and VTU files are supported.
    return pyvista.read(file)

def save_mesh(mesh, file):  # Save mesh as a file. STL, OBJ, PLY, VTK, VTU, VTP, GLB and PVD files are supported.
    mesh.save(file)
