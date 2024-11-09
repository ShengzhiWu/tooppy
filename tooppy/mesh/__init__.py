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

def get_faces(mesh):  # Get faces. This function is expensive.
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

    return pyvista.PolyData(points, faces_for_pyvista)

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

def save_mesh(mesh, file):
    mesh.save(file)