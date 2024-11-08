import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import pyvista

def convert_to_mesh(array,
                    pad: int=2,
                    upsample_factor: float=2,
                    upsample_order: int=3,
                    sharpen_strength: float=3,
                    sharpen_radius: float=4,
                    smoothen_radius: float=1.2,
                    iso=0.37):
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

    mesh = grid.contour([iso], array.flatten(order="F"), method='flying_edges')  # 其他常用的method还有'marching_cubes'，二者没有明显区别

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