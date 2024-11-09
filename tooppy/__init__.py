import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve  # , cg, minres
# import pyamg
import time
from sympy import symbols, Matrix, diff, expand
import itertools
import os
import pyvista

__version__ = "2.3.0"

def get_M_n(d:int, n:int, E=1, nu=1/3):
    return E / (1-(n - 1) * nu - (d - n) * n / (1 - max(0, d - n - 1) * nu) * nu ** 2)

def get_element_stiffness_matrix(E=1, nu=1/3, dimensional=2):
    dof = dimensional * 2 ** dimensional
    M_1 = get_M_n(dimensional, 1, E=E, nu=nu)
    M_2 = get_M_n(dimensional, 2, E=E, nu=nu)
    G = E / (2 * (1 + nu))  # modulus of rigidity

    C = np.zeros([dimensional + dimensional * (dimensional - 1) // 2] * 2)
    C[:dimensional, :dimensional] = np.eye(dimensional) * (M_1 * 2 - M_2) + np.ones([dimensional] * 2) * (M_2 - M_1)
    C[dimensional:, dimensional:] = np.eye(dimensional * (dimensional - 1) // 2) * G

    C = Matrix(C)  # Constitutive (material property) matrix:
    
    # Initialize SymPy symbols
    xs = ''
    for i in range(dimensional):
        xs += (' ' if i > 0 else '') + 'x' + str(i)
    xs = symbols(xs)

    Ns = []
    for I in itertools.product(*([[-1, 1]] * dimensional)):
        Ns.append(1)
        for j, i in enumerate(I):
            Ns[-1] *= 0.5 + i * xs[j]
        
    # Create strain-displacement matrix B:
    B = [[0] * dof for i in range(dimensional)]
    for i, x in enumerate(xs):
        for j, N in enumerate(Ns):
            B[i][j * dimensional + i] = diff(N, x)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            if j <= i:
                continue
            B.append([0] * dof)
            for k, N in enumerate(Ns):
                B[-1][k * dimensional + i] = diff(N, y)
                B[-1][k * dimensional + j] = diff(N, x)
    B = Matrix(B)

    dK = B.T * C * B

    # Because dK is symmetric, only need to integrate about half of it. 
    K = np.zeros([dof] * 2)
    for i in range(dof):
        for j in range(0, i + 1):
            K[i, j] = expand(dK[i * dof + j]).integrate(*[[x, -0.5, 0.5] for x in xs])
    for i in range(dof):
        for j in range(i + 1, dof):
            K[i, j] = K[j, i]
    
    return K

def get_smoothen_kernel(rmin, resolution):
    iH  =  []
    jH  =  []
    sH  =  []
    for row, I in enumerate(itertools.product(*[range(e) for e in resolution])):
        KK1 = [int(np.maximum(i - (np.ceil(rmin) - 1), 0)) for i in I]
        KK2 = [int(np.minimum(i + np.ceil(rmin), nel)) for i, nel in zip(I, resolution)]
        for J in itertools.product(*[range(e_1, e_2) for e_1, e_2 in zip(KK1, KK2)]):
            col = 0
            for a, b in zip(J, resolution):
                col = col * b + a
            fac = rmin - np.sqrt(np.sum([(i - j) ** 2 for i, j in zip(I, J)]))
            iH.append(row)
            jH.append(col)
            sH.append(np.maximum(0., fac))
    H = coo_matrix((sH, (iH, jH)), shape = (np.prod(resolution), np.prod(resolution))).tocsc()
    Hs = H.sum(1)
    H /= Hs
    return H
    
def optimality_criteria(x, dc, dv, g, mask=None):  # Used to update the design variables
    # dc: Sensitivity of the compliance (objective function) with respect to the design variables.
    # dv: Sensitivity of the volume with respect to the design variables.
    # g: Constraint term used in the optimization.
    
    l1 = 0  # Bounds for the Lagrange multiplier
    l2 = 1e9
    move = 0.2  # Maximum change allowed in the design variables in one iteration
    while (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        xnew =  np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        if not mask is None:
            xnew[np.logical_not(mask)] = 0
        gt = g + np.sum((dv * (xnew - x)))
        if gt > 0 :
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)

def solve(get_fixed,
          get_load,
          resolution,
          volfrac,
          penal,
          rmin,
          ft,
          E=1,
          nu=1/3,
          iterations=20,
          get_mask=None,
          change_threshold=0,
          initial_noise_strength=0,
          intermediate_results_saving_path=None,
          element_stiffness_matrix_file_dir='./element_stiffness_matrices/',
          skip_calculating_element_stiffness_matrix_if_exists=True):

    if not intermediate_results_saving_path is None:
        if not os.path.exists(intermediate_results_saving_path):
            os.makedirs(intermediate_results_saving_path)
    
    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0
    
    # Degrees of freedom (DOFs)
    dof = len(resolution) * np.prod([e + 1 for e in resolution])
    print('degrees of freedom =', dof)

    if type(penal) in [int, float]:
        penal = [penal] * 2
    
    # Set mask
    if not get_mask is None:
        # Calculate location of vertices
        slices = [slice(0, e) for e in resolution]
        coordinates = np.mgrid[slices]
        coordinates = [e.flatten() + 0.5 for e in coordinates]
        mask = get_mask(resolution, np.prod(resolution), np.array(coordinates))
    else:
        mask = None
    
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac / (1 if mask is None else np.mean(mask)) * np.ones(np.prod(resolution), dtype=float)
    if initial_noise_strength != 0:
        x += (np.random.rand(np.prod(resolution)) * 2 - 1) * initial_noise_strength
    if not mask is None:
        x[np.logical_not(mask)] = 0
    xold = x.copy()
    xPhys = x.copy()
    
    if not os.path.exists(element_stiffness_matrix_file_dir):
        os.makedirs(element_stiffness_matrix_file_dir)
    element_stiffness_matrix_file_name = 'KE_' + str(len(resolution)) + 'd_' + str(E) + ',' + str(nu) + '.npy'
    if skip_calculating_element_stiffness_matrix_if_exists and os.path.exists(element_stiffness_matrix_file_dir + element_stiffness_matrix_file_name):
        KE = np.load(element_stiffness_matrix_file_dir + element_stiffness_matrix_file_name)
    else:
        t_0 = time.time()
        KE = get_element_stiffness_matrix(E=E, nu=nu, dimensional=len(resolution))  # Element Stiffness Matrix
        np.save(element_stiffness_matrix_file_dir + element_stiffness_matrix_file_name, KE)
        print('time escaped in calculating element stiffness matrix =', time.time() - t_0)
    
    t_0 = time.time()

    dof_per_element = len(resolution) * 2 ** len(resolution)
    edofMat = np.zeros((np.prod(resolution), dof_per_element), dtype=int)
    for el, EL in enumerate(itertools.product(*[range(e) for e in resolution])):
        n1 = 0
        for a, b in zip(resolution, EL):
            n1 = (a + 1) * n1 + b
        indices = n1 * len(resolution) + np.arange(2 * len(resolution))
        j = len(resolution)
        for nel in resolution[-1:0:-1]:
            j *= nel + 1
            indices = list(indices) + list(np.array(indices) + j)
        edofMat[el] = indices
            
    # Construct the index pointers for the coo format
    iK  =  np.kron(edofMat, np.ones((dof_per_element, 1), dtype=np.int32)).flatten()
    jK  =  np.kron(edofMat, np.ones((1, dof_per_element), dtype=np.int32)).flatten()

    print('time escaped in calculating edofMat =', time.time() - t_0)

    # Construct a kernel for smoothening the design variables for regularization
    t_0 = time.time()
    H = get_smoothen_kernel(rmin, resolution)
    print('time escaped in calculating kernel for smoothening =', time.time() - t_0)
    
    t_0 = time.time()

    # Calculate location of vertices
    slices = [slice(0, e + 1) for e in resolution]
    coordinates = np.mgrid[slices]
    coordinates = np.array([e.flatten() for e in coordinates])

    # Boundary Conditions and support
    fixed = get_fixed(resolution, dof, coordinates)
    free = np.setdiff1d(np.arange(dof), fixed)
    
    # Set load
    f = get_load(resolution, dof, coordinates)
    f = f.reshape(dof, -1)
    if np.min(np.max(np.abs(f[free]), axis=0)) == 0:
        raise ValueError("No load found on free dofs.")
    
    u = np.zeros(f.shape)

    print('time escaped in preparing other things =', time.time() - t_0)
    
    t_0 = time.time()
    loop = 0
    change = 1
    g = 0  # A constraint or a measure related to the volume of the design
    penal_iter = iter(np.linspace(penal[0], penal[1], iterations))
    while change > change_threshold and loop < iterations:
        loop += 1

        penal_in_iteration = next(penal_iter)
        
        t_1 = time.time()

        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal_in_iteration*(Emax-Emin))).flatten(order = 'F')
        K  =  coo_matrix((sK, (iK, jK)), shape = (dof, dof)).tocsc()
        
        # Remove constrained dofs from matrix
        K  =  K[free, :][:, free]

        # print('time escaped in generating K =', time.time() - t_1)
        t_1 = time.time()
        
        # Solve system 
        u[free, :] = np.reshape(spsolve(K, f[free]), [len(free), -1])
        # u[free, :] = np.reshape(cg(K, f[free])[0], [len(free), -1])
        # u[free, :] = np.reshape(minres(K, f[free])[0], [len(free), -1])
        # ml = pyamg.ruge_stuben_solver(K)
        # u[free, :] = np.reshape(ml.solve(f[free], tol=1e-8), [len(free), -1])

        # print('time escaped in solving the linear system =', time.time() - t_1)
        t_1 = time.time()
        
        # Objective and sensitivity
        ce = np.zeros(np.prod(resolution))
        for e in u.T:
            u_element = e[edofMat].reshape(np.prod(resolution), dof_per_element)
            ce = np.maximum(ce, (np.dot(u_element, KE) * u_element).sum(1))  # Compliance energy
        obj = ((Emin + xPhys ** penal_in_iteration * (Emax - Emin)) * ce ).sum()
        dc = (-penal_in_iteration * xPhys ** (penal_in_iteration - 1) * (Emax - Emin)) * ce  # Ignore contribution of d ce /d obj
        dv = np.ones(np.prod(resolution))
        
        # Sensitivity filtering:
        if ft == 0:
            dc  =  np.asarray((H * (x * dc))[np.newaxis].T)[:, 0] / np.maximum(0.001, x)
        elif ft == 1:
            dc  =  np.asarray(H * (dc[np.newaxis].T))[:, 0]
            dv  =  np.asarray(H * (dv[np.newaxis].T))[:, 0]
            
        # Optimality criteria (Update design variables)
        xold = x
        (x, g) = optimality_criteria(x, dc, dv, g, mask=mask)

        # print('time escaped in optimality criteria =', time.time() - t_1)
        t_1 = time.time()
        
        # Regularize the design variables
        if ft == 0:
            xPhys = x
        elif ft == 1:  # Directly adjusts the design variables x themselves based on their average within a neighborhood defined by rmin.
            xPhys = np.asarray(H * x[np.newaxis].T)[:, 0]
        
        # Compute the change by the inf. norm
        change = np.linalg.norm(x.reshape(np.prod(resolution), 1) - xold.reshape(np.prod(resolution), 1), np.inf)

        if not intermediate_results_saving_path is None:
            np.save(intermediate_results_saving_path + 'result_' + str(len(resolution)) + 'd_' + str(loop) + '.npy', xPhys.reshape(resolution))
        
        print('iteration ', loop,
              ', loss = ', obj,
              ', change = ', change,
              sep='')
    print('time escaped in main loop =', time.time() - t_0)
    
    return xPhys.reshape(resolution)

def get_indices_on_boundary_elements(resolution, axis_selection):
    indices = []
    for a, b in zip(resolution, axis_selection):
        if b is None:
            indices.append(range(a))
        else:
            indices.append(([0] if b[0] else []) + ([a - 1] if b[1] else []))
    indices = np.array(list(itertools.product(*indices)))
    indices *= [int(np.prod(resolution[i:])) for i in range(1, len(resolution) + 1)]
    return np.sum(indices, axis=-1)

def get_indices_on_face(resolution, axis, start=False, end=False):
    axis_selection = [None] * len(resolution)
    axis_selection[axis] = [start, end]
    
    return get_indices_on_boundary_elements(resolution, axis_selection)

def mirrow_first_axis(array):
    shape = list(array.shape)
    shape[0] *= 2
    result = np.zeros(shape, dtype=array.dtype)
    result[:array.shape[0]] = array[::-1]
    result[array.shape[0]:] = array
    return result

def mirror(array,
           mirror_x=False,
           mirror_y=False,
           mirror_z=False):
    if mirror_x:
        array = mirrow_first_axis(array)
    array = np.transpose(array, [1, 2, 0])
    if mirror_y:
        array = mirrow_first_axis(array)
    array = np.transpose(array, [1, 2, 0])
    if mirror_z:
        array = mirrow_first_axis(array)
    array = np.transpose(array, [1, 2, 0])

    return array

# Visualize the result with pyvista
def plot_3d_array(array,
                  mirror_x=False,
                  mirror_y=False,
                  mirror_z=False,
                  volume_quality=5,
                  additional_meshes=[],
                  notebook=False):
    array = mirror(array, mirror_x=mirror_x, mirror_y=mirror_y, mirror_z=mirror_z)

    plotter = pyvista.Plotter(notebook=notebook)

    grid = pyvista.ImageData()
    grid.dimensions = np.array(array.shape) + 1
    grid.spacing = [volume_quality] * 3
    grid.cell_data["values"] = array.flatten(order="F")

    plotter.add_volume(grid, opacity=[0, 1, 1], cmap='magma')

    for mesh in additional_meshes:
        mesh.points *= volume_quality
        plotter.add_mesh(mesh)

    plotter.show()
