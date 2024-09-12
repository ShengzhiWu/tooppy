# tooppy
tooppy is a powerful **to**pological **op**timization library for **py**thon 3.

In principle, tooppy is capable of handling topology optimization problems in any dimension, not just the usual 2D and 3D cases.

## Install

```bash
pip install tooppy
```

## Features

### 2D

In this example, we seek a structure with both ends fixed (vertically fixed, horizontally slidable), and a vertical force applied at the center as the load. See `tests\test_2d.py`.

<img src="README.assets/result_2d.png" alt="result_2d" style="zoom:50%;" />

### 3D

This example is the 3D version of the one above, with the structure fixed at the four corners (slidable) and a load applied at the center. See `tests\test_3d.py`.

<img src="README.assets/result_3d_0.png" alt="result_3d_0" style="zoom:67%;" />

### Multiple Loads

tooppy supports solving for structures that are robust under multiple load conditions. The following example designs a structure with 4 fixed corners, where the center may be subjected to loads in the X, Y, and Z directions, requiring robustness under all three load conditions. Note that, since there may be horizontal loads, the previously used slidable supports can no longer be applied. All supports here are fully fixed. See `tests\test_3d_multiload.py`.

<img src="README.assets/result_3d_1.png" alt="result_3d_1" style="zoom:50%;" />

Requiring the structure to be robust under various load conditions often leads to relatively moderate designs.

### Mask

Another powerful feature of tooppy is the mask. Users can specify certain regions where material placement is prohibited. This is particularly useful when the part is in a constrained environment or needs to fit with other parts in space. The regions are specified using a boolean array.

This example has the same configuration as the first 3D example, except that material cannot be placed within the specified four cylindrical regions. See `tests\test_3d_mask.py`.

<img src="README.assets/result_3d_2.png" alt="result_3d_2" style="zoom:67%;" />

## Element Stiffness Matrix Cache

The element stiffness matrix is automatically cached in`element_stiffness_matrices/` on the first time you solve a problem by default. In subsequent calculations, this matrix will be automatically loaded to save time. When you use new materials (with new Young's modulus `E` or Poisson's ratio `nu`), or calculate problems in different dimensions, the matrix needs to be recalculated.

You can disable this behavior by using the `skip_calculating_element_stiffness_matrix_if_exists=False` option. You can also use the `element_stiffness_matrix_file_dir` option to specify the location where the matrix is stored or to be stored.
