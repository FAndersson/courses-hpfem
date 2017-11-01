import math
import numpy as np

from dolfin import IntervalMesh
from dolfin.fem.interpolation import interpolate
from dolfin.functions import FunctionSpace, Function, Expression


def compute_interpolation_error(mesh_resolution=20):
    # Define domain and mesh
    a, b = 0, 1
    mesh = IntervalMesh(mesh_resolution, a, b)

    # Define finite element function space
    p_order = 1
    V = FunctionSpace(mesh, "CG", p_order)

    # Extract vertices of the mesh
    x = V.tabulate_dof_coordinates()
    # Note: Not the same as x = mesh.coordinates() (apparently has been re-ordered)

    # Express the analytical function
    u = Expression("1 + x[0] * sin(10 * x[0])", degree=5)

    # Interpolate u onto V and extract the values in the mesh nodes
    Iu = interpolate(u, V)
    Iua = Iu.vector().array()

    # Evaluate u at the mesh vertices
    Eua = np.empty(len(x))
    for i in range(len(x)):
        Eua[i] = 1 + x[i] * math.sin(10 * x[i])

    # Compute max interpolation error in the nodes
    e = Eua - Iua
    e_abs = np.absolute(e)
    error = np.amax(e_abs)
    return error
