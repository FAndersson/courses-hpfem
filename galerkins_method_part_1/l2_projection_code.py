import numpy as np

from dolfin import IntervalMesh
from dolfin.fem.projection import project
from dolfin.fem.solving import solve
from dolfin.functions import FunctionSpace, Function, TestFunction, Expression
from ufl import inner, dx


def compute_projection_error(mesh_resolution=20, p_order=1):
    # Define domain and mesh
    a, b = 0, 1
    mesh = IntervalMesh(mesh_resolution, a, b)

    # Define finite element function space
    V = FunctionSpace(mesh, "CG", p_order)

    # Extract vertices of the mesh
    x = V.dofmap().tabulate_all_coordinates(mesh)
    indices = np.argsort(x)

    # Express the analytical function
    u = Expression("1 + 4 * x[0] * x[0] - 5 * x[0] * x[0] * x[0]", degree=5)

    # Project u onto V and extract the values in the mesh nodes
    Pu = project(u, V)
    Pua = Pu.vector().array()

    # Create a function in the finite element function space V
    Eu = Function(V)
    Eua = Eu.vector().array()

    # Evaluate function in the mesh nodes
    for j in indices:
        Eua[j] = 1 + 4 * x[j] * x[j] - 5 * x[j] * x[j] * x[j]

    # Compute sum of projection error in the nodes
    e = Eua - Pua
    error = 0
    for i in range(len(e)):
        error += abs(e[i])
    return error


def compute_projection_error_solve(mesh_resolution=20, p_order=1):
    # Define domain and mesh
    a, b = 0, 1
    mesh = IntervalMesh(mesh_resolution, a, b)

    # Define finite element function space
    V = FunctionSpace(mesh, "CG", p_order)

    # Extract vertices of the mesh
    x = V.dofmap().tabulate_all_coordinates(mesh)
    indices = np.argsort(x)

    # Express the analytical function
    u = Expression("1 + 4 * x[0] * x[0] - 5 * x[0] * x[0] * x[0]", degree=5)

    # Project u onto V and extract the values in the mesh nodes
    Pu = Function(V)
    v = TestFunction(V)
    r = inner(Pu - u, v)*dx
    solve(r == 0, Pu)
    Pua = Pu.vector().array()

    # Create a function in the finite element function space V
    Eu = Function(V)
    Eua = Eu.vector().array()

    # Evaluate function in the mesh nodes
    for j in indices:
        Eua[j] = 1 + 4 * x[j] * x[j] - 5 * x[j] * x[j] * x[j]

    # Compute sum of projection error in the nodes
    e = Eua - Pua
    error = 0
    for i in range(len(e)):
        error += abs(e[i])
    return error


def p_order_1_3():
    import logging
    logging.getLogger('FFC').setLevel(logging.WARNING)

    s1 = compute_projection_error(10, 1)
    s3 = compute_projection_error(10, 3)

    s1_2 = compute_projection_error_solve(10, 1)
    s3_2 = compute_projection_error_solve(10, 3)

    print(s1)
    print(s3)
    print(s1_2)
    print(s3_2)


if __name__ == "__main__":
    p_order_1_3()
