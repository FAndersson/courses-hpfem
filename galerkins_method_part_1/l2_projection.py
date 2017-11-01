from dolfin import IntervalMesh
from dolfin.fem.projection import project
from dolfin.functions import FunctionSpace, Function, Expression


def compute_projection_error(mesh_resolution=20, p_order=1):
    # Define domain and mesh
    a, b = 0, 1
    mesh = IntervalMesh(mesh_resolution, a, b)

    # Define finite element function space
    V = FunctionSpace(mesh, "CG", p_order)

    # Extract vertices of the mesh
    x = V.tabulate_dof_coordinates()
    # Note: Not the same as x = mesh.coordinates() (apparently has been re-ordered)

    # Express the analytical function
    u = Expression("1 + 4 * x[0] * x[0] - 5 * x[0] * x[0] * x[0]", degree=5)

    # Project u onto V and extract the values in the mesh nodes
    Pu = project(u, V)
    Pua = Pu.vector().array()

    # Create a function in the finite element function space V
    Eu = Function(V)
    Eua = Eu.vector().array()

    # Evaluate function in the mesh nodes
    for i in range(len(x)):
        Eua[i] = 1 + 4 * x[i]**2 - 5 * x[i]**3

    # Compute sum of projection error in the nodes
    e = Eua - Pua
    error = 0
    for i in range(len(e)):
        error += abs(e[i])
    return error
