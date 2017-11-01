from dolfin import UnitIntervalMesh
from dolfin.fem.assembling import assemble
from dolfin.fem.interpolation import interpolate
from dolfin.fem.projection import project
from dolfin.functions import FunctionSpace, Function, Expression
from ufl import sqrt, dx


def compute_error(u1, u2):
    # Reference mesh
    mesh_resolution_ref = 500
    mesh_ref = UnitIntervalMesh(mesh_resolution_ref)

    # Reference function space
    V_ref = FunctionSpace(mesh_ref, "CG", 1)

    # Evaluate the input functions on the reference mesh
    Iu1 = interpolate(u1, V_ref)
    Iu2 = interpolate(u2, V_ref)

    # Compute the error
    e = Iu1 - Iu2
    error = sqrt(assemble(e * e * dx))
    return error


def compare_errors(p_order=1):
    # Express the analytical function
    u = Expression("1 + sin(10*x[0])", degree=5)

    mesh_resolutions = [10, 50, 100, 200]

    projection_errors = []
    interpolation_errors = []
    for mesh_resolution in mesh_resolutions:
        # Define mesh
        mesh = UnitIntervalMesh(mesh_resolution)

        # Define finite element function space
        V = FunctionSpace(mesh, "CG", p_order)

        # Compute projection
        up = project(u, V)

        # Compute interpolation
        ui = interpolate(u, V)

        # Compute errors
        projection_errors.append(compute_error(u, up))
        interpolation_errors.append(compute_error(u, ui))

    print(projection_errors)
    print(interpolation_errors)
