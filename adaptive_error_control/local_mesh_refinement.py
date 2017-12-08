from fenics import Expression, Function, FunctionSpace, MeshFunction, Point, \
    assemble, dx, cells, interpolate, refine, sqrt
from mshr import Rectangle, generate_mesh


# Set log level
import logging
logging.getLogger("FFC").setLevel(logging.WARNING)


def generate_rectangle_mesh(mesh_resolution):
    # Generate domain and mesh
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    mesh = generate_mesh(Rectangle(Point(xmin, ymin), Point(xmax, ymax)), mesh_resolution)
    return mesh


def local_refine(mesh, center, r):
    xc, yc = center
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    for c in cells(mesh):
        mp = c.midpoint()
        cell_markers[c] = sqrt( (mp[0] - xc)*(mp[0] - xc) + (mp[1] - yc)*(mp[1] - yc))<r
    mesh = refine(mesh, cell_markers)
    return mesh


def compute_error(u1, u2):
    """
    L1 error between two functions u1 and u2

    :param u1: FEniCS function
    :param u2: FEniCS function
    :return: Approximate L1 error between u1 and u2
    """
    mesh_resolution_ref = 400
    mesh_ref = generate_rectangle_mesh(mesh_resolution_ref)
    V_ref = FunctionSpace(mesh_ref, "CG", degree=1)
    Iu1 = interpolate(u1, V_ref)
    Iu2 = interpolate(u2, V_ref)
    error = assemble(abs(Iu1-Iu2) * dx)
    return error


def compute_mesh_error(do_local_refine, mesh_resolution, num_refinements):
    mesh = generate_rectangle_mesh(mesh_resolution)

    # Define heat kernel
    t0 = 0.01
    u = Expression("exp(-(x[0]*x[0]+x[1]*x[1])/(4*t))/(4*pi*t)", t=t0, domain=mesh, degree=3)

    # Define finite element function space
    degree = 1
    V = FunctionSpace(mesh, "CG", degree)

    # Refine mesh
    r = 0.4
    xc, yc = 0.0, 0.0
    for i in range(0, num_refinements):
        if do_local_refine:
            mesh = local_refine(mesh, [xc, yc], r)
        else:
            mesh = refine(mesh)

    # Interpolate the heat kernel into the function space
    Iu = interpolate(u, V)

    # Compute L1 error between u and its interpolant
    error = compute_error(u, Iu)
    return error


def solve_problem():
    mesh_resolution = 5
    num_refinements = 4
    do_local_refine = False
    error = compute_mesh_error(do_local_refine, mesh_resolution, num_refinements)
    print(error)

    mesh_resolution = 5
    num_refinements = 5
    do_local_refine = True
    error = compute_mesh_error(do_local_refine, mesh_resolution, num_refinements)
    print(error)


if __name__ == "__main__":
    solve_problem()
