import matplotlib.pyplot as plt

from dolfin import Point
from dolfin.common.plotting import plot
from dolfin.fem import DirichletBC
from dolfin.fem.norms import errornorm, norm
from dolfin.fem.projection import project
from dolfin.fem.solving import solve
from dolfin.fem.assembling import assemble
from dolfin.functions import FunctionSpace, TestFunction, Function, Expression
from mshr import Rectangle, generate_mesh
from ufl import dx, inner, grad, sqrt


def setup_geometry():
    # Generate mesh
    xmin, xmax = -2.0, 2.0
    ymin, ymax = -2.0, 2.0
    geometry = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
    mesh_resolution = 20
    mesh = generate_mesh(geometry, mesh_resolution)

    # Define boundary
    def boundary(x, on_boundary):
        return on_boundary

    return mesh, boundary


def solve_heat_equation(k, time_stepping_method):
    """
    Solve the heat equation on a hard-coded mesh with a hard-coded initial and boundary conditions

    :param k: Thermal conductivity
    :param time_stepping_method: Time stepping method. Can be one of ["forward_euler", "backward_euler", "trapezoidal"]
    """
    mesh, boundary = setup_geometry()

    # Exact solution (Gauss curve)
    ue = Expression("exp(-(x[0]*x[0]+x[1]*x[1])/(4*a*t))/(4*pi*a*t)", a=k, t=1e-7, domain=mesh, degree=1)

    # Polynomial degree
    r = 1

    # Setup FEM function space
    V = FunctionSpace(mesh, "CG", r)

    # Create boundary condition
    bc = DirichletBC(V, ue, boundary)

    # Setup FEM functions
    v = TestFunction(V)
    u = Function(V)

    # Time parameters
    time_step = 0.001
    t_start, t_end = 0.0, 20.0

    # Time stepping
    t = t_start
    if time_stepping_method == "forward_euler":
        theta = 0.0
    if time_stepping_method == "backward_euler":
        theta = 1.0
    if time_stepping_method == "trapezoidal":
        theta = 0.5
    u0 = ue
    step = 0
    while t < t_end:
        # Intermediate value for u (depending on the chosen time stepping method)
        um = (1.0 - theta) * u0 + theta * u

        # Weak form of the heat equation
        a = (u - u0) / time_step * v * dx + k * inner(grad(um), grad(v)) * dx

        # Solve the heat equation (one time step)
        solve(a == 0, u, bc)

        # Advance time in exact solution
        t += time_step
        ue.t = t

        if step % 100 == 0:
            # Compute error in L2 norm
            error_L2 = errornorm(ue, u, 'L2')
            # or equivalently
            # sqrt(assemble((ue - u) * (ue - u) * dx))
            # Compute norm of exact solution
            nue = norm(ue)
            # Print relative error
            print("Relative error = {}".format(error_L2 / nue))

        # Shift to next time step
        u0 = project(u, V)
        step += 1


def check_time_stepping_methods():
    import logging
    logging.getLogger("FFC").setLevel(logging.WARNING)

    for tsm in ["forward_euler", "backward_euler", "trapezoidal"]:
        print("----------------------------")
        print(tsm)
        print("----------------------------")
        solve_heat_equation(1.0, tsm)


if __name__ == "__main__":
    check_time_stepping_methods()
