import numpy as np
import matplotlib.pyplot as plt

from dolfin import Point
from dolfin.common.plotting import plot
from dolfin.fem import DirichletBC
from dolfin.fem.norms import errornorm
from dolfin.fem.projection import project
from dolfin.fem.solving import solve
from dolfin.functions import (
    FunctionSpace, VectorFunctionSpace, TestFunctions, TrialFunction, TrialFunctions, Expression, CellSize)
from mshr import Rectangle, generate_mesh
from ufl import dx, inner, grad


def setup_geometry():
    # Generate mesh
    xmin, xmax = -10.0, 10.0
    ymin, ymax = -0.5, 0.5
    geometry = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
    mesh_resolution = 50
    mesh = generate_mesh(geometry, mesh_resolution)

    # Define boundary
    def boundary(x, on_boundary):
        return on_boundary

    return mesh, boundary


def solve_wave_equation(a, symmetric=True):
    """
    Solve the wave equation on a hard-coded mesh with a hard-coded initial and boundary conditions

    :param a: Wave propagation factor
    :param symmetric: Whether or not the problem is symmetric
    """
    mesh, boundary = setup_geometry()

    # Exact solution
    if symmetric:
        ue = Expression("(1-pow(a*t-x[0],2))*exp(-pow(a*t-x[0],2)) + (1-pow(a*t+x[0],2))*exp(-pow(a*t+x[0],2))",
                        a=a, t=0, domain=mesh, degree=2)
        ve = Expression("2*a*(a*t-x[0])*(pow(a*t-x[0],2)-2)*exp(-pow(a*t-x[0],2))"
                        "+ 2*a*(a*t+x[0])*(pow(a*t+x[0],2)-2)*exp(-pow(a*t+x[0],2))",
                        a=a, t=0, domain=mesh, degree=2)
    else:
        ue = Expression("(1-pow(a*t+x[0],2))*exp(-pow(a*t+x[0],2))", a=a, t=0, domain=mesh, degree=2)
        ve = Expression("2*a*(a*t+x[0])*(pow(a*t+x[0],2)-2)*exp(-pow(a*t+x[0],2))", a=a, t=0, domain=mesh, degree=2)

    # Polynomial degree
    r = 1

    # Setup FEM function spaces
    Q = FunctionSpace(mesh, "CG", r)
    W = VectorFunctionSpace(mesh, "CG", r, dim=2)

    # Create boundary conditions
    bcu = DirichletBC(W.sub(0), ue, boundary)
    bcv = DirichletBC(W.sub(1), ve, boundary)
    bcs = [bcu, bcv]

    # Setup FEM functions
    p, q = TestFunctions(W)
    w = Function(W)
    u, v = w[0], w[1]

    # Time parameters
    time_step = 0.05
    t_start, t_end = 0.0, 5.0

    # Time stepping
    t = t_start
    u0 = ue
    v0 = ve
    step = 0
    while t < t_end:
        # Weak form of the wave equation
        um = 0.5 * (u + u0)
        vm = 0.5 * (v + v0)
        a1 = (u - u0) / time_step * p * dx - vm * p * dx
        a2 = (v - v0) / time_step * q * dx + a**2 * inner(grad(um), grad(q)) * dx

        # Solve the wave equation (one time step)
        solve(a1 + a2 == 0, w, bcs)

        # Advance time in exact solution
        t += time_step
        ue.t = t
        ve.t = t

        if step % 10 == 0:
            # Plot solution at current time step
            fig = plt.figure()
            plot(u, fig=fig)
            plt.show()

            # Compute max error at vertices
            vertex_values_ue = ue.compute_vertex_values(mesh)
            vertex_values_w = w.compute_vertex_values(mesh)
            vertex_values_u = np.split(vertex_values_w, 2)[0]
            error_max = np.max(np.abs(vertex_values_ue - vertex_values_u))
            # Print error
            print(error_max)

        # Shift to next time step
        u0 = project(u, Q)
        v0 = project(v, Q)
        step += 1
