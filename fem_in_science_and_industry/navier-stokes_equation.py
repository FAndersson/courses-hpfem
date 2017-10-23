import numpy as np
import matplotlib.pyplot as plt

from dolfin import Point
from dolfin.common.plotting import plot
from dolfin.fem import DirichletBC
from dolfin.fem.norms import errornorm
from dolfin.fem.projection import project
from dolfin.fem.solving import solve
from dolfin.functions import (
    FunctionSpace, VectorFunctionSpace, TestFunction, Function, TrialFunctions, Expression, CellSize)
from mshr import Circle, Rectangle, generate_mesh
from ufl import dx, ds, inner, grad, div, VectorElement, FiniteElement


def setup_geometry():
    # Generate mesh
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 1.0
    mesh_resolution = 30
    geometry1 = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
    geometry2 = Circle(Point(0.5, 0.5), 0.1)
    mesh = generate_mesh(geometry1 - geometry2, mesh_resolution)

    # Mark regions for boundary conditions
    eps = 1e-5
    # Part of the boundary with zero pressure
    om = Expression("x[0] > XMAX - eps ? 1. : 0.", XMAX=xmax, eps=eps, degree=1)
    # Part of the boundary with prescribed velocity
    im = Expression("x[0] < XMIN + eps ? 1. : 0.", XMIN=xmin, eps=eps, degree=1)
    # Part of the boundary with zero velocity
    nm = Expression("x[0] > XMIN + eps && x[0] < XMAX - eps ? 1. : 0.", XMIN=xmin, XMAX=xmax, eps=eps, degree=1)

    return mesh, om, im, nm, ymax


def solve_navier_stokes_equation():
    """
    Solve the Navier-Stokes equation on a hard-coded mesh with hard-coded initial and boundary conditions
    """
    mesh, om, im, nm, ymax = setup_geometry()

    # Setup FEM function spaces
    # Function space for the velocity
    P1 = VectorElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)
    # Function space for the pressure
    P2 = FiniteElement("CG", mesh.ufl_cell(), 1)
    Q = FunctionSpace(mesh, P2)
    # Mixed function space for velocity and pressure
    M = P1 * P2
    W = FunctionSpace(mesh, M)

    # Setup FEM functions
    v, q = TestFunctions(W)
    w = Function(W)
    u, p = split(w)
    u0 = Function(V)

    # Inlet velocity
    uin = Expression(("4*(x[1]*(YMAX-x[1]))/(YMAX*YMAX)", "0."), YMAX=ymax, degree=1)

    # Viscosity and stabilization parameters
    nu = 1e-6
    h = CellSize(mesh)
    d = 0.2 * h**(3.0 / 2.0)

    # Time parameters
    time_step = 0.1
    t_start, t_end = 0.0, 10.0

    # Penalty parameter
    gamma = 10 / h

    # Time stepping
    t = t_start
    step = 0
    while t < t_end:
        # Time discretization (Crank–Nicolson method)
        um = 0.5 * u + 0.5 * u0

        # Navier-Stokes equations in weak residual form (stabilized FEM)
        # Basic residual
        r = (inner((u - u0) / time_step + grad(p) + grad(um) * um, v) + nu * inner(grad(um), grad(v)) + div(um) * q) * dx
        # Weak boundary conditions
        r += gamma * (om * p * q + im * inner(u - uin, v) + nm * inner(u, v)) * ds
        # Stabilization
        r += d * (inner(grad(p) + grad(um) * um, grad(q) + grad(um) * v) + inner(div(um), div(v))) * dx

        # Solve the Navier-Stokes equation (one time step)
        solve(r == 0, w)

        if step % 5 == 0:
            # Plot norm of velocity at current time step
            nov = project(sqrt(inner(u, u)), Q)
            fig = plt.figure()
            plot(nov, fig=fig)
            plt.show()

        # Shift to next time step
        t += time_step
        step += 1
        u0 = project(u, V)
