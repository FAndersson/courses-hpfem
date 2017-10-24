import numpy as np
import matplotlib.pyplot as plt

from dolfin import Point, cells, facets, refine, MeshFunction, SubDomain
from dolfin.common.plotting import plot
from dolfin.fem import DirichletBC
from dolfin.fem.norms import errornorm
from dolfin.fem.projection import project
from dolfin.fem.solving import solve
from dolfin.fem.assembling import assemble
from dolfin.functions import (
    FunctionSpace, VectorFunctionSpace, TestFunction, Function, TrialFunctions, Expression, CellSize, FacetNormal)
from mshr import Circle, Rectangle, generate_mesh
from ufl import dx, ds, inner, grad, div, VectorElement, FiniteElement, Measure


def setup_geometry(interior_circle=True, num_mesh_refinements=0):
    # Generate mesh
    xmin, xmax = 0.0, 4.0
    ymin, ymax = 0.0, 1.0
    mesh_resolution = 50
    geometry1 = Rectangle(Point(xmin, ymin), Point(xmax, ymax))
    center = Point(0.5, 0.5)
    r = 0.1
    side_length = 0.1
    if interior_circle:
        geometry2 = Circle(center, r)
    else:
        l2 = side_length / 2
        geometry2 = Rectangle(Point(center[0] - l2, center[1] - l2), Point(center[0] + l2, center[1] + l2))
    mesh = generate_mesh(geometry1 - geometry2, mesh_resolution)

    # Refine mesh around the interior boundary
    for i in range(0, num_mesh_refinements):
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        for c in cells(mesh):
            p = c.midpoint()
            cell_markers[c] = (abs(p[0] - .5) < .5 and abs(p[1] - .5) < .3 and c.h() > .1) or c.h() > .2
        mesh = refine(mesh, cell_markers)

    # Mark regions for boundary conditions
    eps = 1e-5
    # Part of the boundary with zero pressure
    om = Expression("x[0] > XMAX - eps ? 1. : 0.", XMAX=xmax, eps=eps, degree=1)
    # Part of the boundary with prescribed velocity
    im = Expression("x[0] < XMIN + eps ? 1. : 0.", XMIN=xmin, eps=eps, degree=1)
    # Part of the boundary with zero velocity
    nm = Expression("x[0] > XMIN + eps && x[0] < XMAX - eps ? 1. : 0.", XMIN=xmin, XMAX=xmax, eps=eps, degree=1)

    # Define interior boundary
    class InteriorBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # Compute squared distance to interior object midpoint
            d2 = (x[0] - center[0])**2 + (x[1] - center[1])**2
            return on_boundary and d2 < (2 * r)**2

    # Create mesh function over the cell facets
    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # Mark all facets as sub domain 0
    sub_domains.set_all(0)
    # Mark interior boundary facets as sub domain 1
    interior_boundary = InteriorBoundary()
    interior_boundary.mark(sub_domains, 1)

    return mesh, om, im, nm, ymax, sub_domains


def solve_navier_stokes_equation(interior_circle=True, num_mesh_refinements=0):
    """
    Solve the Navier-Stokes equation on a hard-coded mesh with hard-coded initial and boundary conditions
    """
    mesh, om, im, nm, ymax, sub_domains = setup_geometry(interior_circle, num_mesh_refinements)
    dsi = Measure("ds", domain=mesh, subdomain_data=sub_domains)

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

            # Compute drag force on circle
            n = FacetNormal(mesh)
            drag_force_measure = p * n[0] * dsi(1)  # Drag (only pressure)
            drag_force = assemble(drag_force_measure)
            print("Drag force = " + str(drag_force))

        # Shift to next time step
        t += time_step
        step += 1
        u0 = project(u, V)
