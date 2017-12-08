get_ipython().magic('matplotlib inline')
get_ipython().magic('run /home/fenics/fenics-matplotlib.py')
from dolfin import *
from mshr import *
from IPython.display import display, clear_output
import time
import logging

logging.getLogger('FFC').setLevel(logging.WARNING)

import time
import os

set_log_active(False)

Y_MAX = 0.41
X_MAX = 2.5
XMIN = 0.
YMIN = 0
G = [XMIN, X_MAX, YMIN, Y_MAX]

ghole = [0.15, 0.25, 0.15, 0.25]
resolution = 10
mesh = generate_mesh(
    Rectangle(Point(G[0], G[2]), Point(G[1], G[3])) - Rectangle(Point(ghole[0], ghole[2]), Point(ghole[1], ghole[3])),
    resolution)


# SubDomains for defining boundary conditions
class WallBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[1] > Y_MAX - DOLFIN_EPS or x[1] < DOLFIN_EPS)


class InnerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
                x[0] > DOLFIN_EPS and x[1] > DOLFIN_EPS and x[0] < X_MAX - DOLFIN_EPS and x[1] < Y_MAX - DOLFIN_EPS)


class InflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] < DOLFIN_EPS)


class OutflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] > X_MAX - DOLFIN_EPS)


# Coefficients for defining boundary conditions
class Inflow(Expression):
    def eval(self, values, x):
        values[0] = 4 * 1.0 * (x[1] * (Y_MAX - x[1])) / (Y_MAX * Y_MAX)
        values[1] = 0

    def value_shape(self):
        return (2,)


class PsiMarker(Expression):
    def eval(self, values, x):
        ib = InnerBoundary()

        if (ib.inside(x, True)):
            values[0] = 1.0
        else:
            values[0] = 0.0


# Known coefficients
u0_0 = Constant((0.0, 0.0))
f = Constant((0.0, 0.0))
u0_0p = Constant(0.0)
psimarker = PsiMarker()

# Use compiler optimizations
parameters["form_compiler"]["cpp_optimize"] = True

# Allow approximating values for points that may be generated outside
# of domain (because of numerical inaccuracies)
parameters["allow_extrapolation"] = True

###### Modify for exercise below ######

# Viscosity
nu = 4e-3


def M(mesh, u, p, i):
    def epsilon(z):
        return 0.5 * (grad(z) + grad(z).T)

    n = FacetNormal(mesh)

    I = Identity(2)
    sigma = p * I - nu * epsilon(u)
    theta = Constant((1.0, 0.0))

    g = Expression(("200.0*exp(-200.0*(pow(x[0] - 0.5, 2) + pow(x[1] - 0.3, 2)))", "0.0"))

    M1 = psimarker * p * n[0] * ds  # Drag (only pressure)
    M2 = psimarker * p * n[1] * ds  # Lift (only pressure)
    M3 = inner(g, u) * dx  # Mean of the velocity in a region
    M4 = psimarker * dot(dot(sigma, n), theta) * ds  # Drag (full stress)
    M5 = u[0] * dx  # Mean of the x-velocity in the whole domain

    m = [M1, M2, M3, M4, M5]

    return m[i]


# The strong residual with w the solution (w2 used for simple linerization)
def R(w, w2):
    (u, p) = (as_vector((w[0], w[1])), w[2])
    (u2, p2) = (as_vector((w2[0], w2[1])), w2[2])

    Au = grad(p) + grad(u2) * u
    Ap = div(u)

    Aui = [Au[i] for i in range(0, 2)]

    return as_vector(Aui + [Ap])


# The weak residual with w the solution and wt the test function
def r(W, w, wt, ei_mode=True, stab=True):
    (u, p) = (as_vector((w[0], w[1])), w[2])
    (v, q) = (as_vector((wt[0], wt[1])), wt[2])

    h = CellSize(W.mesh())
    delta = h

    Z = FunctionSpace(W.mesh(), "DG", 0)
    z = TestFunction(Z)

    if (not stab):
        delta = 0.0

    if (not ei_mode):
        z = 1.0

    # Define variational forms
    # Multiply by z to be able to extract error indicators for each cell
    dx = Measure("dx", domain=W.mesh())
    a_G = z * (nu * inner(grad(u), grad(v)) + inner(grad(p) + grad(u) * u, v) + div(u) * q) * dx
    a_stab = z * delta * inner(R(w, w), R(wt, w)) * dx
    a = a_G + a_stab
    L = z * inner(f, v) * dx
    F = a - L

    return F


###### Modify for exercise above ######


# Solve primal and dual equations and compute error indicators
def adaptive_solve(mesh, i):
    h = CellSize(mesh)

    # Define function spaces
    Z = FunctionSpace(mesh, "DG", 0)
    V = VectorFunctionSpace(mesh, "CG", 1)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V * Q

    # primal solver boundary conditions
    bc_0_0 = DirichletBC(W.sub(0), u0_0, WallBoundary())
    bc_0_1 = DirichletBC(W.sub(0), u0_0, InnerBoundary())
    bc_0_2 = DirichletBC(W.sub(0), Inflow(), InflowBoundary())
    bc_0_3p = DirichletBC(W.sub(1), u0_0p, OutflowBoundary())

    # Define unknown and test function(s)
    (v, q) = TestFunctions(W)
    wt = TestFunction(W)
    w = Function(W)

    phi = Function(W)
    (u, p) = (as_vector((w[0], w[1])), w[2])

    n = FacetNormal(mesh)

    # The variational form
    F = r(W, w, wt, ei_mode=False, stab=True)

    # Primal boundary conditions
    bcs = [bc_0_0, bc_0_1, bc_0_2, bc_0_3p]

    # Solve the primal problem
    solve(F == 0, w, bcs)

    # Compute output
    output = assemble(M(mesh, u, p, i))

    # Project output functional
    (ut, pt) = TrialFunctions(W)
    a_psi = inner(ut, v) * dx + inner(pt, q) * dx
    L_psi = M(mesh, v, q, i)
    psi = Function(W)
    solve(a_psi == L_psi, psi)

    # Generate the dual problem
    # a_star = derivative(F, w, wt)
    a_star = adjoint(derivative(F, w))
    (phi_u, phi_p) = (as_vector((phi[0], phi[1])), phi[2])

    L_star = M(mesh, v, q, i)

    # Generate dual boundary conditions
    for bc in bcs:
        bc.homogenize()

    # Solve the dual problem
    solve(a_star == L_star, phi, bcs)

    # Generate error indicators
    z = TestFunction(Z)

    (u, p) = w.split()
    (phi_u, phi_p) = phi.split()
    (psi_u, psi_p) = psi.split()

    # Compute error indicators ei
    LR1 = r(W, w, phi, ei_mode=True, stab=False)
    ei = Function(Z)
    ei.vector()[:] = assemble(LR1).array()

    return (u, p, phi_u, phi_p, psi_u, psi_p, output, ei)


# Refine the mesh based on error indicators
def adaptive_refine(mesh, ei, adapt_ratio, adaptive):
    gamma = abs(ei.vector().array())

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    gamma_0 = sorted(gamma, reverse=True)[int(len(gamma) * adapt_ratio) - 1]
    for c in cells(mesh):
        cell_markers[c] = gamma[c.index()] > gamma_0

    # Refine mesh
    if adaptive:
        mesh = refine(mesh, cell_markers)
    else:
        mesh = refine(mesh)

    return mesh


def solve_problem(maxiters, mesh, adapt_ratio, mi=0, adaptive=True):
    results = []

    # Adaptive loop
    for i in range(0, maxiters):
        # Solve primal and dual equations and compute error indicators
        (u, p, phi_u, phi_p, psi_u, psi_p, output, ei) = adaptive_solve(mesh, mi)

        # Compute error estimate and compare against reference value
        est = abs(sum(ei.vector().array()))
        ref = 0.14181
        err = abs(ref - output)
        result = (output, est, err, mesh.num_cells())
        results.append(result)
        print("output: %5.5f est: %5.5f err: %5.5f cells: %d vertices: %d" % (output, est, err, mesh.num_cells(), mesh.num_vertices()))
        # Functionspace used for plotting
        Q = FunctionSpace(mesh, "CG", 1)

        plt.figure(figsize=(15.5, 1.))
        plt.suptitle("iter:%d" % (i))
        plt.subplot(1, 3, 1)
        mplot_function(project(sqrt(inner(u, u)), Q))
        plt.subplot(1, 3, 2)
        mplot_function(project(sqrt(phi_u[0] ** 2 + phi_u[1] ** 2), Q))
        plt.subplot(1, 3, 3)
        plt.triplot(mesh2triang(mesh))

        # Refine the mesh
        mesh = adaptive_refine(mesh, ei, adapt_ratio, adaptive)


if __name__ == "__main__":
    # Solve the problem
    maxiters = 10
    # Fraction of the cells to refine
    adapt_ratio = 0.1
    for mi in range(5):
        solve_problem(maxiters, mesh, adapt_ratio, mi, adaptive=True)