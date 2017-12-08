from fenics import Constant, DirichletBC, Expression, Function, FunctionSpace, MeshFunction, TestFunction, TrialFunction, UnitSquareMesh, \
    assemble, dx, cells, grad, inner, interpolate, refine, solve, sqrt


# Set log level
import logging
logging.getLogger("FFC").setLevel(logging.WARNING)


def compute_error(u1, u2):
    """
    L1 error between two functions u1 and u2

    :param u1: FEniCS function
    :param u2: FEniCS function
    :return: Approximate L1 error between u1 and u2
    """
    mesh_resolution_ref = 400
    mesh_ref = UnitSquareMesh(mesh_resolution_ref, mesh_resolution_ref)
    V_ref = FunctionSpace(mesh_ref, "CG", degree=1)
    Iu1 = interpolate(u1, V_ref)
    Iu2 = interpolate(u2, V_ref)
    error = assemble(abs(Iu1-Iu2) * dx)
    return error


def dual_error_estimates(resolution):
    mesh = UnitSquareMesh(resolution, resolution)

    def all_boundary(_, on_boundary):
        return on_boundary

    zero = Constant(0.0)

    def a(u, v):
        return inner(grad(u), grad(v)) * dx

    def L(f, v):
        return f * v * dx

    # Primal problem
    f = Expression("32.*x[0]*(1. - x[0])+32.*x[1]*(1. - x[1])", domain=mesh, degree=5)
    ue = Expression("16.*x[0]*(1. - x[0])*x[1]*(1. - x[1])", domain=mesh, degree=5)

    Qp = FunctionSpace(mesh,'CG',1)
    bcp = DirichletBC(Qp, zero, all_boundary)

    u = TrialFunction(Qp)
    v = TestFunction(Qp)

    U = Function(Qp)
    solve(a(u, v) == L(f, v), U, bcp)

    # Dual problem
    Qd = FunctionSpace(mesh, 'CG', 2)
    psi = Constant(1.0)
    bcd = DirichletBC(Qd, zero, all_boundary)

    w = TestFunction(Qd)
    phi = TrialFunction(Qd)
    Phi = Function(Qd)
    solve(a(w, phi) == L(psi, w), Phi, bcd)

    # Compute errors
    e1 = compute_error(ue, U)
    e2 = assemble((inner(grad(U), grad(Phi)) - f * Phi) * dx)
    print("e1 = {}".format(e1))
    print("e2 = {}".format(e2))


def part_1():
    print("Part 1:")
    resolution = 10
    dual_error_estimates(resolution)


def part_2():
    print("Part 2:")
    for resolution in range(5, 10):
        dual_error_estimates(resolution)


if __name__ == "__main__":
    part_1()
    print()
    print()
    part_2()
