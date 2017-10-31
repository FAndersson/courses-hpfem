from dolfin import IntervalMesh
from dolfin.fem.interpolation import interpolate
from dolfin.functions import FunctionSpace, TestFunction, Function, Expression


# Define domain and mesh
a, b = 0, 1
mesh_resolution = 20
mesh = IntervalMesh(mesh_resolution, a, b)

# Define finite element function space
p_order = 1
V = FunctionSpace(mesh, "CG", p_order)

# Interpolate function
u = Expression("1 + 4.0 * x[0] * x[0] - 5.0 * x[0] * x[0] * x[0]", degree=5)
Iu = interpolate(u, V)
Iua = Iu.vector().array()
print(Iua.max())
