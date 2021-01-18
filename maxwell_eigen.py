# Based on [1]

# References:
# [1]: https://fenicsproject.org/olddocs/dolfin/latest/python/demos/maxwell-eigenvalues/demo_maxwell-eigenvalues.py.html
# [2]: https://bitbucket.org/fenics-project/dolfin/src/master/dolfin/la/SLEPcEigenSolver.cpp
# [3]: https://slepc.upv.es/slepc4py-current/docs/apiref/slepc4py.SLEPc-module.html

import dolfinx
from slepc4py import SLEPc
from ufl import dx, curl, inner, TrialFunction, TestFunction
import numpy as np
from dolfinx import (DirichletBC, Function, FunctionSpace, RectangleMesh,
                     VectorFunctionSpace)
from mpi4py import MPI
from dolfinx.fem import assemble_matrix, locate_dofs_geometrical
from petsc4py import PETSc
from dolfinx.cpp.mesh import CellType


def eigenvalues(n_eigs, shift, V, bcs):
    # Define problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(curl(u), curl(v)) * dx
    b = inner(u, v) * dx

    # Assemble matrices
    # TODO Check this preserves symmetry, see comment in [1]
    A = assemble_matrix(a, bcs)
    A.assemble()
    B = assemble_matrix(b, bcs)
    B.assemble()

    # Zero rows of boundary DOFs of B. See [1]
    # FIXME This is probably a stupid way of doing it
    for bc in bcs:
        dof_indices = bc.dof_indices()[0]
        for index in dof_indices:
            B.setValue(index, index, 0)
    B.assemble()

    # Create SLEPc Eigenvalue solver
    # Settings found by following [1] and finding actual SLEPc settings from
    # old FEniCS SLEPcEigenSolver.cpp documentation [2] and comparing with
    # slepc4py documentation [3].
    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A, B)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(eps.Which.TARGET_MAGNITUDE)
    eps.setTarget(shift)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(shift)

    eps.setDimensions(n_eigs, PETSc.DECIDE, PETSc.DECIDE)
    eps.setFromOptions()
    eps.solve()

    its = eps.getIterationNumber()
    print(f"Number of iterations: {its}")

    eps_type = eps.getType()
    print(f"Solution method: {eps_type}")

    n_ev, n_cv, mpd = eps.getDimensions()
    print(f"Number of requested eigenvalues: {n_ev}")

    tol, max_it = eps.getTolerances()
    print(f"Stopping condition: tol={tol}, maxit={max_it}")

    n_conv = eps.getConverged()
    print(f"Number of converged eigenpairs: {n_conv}")

    computed_eigenvalues = []
    for i in range(n_conv):
        lmbda = eps.getEigenvalue(i)
        # Ignore zero eigenvalues, see [1]
        # FIXME See if zero eigenvalue check is still needed
        if not np.isclose(lmbda, 0) and len(computed_eigenvalues) < n_eigs:
            computed_eigenvalues.append(np.round(np.real(lmbda), 1))
    return np.sort(computed_eigenvalues)


def boundary(x):
    lr = boundary_lr(x)
    tb = boundary_tb(x)
    return np.logical_or(lr, tb)


def boundary_lr(x):
    return np.logical_or(np.isclose(x[0], 0.0),
                         np.isclose(x[0], np.pi))


def boundary_tb(x):
    return np.logical_or(np.isclose(x[1], 0.0),
                         np.isclose(x[1], np.pi))


# Number of element in wach direction
n = 40
# Number of eigernvalues to compute
n_eigs = 12
# Find eigenvalues near
shift = 5.5

# Create mesh and function space
mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([np.pi, np.pi, 0])], [n, n],
    CellType.triangle, dolfinx.cpp.mesh.GhostMode.none,
    diagonal="right")
    # diagonal="crossed")

# Nédélec
V_nedelec = FunctionSpace(mesh, ("N1curl", 1))

# Set boundary DOFs to 0 (u x n = 0 on \partial \Omega).
ud_nedelec = Function(V_nedelec)
with ud_nedelec.vector.localForm() as bc_local:
    bc_local.set(0.0)
bcs_nedelec = [DirichletBC(ud_nedelec,
                           locate_dofs_geometrical(V_nedelec, boundary))]

# Solve Maxwell eigenvalue problem
eigenvalues_nedelec = eigenvalues(n_eigs, shift, V_nedelec, bcs_nedelec)

# Lagrange
V_vec_lagrange = VectorFunctionSpace(mesh, ("Lagrange", 1))
V_lagrange = FunctionSpace(mesh, ("Lagrange", 1))

# Zero function
ud_lagrange = Function(V_lagrange)
with ud_lagrange.vector.localForm() as ud_lagrange_local:
    ud_lagrange_local.set(0.0)
# Find correct DOFs to constrain. Must constrain horizontal DOFs on
# horizontal faces and vertical DOFs on vertical faces
dofs_0 = dolfinx.fem.locate_dofs_geometrical(
    (V_vec_lagrange.sub(0), V_lagrange), boundary_tb)
dofs_1 = dolfinx.fem.locate_dofs_geometrical(
    (V_vec_lagrange.sub(1), V_lagrange), boundary_lr)
bcs_lagrange = [DirichletBC(ud_lagrange, dofs_0, V_vec_lagrange.sub(0)),
                DirichletBC(ud_lagrange, dofs_1, V_vec_lagrange.sub(1))]

# Solve Maxwell eigenvalue problem
eigenvalues_lagrange = eigenvalues(n_eigs, shift, V_vec_lagrange, bcs_lagrange)

# Print results
np.set_printoptions(formatter={'float': '{:5.1f}'.format})
eigenvalues_exact = np.sort(np.array([float(m**2 + n**2)
                                      for m in range(6)
                                      for n in range(6)]))[1:13]
print(f"Exact    = {eigenvalues_exact}")
print(f"Nédélec  = {eigenvalues_nedelec}")
print(f"Lagrange = {eigenvalues_lagrange}")
