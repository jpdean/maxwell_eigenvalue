# Based on [1]

# References:
# [1]: https://fenicsproject.org/olddocs/dolfin/latest/python/demos/maxwell-eigenvalues/demo_maxwell-eigenvalues.py.html

import dolfinx
from slepc4py import SLEPc
from ufl import ds, dx, curl, inner, TrialFunction, TestFunction
import numpy as np
import ufl
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh, fem
from dolfinx.fem import locate_dofs_topological
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI
from dolfinx.fem import assemble_matrix, locate_dofs_geometrical
from petsc4py import PETSc
from dolfinx.cpp.mesh import CellType
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix


def eigenvalues(n_eigs, V, bc):
    # Define problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(curl(u), curl(v)) * dx
    b = inner(u, v) * dx

    # Assemble matrices
    # TODO Check this preserves symmetry, see comment in [1]
    A = assemble_matrix(a, [bc])
    A.assemble()
    B = assemble_matrix(b, [bc])
    B.assemble()

    # Zero rows of boundary DOFs of B. See [1]
    # FIXME This is probably a stupid way of doing it
    dof_indices = bc.dof_indices()[0]
    for index in dof_indices:
        B.setValue(index, index, 0)
    B.assemble()
    # bi, bj, bv = B.getValuesCSR()
    # Bsp = csr_matrix((bv, bj, bi))
    # with open('B_new.npy', 'wb') as f:
    #     np.save(f, Bsp.toarray())

    # create SLEPc Eigenvalue solver
    E = SLEPc.EPS().create(PETSc.COMM_WORLD)
    E.setOperators(A, B)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)

    st = E.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(5.5)

    E.setDimensions(n_eigs, PETSc.DECIDE, PETSc.DECIDE)
    E.setFromOptions()
    E.solve()

    its = E.getIterationNumber()
    print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    print("Solution method: %s" % eps_type)

    nev, ncv, mpd = E.getDimensions()
    print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    n_conv = E.getConverged()
    print("Number of converged eigenpairs %d" % n_conv)

    # if nconv > 0:
    #     # Create the results vectors
    #     vr, vi = A.createVecs()
    #     for i in range(nconv):
    #         k = E.getEigenpair(i, vr, vi)
    #         error = E.computeError(i)
    #         print(f"Eigenvalue = {round(k.real, ndigits=2)},  Error = {error}")
    #         # print(E.getEigenvalue(i))

    computed_eigenvalues = []
    for i in range(n_conv):
        lmbda = E.getEigenvalue(i)
        if not np.isclose(lmbda, 0) and len(computed_eigenvalues) < n_eigs:
            computed_eigenvalues.append(np.round(np.real(lmbda)))
    return np.sort(computed_eigenvalues)

    # ai, aj, av = A.getValuesCSR()
    # Asp = csr_matrix((av, aj, ai))
    # bi, bj, bv = B.getValuesCSR()
    # Bsp = csr_matrix((bv, bj, bi))
    # vals, vecs = eigs(Asp, M=Bsp, sigma=5.5, k=12)
    # print(np.sort(np.real(vals)))


def boundary(x):
    lr = np.logical_or(np.isclose(x[0], 0.0),
                       np.isclose(x[0], np.pi))
    tb = np.logical_or(np.isclose(x[1], 0.0),
                       np.isclose(x[1], np.pi))
    return np.logical_or(lr, tb)


# Number of element in wach direction
n = 40
# Number of eigernvalues to compute
n_eigs = 12

# Create mesh and function space
mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([np.pi, np.pi, 0])], [n, n],
    CellType.triangle, dolfinx.cpp.mesh.GhostMode.none,
    diagonal="right")
V = FunctionSpace(mesh, ("N1curl", 1))

# Set boundart DOFs to 0 (u x n = 0 on \partial \Omega).
ud = Function(V)
with ud.vector.localForm() as bc_local:
    bc_local.set(0.0)
bc = DirichletBC(ud, locate_dofs_geometrical(V, boundary))

# Solve Maxwell eigenvalue problem
nedelec_eigenvalues = eigenvalues(n_eigs, V, bc)

# Print results
np.set_printoptions(formatter={'float': '{:5.1f}'.format})
exact_eigenvalues = np.sort(np.array([float(m**2 + n**2)
                                      for m in range(6)
                                      for n in range(6)]))[1:13]
print(f"Exact   = {exact_eigenvalues}")
print(f"Nédélec = {nedelec_eigenvalues}")
