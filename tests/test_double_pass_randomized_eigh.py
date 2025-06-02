"""
Tests for the double_pass_randomized_eigh algorithm.

Assumptions:
    - The input matrix A is symmetric. For correct behavior, A should be
      real symmetric (or Hermitian if complex) because jnp.linalg.eigh is used.
    - The algorithm performs QR re-orthonormalization during power iterations,
      ensuring the subspace Q remains (approximately) orthonormal.
    - The projected matrix T = Qᵀ A Q is symmetric, so jnp.linalg.eigh returns
      its eigen-decomposition correctly, with eigenvalues in ascending order.
      These are then sorted in descending order.
    - The tests compare the computed eigenpairs to those from np.linalg.eigh
      on the full matrix A.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from randlax import double_pass_randomized_eigh


def test_small_matrix():
    """
    Test the double_pass_randomized_eigh algorithm on a small 2x2 symmetric
    positive-definite matrix.

    This test verifies:
      - The computed dominant eigenvalue is close to the true dominant eigenvalue.
      - The eigenvalue equation residual ||A*v - λ*v|| is small for the computed eigenpair.

    Assumptions:
      - The matrix A is symmetric and positive-definite.
      - np.linalg.eigh returns eigenvalues in ascending order; we reverse them.
    """
    key = jax.random.PRNGKey(0)
    # Define a small symmetric positive-definite matrix.
    A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    r = 1
    p = 2
    power_iters = 3

    # Compute eigenpairs using the randomized algorithm.
    comp_eigvals, comp_evecs = double_pass_randomized_eigh(key, A, r, p, power_iters)

    # Compute "true" eigenpairs using numpy.
    A_np = np.array(A)
    true_eigvals, true_eigvecs = np.linalg.eigh(A_np)
    # Sort eigenvalues in descending order.
    desc_order = np.argsort(true_eigvals)[::-1]
    true_eigvals = true_eigvals[desc_order]
    true_eigvecs = true_eigvecs[:, desc_order]

    # Verify computed eigenvalues are close to the true dominant eigenvalue.
    np.testing.assert_allclose(
        np.array(comp_eigvals),
        true_eigvals[:r],
        rtol=0.05,
        atol=1e-3,
        err_msg="Eigenvalues do not match for small matrix.",
    )

    # Check that the eigenvalue equation residual is small.
    for i in range(r):
        v = np.array(comp_evecs)[:, i]
        lam = np.array(comp_eigvals)[i]
        residual = np.linalg.norm(A_np @ v - lam * v)
        assert residual < 1e-2, f"Residual too high for eigenpair {i}: {residual}"


def test_large_matrix():
    """
    Test the double_pass_randomized_eigh algorithm on a larger 10x10 symmetric
    positive-definite matrix.

    This test verifies:
      - The computed top r eigenvalues are close to the true eigenvalues.
      - The eigenvalue equation residuals ||A*v - λ*v|| are small.

    Assumptions:
      - The matrix A is constructed as A = Xᵀ X + 0.1 I, ensuring it is symmetric
        and positive definite.
      - The QR re-orthonormalization in the power iterations maintains a good
        approximation of the subspace.
    """
    key = jax.random.PRNGKey(42)
    n = 10
    r = 5
    p = 7
    power_iters = 5

    # Create a random symmetric positive-definite matrix:
    # A = Xᵀ X + 0.1 * I.
    X = jax.random.normal(key, (n, n))
    A = X.T @ X + 0.1 * jnp.eye(n)

    comp_eigvals, comp_evecs = double_pass_randomized_eigh(key, A, r, p, power_iters)

    A_np = np.array(A)
    true_eigvals, true_eigvecs = np.linalg.eigh(A_np)
    # Sort eigenvalues in descending order.
    desc_order = np.argsort(true_eigvals)[::-1]
    true_eigvals = true_eigvals[desc_order]
    true_eigvecs = true_eigvecs[:, desc_order]

    # Verify computed eigenvalues are close to the true eigenvalues.
    np.testing.assert_allclose(
        np.array(comp_eigvals),
        true_eigvals[:r],
        rtol=0.1,
        atol=1e-2,
        err_msg="Eigenvalues do not match for large matrix.",
    )

    # Check the eigenvalue equation residual for each computed eigenpair.
    for i in range(r):
        v = np.array(comp_evecs)[:, i]
        lam = np.array(comp_eigvals)[i]
        residual = np.linalg.norm(A_np @ v - lam * v)
        assert residual < 1e-1, f"Residual too high for eigenpair {i}: {residual}"


if __name__ == "__main__":
    test_small_matrix()
    test_large_matrix()
    print("All tests passed!")
