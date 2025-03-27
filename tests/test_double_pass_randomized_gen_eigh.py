import jax
import jax.numpy as jnp
import numpy as np

from randlax import double_pass_randomized_gen_eigh


def test_small_matrix():
    key = jax.random.PRNGKey(0)
    # Define a small SPD matrix
    A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    C_inv = jnp.eye(2)
    r = 1
    p = 2

    # Compute the eigenpairs using your double pass routine
    computed_eigvals, computed_evecs = double_pass_randomized_gen_eigh(
        key, A, C_inv, r, p, power_iters=2, reorthog_iter=2
    )

    # Compute true eigenpairs using numpy
    A_np = np.array(A)
    true_eigvals, true_eigvecs = np.linalg.eigh(A_np)
    # Sort in descending order (largest eigenvalues first)
    desc_order = np.argsort(true_eigvals)[::-1]
    true_eigvals = true_eigvals[desc_order]
    true_eigvecs = true_eigvecs[:, desc_order]

    # Check that the computed eigenvalues are close to the top eigenvalue(s)
    np.testing.assert_allclose(
        np.array(computed_eigvals),
        true_eigvals[:r],
        rtol=0.05,
        atol=1e-3,
        err_msg="Eigenvalues do not match for small matrix.",
    )

    # Check the eigenvalue equation residual for each computed eigenpair
    for i in range(r):
        v = np.array(computed_evecs)[:, i]
        lambda_val = np.array(computed_eigvals)[i]
        res = np.linalg.norm(A_np @ v - lambda_val * v)
        assert res < 1e-2, f"Residual too high for eigenpair {i}: {res}"


def test_large_matrix():
    key = jax.random.PRNGKey(42)
    n = 10
    r = 5
    p = 7
    # Create a random SPD matrix: A = X.T @ X + 0.1 * I
    X = jax.random.normal(key, (n, n))
    A = X.T @ X + 0.1 * jnp.eye(n)
    C_inv = jnp.eye(n)

    # Compute the eigenpairs using your double pass routine
    computed_eigvals, computed_evecs = double_pass_randomized_gen_eigh(
        key, A, C_inv, r, p, power_iters=3, reorthog_iter=5
    )

    # Compute true eigenpairs using numpy
    A_np = np.array(A)
    true_eigvals, true_eigvecs = np.linalg.eigh(A_np)
    desc_order = np.argsort(true_eigvals)[::-1]
    true_eigvals = true_eigvals[desc_order]
    true_eigvecs = true_eigvecs[:, desc_order]

    # Compare computed eigenvalues with the top r true eigenvalues
    np.testing.assert_allclose(
        np.array(computed_eigvals),
        true_eigvals[:r],
        rtol=0.1,
        atol=1e-2,
        err_msg="Eigenvalues do not match for large matrix.",
    )

    # Check the eigenvalue equation residual for each computed eigenpair
    for i in range(r):
        v = np.array(computed_evecs)[:, i]
        lambda_val = np.array(computed_eigvals)[i]
        res = np.linalg.norm(A_np @ v - lambda_val * v)
        assert res < 1.5e-1, f"Residual too high for eigenpair {i}: {res}"


if __name__ == "__main__":
    test_small_matrix()
    test_large_matrix()
    print("All tests passed!")
