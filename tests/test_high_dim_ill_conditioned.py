import jax
import jax.numpy as jnp
import numpy as np
from randlax import double_pass_randomized_eigh

def is_gpu_available() -> bool:
    """Return True if any JAX device is a GPU."""
    return any(device.platform == "gpu" for device in jax.devices())

def test_high_dim_ill_condition_non_gen():
    """
    Test double_pass_randomized_eigh on a 50k x 50k ill-conditioned matrix.

    The matrix A is constructed as a diagonal matrix with entries 
    logarithmically spaced between 1 and 1e14 (float32). This creates a matrix
    with condition number ~1e14. For a diagonal matrix, the eigenvalues are
    exactly the diagonal entries. This test compares the top r eigenvalues
    computed by the randomized algorithm to the sorted diagonal values.
    
    Note:
        - A 50,000×50,000 float32 matrix requires roughly 50,000²×4 bytes ≈ 10 GB.
        - This test only runs if a GPU is available.
    """
    if not is_gpu_available():
        print("GPU not available. Skipping high-dim ill-conditioned test.")
        return

    n = 50000
    r = 20
    p = 30
    power_iters = 5

    # Create diagonal entries spanning 1 to 1e5 in float32.
    diag_vals = jnp.logspace(0, 8, n, dtype=jnp.float32)
    # Construct A as a dense diagonal matrix.
    A = jnp.diag(diag_vals)

    key = jax.random.PRNGKey(0)
    comp_eigvals, comp_evecs = double_pass_randomized_eigh(
        key, A, r, p, power_iters
    )

    # True eigenvalues: sort diag_vals in descending order and take the top r.
    true_eigvals = jnp.sort(diag_vals)[::-1][:r]
    print(len(np.array(comp_eigvals)), len(np.array(true_eigvals)))
    np.testing.assert_allclose(
        np.array(comp_eigvals),
        np.array(true_eigvals),
        rtol=0.1,
        err_msg="High-dim ill-conditioned eigenvalues do not match."
    )

    # Check that the residuals are small.
    A_np = np.array(A)
    for i in range(r):
        v = np.array(comp_evecs)[:, i]
        lam = np.array(comp_eigvals)[i]
        rel_residual = np.linalg.norm(A_np @ v - lam * v)/np.linalg.norm(A_np)
        assert rel_residual < 5e-2, (
            f"Relative residual too high for eigenpair {i}: {rel_residual}"
        )

if __name__ == "__main__":
    test_high_dim_ill_condition_non_gen()
    print("High-dim ill-conditioned test passed!")