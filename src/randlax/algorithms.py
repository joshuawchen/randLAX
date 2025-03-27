from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


def __orthonormalize_inner(
    q: jnp.ndarray, Q_prev: jnp.ndarray, B: jnp.ndarray, iters: int, i: int
) -> jnp.ndarray:
    """
    Re-orthogonalizes vector q against previously orthonormalized columns in
    Q_prev using the B-inner product.

    Args:
        q: (n,) vector to orthonormalize.
        Q_prev: (n, r) matrix containing already orthonormalized vectors (only
            the first `i` columns are used).
        B: (n, n) symmetric positive-definite (SPD) matrix.
        iters: Number of re-orthogonalization iterations.
        i: Current index; only the first `i` columns of Q_prev are used.

    Returns:
        (n,) re-orthogonalized vector.
    """

    def body_fn(_, q):
        # Mask the first i columns of Q_prev
        mask = jnp.arange(Q_prev.shape[1]) < i
        Q_masked = Q_prev * mask[None, :]
        correction = jnp.einsum("ij,kj,kl,l->i", Q_masked, Q_masked, B, q)
        return q - correction

    return jax.lax.fori_loop(0, iters, body_fn, q)


def __mgs_b_orthonormalize(
    Q: jnp.ndarray, B: jnp.ndarray, reorthog_iter: int = 2
) -> jnp.ndarray:
    """
    Performs modified Gram-Schmidt orthonormalization with respect to the
    B-inner product.

    Args:
        Q: (n, p) matrix whose columns are to be orthonormalized.
        B: (n, n) symmetric positive-definite (SPD) matrix.
        reorthog_iter: Number of re-orthogonalization iterations per column.

    Returns:
        Q_out: (n, p) matrix with columns orthonormal with respect to B,
               i.e. Q_outᵀ B Q_out = I.
    """
    n, p = Q.shape
    Q_out = jnp.zeros((n, p))  # Preallocate output matrix

    # Initialize the first column.
    q0 = Q[:, 0]
    t0 = jnp.sqrt(jnp.einsum("i,ij,j->", q0, B, q0))
    needs_norm_0 = t0 > 0.0
    Q_out = Q_out.at[:, 0].set(jnp.where(needs_norm_0, q0 / t0, q0))

    def body_fn(carry, i):
        Q_out = carry
        # Extract the i-th column from Q.
        q_i = Q[:, i]
        # Re-orthogonalize q_i using already computed columns.
        q_orth = __orthonormalize_inner(q_i, Q_out, B, reorthog_iter, i)
        t = jnp.sqrt(jnp.einsum("i,ij,j->", q_orth, B, q_orth))
        needs_norm = t > 0.0
        q_norm = jnp.where(needs_norm, q_orth / t, q_orth)
        Q_out = Q_out.at[:, i].set(q_norm)
        return Q_out, None

    return jax.lax.scan(body_fn, Q_out, jnp.arange(1, p))[0]


def __power_iteration(
    Q: jnp.ndarray, A: jnp.ndarray, B: jnp.ndarray, power_iters: int
) -> jnp.ndarray:
    """
    Applies power iterations with preconditioning via B.

    At each iteration, solves for Q in the linear system:
        B * Q_new = A @ Q

    Args:
        Q: (n, p) initial subspace.
        A: (n, n) matrix.
        B: (n, n) symmetric positive-definite (SPD) matrix.
        power_iters: Number of power iterations to perform.

    Returns:
        Q after applying the power iterations.
    """
    for _ in range(power_iters):
        Q = jax.lax.stop_gradient(jax.scipy.linalg.solve(B, A @ Q))
    return Q


def __power_iteration_cholesky(
    Q: jnp.ndarray, A: jnp.ndarray, L: jnp.ndarray, power_iters: int
) -> jnp.ndarray:
    """
    Applies power iterations with preconditioning using the Cholesky
    factor of B.

    Given that B = L Lᵀ, we solve B * Q_new = A @ Q by computing:
        Y = A @ Q,
        Z = solve_triangular(L, Y, lower=True),
        Q_new = solve_triangular(L.T, Z, lower=False).
    
    Args:
        Q: (n, p) initial subspace.
        A: (n, n) matrix.
        L: (n, n) lower-triangular Cholesky factor of B.
        power_iters: Number of power iterations to perform.
    
    Returns:
        Q after applying the power iterations.
    """
    for _ in range(power_iters):
        Y = A @ Q
        Z = solve_triangular(L, Y, lower=True)
        Q = solve_triangular(L.T, Z, lower=False)
        Q = jax.lax.stop_gradient(Q)
    return Q

__jitted_power_iteration_cholesky = jax.jit(
    __power_iteration_cholesky, static_argnums=(3,), donate_argnums=(0,)
)

def double_pass_randomized_gen_eigh(
    key: jax.random.PRNGKey,
    A: jnp.ndarray,
    B: jnp.ndarray,
    r: int,
    p: int,
    power_iters: int = 1,
    reorthog_iter: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the dominant generalized eigenpairs for the problem:

        A u = λ B u,

    where A is a dense (n, n) matrix and B is a symmetric positive-definite
    (SPD) (n, n) matrix.

    The algorithm performs the following steps:
      1. Generates an initial subspace Q of shape (n, p) using a random normal
         distribution.
      2. Applies power iterations with preconditioning via B to enhance the
         separation of dominant eigenpairs.
      3. Orthonormalizes Q with respect to the B-inner product using a modified
         Gram-Schmidt routine.
      4. Computes the reduced eigen-decomposition on the projected matrix
         T = Qᵀ A Q.
      5. Maps the reduced eigenvectors back to the full space.

    Args:
        key: A JAX PRNGKey for random number generation.
        A: A (n, n) dense JAX array representing the matrix A.
        B: A (n, n) symmetric positive-definite (SPD) JAX array.
        r: The number of dominant eigenpairs to extract.
        p: The number of probing vectors (p must be at least r).
        power_iters: The number of power iterations to perform (default is 1).
        reorthog_iter: The number of re-orthogonalization iterations in
                       modified Gram-Schmidt (default is 3).

    Returns:
        A tuple (eigvals, eigvecs) where:
            eigvals: (r,) JAX array of the dominant eigenvalues (sorted in
                     descending order).
            eigvecs: (n, r) JAX array of the corresponding eigenvectors.
    """
    # Precompute the Cholesky factorization of B.
    L = jnp.linalg.cholesky(B)
    # Generate an initial subspace Q of shape (n, p).
    Q = jax.random.normal(key, (B.shape[0], p), dtype=A.dtype)
    # Apply power iterations using the Cholesky factor.
    Q = __jitted_power_iteration_cholesky(Q, A, L, power_iters)
    # # Generate an initial subspace Q of shape (n, p)
    # Q = jax.random.normal(key, (B.shape[0], p), dtype=A.dtype)
    # Q = __jitted_power_iteration(Q, A, B, power_iters)
    Q = __mgs_b_orthonormalize(Q, B, reorthog_iter=reorthog_iter)

    # Compute the projected matrix T = Qᵀ A Q and perform eigen-decomposition.
    T = jnp.einsum("ia,ij,jb->ab", Q, A, Q)
    evals, evecs = jnp.linalg.eigh(T)
    perm_r = jnp.argsort(evals)[::-1][:r]

    return evals[perm_r], Q @ (evecs[:, perm_r])


def double_pass_randomized_eigh(
    key: jax.random.PRNGKey,
    A: jnp.ndarray,
    r: int,
    p: int,
    power_iters: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the dominant eigenpairs of matrix A using a double-pass
    randomized algorithm.

    The algorithm performs the following steps:
      1. Generates an initial subspace Q of shape (n, p) using a random normal
         distribution.
      2. Applies power iterations with QR re-orthonormalization:
         For a given number of iterations, re-orthonormalizes Q by computing
         the QR factorization of A @ Q.
      3. Forms the projected matrix T = Qᵀ A Q using an einsum operation.
      4. Computes the eigen-decomposition of T.
      5. Sorts the eigenvalues in descending order and selects the top r eigenpairs,
         mapping the eigenvectors back to the original space.

    Args:
        key: A JAX PRNGKey for random number generation.
        A: A (n, n) JAX array representing the matrix to decompose.
        r: The number of dominant eigenpairs to extract.
        p: The number of probing vectors (must be at least r).
        power_iters: The number of power iterations to perform.

    Returns:
        A tuple (eigvals, eigvecs) where:
            eigvals: (r,) JAX array of the dominant eigenvalues (sorted in
                     descending order).
            eigvecs: (n, r) JAX array of the corresponding eigenvectors.
    """
    # Generate an initial subspace Q of shape (n, p)
    Q = jax.random.normal(key, (A.shape[0], p), dtype=A.dtype)

    # --- Power iterations: Build the subspace ---
    for _ in range(power_iters):
        Q, _ = jnp.linalg.qr(A @ Q)

    # --- Reduced eigen-decomposition ---
    # Compute the eigen-decomposition of Qᵀ A Q
    evals, evecs = jnp.linalg.eigh(jnp.einsum("ij,jk,kl->il", Q.T, A, Q))
    # Sort eigenvalues in descending order and select the top r
    perm_r = jnp.argsort(evals)[::-1][:r]
    return evals[perm_r], Q @ (evecs[:, perm_r])