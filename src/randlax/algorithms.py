from typing import Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular


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
        Z = solve_triangular(L, A @ Q, lower=True)
        Q = solve_triangular(L.T, Z, lower=False)
        Q = jax.lax.stop_gradient(Q)
    return Q


__jitted_power_iteration_cholesky = jax.jit(
    __power_iteration_cholesky, static_argnums=(3,), donate_argnums=(0,)
)


def __orthonormalize_inner(
    q: jnp.ndarray,  # shape (n,), dtype=float64
    Q_prev: jnp.ndarray,  # shape (n, p), dtype=float64; first i cols are B-orthonormal
    B: jnp.ndarray,  # shape (n, n), dtype=float64 (SPD)
    i: int,  # project onto columns 0..(i-1)
) -> jnp.ndarray:
    """
    Re-orthogonalize vector q against Q_prev[:, 0..(i-1)] in the B-inner product,
    using Rutishauser’s “while (tt > 10 eps t and tt < t/10) do re-orth” logic.
    Returns a unit-B-norm result (or zero if the norm collapses).
    """
    # Force float64 everywhere
    q = q.astype(jnp.float64)
    Q_prev = Q_prev.astype(jnp.float64)
    B = B.astype(jnp.float64)

    eps = jnp.finfo(jnp.float64).eps

    # Initial norm: t0 = sqrt(q^T B q)
    Bq0 = B @ q
    t0 = jnp.sqrt(q @ Bq0)

    # Loop state: (q_cur, t_old, t_last, keep_looping)
    def cond_fn(state):
        _, _, _, do_again = state
        return do_again

    def body_fn(state):
        q_cur, t_old, _unused_last, _unused_flag = state

        # 1) Compute raw projections against all columns, then mask j >= i
        Bq_cur = B @ q_cur  # (n,)
        raw_coeffs = Q_prev.T @ Bq_cur  # (p,)
        mask = (jnp.arange(Q_prev.shape[1]) < i).astype(jnp.float64)  # (p,)
        proj_coeffs = raw_coeffs * mask  # zero out j>=i

        # 2) Subtract the B-inner projection
        correction = Q_prev @ proj_coeffs  # (n,)
        q_new = q_cur - correction

        # 3) Compute new norm: tt = sqrt(q_new^T B q_new)
        Bq_new = B @ q_new
        t_new = jnp.sqrt(q_new @ Bq_new)

        # 4) Check Rutishauser’s criterion
        cond = (t_new > (t_old * 10.0 * eps)) & (t_new < (t_old / 10.0))

        # If stopping and t_new is “tiny,” force t_new=0
        t_new_final = jnp.where((~cond) & (t_new < (10.0 * eps * t_old)), 0.0, t_new)

        t_old_next = jnp.where(cond, t_new, t_old)
        cont_next = cond

        return (q_new, t_old_next, t_new_final, cont_next)

    # Initialize: q_cur=q, t_old=t0, t_last=t0, keep looping = True
    init_state = (q, t0, t0, True)
    q_fin, t_old_fin, t_last_fin, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # Finally, normalize (or leave zero if t_last_fin=0)
    q_out = jnp.where(t_last_fin > 0.0, q_fin / t_last_fin, q_fin)
    return q_out


@jax.jit
def mgs_b_orthonormalize_jit(Q: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-compiled wrapper for __mgs_b_orthonormalize.  Takes Q (n,p), B (n,n),
    returns Q_B (n,p) so that Q_B^T B Q_B = I_p up to ~1e-15.
    """
    return __mgs_b_orthonormalize(Q, B)


def __mgs_b_orthonormalize(
    Q: jnp.ndarray,  # (n, p), dtype=float64
    B: jnp.ndarray,  # (n, n), dtype=float64 (SPD)
) -> jnp.ndarray:
    """
    Perform Modified Gram–Schmidt with dynamic Rutishauser re-orthogonalization
    so that columns of Q become B-orthonormal.  Returns Q_out with
    Q_out^T B Q_out = I_p (to ~1e-15).
    """
    # Force float64
    Q = Q.astype(jnp.float64)
    B = B.astype(jnp.float64)

    n, p = Q.shape
    Q_out = jnp.zeros((n, p), dtype=jnp.float64)

    # Normalize the very first column in the B-norm:
    q0 = Q[:, 0]
    Bq0 = B @ q0
    t0 = jnp.sqrt(q0 @ Bq0)
    needs0 = t0 > 0.0
    q0n = jnp.where(needs0, q0 / t0, q0)
    Q_out = Q_out.at[:, 0].set(q0n)

    # For k=1..(p-1), re-orthonormalize against Q_out[:, 0..(k-1)]
    def scan_body(Q_acc, k):
        qk = Q[:, k]
        qk_orth = __orthonormalize_inner(qk, Q_acc, B, k)
        Q_acc = Q_acc.at[:, k].set(qk_orth)
        return Q_acc, None

    ks = jnp.arange(1, p)
    Q_final, _ = jax.lax.scan(scan_body, Q_out, ks)

    return Q_final


def double_pass_randomized_gen_eigh(
    key: jax.random.PRNGKey,
    A: jnp.ndarray,
    B: jnp.ndarray,
    r: int,
    p: int,
    power_iters: int = 1,
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

    Returns:
        A tuple (eigvals, eigvecs) where:
            eigvals: (r,) JAX array of the dominant eigenvalues (sorted in
                     descending order).
            eigvecs: (n, r) JAX array of the corresponding eigenvectors.
    """

    A = A.astype(jnp.float64)
    B = B.astype(jnp.float64)

    # Precompute the Cholesky factorization of B.
    L = jnp.linalg.cholesky(B)
    # Generate an initial subspace Q of shape (n, p).
    Q = jax.random.normal(key, (B.shape[0], p), dtype=A.dtype)
    # Apply power iterations using the Cholesky factor.
    Q = __jitted_power_iteration_cholesky(Q, A, L, power_iters)
    # # Generate an initial subspace Q of shape (n, p)
    Q = mgs_b_orthonormalize_jit(Q, B)
    # err = jnp.max(jnp.abs(Q.T @ B @ Q - jnp.eye(p)))
    # print("max |B_inner - I| =", err)
    # Compute the projected matrix T = Q^T A Q and perform eigen-decomposition.
    T = jnp.einsum("ia,ij,jb->ab", Q, A, Q)
    # assert jnp.allclose(T, T.T, atol=1e-6)
    # sym_err = jnp.max(jnp.abs(T - T.T))
    # print("max |T − T^T| =", sym_err)     # expect ~1e-15…1e-13

    evals, evecs = jnp.linalg.eigh(T)
    perm_r = jnp.argsort(evals)[::-1][:r]

    return evals[perm_r], Q @ (evecs[:, perm_r])


def double_pass_randomized_eigh(
    key: jax.random.PRNGKey, A: jnp.ndarray, r: int, p: int, power_iters: int
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
    # Compute the eigen-decomposition of Q^T A Q
    evals, evecs = jnp.linalg.eigh(jnp.einsum("ij,jk,kl->il", Q.T, A, Q))
    # Sort eigenvalues in descending order and select the top r
    perm_r = jnp.argsort(evals)[::-1][:r]
    
    return evals[perm_r], Q @ (evecs[:, perm_r])
