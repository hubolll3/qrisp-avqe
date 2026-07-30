import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# Enable double precision (64-bit) floating-point calculations in JAX.
# Double precision is necessary for accurately computing geometric metric tensors (QFIM)
# and energy standard deviations (variance), where loss of precision leads to numerical instabilities.
jax.config.update("jax_enable_x64", True)

import qrisp
from qrisp.operators import X, Y, Z
from qiskit_aer import AerSimulator

# Set up AerSimulator as the virtual backend for Qrisp circuit synthesis and testing
qrisp.environments.virtual_backend = AerSimulator()


# ==========================================
# 1. UTILITIES & PRE-COMPILATION
# ==========================================

def pauli_string_to_matrix(p_str):
    """
    Converts a Pauli string representation (e.g., 'IXYZ') into its full 2^n x 2^n 
    dense complex matrix via Kronecker tensor products.
    """
    # Define single-qubit fundamental Pauli matrices in JAX complex128 format
    PAULIS = {
        'I': jnp.array([[1, 0], [0, 1]], dtype=jnp.complex128),
        'X': jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128),
        'Y': jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128),
        'Z': jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128),
    }
    # Initialize full matrix with the first qubit's operator
    mat = PAULIS[p_str[0]]
    # Successively compute tensor product for subsequent qubits: mat = mat ⊗ P_i
    for char in p_str[1:]:
        mat = jnp.kron(mat, PAULIS[char])
    return mat


def qrisp_to_jax_matrix(op, n_qubits):
    """
    Converts a Qrisp Operator object into a dense JAX matrix representation.
    """
    np_mat = op.to_array(factor_amount=n_qubits)
    return jnp.array(np_mat, dtype=jnp.complex128)


def precompute_generator_matrices(P):
    """
    Pre-computes and caches matrix representations for all Pauli string generators P_j
    to avoid repeatedly converting strings during gradient descent iterations.
    """
    return [pauli_string_to_matrix(p_str) for p_str in P]


def ansatz_to_qrisp_circuit(P, params):
    """
    Constructs an equivalent Qrisp QuantumCircuit for the variational state |ψ(θ)⟩
    by trotterizing and sequentially applying single-parameter Pauli rotations e^(-i * θ_j * P_j).
    """
    n_qubits = len(P[0])
    qv = qrisp.QuantumVariable(n_qubits)

    for j, p_str in enumerate(P):
        theta = float(params[j])
        term = None
        # Build composite multi-qubit Pauli operator term
        for i, char in enumerate(p_str):
            op = None
            if char == 'X': op = X(i)
            elif char == 'Y': op = Y(i)
            elif char == 'Z': op = Z(i)

            if op is not None:
                term = op if term is None else term * op

        # Apply exponentiated unitary gate exp(-i * theta * P_j) onto quantum variable
        if term is not None:
            U = term.trotterization()
            U(qv, t=theta)

    return qv.qs


# ==========================================
# 2. METRICS & STATE CALCULATIONS
# ==========================================

def get_statevector(params, P_mats):
    """
    Simulates statevector evolution under the variational ansatz |ψ(θ)⟩ = ∏ e^(-i θ_j P_j) |0⟩.
    Exploits Pauli involutive identity: e^(-i θ P) = cos(θ) I - i sin(θ) P.
    """
    n_qubits = int(np.log2(P_mats[0].shape[0]))
    # Initialize state vector to |0...0⟩
    psi = jnp.zeros(2**n_qubits, dtype=jnp.complex128).at[0].set(1.0 + 0.0j)

    # Sequentially apply parameter-dependent unitary rotations
    for j, P_mat in enumerate(P_mats):
        theta = params[j]
        psi = jnp.cos(theta) * psi - 1j * jnp.sin(theta) * (P_mat @ psi)

    return psi


def measure_val(H_mat, P_mats, params):
    """
    Computes variational expectation value of energy: E(θ) = ⟨ψ(θ)| H |ψ(θ)⟩.
    """
    psi = get_statevector(params, P_mats)
    return jnp.real(jnp.vdot(psi, H_mat @ psi))


def get_energy_and_grad(params, H_mat, P_mats):
    """
    Computes variational energy expectation E(θ) along with its exact analytical 
    gradient vector ∇E(θ) using JAX automatic differentiation (`value_and_grad`).
    """
    def loss(p):
        return measure_val(H_mat, P_mats, p)
    return jax.value_and_grad(loss)(params)


def get_sigma(H_mat, P_mats, params):
    """
    Computes energy standard deviation (variance certificate): 
    σ_ψ(H) = √( ⟨ψ| H² |ψ⟩ - ⟨ψ| H |ψ⟩² ).
    A value σ_ψ(H) → 0 confirms |ψ⟩ is an exact eigenstate of H.
    """
    psi = get_statevector(params, P_mats)
    E = jnp.real(jnp.vdot(psi, H_mat @ psi))
    H2_psi = H_mat @ (H_mat @ psi)
    E2 = jnp.real(jnp.vdot(psi, H2_psi))
    # Enforce non-negativity under square root to handle minor float roundoff issues
    return jnp.sqrt(jnp.maximum(0.0, E2 - E**2))


def compute_qfim(params, P_mats):
    """
    Computes the Quantum Fisher Information Matrix (Fubini-Study Metric Tensor) g_μν(θ):
    g_μν = Re( ⟨∂_μ ψ | ∂_ν ψ⟩ - ⟨∂_μ ψ | ψ⟩ ⟨ψ | ∂_ν ψ⟩ ).
    Calculated efficiently using forward-mode automatic differentiation Jacobian J.
    """
    psi = get_statevector(params, P_mats)
    # Forward Jacobian of the state vector with respect to variational parameters
    J = jax.jacfwd(get_statevector, argnums=0)(params, P_mats)
    
    psi_col = jnp.reshape(psi, (-1, 1))
    J_dag = jnp.conj(J.T)
    v = J_dag @ psi_col
    
    # Raw Fubini-Study metric tensor
    g = jnp.real((J_dag @ J) - (v @ jnp.conj(v.T)))
    # Symmetrize metric matrix to preserve mathematical Hermitian symmetry
    return 0.5 * (g + g.T)


def get_fidelity(P_mats, params, target_state):
    """
    Calculates quantum state fidelity F = |⟨ψ_target | ψ(θ)⟩|² relative to an exact target state.
    """
    psi = get_statevector(params, P_mats)
    return float(jnp.abs(jnp.vdot(target_state, psi)) ** 2)


# ==========================================
# 3. OPTIMIZER STEP
# ==========================================

def qng_step_baseline(params, H_mat, P_mats, lr=0.08, eps=1e-3, max_grad_norm=0.5):
    """
    Performs one step of Quantum Natural Gradient (QNG) descent:
    θ^(k+1) = θ^(k) - lr * (g + εI)^(-1) ∇E(θ).
    
    Includes adaptive Tikhonov regularization, natural gradient norm clipping, 
    and line-search safeguards against energy increases.
    """
    energy, grad = get_energy_and_grad(params, H_mat, P_mats)
    grad_norm = jnp.linalg.norm(grad)

    # Stationary point safeguard
    if grad_norm < 1e-8:
        return params, energy, grad_norm

    # Compute Quantum Fisher Information Matrix (QFIM)
    g = compute_qfim(params, P_mats)

    current_eps = eps
    # Adaptive regularization loop: increase diagonal shift if state update fails
    for _ in range(8):
        g_reg = g + current_eps * jnp.eye(g.shape[0], dtype=g.dtype)
        try:
            # Solve linear equation (g + εI) * nat_grad = grad
            nat_grad = jnp.linalg.solve(g_reg, grad)
        except Exception:
            # Fallback to pseudo-inverse if linear solve encounters singular matrix
            nat_grad = jnp.linalg.pinv(g_reg, rcond=1e-4) @ grad

        # Clip natural gradient magnitude to ensure optimization stability
        gnorm = jnp.linalg.norm(nat_grad)
        if gnorm > max_grad_norm:
            nat_grad = nat_grad * (max_grad_norm / gnorm)

        candidate_params = params - lr * nat_grad
        candidate_energy = float(measure_val(H_mat, P_mats, candidate_params))

        # Accept step if energy decreases or stays within tolerance
        if candidate_energy <= energy + 1e-12:
            return candidate_params, candidate_energy, grad_norm

        # Increase regularization strength if step increased energy
        current_eps *= 4.0

    # Fallback to standard steepest Euclidean gradient descent if QNG fails
    fallback_params = params - 0.02 * (grad / (grad_norm + 1e-8))
    fallback_energy = float(measure_val(H_mat, P_mats, fallback_params))
    if fallback_energy < energy:
        return fallback_params, fallback_energy, grad_norm

    # Return unchanged parameters if energy could not be reduced
    return params, energy, grad_norm


# ==========================================
# 4. AVQE MAIN LOOP
# ==========================================

def self_verifying_AVQE_jax(
    H_i, H_f, P, params_0, dl_A, K, delta_C, lr=0.08, eps=1e-3, max_grad_norm=0.5,
    name=None, live_plot=False, track_exact=False
):
    """
    Parameters
    ----------
    H_i, H_f : Qrisp QubitOperators defining the initial and target Hamiltonians.
    P_strings : List of Pauli strings representing ansatz generators, shape (m,n), where n is the number of qubits.
    params_0 : Initial variational parameter array of length m.
    dl_A : Maximum adiabatic path step size.
    K : Number of optimization steps between verification variance checks.
    delta_C : Gap threshold certifying ground-state tracking (σ_H < Δ_C / 2).
    lr : Optimization learning rate.
    name : Output filename for metric plot generation.
    live_plot : Enables real-time progress plotting.
    track_exact : Enables exact sparse ground-state diagonalizations for verification benchmarks.
    """
    
    n_qubits = len(P[0])

    # Convert initial/final Hamiltonians and precompute Pauli generator matrices
    H_i_mat = qrisp_to_jax_matrix(H_i, n_qubits)
    H_f_mat = qrisp_to_jax_matrix(H_f, n_qubits)
    H_diff_mat = H_f_mat - H_i_mat
    P_mats = precompute_generator_matrices(P)

    # Define linear adiabatic Hamiltonian schedule H(λ) = (1 - λ) H_i + λ H_f
    def H_mat(t):
        return (1.0 - t) * H_i_mat + t * H_f_mat

    lbd = 0.0
    # Add small Gaussian noise to initial parameters to break potential symmetries
    np.random.seed(42)
    params = jnp.array(params_0, dtype=jnp.float64) + jnp.array(np.random.normal(0, 0.01, size=len(params_0)))

    start_time = time.time()
    last_check_time = time.time()

    # Configure real-time interactive plotting environment if enabled
    if live_plot:
        plt.ion()
        fig, axes = plt.subplots(3 if track_exact else 2, 1, figsize=(9, 9.5), sharex=True)
        ax_energy, ax_sigma = axes[0], axes[1]
        ax_fid = axes[2] if track_exact else None

    check_steps, energies, exact_energies, sigmas, infidelities = [], [], [], [], []
    lambda_change_steps = []

    check_count = 0
    step = 0
    prev_energy = float('inf')
    stagnant_checks = 0

    print(f"\n--- Starting JAX AVQE ({n_qubits} Qubits, {len(P)} Generators) ---")

    # Main adiabatic progression loop across schedule λ ∈ [0, 1]
    while lbd < 1.0:
        step += 1

        # Inner verification optimization loop
        while True:
            # Execute K Quantum Natural Gradient optimization steps on current H(λ)
            for _ in range(K):
                params, _, grad_norm = qng_step_baseline(
                    params, H_mat(lbd), P_mats, lr=lr, eps=eps, max_grad_norm=max_grad_norm
                )

            check_count += 1
            now = time.time()
            check_duration = now - last_check_time
            total_elapsed = now - start_time
            last_check_time = now

            # Compute verification energy and variance metrics
            current_energy = float(measure_val(H_mat(lbd), P_mats, params))
            sigma_H = float(get_sigma(H_mat(lbd), P_mats, params))

            check_steps.append(check_count)
            energies.append(current_energy)
            sigmas.append(sigma_H)

            # Perform optional exact diagonalizations for ground-state benchmark tracking
            infidelity_str = ""
            if track_exact:
                evals, evecs = np.linalg.eigh(np.array(H_mat(lbd)))
                exact_E0 = float(evals[0])
                fidelity = get_fidelity(P_mats, params, evecs[:, 0])
                infidelity = abs(1.0 - fidelity)

                exact_energies.append(exact_E0)
                infidelities.append(infidelity)
                infidelity_str = f", Infidelity = {infidelity:.4e}"

            # Render real-time convergence diagnostics
            if live_plot:
                ax_energy.clear()
                ax_sigma.clear()
                if ax_fid: ax_fid.clear()

                mins, secs = divmod(int(total_elapsed), 60)
                time_str = f"Total Time: {mins:02d}m {secs:02d}s  |  Last {K} Steps: {check_duration:.2f}s"
                fig.suptitle(f"AVQE (JAX, {n_qubits}-Qubit, K={K})\n{time_str}", fontsize=13, fontweight="bold")

                ax_energy.plot(check_steps, energies, 'm-^', markersize=4, label=r"Prepared $\langle H \rangle$")
                if track_exact:
                    ax_energy.plot(check_steps, exact_energies, 'k--', alpha=0.7, label=r"Exact $E_0(\lambda)$")
                ax_energy.set_ylabel("Energy")
                ax_energy.grid(True, alpha=0.3)
                ax_energy.legend(loc="lower right")

                ax_sigma.plot(check_steps, sigmas, 'b-o', markersize=4, label=r"$\sigma(H)$")
                ax_sigma.axhline(delta_C / 2.0, color='r', linestyle='--', label=r"$\Delta_C / 2$ Threshold")
                ax_sigma.set_ylabel(r"$\sigma(H)$")
                ax_sigma.grid(True, alpha=0.3)
                ax_sigma.legend(loc="lower right")

                if track_exact and ax_fid:
                    ax_fid.plot(check_steps, infidelities, 'g-s', markersize=4, label=r"Infidelity $1 - F$")
                    ax_fid.set_yscale('log')
                    ax_fid.set_xlabel(f"Verification Checks (Every {K} QNG Steps)")
                    ax_fid.set_ylabel(r"Infidelity $1 - F$")
                    ax_fid.grid(True, alpha=0.3, which='both')
                    ax_fid.legend(loc="upper right")

                plt.tight_layout()
                plt.pause(0.01)

            energy_change = abs(prev_energy - current_energy)
            prev_energy = current_energy

            # Verification check: If variance is below certified gap threshold σ(H) < Δ_C/2, exit inner optimization loop
            if sigma_H < delta_C / 2.0:
                stagnant_checks = 0
                break

            # Anti-stagnation escape: If optimization plateaus without passing verification, inject random perturbation
            if energy_change < 1e-5 or grad_norm < 1e-4:
                stagnant_checks += 1
                if stagnant_checks >= 2:
                    kick_scale = min(0.50, 0.15 * (stagnant_checks - 1))
                    print(f"  [Kick Scale {kick_scale:.2f}] Perturbing params at lambda={lbd:.4f}.")
                    params = params + jnp.array(np.random.normal(0, kick_scale, size=len(params)))

        # Evaluate differential variance σ_ψ(H_f - H_i) to set self-verifying step size limit δλ_V
        sigma_delta = float(get_sigma(H_diff_mat, P_mats, params))
        # Self-verifying step size calculation: δλ_V = (Δ_C / 2 - σ_ψ(H)) / σ_ψ(H_f - H_i)
        dl_V = dl_A if sigma_delta < 1e-9 else (delta_C / 2.0 - sigma_H) / sigma_delta
        # Select adaptive step size δλ = min(δλ_A, δλ_V, 1 - λ) bounded below by 0.005
        dl = min(dl_A, max(0.005, dl_V), 1.0 - lbd)
        lbd += dl
        lambda_change_steps.append((check_count, lbd))

        print(f"Step {step}: lambda = {lbd:.4f}, Energy = {current_energy:.5f}, "
              f"Sigma(H) = {sigma_H:.5f}{infidelity_str} | Step Time: {check_duration:.2f}s")

    total_run_time = time.time() - start_time
    total_mins, total_secs = divmod(int(total_run_time), 60)
    print(f"\nAVQE Complete in {total_mins:02d}m {total_secs:02d}s ({total_run_time:.2f}s)\n")

    # Save final plot summary if active
    if live_plot:
        plt.ioff()
        if name:
            plt.savefig(f"{name}_metrics.pdf")
            plt.close()
        else:
            plt.show()

    # Reconstruct and return final variational quantum circuit in Qrisp
    final_qc = ansatz_to_qrisp_circuit(P, params)
    return params, final_qc