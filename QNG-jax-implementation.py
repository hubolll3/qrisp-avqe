import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# Enable 64-bit precision for high-accuracy metric and variance operations
jax.config.update("jax_enable_x64", True)

# Qrisp imports
import qrisp
from qrisp.operators import X, Y, Z

from qiskit_aer import AerSimulator
qrisp.environments.virtual_backend = AerSimulator()


# ==========================================
# 1. QRISP UTILITIES & PRE-COMPILATION
# ==========================================

def pauli_string_to_matrix(p_str):
    """Converts Pauli character list (e.g. ['X', 'Z', 'I']) to JAX complex matrix."""
    PAULIS = {
        'I': jnp.array([[1, 0], [0, 1]], dtype=jnp.complex128),
        'X': jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128),
        'Y': jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128),
        'Z': jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128),
    }
    mat = PAULIS[p_str[0]]
    for char in p_str[1:]:
        mat = jnp.kron(mat, PAULIS[char])
    return mat


def qrisp_to_jax_matrix(op, n_qubits):
    """Converts Qrisp QubitOperator to dense JAX matrix."""
    np_mat = op.to_array(factor_amount=n_qubits)
    return jnp.array(np_mat, dtype=jnp.complex128)


def precompute_generator_matrices(P):
    """Pre-compiles list of Pauli strings into JAX matrices to prevent runtime overhead."""
    return [pauli_string_to_matrix(p_str) for p_str in P]


def ansatz_to_qrisp_circuit(P, params):
    """
    Translates optimized ansatz generators and parameters into a native
    Qrisp QuantumCircuit for backend execution or compilation.
    """
    n_qubits = len(P[0])
    qv = qrisp.QuantumVariable(n_qubits)

    for j, p_str in enumerate(P):
        theta = float(params[j])

        term = None
        for i, char in enumerate(p_str):
            op = None
            if char == 'X':
                op = X(i)
            elif char == 'Y':
                op = Y(i)
            elif char == 'Z':
                op = Z(i)

            if op is not None:
                term = op if term is None else term * op

        if term is not None:
            U = term.trotterization()
            U(qv, t=theta)

    return qv.qs  # Returns underlying qrisp.QuantumCircuit


# ==========================================
# 2. 2D GRID ANSATZ BUILDER
# ==========================================

def build_2d_grid_ansatz(n_rows=3, n_cols=4, layers=2):
    """Constructs 2D grid hardware-efficient ansatz for an (n_rows x n_cols) grid."""
    n_qubits = n_rows * n_cols
    P = []

    # Layer 0: Single-qubit Y-rotations
    for i in range(n_qubits):
        p = ['I'] * n_qubits
        p[i] = 'Y'
        P.append(p)

    def node(r, c):
        return r * n_cols + c

    horiz_edges = [(node(r, c), node(r, c + 1)) for r in range(n_rows) for c in range(n_cols - 1)]
    vert_edges = [(node(r, c), node(r + 1, c)) for r in range(n_rows - 1) for c in range(n_cols)]
    diag_edges = []
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            diag_edges.append((node(r, c), node(r + 1, c + 1)))
            diag_edges.append((node(r + 1, c), node(r, c + 1)))

    all_edges = horiz_edges + vert_edges + diag_edges

    for _ in range(layers):
        for i, j in all_edges:
            p1 = ['I'] * n_qubits
            p1[i], p1[j] = 'Y', 'X'
            P.append(p1)

            p2 = ['I'] * n_qubits
            p2[i], p2[j] = 'X', 'Y'
            P.append(p2)

        for i in range(n_qubits):
            p = ['I'] * n_qubits
            p[i] = 'Y'
            P.append(p)

    params_0 = [np.pi / 4.0] * n_qubits + [0.0] * (len(P) - n_qubits)
    return P, params_0


# ==========================================
# 3. METRICS & STATE CALCULATIONS
# ==========================================

def get_statevector(params, P_mats):
    """Computes |psi(params)> using pre-compiled generator matrices."""
    n_qubits = int(np.log2(P_mats[0].shape[0]))
    psi = jnp.zeros(2**n_qubits, dtype=jnp.complex128).at[0].set(1.0 + 0.0j)

    for j, P_mat in enumerate(P_mats):
        theta = params[j]
        psi = jnp.cos(theta) * psi - 1j * jnp.sin(theta) * (P_mat @ psi)

    return psi


def measure_val(H_mat, P_mats, params):
    psi = get_statevector(params, P_mats)
    return jnp.real(jnp.vdot(psi, H_mat @ psi))


def get_energy_and_grad(params, H_mat, P_mats):
    def loss(p):
        return measure_val(H_mat, P_mats, p)
    return jax.value_and_grad(loss)(params)


def get_sigma(H_mat, P_mats, params):
    psi = get_statevector(params, P_mats)
    E = jnp.real(jnp.vdot(psi, H_mat @ psi))
    H2_psi = H_mat @ (H_mat @ psi)
    E2 = jnp.real(jnp.vdot(psi, H2_psi))
    return jnp.sqrt(jnp.maximum(0.0, E2 - E**2))


def compute_qfim(params, P_mats):
    """Computes standard Fubini-Study Metric Tensor g_mu_nu."""
    psi = get_statevector(params, P_mats)
    J = jax.jacfwd(get_statevector, argnums=0)(params, P_mats)
    psi_col = jnp.reshape(psi, (-1, 1))
    J_dag = jnp.conj(J.T)
    v = J_dag @ psi_col
    g = jnp.real((J_dag @ J) - (v @ jnp.conj(v.T)))
    return 0.5 * (g + g.T)


def get_fidelity(P_mats, params, target_state):
    psi = get_statevector(params, P_mats)
    return float(jnp.abs(jnp.vdot(target_state, psi)) ** 2)


# ==========================================
# 4. BASELINE QNG OPTIMIZER (NON-STOKES)
# ==========================================

def qng_step_baseline(params, H_mat, P_mats, lr=0.08, eps=1e-3, max_grad_norm=0.5):
    """
    Standard Quantum Natural Gradient step using diagonal shift regularization
    and Euclidean norm gradient clipping.
    """
    energy, grad = get_energy_and_grad(params, H_mat, P_mats)
    grad_norm = jnp.linalg.norm(grad)

    if grad_norm < 1e-8:
        return params, energy, grad_norm

    g = compute_qfim(params, P_mats)

    current_eps = eps
    for _ in range(8):
        g_reg = g + current_eps * jnp.eye(g.shape[0], dtype=g.dtype)
        try:
            nat_grad = jnp.linalg.solve(g_reg, grad)
        except Exception:
            nat_grad = jnp.linalg.pinv(g_reg, rcond=1e-4) @ grad

        gnorm = jnp.linalg.norm(nat_grad)
        if gnorm > max_grad_norm:
            nat_grad = nat_grad * (max_grad_norm / gnorm)

        candidate_params = params - lr * nat_grad
        candidate_energy = float(measure_val(H_mat, P_mats, candidate_params))

        if candidate_energy <= energy + 1e-12:
            return candidate_params, candidate_energy, grad_norm

        current_eps *= 4.0

    fallback_params = params - 0.02 * (grad / (grad_norm + 1e-8))
    fallback_energy = float(measure_val(H_mat, P_mats, fallback_params))
    if fallback_energy < energy:
        return fallback_params, fallback_energy, grad_norm

    return params, energy, grad_norm


# ==========================================
# 5. AVQE MAIN LOOP WITH TIMERS
# ==========================================

def self_verifying_AVQE_QNG(
    H_i, H_f, P, params_0, dl_A, K, delta_C, lr=0.08, eps=1e-3, max_grad_norm=0.5, name=None
):
    n_qubits = len(P[0])

    H_i_mat = qrisp_to_jax_matrix(H_i, n_qubits)
    H_f_mat = qrisp_to_jax_matrix(H_f, n_qubits)
    H_diff_mat = H_f_mat - H_i_mat
    
    P_mats = precompute_generator_matrices(P)

    def H_mat(t):
        return (1.0 - t) * H_i_mat + t * H_f_mat

    lbd = 0.0

    np.random.seed(42)
    params = jnp.array(params_0, dtype=jnp.float64) + jnp.array(np.random.normal(0, 0.01, size=len(params_0)))

    # --- TIMER INITIALIZATION ---
    start_time = time.time()
    last_check_time = time.time()

    plt.ion()
    fig, (ax_energy, ax_sigma, ax_fid) = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)

    check_steps, energies, exact_energies, sigmas, infidelities = [], [], [], [], []
    lambda_change_steps = []

    check_count = 0
    step = 0
    prev_energy = float('inf')
    stagnant_checks = 0

    print(f"\n--- Starting AVQE ({n_qubits} Qubits, {len(P)} Generators) ---")

    while lbd < 1.0:
        step += 1

        while True:
            # 1. Execute K Baseline QNG steps
            for _ in range(K):
                params, _, grad_norm = qng_step_baseline(
                    params, H_mat(lbd), P_mats, lr=lr, eps=eps, max_grad_norm=max_grad_norm
                )

            # 2. Evaluate metrics and measure elapsed time
            check_count += 1
            now = time.time()
            check_duration = now - last_check_time
            total_elapsed = now - start_time
            last_check_time = now

            current_energy = float(measure_val(H_mat(lbd), P_mats, params))
            sigma_H = float(get_sigma(H_mat(lbd), P_mats, params))

            evals, evecs = np.linalg.eigh(np.array(H_mat(lbd)))
            exact_E0 = float(evals[0])
            fidelity = get_fidelity(P_mats, params, evecs[:, 0])
            infidelity = abs(1.0 - fidelity)

            check_steps.append(check_count)
            energies.append(current_energy)
            exact_energies.append(exact_E0)
            sigmas.append(sigma_H)
            infidelities.append(infidelity)

            # --- Live Plotting with Timers ---
            ax_energy.clear()
            ax_sigma.clear()
            ax_fid.clear()

            mins, secs = divmod(int(total_elapsed), 60)
            time_str = f"Total Time: {mins:02d}m {secs:02d}s  |  Last {K} Steps: {check_duration:.2f}s"

            fig.suptitle(
                f"Self-Verifying AVQE (Baseline QNG, {n_qubits}-Qubit, K={K})\n{time_str}", 
                fontsize=13, fontweight="bold"
            )

            ax_energy.plot(check_steps, energies, 'm-^', markersize=4, label=r"Prepared $\langle H \rangle$")
            ax_energy.plot(check_steps, exact_energies, 'k--', alpha=0.7, label=r"Exact $E_0(\lambda)$")
            ax_energy.set_ylabel("Energy")
            ax_energy.grid(True, alpha=0.3)
            ax_energy.legend(loc="lower right")

            ax_sigma.plot(check_steps, sigmas, 'b-o', markersize=4, label=r"$\sigma(H)$")
            ax_sigma.axhline(delta_C / 2.0, color='r', linestyle='--', label=r"$\Delta_C / 2$ Threshold")
            ax_sigma.set_ylabel(r"$\sigma(H)$")
            ax_sigma.grid(True, alpha=0.3)
            ax_sigma.legend(loc="lower right")

            ax_fid.plot(check_steps, infidelities, 'g-s', markersize=4, label=r"Infidelity $1 - F$")
            ax_fid.set_yscale('log')
            ax_fid.set_xlabel(f"Verification Checks (Every {K} QNG Steps)")
            ax_fid.set_ylabel(r"Infidelity $1 - F$")
            ax_fid.set_title("State Infidelity Over Time")
            ax_fid.grid(True, alpha=0.3, which='both')
            ax_fid.legend(loc="upper right")

            for trans_step, _ in lambda_change_steps:
                ax_energy.axvline(x=trans_step, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
                ax_sigma.axvline(x=trans_step, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)
                ax_fid.axvline(x=trans_step, color='purple', linestyle=':', linewidth=1.5, alpha=0.8)

            plt.tight_layout()
            plt.pause(0.01)

            energy_change = abs(prev_energy - current_energy)
            prev_energy = current_energy

            # STRICT GATE: Only advance lambda if variance threshold is met
            if sigma_H < delta_C / 2.0:
                stagnant_checks = 0
                break

            # ESCALATING THERMAL KICK DETECTOR
            if energy_change < 1e-5 or grad_norm < 1e-4:
                stagnant_checks += 1
                if stagnant_checks >= 2:
                    kick_scale = min(0.50, 0.15 * (stagnant_checks - 1))
                    print(f"  [Kick Scale {kick_scale:.2f}] Trapped at lambda={lbd:.4f} (sigma_H={sigma_H:.4f}). Perturbing parameters.")
                    kick = jnp.array(np.random.normal(0, kick_scale, size=len(params)))
                    params = params + kick

        # Calculate adaptive step size ONLY after passing verification
        sigma_delta = float(get_sigma(H_diff_mat, P_mats, params))
        dl_V = dl_A if sigma_delta < 1e-9 else (delta_C / 2.0 - sigma_H) / sigma_delta

        dl = min(dl_A, max(0.005, dl_V), 1.0 - lbd)
        lbd += dl
        lambda_change_steps.append((check_count, lbd))

        print(f"Step {step}: lambda = {lbd:.4f}, Energy = {current_energy:.5f}, "
              f"Sigma(H) = {sigma_H:.5f}, Infidelity = {infidelity:.4e} | "
              f"Step Time: {check_duration:.2f}s, Total: {total_elapsed:.1f}s")

    # --- FINAL TIMER DISPLAY ---
    total_run_time = time.time() - start_time
    total_mins, total_secs = divmod(int(total_run_time), 60)
    print(f"\n==========================================")
    print(f"       AVQE RUN COMPLETE")
    print(f" Total Execution Time: {total_mins:02d}m {total_secs:02d}s ({total_run_time:.2f} seconds)")
    print(f"==========================================\n")

    plt.ioff()
    if name is not None:
        plt.savefig(f"{name}_metrics.pdf")
        plt.clf()
        plt.close()
    else:
        plt.show()

    final_qc = ansatz_to_qrisp_circuit(P, params)
    print(f"\n[Qrisp] Exported ground state to Qrisp QuantumCircuit ({final_qc.num_qubits} qubits).")
    return params, final_qc





# ==========================================
# 5. EXECUTING ALL EXAMPLES
# ==========================================
def build_tfim_ansatz(n_qubits, layers=2):
    """
    Constructs a layered hardware-efficient ansatz using pure-real entanglers.
    Includes alternating Y_i X_{i+1} and X_i Y_{i+1} gates to ensure full expressivity.
    """
    P = []
    # Initial single-qubit Y rotations (prepares |+> state when theta = pi/4)
    for i in range(n_qubits):
        p = ['I'] * n_qubits
        p[i] = 'Y'
        P.append(p)

    for _ in range(layers):
        # Entangling pairs along nearest neighbors
        for i in range(n_qubits - 1):
            p1 = ['I'] * n_qubits
            p1[i], p1[i + 1] = 'Y', 'X'
            P.append(p1)

            p2 = ['I'] * n_qubits
            p2[i], p2[i + 1] = 'X', 'Y'
            P.append(p2)

        # Single-qubit Y rotations
        for i in range(n_qubits):
            p = ['I'] * n_qubits
            p[i] = 'Y'
            P.append(p)

    params_0 = [np.pi / 4.0] * n_qubits + [0.0] * (len(P) - n_qubits)
    return P, params_0

def build_2d_grid_ansatz(n_rows=2, n_cols=4, layers=2):
    """
    Constructs a 2D hardware-efficient ansatz for an (n_rows x n_cols) grid.
    Connects nearest-neighbor horizontal, vertical, and diagonal edges
    using real skew-symmetric generators (Y_i X_j and X_i Y_j).
    """
    n_qubits = n_rows * n_cols
    P = []

    # Layer 0: Single-qubit Y-rotations (prepares |+>_8 state at theta = pi/4)
    for i in range(n_qubits):
        p = ['I'] * n_qubits
        p[i] = 'Y'
        P.append(p)

    def node(r, c):
        return r * n_cols + c

    # Extract 2D grid edges
    horiz_edges = [(node(r, c), node(r, c + 1)) for r in range(n_rows) for c in range(n_cols - 1)]
    vert_edges = [(node(r, c), node(r + 1, c)) for r in range(n_rows - 1) for c in range(n_cols)]
    diag_edges = []
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            diag_edges.append((node(r, c), node(r + 1, c + 1)))
            diag_edges.append((node(r + 1, c), node(r, c + 1)))

    all_edges = horiz_edges + vert_edges + diag_edges

    for _ in range(layers):
        # Entangling blocks across 2D grid connectivity
        for i, j in all_edges:
            p1 = ['I'] * n_qubits
            p1[i], p1[j] = 'Y', 'X'
            P.append(p1)

            p2 = ['I'] * n_qubits
            p2[i], p2[j] = 'X', 'Y'
            P.append(p2)

        # Single-qubit Y-rotations
        for i in range(n_qubits):
            p = ['I'] * n_qubits
            p[i] = 'Y'
            P.append(p)

    params_0 = [np.pi / 4.0] * n_qubits + [0.0] * (len(P) - n_qubits)
    return P, params_0


if __name__ == "__main__":

    # --------------------------------------
    # EXAMPLE 1: 1-QUBIT TOY PROBLEM
    # --------------------------------------
    print("==========================================")
    print("         1-QUBIT TOY PROBLEM              ")
    print("==========================================")
    Hi_1q = 1.0 + 1.0 * X(0)
    Hf_1q = 2.0 - 1.0 * Z(0)

    P_1q = [['Y']]
    params_0_1q = [-np.pi / 4]

    self_verifying_AVQE_QNG(        H_i=Hi_1q, H_f=Hf_1q, P=P_1q, params_0=params_0_1q,        dl_A=0.2, K=10, delta_C=0.4, lr=0.08, name="1-qubit-toy-problem-QNG"    )

    # --------------------------------------
    # EXAMPLE 2: 2-QUBIT TARGET PROBLEM
    # --------------------------------------
    print("==========================================")
    print("         2-QUBIT TARGET PROBLEM           ")
    print("==========================================")
    Hi_2q = -1.0 * X(0) - 1.0 * X(1)
    Hf_2q = -1.0 * Z(0) * Z(1) - 0.2 * X(0) - 0.2 * X(1)

    P_2q = [
        ['Y', 'I'],
        ['I', 'Y'],
        ['Y', 'X'],
        ['Y', 'I'],
        ['I', 'Y']
    ]
    params_0_2q = [np.pi / 4, np.pi / 4, 0.0, 0.0, 0.0]

    self_verifying_AVQE_QNG(        H_i=Hi_2q, H_f=Hf_2q, P=P_2q, params_0=params_0_2q,        dl_A=0.15, K=10, delta_C=0.4, lr=0.08, name="2-qubit-toy-problem-QNG"    )

    # --------------------------------------
    # EXAMPLE 3: 3-QUBIT ISING CHAIN (2-LAYER ANSATZ)
    # --------------------------------------
    print("==========================================")
    print("    3-QUBIT TRANSVERSE FIELD ISING CHAIN   ")
    print("==========================================")
    Hi_3q = -1.0 * X(0) - 1.0 * X(1) - 1.0 * X(2)
    Hf_3q = -1.0 * (Z(0) * Z(1) + Z(1) * Z(2)) - 0.3 * (X(0) + X(1) + X(2))

    P_3q, params_0_3q = build_tfim_ansatz(n_qubits=3, layers=2)

    self_verifying_AVQE_QNG(        H_i=Hi_3q, H_f=Hf_3q, P=P_3q, params_0=params_0_3q,        dl_A=0.15, K=10, delta_C=0.4, lr=0.08, name="3-qubit-TFIM-QNG"    )

    # --------------------------------------
    # EXAMPLE 4: 4-QUBIT ISING CHAIN (2-LAYER ANSATZ)
    # --------------------------------------
    print("==========================================")
    print("    4-QUBIT TRANSVERSE FIELD ISING CHAIN   ")
    print("==========================================")
    Hi_4q = sum([-1.0 * X(i) for i in range(4)])
    Hf_4q = -1.0 * (Z(0)*Z(1) + Z(1)*Z(2) + Z(2)*Z(3)) - 0.3 * sum([X(i) for i in range(4)])

    P_4q, params_0_4q = build_tfim_ansatz(n_qubits=4, layers=2)

    self_verifying_AVQE_QNG(        H_i=Hi_4q, H_f=Hf_4q, P=P_4q, params_0=params_0_4q,        dl_A=0.15, K=10, delta_C=0.4, lr=0.08, name="4-qubit-TFIM-QNG"    )

# ==========================================
# EXAMPLE 5: 10-QUBIT 2D FRUSTRATED MAGNET
# ==========================================

if __name__ == "__main__":
    n_rows, n_cols = 2, 5
    n_qubits = n_rows * n_cols

    # Construct Initial Hamiltonian: - \sum X_i
    Hi_8q = sum([-1.0 * X(i) for i in range(n_qubits)])

    # Construct Final Target Hamiltonian: 2D J1-J2 Frustrated Magnet
    J1, J2, h = 1.0, 0.4, 0.2

    Hf_terms = []
    # Nearest-neighbor coupling (J1)
    for r in range(n_rows):
        for c in range(n_cols - 1):
            i, j = r * n_cols + c, r * n_cols + (c + 1)
            Hf_terms.append(-J1 * Z(i) * Z(j))
    for r in range(n_rows - 1):
        for c in range(n_cols):
            i, j = r * n_cols + c, (r + 1) * n_cols + c
            Hf_terms.append(-J1 * Z(i) * Z(j))

    # Diagonal frustrated coupling (J2)
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            i1, j1 = r * n_cols + c, (r + 1) * n_cols + (c + 1)
            i2, j2 = (r + 1) * n_cols + c, r * n_cols + (c + 1)
            Hf_terms.append(-J2 * Z(i1) * Z(j1))
            Hf_terms.append(-J2 * Z(i2) * Z(j2))

    # Residual Transverse Field (h)
    for i in range(n_qubits):
        Hf_terms.append(-h * X(i))

    Hf_8q = sum(Hf_terms)

    # Build 2D Lattice Ansatz
    P_8q, params_0_8q = build_2d_grid_ansatz(n_rows=2, n_cols=5, layers=2)

    # Run AVQE Optimization
    self_verifying_AVQE_QNG(        H_i=Hi_8q, H_f=Hf_8q, P=P_8q, params_0=params_0_8q,        dl_A=0.15, K=10, delta_C=0.4, lr=0.08, name="10-qubit-2D-frustrated-spin-grid"    )



# ==========================================
# 6. 12-QUBIT FRUSTRATED MAGNET EXECUTION
# ==========================================

if __name__ == "__main__":
    print("==========================================")
    print("    12-QUBIT (3x4) FRUSTRATED MAGNET      ")
    print("==========================================")

    n_rows, n_cols = 3, 4
    n_qubits = n_rows * n_cols  # 12 Qubits

    # Initial Hamiltonian: - \sum X_i
    Hi_12q = sum([-1.0 * X(i) for i in range(n_qubits)])

    # Target Hamiltonian: 2D J1-J2 Frustrated Magnet
    J1, J2, h = 1.0, 0.4, 0.2
    Hf_terms = []

    def node(r, c):
        return r * n_cols + c

    # Nearest-neighbor coupling (J1)
    for r in range(n_rows):
        for c in range(n_cols - 1):
            Hf_terms.append(-J1 * Z(node(r, c)) * Z(node(r, c + 1)))
    for r in range(n_rows - 1):
        for c in range(n_cols):
            Hf_terms.append(-J1 * Z(node(r, c)) * Z(node(r + 1, c)))

    # Next-nearest-neighbor diagonal coupling (J2)
    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            Hf_terms.append(-J2 * Z(node(r, c)) * Z(node(r + 1, c + 1)))
            Hf_terms.append(-J2 * Z(node(r + 1, c)) * Z(node(r, c + 1)))

    # Residual Transverse Field (h)
    for i in range(n_qubits):
        Hf_terms.append(-h * X(i))

    Hf_12q = sum(Hf_terms)

    # Construct 2-Layer 2D Grid Ansatz (~152 parameters)
    P_12q, params_0_12q = build_2d_grid_ansatz(n_rows=n_rows, n_cols=n_cols, layers=2)

    print(f"System Configuration: {n_qubits} Qubits ({n_rows}x{n_cols} grid)")
    print(f"Ansatz Generators: {len(P_12q)}")

    # Run AVQE Optimization
    params, final_qc = self_verifying_AVQE_QNG(
        H_i=Hi_12q,
        H_f=Hf_12q,
        P=P_12q,
        params_0=params_0_12q,
        dl_A=0.15,
        K=10,
        delta_C=0.35,
        lr=0.08,
        eps=1e-3,
        max_grad_norm=0.5,
        name="12-qubit-2D-frustrated-spin-grid-baseline"
    )