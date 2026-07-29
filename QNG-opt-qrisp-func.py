import time
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Qrisp Core and Operator Imports
import qrisp
from qrisp.operators import X, Y, Z, QubitOperator


from qiskit_aer import AerSimulator
qrisp.environments.virtual_backend = AerSimulator()


# ==========================================
# 1. ANSATZ & STATE PREPARATION GENERATORS
# ==========================================

def build_2d_grid_ansatz_qrisp(n_rows=3, n_cols=4, layers=2):
    """
    Constructs hardware-efficient 2D grid generators and compiles them 
    into native Qrisp trotterized unitary operations.
    """
    n_qubits = n_rows * n_cols
    P_strings = []

    # Layer 0: Single-qubit Y-rotations
    for i in range(n_qubits):
        p = ['I'] * n_qubits
        p[i] = 'Y'
        P_strings.append(p)

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
            P_strings.append(p1)

            p2 = ['I'] * n_qubits
            p2[i], p2[j] = 'X', 'Y'
            P_strings.append(p2)

        for i in range(n_qubits):
            p = ['I'] * n_qubits
            p[i] = 'Y'
            P_strings.append(p)

    # Pre-compile Qrisp trotterized unitary functions for fast execution
    trotter_unitaries = []
    for p_str in P_strings:
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

        trotter_unitaries.append(term.trotterization() if term is not None else None)

    params_0 = [np.pi / 4.0] * n_qubits + [0.0] * (len(P_strings) - n_qubits)
    return P_strings, trotter_unitaries, params_0


def create_qrisp_state_prep(trotter_unitaries, n_qubits):
    """
    Generates a Qrisp state preparation closure compatible with 
    QubitOperator.expectation_value().
    """
    def state_prep(params):
        qv = qrisp.QuantumVariable(n_qubits)
        for j, U in enumerate(trotter_unitaries):
            if U is not None:
                U(qv, t=float(params[j]))
        return qv

    return state_prep


# ==========================================
# 2. SCALABLE METRICS & QFIM CALCULATORS
# ==========================================

def get_exact_ground_state_energy_sparse(H_op, n_qubits):
    """
    Computes exact ground state energy classically using SciPy sparse Lanczos solver 
    (Avoids $2^N x 2^N$ dense memory allocation).
    """
    sparse_H = H_op.to_sparse_matrix(factor_amount=n_qubits)
    evals, _ = scipy.sparse.linalg.eigsh(sparse_H, k=1, which='SA')
    return float(evals[0])


def compute_statevector_1d(params, trotter_unitaries, n_qubits):
    """Returns 1D complex statevector array of length 2^N."""
    qv = qrisp.QuantumVariable(n_qubits)
    for j, U in enumerate(trotter_unitaries):
        if U is not None:
            U(qv, t=float(params[j]))
    return qv.qs.statevector_array()


def compute_qfim_sparse(params, trotter_unitaries, n_qubits, eps=1e-5):
    """
    Computes Quantum Fisher Information Matrix (QFIM) using 1D statevectors,
    scaling as O(2^N) memory instead of O(2^{2N}) dense matrices.
    """
    m = len(params)
    psi = compute_statevector_1d(params, trotter_unitaries, n_qubits)
    
    # Finite-difference jacobian of statevector
    J = np.zeros((len(psi), m), dtype=np.complex128)
    for j in range(m):
        p_plus = np.array(params, copy=True)
        p_minus = np.array(params, copy=True)
        p_plus[j] += eps
        p_minus[j] -= eps

        psi_plus = compute_statevector_1d(p_plus, trotter_unitaries, n_qubits)
        psi_minus = compute_statevector_1d(p_minus, trotter_unitaries, n_qubits)
        J[:, j] = (psi_plus - psi_minus) / (2.0 * eps)

    v = J.conj().T @ psi
    g = np.real((J.conj().T @ J) - np.outer(v, v.conj()))
    return 0.5 * (g + g.T)


# ==========================================
# 3. NATIVE QRISP QNG OPTIMIZER
# ==========================================

def compute_gradient_parameter_shift(ev_H, params):
    """
    Computes exact analytical gradient using the Parameter-Shift Rule.
    Requires 2 evaluations per parameter, with ZERO numerical error.
    """
    grad = np.zeros_like(params)
    shift = np.pi / 4.0  # Exact shift for e^{-i theta P} generators

    for j in range(len(params)):
        p_plus = np.array(params, copy=True)
        p_minus = np.array(params, copy=True)

        p_plus[j] += shift
        p_minus[j] -= shift

        # Exact derivative: E(theta + pi/4) - E(theta - pi/4)
        grad[j] = float(ev_H(p_plus)) - float(ev_H(p_minus))

    return grad

def qng_step_qrisp(
    params, H_op, trotter_unitaries, n_qubits, lr=0.08, eps=1e-3, max_grad_norm=0.5
):
    """
    Executes a Quantum Natural Gradient step using Qrisp expectation value 
    evaluators and sparse statevector QFIM calculations.
    """
    state_prep = create_qrisp_state_prep(trotter_unitaries, n_qubits)
    ev_H = H_op.expectation_value(state_prep)

    # 1. Compute Energy and Euclidean Gradient via finite differences
    current_E = float(ev_H(params))
    grad = compute_gradient_parameter_shift(ev_H, params)

    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-8:
        return params, current_E, grad_norm

    # 2. Compute QFIM using 1D statevector method
    g = compute_qfim_sparse(params, trotter_unitaries, n_qubits)

    # 3. Solve regularized system
    current_eps = eps
    for _ in range(6):
        g_reg = g + current_eps * np.eye(g.shape[0])
        try:
            nat_grad = np.linalg.solve(g_reg, grad)
        except np.linalg.LinAlgError:
            nat_grad = np.linalg.pinv(g_reg, rcond=1e-4) @ grad

        gnorm = np.linalg.norm(nat_grad)
        if gnorm > max_grad_norm:
            nat_grad = nat_grad * (max_grad_norm / gnorm)

        candidate_params = params - lr * nat_grad
        candidate_E = float(ev_H(candidate_params))

        if candidate_E <= current_E + 1e-12:
            return candidate_params, candidate_E, grad_norm

        current_eps *= 4.0

    # Fallback to standard gradient step if regularized QNG fails
    fallback_params = params - 0.02 * (grad / (grad_norm + 1e-8))
    fallback_E = float(ev_H(fallback_params))
    if fallback_E < current_E:
        return fallback_params, fallback_E, grad_norm

    return params, current_E, grad_norm


# ==========================================
# 4. MAIN AVQE LOOP WITH QRISP & TIMERS
# ==========================================

def self_verifying_AVQE_qrisp(
    H_i, H_f, P_strings, trotter_unitaries, params_0, dl_A, K, delta_C, lr=0.08, name=None
):
    n_qubits = len(P_strings[0])
    H_diff = H_f - H_i

    state_prep = create_qrisp_state_prep(trotter_unitaries, n_qubits)

    lbd = 0.0
    np.random.seed(42)
    params = np.array(params_0, dtype=np.float64) + np.random.normal(0, 0.01, size=len(params_0))

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

    print(f"\n--- Starting Native Qrisp AVQE ({n_qubits} Qubits) ---")

    while lbd < 1.0:
        step += 1
        H_curr = ((1.0 - lbd) * H_i + lbd * H_f).hermitize()
        H2_curr = (H_curr * H_curr).hermitize()

        ev_H = H_curr.expectation_value(state_prep)
        ev_H2 = H2_curr.expectation_value(state_prep)

        while True:
            # 1. Perform K Native QNG steps
            for _ in range(K):
                params, _, grad_norm = qng_step_qrisp(
                    params, H_curr, trotter_unitaries, n_qubits, lr=lr
                )

            # 2. Evaluate physical metrics using Qrisp
            check_count += 1
            now = time.time()
            check_duration = now - last_check_time
            total_elapsed = now - start_time
            last_check_time = now

            E_val = float(np.real(ev_H(params)))
            E2_val = float(np.real(ev_H2(params)))              #precision 1e-2
            sigma_H = np.sqrt(max(0.0, E2_val - E_val**2))      #precision 1e-4
            if sigma_H == 0:
                print("sigma=0, hamiltonians: H=", H_curr, "    H2:", H2_curr)
                print("E_val=", E_val, "    E2_val=", E2_val)

            # Compute ground state energy classically via sparse Lanczos solver
            exact_E0 = float(H_curr.ground_state_energy())

            # Fidelity against exact ground state vector (1D sparse)
            sparse_H = H_curr.to_sparse_matrix(factor_amount=n_qubits)
            _, evecs = scipy.sparse.linalg.eigsh(sparse_H, k=1, which='SA')
            psi = compute_statevector_1d(params, trotter_unitaries, n_qubits)
            fidelity = float(np.abs(np.vdot(evecs[:, 0], psi))**2)
            infidelity = abs(1.0 - fidelity)

            check_steps.append(check_count)
            energies.append(E_val)
            exact_energies.append(exact_E0)
            sigmas.append(sigma_H)
            infidelities.append(infidelity)

            # --- Live Plotting with Timer Header ---
            ax_energy.clear()
            ax_sigma.clear()
            ax_fid.clear()

            mins, secs = divmod(int(total_elapsed), 60)
            time_str = f"Total Time: {mins:02d}m {secs:02d}s  |  Last {K} Steps: {check_duration:.2f}s"

            fig.suptitle(
                f"Self-Verifying AVQE (Native Qrisp, {n_qubits}-Qubit, K={K})\n{time_str}",
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

            energy_change = abs(prev_energy - E_val)
            prev_energy = E_val

            # Verification Check
            if sigma_H < delta_C / 2.0:
                stagnant_checks = 0
                break

            # Trapping Recovery
            if energy_change < 1e-5 or grad_norm < 1e-4:
                stagnant_checks += 1
                if stagnant_checks >= 2:
                    kick_scale = min(0.50, 0.15 * (stagnant_checks - 1))
                    print(f"  [Kick Scale {kick_scale:.2f}] Trapped at lambda={lbd:.4f} (sigma_H={sigma_H:.4f}). Perturbing.")
                    params += np.random.normal(0, kick_scale, size=len(params))

        # Adaptive Lambda Step Calculation
        ev_Hdiff = H_diff.expectation_value(state_prep)
        ev_Hdiff2 = (H_diff * H_diff).expectation_value(state_prep)
        diff_E = float(ev_Hdiff(params))
        diff_E2 = float(ev_Hdiff2(params))
        sigma_delta = np.sqrt(max(0.0, diff_E2 - diff_E**2))

        dl_V = dl_A if sigma_delta < 1e-9 else (delta_C / 2.0 - sigma_H) / sigma_delta
        dl = min(dl_A, max(0.005, dl_V), 1.0 - lbd)
        lbd += dl
        lambda_change_steps.append((check_count, lbd))

        print(f"Step {step}: lambda = {lbd:.4f}, Energy = {E_val:.5f}, "
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

    final_qv = state_prep(params)
    return params, final_qv.qs






# ==========================================
# 2-QUBIT EXECUTION EXAMPLE
# ==========================================

if __name__ == "__main__":
    print("==========================================")
    print("      2-QUBIT TRANSVERSE ISING MODEL      ")
    print("==========================================")

    n_rows, n_cols = 1, 2
    n_qubits = n_rows * n_cols  # 2 Qubits

    # Initial Hamiltonian: - (X_0 + X_1)
    Hi_2q = -1.0 * X(0) - 1.0 * X(1)

    # Target Hamiltonian: - J * Z_0 Z_1 - h * (X_0 + X_1)
    J = 1.0
    h = 0.2
    Hf_2q = -J * Z(0) * Z(1) - h * X(0) - h * X(1)

    # Build 2-Qubit Grid Ansatz (1x2 topology)
    P_strings, trotter_unitaries, params_0 = build_2d_grid_ansatz_qrisp(
        n_rows=n_rows, n_cols=n_cols, layers=2
    )

    print(f"System Configuration: {n_qubits} Qubits ({n_rows}x{n_cols} grid)")
    print(f"Ansatz Generators: {len(P_strings)}")

    # Run Native Qrisp AVQE Algorithm
    params, final_qc = self_verifying_AVQE_qrisp(
        H_i=Hi_2q,
        H_f=Hf_2q,
        P_strings=P_strings,
        trotter_unitaries=trotter_unitaries,
        params_0=params_0,
        dl_A=0.20,       # Adiabatic step size
        K=5,             # Check metrics every 5 QNG steps
        delta_C=0.40,    # Strict variance threshold (delta_C / 2 = 0.05)
        lr=0.08,         # Learning rate
        name="2-qubit-qrisp-avqe"
    )




# ==========================================
# 5. EXECUTION EXAMPLE
# ==========================================

if __name__ == "__main__":
    n_rows, n_cols = 2, 2
    n_qubits = n_rows * n_cols  # 12 Qubits

    # Build Qrisp Initial and Target Hamiltonians
    Hi_12q = sum([-1.0 * X(i) for i in range(n_qubits)])

    J1, J2, h = 1.0, 0.4, 0.2
    Hf_terms = []

    def node(r, c):
        return r * n_cols + c

    for r in range(n_rows):
        for c in range(n_cols - 1):
            Hf_terms.append(-J1 * Z(node(r, c)) * Z(node(r, c + 1)))
    for r in range(n_rows - 1):
        for c in range(n_cols):
            Hf_terms.append(-J1 * Z(node(r, c)) * Z(node(r + 1, c)))

    for r in range(n_rows - 1):
        for c in range(n_cols - 1):
            Hf_terms.append(-J2 * Z(node(r, c)) * Z(node(r + 1, c + 1)))
            Hf_terms.append(-J2 * Z(node(r + 1, c)) * Z(node(r, c + 1)))

    for i in range(n_qubits):
        Hf_terms.append(-h * X(i))

    Hf_12q = sum(Hf_terms)

    # Build 2D Grid Ansatz Generator Suite
    P_strings, trotter_unitaries, params_0 = build_2d_grid_ansatz_qrisp(n_rows=n_rows, n_cols=n_cols, layers=2)

    # Run Pure Qrisp Scalable AVQE Algorithm
    params, final_qc = self_verifying_AVQE_qrisp(
        H_i=Hi_12q,
        H_f=Hf_12q,
        P_strings=P_strings,
        trotter_unitaries=trotter_unitaries,
        params_0=params_0,
        dl_A=0.15,
        K=10,
        delta_C=0.35,
        lr=0.08,
        name="4-qubit-qrisp-avqe"
    )