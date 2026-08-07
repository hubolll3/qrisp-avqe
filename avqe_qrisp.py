import time
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import qrisp
from qrisp.operators import X, Y, Z
from qiskit_aer import AerSimulator

# Configure Aer for multi-threaded statevector simulation natively in Qrisp
qrisp.environments.virtual_backend = AerSimulator(
    method="statevector",
    max_parallel_threads=0  # Automatically use all CPU cores
)

# ==========================================
# 1. STATE PREP
# ==========================================

def create_qrisp_state_prep(trotter_unitaries, n_qubits):
    def state_prep(params):
        qv = qrisp.QuantumVariable(n_qubits)
        for j, U in enumerate(trotter_unitaries):
            if U is not None:
                U(qv, t=float(params[j]))
        return qv
    return state_prep


# ==========================================
# 2. OPTIMIZERS (VANILLA, QNP/QNG, ADAM)
# ==========================================

def compute_gradient_parameter_shift(ev_H, params):
    m = len(params)
    grad = np.zeros(m)
    shift = np.pi / 4.0

    for j in range(m):
        p_plus = np.array(params, copy=True)
        p_minus = np.array(params, copy=True)
        p_plus[j] += shift
        p_minus[j] -= shift
        grad[j] = float(np.real(ev_H(p_plus))) - float(np.real(ev_H(p_minus)))

    return grad


def compute_qfim_forward_with_base(params, psi_base, state_prep, eps=1e-5):
    m = len(params)
    J = np.zeros((len(psi_base), m), dtype=np.complex128)

    for j in range(m):
        p_plus = np.array(params, copy=True)
        p_plus[j] += eps

        qv_plus = state_prep(p_plus)
        psi_plus = qv_plus.qs.statevector_array()
        J[:, j] = (psi_plus - psi_base) / eps

    v = J.conj().T @ psi_base
    g = np.real((J.conj().T @ J) - np.outer(v, v.conj()))
    return 0.5 * (g + g.T)


def vanilla_gd_step_qrisp(params, ev_H, state_prep, lr=0.08):
    qv_base = state_prep(params)
    psi_base = qv_base.qs.statevector_array()

    grad = compute_gradient_parameter_shift(ev_H, params)
    grad_norm = np.linalg.norm(grad)

    new_params = params - lr * grad
    new_E = float(np.real(ev_H(new_params)))

    return new_params, new_E, grad_norm, psi_base


def qnp_step_qrisp(params, ev_H, state_prep, lr=0.08, eps=1e-3, max_grad_norm=0.5):
    qv_base = state_prep(params)
    psi_base = qv_base.qs.statevector_array()

    current_E = float(np.real(ev_H(params)))
    grad = compute_gradient_parameter_shift(ev_H, params)
    grad_norm = np.linalg.norm(grad)

    if grad_norm < 1e-8:
        return params, current_E, grad_norm, psi_base

    g = compute_qfim_forward_with_base(params, psi_base, state_prep, eps=eps)

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
        candidate_E = float(np.real(ev_H(candidate_params)))

        if candidate_E <= current_E + 1e-12:
            return candidate_params, candidate_E, grad_norm, psi_base

        current_eps *= 4.0

    fallback_params = params - 0.02 * (grad / (grad_norm + 1e-8))
    fallback_E = float(np.real(ev_H(fallback_params)))
    if fallback_E < current_E:
        return fallback_params, fallback_E, grad_norm, psi_base

    return params, current_E, grad_norm, psi_base


def adam_step_qrisp(params, ev_H, state_prep, opt_state, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Executes one step of Adam optimization.
    """
    qv_base = state_prep(params)
    psi_base = qv_base.qs.statevector_array()

    grad = compute_gradient_parameter_shift(ev_H, params)
    grad_norm = np.linalg.norm(grad)

    opt_state["t"] += 1
    t = opt_state["t"]

    opt_state["m"] = beta1 * opt_state["m"] + (1.0 - beta1) * grad
    opt_state["v"] = beta2 * opt_state["v"] + (1.0 - beta2) * (grad ** 2)

    m_hat = opt_state["m"] / (1.0 - (beta1 ** t))
    v_hat = opt_state["v"] / (1.0 - (beta2 ** t))

    new_params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
    new_E = float(np.real(ev_H(new_params)))

    return new_params, new_E, grad_norm, psi_base


# ==========================================
# 3. AVQE MAIN LOOP
# ==========================================

def self_verifying_AVQE_qrisp(
    H_i,
    H_f,
    P_strings,
    params_0,
    dl_A,
    K,
    delta_C,
    lr=0.08,
    kappa=0.9,
    optimizer_type="vanilla",  # Choice between 'qnp', 'vanilla', and 'adam'
    name=None,
    live_plot=False,
    track_exact=False,
    const_step=False
):
    """
    Implements AVQE with option for constant step size or adaptive verification.
    """
    n_qubits = len(P_strings[0])

    trotter_unitaries = []
    for p_str in P_strings:
        term = None
        for i, char in enumerate(p_str):
            op = None
            if char == "X":
                op = X(i)
            elif char == "Y":
                op = Y(i)
            elif char == "Z":
                op = Z(i)
            if op is not None:
                term = op if term is None else term * op

        trotter_unitaries.append(term.trotterization() if term is not None else None)

    state_prep = create_qrisp_state_prep(trotter_unitaries, n_qubits)

    H_diff = (H_f - H_i).hermitize()
    H_diff_sq = (H_diff * H_diff).hermitize()
    ev_Hdiff = H_diff.expectation_value(state_prep)
    ev_Hdiff2 = H_diff_sq.expectation_value(state_prep)

    lbd = 0.0
    np.random.seed(42)
    params = np.array(params_0, dtype=np.float64) + np.random.normal(0, 0.01, size=len(params_0))

    H_0 = H_i.hermitize()
    H_0_sq = (H_0 * H_0).hermitize()
    ev_H0 = H_0.expectation_value(state_prep)
    ev_H02 = H_0_sq.expectation_value(state_prep)

    E_val0 = float(np.real(ev_H0(params)))
    E2_val0 = float(np.real(ev_H02(params)))
    sigma_H = np.sqrt(max(0.0, E2_val0 - E_val0**2))

    start_time = time.time()
    last_check_time = time.time()

    if live_plot:
        plt.ion()
        fig, axes = plt.subplots(3 if track_exact else 2, 1, figsize=(9, 9.5), sharex=True)
        ax_energy, ax_sigma = axes[0], axes[1]
        ax_fid = axes[2] if track_exact else None

    check_steps, energies, exact_energies, sigmas, infidelities = [], [], [], [], []
    check_count = 0
    step = 0
    psi_last = state_prep(params).qs.statevector_array()

    mode_label = "CONSTANT STEP" if const_step else "SELF-VERIFYING"
    print(f"\n--- Starting AVQE ({n_qubits} Qubits, Mode: {mode_label}, Optimizer: {optimizer_type.upper()}) ---")

    # Select step function
    opt_choice = optimizer_type.lower()
    if opt_choice in ["qnp", "qng"]:
        step_fn = lambda p, ev: qnp_step_qrisp(p, ev, state_prep, lr=lr)
    elif opt_choice == "vanilla":
        step_fn = lambda p, ev: vanilla_gd_step_qrisp(p, ev, state_prep, lr=lr)
    elif opt_choice == "adam":
        adam_state = {
            "m": np.zeros(len(params), dtype=np.float64),
            "v": np.zeros(len(params), dtype=np.float64),
            "t": 0
        }
        step_fn = lambda p, ev: adam_step_qrisp(p, ev, state_prep, adam_state, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_type '{optimizer_type}'. Choose 'qnp', 'vanilla', or 'adam'.")

    while lbd < 1.0:
        step += 1

        if const_step:
            dl = min(dl_A, 1.0 - lbd)
        else:
            diff_E = float(np.real(ev_Hdiff(params)))
            diff_E2 = float(np.real(ev_Hdiff2(params)))
            sigma_delta = np.sqrt(max(0.0, diff_E2 - diff_E**2))

            numerator = max(1e-12, (delta_C / 2.0) - sigma_H)
            if sigma_delta < 1e-9:
                dl_V = dl_A
            else:
                dl_V = kappa * (numerator / sigma_delta)

            dl = min(dl_A, max(0.001, dl_V), 1.0 - lbd)

        lbd += dl

        H_curr = ((1.0 - lbd) * H_i + lbd * H_f).hermitize()
        H_curr_sq = (H_curr * H_curr).hermitize()
        ev_H = H_curr.expectation_value(state_prep)
        ev_H2 = H_curr_sq.expectation_value(state_prep)

        while True:
            for _ in range(K):
                params, E_val, grad_norm, psi_last = step_fn(params, ev_H)

            check_count += 1
            now = time.time()
            check_duration = now - last_check_time
            total_elapsed = now - start_time
            last_check_time = now

            E_val = float(np.real(ev_H(params)))
            E2_val = float(np.real(ev_H2(params)))
            sigma_H = np.sqrt(max(0.0, E2_val - E_val**2))

            check_steps.append(check_count)
            energies.append(E_val)
            sigmas.append(sigma_H)

            infidelity_str = ""
            if track_exact:
                exact_E0 = float(H_curr.ground_state_energy())
                sparse_H = H_curr.to_sparse_matrix(factor_amount=n_qubits)
                _, evecs = scipy.sparse.linalg.eigsh(sparse_H, k=1, which="SA")

                fidelity = float(np.abs(np.vdot(evecs[:, 0], psi_last)) ** 2)
                infidelity = abs(1.0 - fidelity)

                exact_energies.append(exact_E0)
                infidelities.append(infidelity)
                infidelity_str = f", Infidelity = {infidelity:.4e}"

            if live_plot:
                ax_energy.clear()
                ax_sigma.clear()
                if ax_fid: ax_fid.clear()

                mins, secs = divmod(int(total_elapsed), 60)
                fig.suptitle(
                    f"AVQE ({optimizer_type.upper()}, {n_qubits}-Qubit, K={K})\n"
                    f"Total Time: {mins:02d}m {secs:02d}s | Step Time: {check_duration:.2f}s",
                    fontsize=13, fontweight="bold"
                )

                ax_energy.plot(check_steps, energies, "m-^", markersize=4, label=r"Prepared $\langle H\rangle$")
                if track_exact:
                    ax_energy.plot(check_steps, exact_energies, "k--", alpha=0.7, label=r"Exact $E_0(\lambda)$")
                ax_energy.set_ylabel("Energy")
                ax_energy.grid(True, alpha=0.3)
                ax_energy.legend(loc="lower right")

                ax_sigma.plot(check_steps, sigmas, "b-o", markersize=4, label=r"$\sigma(H)$")
                ax_sigma.axhline(delta_C / 4.0, color="r", linestyle="--", label=r"$\Delta_C/4$ Threshold")
                ax_sigma.set_ylabel(r"$\sigma(H)$")
                ax_sigma.grid(True, alpha=0.3)
                ax_sigma.legend(loc="lower right")

                if track_exact and ax_fid:
                    ax_fid.plot(check_steps, infidelities, "g-s", markersize=4, label=r"Infidelity $1-F$")
                    ax_fid.set_yscale("log")
                    ax_fid.set_xlabel(f"Verification Checks (Every {K} Steps)")
                    ax_fid.set_ylabel(r"Infidelity $1-F$")
                    ax_fid.grid(True, alpha=0.3, which="both")
                    ax_fid.legend(loc="upper right")

                plt.tight_layout()
                plt.pause(0.01)

            if const_step or sigma_H <= (delta_C / 4.0):
                break
            else:
                print(f"  [Verification Failed] σ(H) = {sigma_H:.5f} > Δ_c/4 ({delta_C/4:.5f}). Retrying {K} steps...")

        print(f"Step {step}: lambda = {lbd:.4f}, Energy = {E_val:.5f}, "
              f"Sigma(H) = {sigma_H:.5f}{infidelity_str} | Step Time: {check_duration:.2f}s")

    total_run_time = time.time() - start_time
    mins, secs = divmod(int(total_run_time), 60)
    print(f"\nAVQE Complete in {mins:02d}m {secs:02d}s ({total_run_time:.2f}s)\n")

    if live_plot:
        plt.ioff()
        if name:
            plt.savefig(f"{name}_metrics.pdf")
            plt.close()
        else:
            plt.show()

    final_qv = state_prep(params)
    return params, final_qv.qs