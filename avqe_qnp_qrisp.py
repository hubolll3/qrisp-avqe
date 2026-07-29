import time
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import qrisp
from qrisp.operators import X, Y, Z, QubitOperator

from qiskit_aer import AerSimulator
qrisp.environments.virtual_backend = AerSimulator()


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
# 2. SCALABLE METRICS & OPTIMIZER
# ==========================================

def compute_statevector_1d(params, trotter_unitaries, n_qubits):
    qv = qrisp.QuantumVariable(n_qubits)
    for j, U in enumerate(trotter_unitaries):
        if U is not None:
            U(qv, t=float(params[j]))
    return qv.qs.statevector_array()


def compute_qfim_sparse(params, trotter_unitaries, n_qubits, eps=1e-5):
    m = len(params)
    psi = compute_statevector_1d(params, trotter_unitaries, n_qubits)

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


def compute_gradient_parameter_shift(ev_H, params):
    grad = np.zeros_like(params)
    shift = np.pi / 4.0

    for j in range(len(params)):
        p_plus = np.array(params, copy=True)
        p_minus = np.array(params, copy=True)
        p_plus[j] += shift
        p_minus[j] -= shift
        grad[j] = float(ev_H(p_plus)) - float(ev_H(p_minus))

    return grad


def qng_step_qrisp(params, H_op, trotter_unitaries, n_qubits, lr=0.08, eps=1e-3, max_grad_norm=0.5):
    state_prep = create_qrisp_state_prep(trotter_unitaries, n_qubits)
    ev_H = H_op.expectation_value(state_prep)

    current_E = float(ev_H(params))
    grad = compute_gradient_parameter_shift(ev_H, params)

    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-8:
        return params, current_E, grad_norm

    g = compute_qfim_sparse(params, trotter_unitaries, n_qubits)

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

    fallback_params = params - 0.02 * (grad / (grad_norm + 1e-8))
    fallback_E = float(ev_H(fallback_params))
    if fallback_E < current_E:
        return fallback_params, fallback_E, grad_norm

    return params, current_E, grad_norm


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
    name=None,
    live_plot=False,
    track_exact=False
):
    """
    Scalable Qrisp implementation of Self-Verifying AVQE.

    Parameters
    ----------
    live_plot : bool
        Enables real-time progress plotting.
    track_exact : bool
        Computes sparse ground-state energy and state infidelity
        (computationally expensive).
    """

    n_qubits = len(P_strings[0])

    # -------------------------------------------------------
    # Automatically build trotterized Pauli rotations
    # -------------------------------------------------------
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

        trotter_unitaries.append(
            term.trotterization() if term is not None else None
        )

    H_diff = H_f - H_i
    state_prep = create_qrisp_state_prep(trotter_unitaries, n_qubits)

    lbd = 0.0
    np.random.seed(42)
    params = np.array(params_0, dtype=np.float64)
    params += np.random.normal(0, 0.01, size=len(params_0))

    start_time = time.time()
    last_check_time = time.time()

    if live_plot:
        plt.ion()
        fig, axes = plt.subplots(
            3 if track_exact else 2,
            1,
            figsize=(9, 9.5),
            sharex=True
        )
        ax_energy, ax_sigma = axes[0], axes[1]
        ax_fid = axes[2] if track_exact else None

    check_steps = []
    energies = []
    exact_energies = []
    sigmas = []
    infidelities = []
    lambda_change_steps = []

    check_count = 0
    step = 0
    prev_energy = float("inf")
    stagnant_checks = 0

    print(f"\n--- Starting Native Qrisp AVQE ({n_qubits} Qubits) ---")

    while lbd < 1.0:

        step += 1

        H_curr = ((1.0 - lbd) * H_i + lbd * H_f).hermitize()
        H2_curr = (H_curr * H_curr).hermitize()

        ev_H = H_curr.expectation_value(state_prep)
        ev_H2 = H2_curr.expectation_value(state_prep)

        while True:

            for _ in range(K):
                params, _, grad_norm = qng_step_qrisp(
                    params,
                    H_curr,
                    trotter_unitaries,
                    n_qubits,
                    lr=lr
                )

            check_count += 1

            now = time.time()
            check_duration = now - last_check_time
            total_elapsed = now - start_time
            last_check_time = now

            E_val = float(np.real(ev_H(params)))
            E2_val = float(np.real(ev_H2(params)))
            sigma_H = np.sqrt(max(0.0, E2_val - E_val ** 2))

            check_steps.append(check_count)
            energies.append(E_val)
            sigmas.append(sigma_H)

            infidelity_str = ""

            if track_exact:

                exact_E0 = float(H_curr.ground_state_energy())

                sparse_H = H_curr.to_sparse_matrix(
                    factor_amount=n_qubits
                )

                _, evecs = scipy.sparse.linalg.eigsh(
                    sparse_H,
                    k=1,
                    which="SA"
                )

                psi = compute_statevector_1d(
                    params,
                    trotter_unitaries,
                    n_qubits
                )

                fidelity = float(np.abs(np.vdot(evecs[:, 0], psi)) ** 2)
                infidelity = abs(1.0 - fidelity)

                exact_energies.append(exact_E0)
                infidelities.append(infidelity)

                infidelity_str = (
                    f", Infidelity = {infidelity:.4e}"
                )

            if live_plot:

                ax_energy.clear()
                ax_sigma.clear()

                if ax_fid:
                    ax_fid.clear()

                mins, secs = divmod(int(total_elapsed), 60)

                fig.suptitle(
                    f"AVQE (Qrisp, {n_qubits}-Qubit, K={K})\n"
                    f"Total Time: {mins:02d}m {secs:02d}s"
                    f" | Last {K} Steps: {check_duration:.2f}s",
                    fontsize=13,
                    fontweight="bold"
                )

                ax_energy.plot(
                    check_steps,
                    energies,
                    "m-^",
                    markersize=4,
                    label=r"Prepared $\langle H\rangle$"
                )

                if track_exact:
                    ax_energy.plot(
                        check_steps,
                        exact_energies,
                        "k--",
                        alpha=0.7,
                        label=r"Exact $E_0(\lambda)$"
                    )

                ax_energy.set_ylabel("Energy")
                ax_energy.grid(True, alpha=0.3)
                ax_energy.legend(loc="lower right")

                ax_sigma.plot(
                    check_steps,
                    sigmas,
                    "b-o",
                    markersize=4,
                    label=r"$\sigma(H)$"
                )

                ax_sigma.axhline(
                    delta_C / 2,
                    color="r",
                    linestyle="--",
                    label=r"$\Delta_C/2$ Threshold"
                )

                ax_sigma.set_ylabel(r"$\sigma(H)$")
                ax_sigma.grid(True, alpha=0.3)
                ax_sigma.legend(loc="lower right")

                if track_exact and ax_fid:
                    ax_fid.plot(
                        check_steps,
                        infidelities,
                        "g-s",
                        markersize=4,
                        label=r"Infidelity $1-F$"
                    )
                    ax_fid.set_yscale("log")
                    ax_fid.set_xlabel(
                        f"Verification Checks (Every {K} QNG Steps)"
                    )
                    ax_fid.set_ylabel(r"Infidelity $1-F$")
                    ax_fid.grid(True, alpha=0.3, which="both")
                    ax_fid.legend(loc="upper right")

                plt.tight_layout()
                plt.pause(0.01)

            energy_change = abs(prev_energy - E_val)
            prev_energy = E_val

            if sigma_H < delta_C / 2:
                stagnant_checks = 0
                break

            if energy_change < 1e-5 or grad_norm < 1e-4:

                stagnant_checks += 1

                if stagnant_checks >= 2:

                    kick_scale = min(
                        0.50,
                        0.15 * (stagnant_checks - 1)
                    )

                    print(
                        f"  [Kick Scale {kick_scale:.2f}] "
                        f"Perturbing params at lambda={lbd:.4f}."
                    )

                    params += np.random.normal(
                        0,
                        kick_scale,
                        size=len(params)
                    )

        ev_Hdiff = H_diff.expectation_value(state_prep)
        ev_Hdiff2 = (H_diff * H_diff).expectation_value(state_prep)

        diff_E = float(ev_Hdiff(params))
        diff_E2 = float(ev_Hdiff2(params))

        sigma_delta = np.sqrt(
            max(0.0, diff_E2 - diff_E ** 2)
        )

        if sigma_delta < 1e-9:
            dl_V = dl_A
        else:
            dl_V = (delta_C / 2 - sigma_H) / sigma_delta

        dl = min(
            dl_A,
            max(0.005, dl_V),
            1.0 - lbd
        )

        lbd += dl
        lambda_change_steps.append((check_count, lbd))

        print(
            f"Step {step}: "
            f"lambda = {lbd:.4f}, "
            f"Energy = {E_val:.5f}, "
            f"Sigma(H) = {sigma_H:.5f}"
            f"{infidelity_str} | "
            f"Step Time: {check_duration:.2f}s"
        )

    total_run_time = time.time() - start_time
    mins, secs = divmod(int(total_run_time), 60)

    print(
        f"\nAVQE Complete in "
        f"{mins:02d}m {secs:02d}s "
        f"({total_run_time:.2f}s)\n"
    )

    if live_plot:
        plt.ioff()
        if name:
            plt.savefig(f"{name}_metrics.pdf")
            plt.close()
        else:
            plt.show()

    final_qv = state_prep(params)

    return params, final_qv.qs