from qrisp.operators import X, Z
from avqe_qnp_jax import self_verifying_AVQE_jax
from avqe_qnp_qrisp import self_verifying_AVQE_qrisp
import numpy as np


def build_tfim_ansatz(n_qubits, layers=2):
    P = []
    for i in range(n_qubits):
        p = ['I'] * n_qubits
        p[i] = 'Y'
        P.append(p)

    for _ in range(layers):
        for i in range(n_qubits - 1):
            p1 = ['I'] * n_qubits
            p1[i], p1[i + 1] = 'Y', 'X'
            P.append(p1)

            p2 = ['I'] * n_qubits
            p2[i], p2[i + 1] = 'X', 'Y'
            P.append(p2)

        for i in range(n_qubits):
            p = ['I'] * n_qubits
            p[i] = 'Y'
            P.append(p)

    params_0 = [np.pi / 4.0] * n_qubits + [0.0] * (len(P) - n_qubits)
    return P, params_0


def run_jax_4qubit_example():
    print("==========================================")
    print("   JAX: 4-Qubit TFIM (Fast Simulation)    ")
    print("==========================================")
    
    Hi_4q = sum([-1.0 * X(i) for i in range(4)])
    Hf_4q = -1.0 * (Z(0)*Z(1) + Z(1)*Z(2) + Z(2)*Z(3)) - 0.3 * sum([X(i) for i in range(4)])
    P_4q, params_0_4q = build_tfim_ansatz(n_qubits=4, layers=2)

    # Note: live_plot and track_exact can be easily toggled on/off here
    self_verifying_AVQE_qrisp(
        H_i=Hi_4q,
        H_f=Hf_4q,
        P=P_4q,
        params_0=params_0_4q,
        dl_A=0.15,
        K=10,
        delta_C=0.4,
        lr=0.08,
        name="4-qubit-jax-tfim",
        live_plot=True,     # Toggle real-time plotting
        track_exact=True    # Toggle exact ground-state diagonalization
    )

def run_qrisp_4qubit_example():
    print("==========================================")
    print("   JAX: 4-Qubit TFIM (Fast Simulation)    ")
    print("==========================================")
    
    Hi_4q = sum([-1.0 * X(i) for i in range(4)])
    Hf_4q = -1.0 * (Z(0)*Z(1) + Z(1)*Z(2) + Z(2)*Z(3)) - 0.3 * sum([X(i) for i in range(4)])
    P_4q, params_0_4q = build_tfim_ansatz(n_qubits=4, layers=2)

    # Note: live_plot and track_exact can be easily toggled on/off here
    self_verifying_AVQE_jax(
        H_i=Hi_4q,
        H_f=Hf_4q,
        P=P_4q,
        params_0=params_0_4q,
        dl_A=0.15,
        K=10,
        delta_C=0.4,
        lr=0.08,
        name="4-qubit-qrisp-tfim",
        live_plot=True,     # Toggle real-time plotting
        track_exact=True    # Toggle exact ground-state diagonalization
    )


def build_2d_grid_ansatz_qrisp(n_rows=3, n_cols=4, layers=2):
    n_qubits = n_rows * n_cols
    P_strings = []

    # Initial single-qubit Y rotations
    for i in range(n_qubits):
        p = ['I'] * n_qubits
        p[i] = 'Y'
        P_strings.append(p)

    def node(r, c):
        return r * n_cols + c

    horiz_edges = [
        (node(r, c), node(r, c + 1))
        for r in range(n_rows)
        for c in range(n_cols - 1)
    ]
    vert_edges = [
        (node(r, c), node(r + 1, c))
        for r in range(n_rows - 1)
        for c in range(n_cols)
    ]

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

    params_0 = [np.pi / 4.0] * n_qubits + [0.0] * (len(P_strings) - n_qubits)

    return P_strings, params_0

def run_qrisp_2qubit_example():
    print("==========================================")
    print("  Qrisp: 2-Qubit Grid (Native Execution)  ")
    print("==========================================")

    Hi_2q = -1.0 * X(0) - 1.0 * X(1)
    Hf_2q = -1.0 * Z(0) * Z(1) - 0.2 * X(0) - 0.2 * X(1)

    P_strings, params_0 = build_2d_grid_ansatz_qrisp(
        n_rows=1, n_cols=2, layers=2
    )

    self_verifying_AVQE_qrisp(
        H_i=Hi_2q,
        H_f=Hf_2q,
        P_strings=P_strings,
        params_0=params_0,
        dl_A=0.20,
        K=5,
        delta_C=0.40,
        lr=0.08,
        name="2-qubit-qrisp-avqe",
        live_plot=True,     # Interactive live tracking enabled
        track_exact=True    # Calculates ground energy baseline and infidelity
    )


if __name__ == "__main__":
    run_jax_4qubit_example()
    run_qrisp_2qubit_example()
    run_qrisp_4qubit_example()