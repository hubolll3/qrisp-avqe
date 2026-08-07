from qrisp.operators import X, Z
from avqe_qrisp import self_verifying_AVQE_qrisp
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

#
#
#
#
#
#
#
#
#

def run_qrisp_10qubit_example():
    print("==========================================")
    print("  Qrisp: 10-Qubit 2x5 Grid (Native Run)   ")
    print("==========================================")

    n_rows, n_cols = 2, 5
    n_qubits = n_rows * n_cols

    # Initial Hamiltonian: Transverse field on all 10 qubits
    Hi_10q = sum([-1.0 * X(i) for i in range(n_qubits)])

    # Helper function to map (row, col) to qubit index
    def node(r, c):
        return r * n_cols + c

    # Construct nearest-neighbor horizontal and vertical ZZ grid interactions
    grid_edges = []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            grid_edges.append((node(r, c), node(r, c + 1)))
    for r in range(n_rows - 1):
        for c in range(n_cols):
            grid_edges.append((node(r, c), node(r + 1, c)))

    # Target Hamiltonian: 2D Grid TFIM with transverse field bias
    Hf_10q = sum([-1.0 * Z(i) * Z(j) for i, j in grid_edges]) - 0.3 * sum([X(i) for i in range(n_qubits)])

    # Build 1-layer 2D grid ansatz for 10 qubits
    P_strings, params_0 = build_2d_grid_ansatz_qrisp(
        n_rows=n_rows, n_cols=n_cols, layers=1
    )

    self_verifying_AVQE_qrisp(
        H_i=Hi_10q,
        H_f=Hf_10q,
        P_strings=P_strings,
        params_0=params_0,
        dl_A=0.10,
        K=5,
        delta_C=0.40,
        lr=0.08,
        optimizer_type="vanilla",
        name="final_plots/10-qubit-qrisp-avqe",
        live_plot=True,     # Interactive live tracking enabled
        track_exact=True    # Calculates ground energy baseline and infidelity via sparse eigensolver
    )

#
#
#
#
#
#
#
#

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
        dl_A=0.15,
        K=5,
        delta_C=0.40,
        lr=0.08,
        optimizer_type="qnp",
        name="final_plots/2-qubit-qrisp-avqe-qnp",
        live_plot=True,     # Interactive live tracking enabled
        track_exact=True,    # Calculates ground energy baseline and infidelity
        const_step=False
    )
    
    self_verifying_AVQE_qrisp(
            H_i=Hi_2q,
            H_f=Hf_2q,
            P_strings=P_strings,
            params_0=params_0,
            dl_A=0.15,
            K=5,
            delta_C=0.40,
            lr=0.08,
            optimizer_type="vanilla",
            name="final_plots/2-qubit-qrisp-avqe-vanilla",
            live_plot=True,     # Interactive live tracking enabled
            track_exact=True,    # Calculates ground energy baseline and infidelity
            const_step=False
    )
    
    self_verifying_AVQE_qrisp(
                H_i=Hi_2q,
                H_f=Hf_2q,
                P_strings=P_strings,
                params_0=params_0,
                dl_A=0.15,
                K=5,
                delta_C=0.40,
                lr=0.08,
                optimizer_type="adam",
                name="final_plots/2-qubit-qrisp-avqe-adam",
                live_plot=True,     # Interactive live tracking enabled
                track_exact=True,    # Calculates ground energy baseline and infidelity
                const_step=False
        )

def run_qrisp_4qubit_example():
    print("==========================================")
    print("  Qrisp: 4-Qubit TFIM (Native Execution)  ")
    print("==========================================")
    
    Hi_4q = sum([-1.0 * X(i) for i in range(4)])
    Hf_4q = -1.0 * (Z(0)*Z(1) + Z(1)*Z(2) + Z(2)*Z(3)) - 0.3 * sum([X(i) for i in range(4)])
    P_4q, params_0_4q = build_tfim_ansatz(n_qubits=4, layers=2)

    # Note: live_plot and track_exact can be easily toggled on/off here
    self_verifying_AVQE_qrisp(
        H_i=Hi_4q,
        H_f=Hf_4q,
        P_strings=P_4q,
        params_0=params_0_4q,
        dl_A=0.15,
        K=10,
        delta_C=0.4,
        lr=0.08,
        optimizer_type="qnp",
        name="final_plots/4-qubit-qrisp-avqe-qnp",
        live_plot=True,     # Toggle real-time plotting
        track_exact=True    # Toggle exact ground-state diagonalization
    )
    self_verifying_AVQE_qrisp(
            H_i=Hi_4q,
            H_f=Hf_4q,
            P_strings=P_4q,
            params_0=params_0_4q,
            dl_A=0.15,
            K=10,
            delta_C=0.4,
            lr=0.08,
            optimizer_type="vanilla",
            name="final_plots/4-qubit-qrisp-avqe-vanilla",
            live_plot=True,     # Toggle real-time plotting
            track_exact=True    # Toggle exact ground-state diagonalization
    )
    self_verifying_AVQE_qrisp(
                H_i=Hi_4q,
                H_f=Hf_4q,
                P_strings=P_4q,
                params_0=params_0_4q,
                dl_A=0.15,
                K=10,
                delta_C=0.4,
                lr=0.08,
                optimizer_type="adam",
                name="final_plots/4-qubit-qrisp-avqe-adam",
                live_plot=True,     # Toggle real-time plotting
                track_exact=True    # Toggle exact ground-state diagonalization
        )
    

#
#
#
#
#
#
#
#



if __name__ == "__main__":
    run_qrisp_2qubit_example()
    run_qrisp_4qubit_example()
    #run_qrisp_10qubit_example()