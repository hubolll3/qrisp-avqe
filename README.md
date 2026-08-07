# Self-Verifying Adiabatic Variational Quantum Eigensolver (AVQE)

This repository contains the specification and algorithmic layout for the **Self-Verifying Adiabatic Variational Quantum Eigensolver (AVQE)**, based on the warm-start adiabatic optimization paradigm.

> **Reference Paper:**  
> *Scalable, self-verifying variational quantum eigensolver using adiabatic warm starts*  
> Bojan Žunkovič, Marco Ballarin, Lewis Wright, and Michael Lubasch (2026)  
> [arXiv:2602.17612v1 [quant-ph]](https://arxiv.org/abs/2602.17612)

---

## 1. Core Idea of the Algorithm

**Self-Verifying AVQE** resolves these challenges through two key mechanisms:
1. **Adiabatic Warm Starts**: Instead of optimizing directly on a single target Hamiltonian, optimization is executed sequentially along a discretized linear adiabatic path. At each path slice $\lambda$, parameters are initialized from the solution at $\lambda - \delta\lambda$. This warm start keeps the optimizer within a locally convex tracking basin along the adiabatic path, provably avoiding barren plateaus and local traps.
2. **Runtime Verification**: Runtime measurements of the energy standard deviation $\sigma_\psi(H(\lambda))$ certify eigenstate accuracy and verify convergence to the global optimum. This enables dynamic adjustment of the adiabatic step size $\delta\lambda$ while providing strict ground-state fidelity certificates.

---

## 2. Repository Structure

```text
├── avqe_qrisp.py        # Qrisp based AVQE
├── run_examples.py      # Main entry point for test cases and examples
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 3. Algorithm Pseudocode

```text
Algorithm: Self-verifying AVQE
Input: Initial Hamiltonian Hi, final Hamiltonian Hf, initial parameters θ0,
       adiabatic tracking step size δλA, learning rate η, number of gradient steps K,
       estimated gap bound Δc
Output: Final variational state |ψ(θ)⟩

 1: Prepare |ψ(θ0)⟩ as the ground state of H(0)
 2: λ ← 0
 3: while λ < 1 do
 4:     Perform K gradient-descent steps on Eλ(θ)
 5:     Measure σψ(H(λ))
 6:     if σψ(H(λ)) > Δc/2 then
 7:         go to line 4
 8:     end if
 9:     Measure σψ(Hf − Hi)
10:     δλV ← (Δc/2 − σψ(H(λ))) / σψ(Hf − Hi)
11:     δλ ← min{δλA, δλV, 1 − λ}
12:     λ ← λ + δλ
13: end while
14: return |ψ(θ)⟩
```

---

## 4. Variational Ansatz

The algorithm uses an $M$-parameter Pauli-rotation variational ansatz defined over $n$ qubits:

$$\vert{}\psi(\theta)\rangle = \prod_{j=0}^{M-1} U_j \vert{}0\rangle = \prod_{j=0}^{M-1} e^{-i P_j \theta_j} \vert{}0\rangle$$

where:
* $\theta = (\theta_0, \theta_1, \dots, \theta_{M-1})^T \in \mathbb{R}^M$ is the vector of variational parameters.
* $P_j \in \{I, X, Y, Z\}^{\otimes n}$ are Pauli string generators acting on $n$ qubits.
* $U_j = e^{-i P_j \theta_j}$ are single-parameter unitary rotations.

---

## 5. Parameter Definitions & Theoretical Quantities

| Parameter / Symbol | Mathematical Definition | Description |
| :--- | :--- | :--- |
| **$H(\lambda)$** | $H(\lambda) = (1-\lambda)H_i + \lambda H_f$ | Linear adiabatic Hamiltonian schedule interpolating between initial Hamiltonian $H_i$ and target Hamiltonian $H_f$ for $\lambda \in [0, 1]$. |
| **$\Delta_{\min}$** | $\Delta_{\min} = \min_{\lambda \in [0, 1]} E_1(\lambda) - E_0(\lambda)$ | Minimum spectral gap encountered across the entire adiabatic path. |
| **$\Delta_c$** | $\Delta_c \le \Delta_{\min}$ | Estimated/certified lower bound on the spectral gap along the path. |
| **$\sigma_\psi(A)$** | $\sigma_\psi(A) = \sqrt{\langle \psi \vert A^2 \vert \psi \rangle - (\langle \psi \vert A \vert \psi \rangle)^2}$ | Standard deviation of an operator $A$ evaluated with respect to state $\vert \psi \rangle$. |
| **$\eta$** | $\eta > 0$ | Optimization learning rate for gradient descent updates: $\theta^{(k+1)} = \theta^{(k)} - \eta \mathcal{G}_t(\theta^{(k)})$. |
| **$K$** | $K \ge c_1 \frac{M \lVert H\rVert_{\text{op}}}{\gamma \Delta_{\min}}$ | Number of gradient-descent optimization steps performed per adiabatic slice. |
| **$\delta\lambda_A$** | $\delta\lambda_A = c_0 \frac{\gamma^2 \Delta_{\min}^2}{M^2 \lVert H\rVert_{\text{op}} \lVert \partial_\lambda H\rVert_{\text{op}}}$ | Maximum adiabatic step size for theoretical ground-state tracking. |
| **$\delta\lambda_V$** | $\delta\lambda_V = \frac{\Delta_c / 2 - \sigma_\psi(H(\lambda))}{\sigma_\psi(H_f - H_i)}$ | Dynamic verification-based step size bound ensuring state fidelity retention. |

---

## 6. Final Fidelity Bound

According to **Theorem 2 (Runtime Verification)**:

Given a lower bound estimate $\Delta_c \le \Delta_{\min}$, if the measured energy standard deviation satisfies:

$$\sigma_{\psi_t}(H(\lambda_t)) < \frac{\Delta_c}{4}$$

and the adiabatic step size satisfies $\delta\lambda < \delta\lambda_V = \frac{\Delta_c - \sigma_{\psi_t}(H(\lambda_t))}{2 \sigma_{\psi_t}(H_f - H_i)}$, then the prepared state $\vert{}\psi_t\rangle$ is uniquely associated with the true ground-state eigenbranch of $H(\lambda_t)$ and satisfies the strict lower fidelity bound:

$$\mathcal{F} = \vert{}\langle \psi_0(\lambda_t) \vert{} \psi_t \rangle\vert{}^2 \ge \frac{8}{9} \approx 88.89\%$$
