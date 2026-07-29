# AVQE with Quantum Natural Gradient (QNP)

This repository contains two implementations of the Adaptive Variational Quantum Eigensolver (AVQE) using the Quantum Natural Gradient (QNP) optimizer.

The goal is to provide a fast version for local testing and a scalable version built with native Qrisp primitives that doesn't run out of memory on larger systems.

## Implementations

* **`avqe_qnp_jax.py` (Fast local testing):** Uses JAX auto-diff (`jax.grad` and `jax.jacfwd`) for fast classical simulations on small systems ($N \le 12$ qubits). 
* **`avqe_qnp_qrisp.py` (Scalable / Qrisp-native):** Uses native Qrisp functions (`expectation_value`) and the parameter-shift rule. This avoids converting Hamiltonians into dense matrices, allowing execution on larger systems ($N \ge 14$ qubits) or shot-based backends without memory issues.

---

## Method Details

### Optimizer
Instead of standard gradient descent, parameters are updated using the Fubini-Study metric tensor (Quantum Fisher Information Matrix):

$$\theta_{k+1} = \theta_k - \eta \, g^+ \nabla E(\theta_k)$$

where $g^+$ is the pseudo-inverse of the metric tensor.

### Convergence Metric
Eigenstate convergence is evaluated using energy variance:

$$\sigma_H = \sqrt{\max\left(0, \langle H^2 \rangle - \langle H \rangle^2\right)}$$

Both scripts include a small numerical floor to keep floating-point cancellation from producing negative numbers under square roots when states are very close to exact ground states.

---

## Setup and Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the JAX version:
   ```bash
   python avqe_qnp_jax.py
   ```

3. Run the Qrisp native version:
   ```bash
   python avqe_qnp_qrisp.py
   ```
