# Self-Verifying AVQE with Quantum Natural Gradient (QNG)

This repository implements the **Self-Verifying Adaptive Variational Quantum Eigensolver (AVQE)** optimized via the **Quantum Natural Gradient (QNG / QNP)**. 

### Overview & Key Features

* **Adiabatic Warm Starts**: Discretizes an adiabatic path $H(\lambda) = (1-\lambda)H_i + \lambda H_f$ into $T$ steps, initializing gradient descent at each step using the optimal parameter set from the previous step[cite: 3].
* **Barren Plateau & Local Minima Mitigation**: By incrementally tracking the ground state along a continuous path, optimization remains within a local convexity/Polyak-Łojasiewicz (PL) basin[cite: 3].
* **Theoretical Convergence Guarantees**: Gradient updates track the instantaneous ground state across the entire path with a total optimization update complexity scaling as $\mathcal{O}(\Delta_{\text{min}}^{-3})$, where $\Delta_{\text{min}}$ is the minimum spectral gap[cite: 3].
* **Runtime Ground-State Certification**: Features an a posteriori verification test using energy standard deviation measurements $\sigma_\psi(H(\lambda))$[cite: 3]. Achieving $\sigma_\psi < \Delta_c / 2$ (for a gap lower bound $\Delta_c \le \Delta_{\text{min}}$) provably certifies convergence to the ground state branch with fidelity $\ge 8/9$[cite: 3].
* **Shot-Noise Robustness**: Maintains certification guarantees under finite measurement shot noise with $\mathcal{O}(\Delta_{\text{min}}^{-4})$ shots per slice[cite: 3].

---

## References

* **Scalable, Self-Verifying Adiabatic VQE (AVQE)**[cite: 3]
  
  ```bibtex
  @article{zunkovic2026scalable,
  title={Scalable, self-verifying variational quantum eigensolver using adiabatic warm starts},
  author={{\v{Z}}unkovi{\v{c}}, Bojan and Ballarin, Marco and Wright, Lewis and Lubasch, Michael},
  journal={arXiv preprint},
  year={2026},
  month={February}
  }


To balance classical simulation speed with scalable quantum execution, the framework provides two backend implementations: a high-speed JAX simulator for rapid benchmarking and a native Qrisp workflow designed for larger systems and real quantum hardware.

## Repository Structure

```text
├── avqe_qnp_jax.py      # JAX backend (fast classical auto-diff, N <= 12)
├── avqe_qnp_qrisp.py    # Native Qrisp backend (scalable execution, N >= 14)
├── run_examples.py      # Main entry point for test cases and examples
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
