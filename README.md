# Quantum state discrimination of mixed 2N-photon polarization states

### Santiago Bustamante & Boris A. Rodríguez

This is a repository containing various numerical and symbolic approaches to the problem of quantum state discrimination (QSD) of mixed 2N-photon polarization states, namely between states

$$
\rho_k = \frac{1}{\binom{2N}{N}} \sum_{\tau \in J_N} | \Psi_k^\tau \rangle \langle \Psi_k^\tau | \qquad k\in\{A,B\}
$$

and

$$
\rho_C = \frac{1}{2^{2N}} \mathrm{I},
$$

where $\mathrm I$ is the identity operator in the 2N-photon polarization state space, $J_N$ is the set of all N-combinations of $J=\{1,\dots,2N\}$, and $\Psi_A^\tau$ ($\Psi_B^\tau$) is the 2N-photon polarization state for which photons with index $j\in\tau$ are in polarization state $H$ ($L$) and the rest in state $V$ ($R$).

### All work within the repository is part of my master's thesis in quantum measurement and information theory, which is not complete yet. _The reader's discretion is kindly solicited._

Libraries required:

 - **numpy** for numerical data manipulation and processing
 - **matplotlib** for data visualization
 - **sympy** for symbolic calculations
 - **picos** for semidefinite programming
 - **strawberryfields** for bosonic quantum circuit simulations
 - **pennylane** for quantum circuit simulations
 - **itertools** for combinatorics

For any inquiries please contact me at santiago.bustamanteq@udea.edu.co
