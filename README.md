Here is a **GitHub-ready README.md** for the current implementation. I’ve kept it accurate to what we actually built today, without claiming that the full QBN-AQVP framework or large-scale validation is already complete.

````markdown
# Quantum Bellman Networks with Adaptive Quantum Value Propagation (QBN-AQVP)

A proof-of-concept research implementation exploring the integration of **quantum graph representations**, **Bellman value propagation**, and **adaptive quantum feedback** for temporal graph decision intelligence.

> **Research Status:** Proof of Concept / Review-2 Prototype

---

## Overview

This project investigates a new hybrid quantum-classical approach for sequential decision-making over temporal graphs.

The work evolves from our previous **Hybrid Quantum Graph Neural Networks with Dynamic Programming (QGNN-DP)** framework. In QGNN-DP, quantum circuits were used primarily for graph representation learning, while Bellman Dynamic Programming was applied as a separate optimization component.

The current research investigates a stronger integration:

```text
Temporal Graph
      ↓
Quantum Graph Representation
      ↓
Quantum Node Scores
      ↓
Bellman Value Propagation
      ↓
Adaptive Quantum Contribution
      ↓
Bellman-Guided Quantum Parameter Update
      ↓
Decision
````

The long-term goal is to develop **Quantum Bellman Networks with Adaptive Quantum Value Propagation (QBN-AQVP)** as a quantum-native framework for temporal graph intelligence.

---

## Current Objectives

The current prototype focuses on validating the following concepts independently and incrementally:

1. Quantum representation of graph nodes.
2. Bellman value propagation over graph states.
3. Interaction between quantum-derived scores and Bellman rewards.
4. Sensitivity of Bellman values to quantum contribution.
5. Adaptive quantum weighting using Bellman temporal-difference error.
6. Bellman-guided updates of quantum circuit parameters.

---

## Project Structure

```text
QBN-AQVP/
│
├── data/
│
├── results/
│   ├── sensitivity_results.csv
│   ├── aqvp_results.csv
│   └── aqvp_feedback_results.csv
│
├── src/
│   ├── quantum_graph.py
│   ├── bellman.py
│   ├── qbn_poc.py
│   ├── sensitivity.py
│   ├── aqvp.py
│   └── aqvp_feedback.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Environment Setup

## Requirements

* Python 3.11+
* Qiskit
* NumPy
* Matplotlib

## Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# Requirements

The current prototype uses:

```text
qiskit>=1.0,<3.0
numpy>=1.26
matplotlib>=3.8
```

---

# Experiments

## 1. Quantum Graph Representation

### File

```text
src/quantum_graph.py
```

This experiment creates a 3-node graph and maps each node to a qubit.

### Graph

```text
        Node 0
       /      \
      /        \
 Node 1 ------ Node 2
```

### Node Features

```text
Node 0 → 0.2
Node 1 → 0.7
Node 2 → 0.9
```

### Quantum Encoding

Node features are converted into rotation angles and encoded using parameterized `RY` gates.

Graph connectivity is represented using CNOT entanglement.

For three qubits, the resulting quantum state exists in:

```text
2^3 = 8
```

computational basis states.

### Purpose

To verify that graph features and graph connectivity can be represented in a small quantum circuit.

---

# 2. Classical Bellman Value Propagation

### File

```text
src/bellman.py
```

The prototype uses the Bellman update:

[
V(s)=\max_a [R(s,a)+\gamma V(s')]
]

with:

```text
γ = 0.9
```

Example rewards:

```text
0 → 1 = 0.7
0 → 2 = 0.9
1 → 2 = 1.0
```

### Observed Result

The prototype produced:

```text
V(Node 0) = 1.6000
```

with the selected decision:

```text
Node 0 → Node 1
```

### Purpose

To independently verify the classical Bellman component before integrating it with the quantum model.

---

# 3. QBN Proof of Concept

### File

```text
src/qbn_poc.py
```

The quantum circuit produces node-level quantum scores using Pauli-Z expectation values.

The score is normalized as:

[
Q_i=\frac{1+\langle Z_i\rangle}{2}
]

Observed quantum node scores:

```text
Node 0 = 0.9045
Node 1 = 0.2622
Node 2 = 0.7795
```

These scores are incorporated into the reward:

[
R_Q(s,a)=R(s,a)+\lambda Q(s')
]

with:

```text
λ = 0.5
```

### Observed Result

```text
QBN Value at Node 0 = 2.0819
```

The selected decision remained:

```text
Node 0 → Node 1
```

### Purpose

To demonstrate that quantum-derived information can influence Bellman value propagation.

---

# 4. Quantum Contribution Sensitivity

### File

```text
src/sensitivity.py
```

The quantum contribution parameter was varied:

[
\lambda \in {0,0.25,0.5,0.75,1.0}
]

### Results

|    λ | Node 0 Value | Decision |
| ---: | -----------: | -------- |
| 0.00 |       1.6000 | Node 1   |
| 0.25 |       1.8409 | Node 1   |
| 0.50 |       2.0819 | Node 1   |
| 0.75 |       2.3228 | Node 1   |
| 1.00 |       2.5638 | Node 1   |

### Observation

Increasing the quantum contribution changes the value estimate, but the selected policy does not change for this simple 3-node graph.

This motivated the development of an adaptive quantum contribution mechanism.

Results are saved to:

```text
results/sensitivity_results.csv
```

---

# 5. Adaptive Quantum Value Propagation — Version 1

### File

```text
src/aqvp.py
```

Instead of using a fixed quantum weight, the prototype adapts the quantum contribution using Bellman temporal-difference error.

The initial adaptive formulation is:

[
\lambda_{t+1}
=============

\operatorname{clip}
\left(
\lambda_t+\eta|\delta_t|,
0,1
\right)
]

where:

* (\lambda_t) = quantum contribution weight
* (\eta) = adaptation rate
* (\delta_t) = Bellman TD error

### Observed Results

```text
Classical Bellman = 1.6000
Fixed QBN         = 2.0819
QBN-AQVP          = 2.1273
```

The adaptive weight evolved from:

```text
0.25 → 0.5934
```

The decision remained:

```text
Node 0 → Node 1
```

### Purpose

To verify that quantum contribution can be adapted dynamically based on Bellman value error.

Results are saved to:

```text
results/aqvp_results.csv
```

---

# 6. Bellman-Guided Quantum Feedback

### File

```text
src/aqvp_feedback.py
```

This experiment extends AQVP by allowing Bellman error to influence the **quantum circuit parameters themselves**.

The prototype uses the update:

[
\theta_{t+1}
============

\theta_t+
\eta\lambda\delta_t
\frac{\partial Q}{\partial\theta}
]

where the quantum gradient is estimated using the parameter-shift rule:

[
\frac{\partial Q}{\partial\theta}
=================================

\frac{
Q(\theta+\pi/2)-Q(\theta-\pi/2)
}{2}
]

### Observed Parameter Change

Initial:

```text
[0.6283, 2.1991, 2.8274]
```

Final:

```text
[0.6357, 2.2383, 2.8396]
```

This demonstrates that Bellman feedback changes the quantum circuit parameters in the current proof-of-concept.

### Quantum Score Change

Initial:

```text
Node 0 = 0.9045
Node 1 = 0.2622
Node 2 = 0.7795
```

Final:

```text
Node 0 = 0.9023
Node 1 = 0.2509
Node 2 = 0.7955
```

### TD Error Behaviour

The Bellman TD error decreased approximately from:

```text
0.8932 → 0.2647 → 0.0047 → ~0
```

The Node-0 value stabilized around:

```text
V(Node 0) ≈ 2.0834
```

### Purpose

This experiment demonstrates the prototype feedback loop:

```text
Quantum State
      ↓
Quantum Score
      ↓
Bellman Value
      ↓
TD Error
      ↓
Quantum Parameter Update
      ↓
New Quantum State
```

Results are saved to:

```text
results/aqvp_feedback_results.csv
```

---

# Overall Prototype Progress

The implementation has progressed through the following stages:

```text
Quantum Graph Encoding
        ↓
Classical Bellman Propagation
        ↓
Quantum-Enhanced Bellman
        ↓
Quantum Contribution Sensitivity
        ↓
Adaptive Quantum Weighting
        ↓
Bellman-Guided Quantum Parameter Feedback
```

This provides the initial experimental foundation for the proposed QBN-AQVP framework.

---

# Current Research Status

### Completed

* [x] Project environment setup
* [x] 3-node quantum graph encoding
* [x] 3-qubit quantum circuit
* [x] Bellman value propagation
* [x] Quantum-Bellman integration
* [x] Quantum contribution sensitivity experiment
* [x] Adaptive quantum weighting prototype
* [x] Bellman-guided quantum parameter update
* [x] Result logging to CSV

### Planned

* [ ] Multi-node temporal graph experiments
* [ ] Multiple temporal snapshots
* [ ] Larger-scale validation
* [ ] METR-LA experiments
* [ ] Comparison with classical and quantum baselines
* [ ] Ablation studies
* [ ] Robustness and scalability analysis
* [ ] Formal mathematical development of QBN-AQVP
* [ ] Final journal-level evaluation

---

# Important Research Note

The current results are **proof-of-concept results on a small synthetic 3-node graph**.

They demonstrate technical feasibility of the proposed mechanisms but should **not yet be interpreted as evidence of quantum advantage or superior predictive performance**.

Large-scale temporal graph experiments and rigorous baseline comparisons are required before making such claims.

---

# Research Direction

The long-term framework is:

```text
Temporal Graph
      ↓
Quantum Graph Representation
      ↓
Quantum Bellman Operator
      ↓
Adaptive Quantum Value Propagation
      ↓
Quantum Temporal Memory
      ↓
Sequential Decision
```

The current implementation represents an initial prototype toward this framework.

---

# License

This repository is currently intended for research and academic development.

---

# Authors

**Sarvesh G**
**Praveena Anand**

Department of Computer Science and Engineering
Amrita School of Computing
Amrita Vishwa Vidyapeetham

````

### One thing I recommend before you commit this README

Use the wording **"proof-of-concept"**, **"prototype"**, and **"observed on the toy graph"** throughout. Don't put claims such as *quantum advantage*, *higher accuracy*, or *improved performance* in the README yet. We haven't established those experimentally.

Save this as:

```text
README.md
````

then run:

```powershell
git add README.md
git commit -m "Add project documentation"
git push
```
