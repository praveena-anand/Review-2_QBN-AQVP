import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ==========================================
# 1. TEMPORAL GRAPH
# ==========================================

# Three nodes with simple example features
node_features = {
    0: 0.2,
    1: 0.7,
    2: 0.9
}

# Graph connections
edges = [
    (0, 1),
    (0, 2),
    (1, 2)
]


# ==========================================
# 2. CREATE QUANTUM CIRCUIT
# ==========================================

# One qubit for each node
qc = QuantumCircuit(3)


# ==========================================
# 3. ENCODE NODE FEATURES
# ==========================================

for node, feature in node_features.items():

    # Convert feature into a rotation angle
    angle = feature * np.pi

    # Encode feature into the qubit
    qc.ry(angle, node)


# ==========================================
# 4. ENCODE GRAPH RELATIONSHIPS
# ==========================================

for node1, node2 in edges:

    # CNOT creates interaction between nodes
    qc.cx(node1, node2)


# ==========================================
# 5. DISPLAY CIRCUIT
# ==========================================

print("\n========== QUANTUM GRAPH CIRCUIT ==========\n")
print(qc.draw())


# ==========================================
# 6. SIMULATE QUANTUM STATE
# ==========================================

state = Statevector.from_instruction(qc)


print("\n========== QUANTUM STATE ==========\n")
print(state)


# ==========================================
# 7. MEASUREMENT PROBABILITIES
# ==========================================

probabilities = state.probabilities()

print("\n========== MEASUREMENT PROBABILITIES ==========\n")

for index, probability in enumerate(probabilities):

    if probability > 0.001:

        binary_state = format(index, "03b")

        print(
            f"|{binary_state}> : "
            f"{probability:.4f}"
        )