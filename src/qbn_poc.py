import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli


# ============================================================
# 1. TEMPORAL GRAPH
# ============================================================

node_features = {
    0: 0.2,
    1: 0.7,
    2: 0.9
}

edges = [
    (0, 1),
    (0, 2),
    (1, 2)
]

# Classical rewards
rewards = {
    (0, 1): 0.7,
    (0, 2): 0.9,
    (1, 2): 1.0
}

gamma = 0.9

# Strength of quantum contribution
lambda_q = 0.5


# ============================================================
# 2. CREATE QUANTUM GRAPH CIRCUIT
# ============================================================

qc = QuantumCircuit(3)

# Encode node features
for node, feature in node_features.items():

    angle = feature * np.pi

    qc.ry(angle, node)


# Encode graph relationships
for node1, node2 in edges:

    qc.cx(node1, node2)


print("\n========== QUANTUM GRAPH CIRCUIT ==========\n")
print(qc.draw())


# ============================================================
# 3. SIMULATE QUANTUM STATE
# ============================================================

state = Statevector.from_instruction(qc)

print("\n========== QUANTUM STATE ==========\n")
print(state)


# ============================================================
# 4. EXTRACT QUANTUM NODE SCORES
# ============================================================

quantum_scores = {}

print("\n========== QUANTUM NODE SCORES ==========\n")

for node in range(3):

    # Z observable for the selected qubit
    pauli_string = ["I", "I", "I"]
    pauli_string[2 - node] = "Z"

    observable = Pauli("".join(pauli_string))

    expectation = np.real(
        state.expectation_value(observable)
    )

    # Convert expectation from [-1, 1] to [0, 1]
    score = (expectation + 1) / 2

    quantum_scores[node] = score

    print(
        f"Node {node}: "
        f"Expectation = {expectation:.4f}, "
        f"Quantum Score = {score:.4f}"
    )


# ============================================================
# 5. QUANTUM-ENHANCED REWARD
# ============================================================

enhanced_rewards = {}

print("\n========== QUANTUM-ENHANCED REWARDS ==========\n")

for (node1, node2), reward in rewards.items():

    quantum_contribution = lambda_q * quantum_scores[node2]

    enhanced_reward = reward + quantum_contribution

    enhanced_rewards[(node1, node2)] = enhanced_reward

    print(
        f"{node1} -> {node2} : "
        f"Classical = {reward:.4f}, "
        f"Quantum Contribution = {quantum_contribution:.4f}, "
        f"Enhanced Reward = {enhanced_reward:.4f}"
    )


# ============================================================
# 6. BELLMAN VALUE PROPAGATION
# ============================================================

graph = {
    0: [1, 2],
    1: [2],
    2: []
}

values = {
    0: 0.0,
    1: 0.0,
    2: 0.0
}


def bellman_update(node):

    next_nodes = graph[node]

    # Terminal node
    if not next_nodes:
        return 0.0

    best_value = float("-inf")

    for next_node in next_nodes:

        reward = enhanced_rewards[(node, next_node)]

        future_value = values[next_node]

        value = reward + gamma * future_value

        if value > best_value:
            best_value = value

    return best_value


# ============================================================
# 7. ITERATIVE VALUE PROPAGATION
# ============================================================

print("\n========== QBN VALUE PROPAGATION ==========\n")

for iteration in range(5):

    new_values = {}

    for node in graph:

        new_values[node] = bellman_update(node)

    values = new_values

    print(f"Iteration {iteration + 1}")

    for node, value in values.items():

        print(f"Node {node}: {value:.4f}")

    print()


# ============================================================
# 8. FINAL DECISION
# ============================================================

best_node = None
best_value = float("-inf")

for next_node in graph[0]:

    reward = enhanced_rewards[(0, next_node)]

    value = reward + gamma * values[next_node]

    if value > best_value:

        best_value = value
        best_node = next_node


print("========== FINAL QBN DECISION ==========\n")

print(
    f"From Node 0 -> Node {best_node}"
)

print(
    f"Final QBN Value = {best_value:.4f}"
)