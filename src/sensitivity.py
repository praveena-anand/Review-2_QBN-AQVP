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

graph = {
    0: [1, 2],
    1: [2],
    2: []
}

rewards = {
    (0, 1): 0.7,
    (0, 2): 0.9,
    (1, 2): 1.0
}

gamma = 0.9


# ============================================================
# 2. BUILD QUANTUM CIRCUIT
# ============================================================

qc = QuantumCircuit(3)

for node, feature in node_features.items():

    angle = feature * np.pi

    qc.ry(angle, node)


for node1, node2 in edges:

    qc.cx(node1, node2)


# ============================================================
# 3. GET QUANTUM STATE
# ============================================================

state = Statevector.from_instruction(qc)


# ============================================================
# 4. EXTRACT QUANTUM NODE SCORES
# ============================================================

quantum_scores = {}

for node in range(3):

    pauli_string = ["I", "I", "I"]

    pauli_string[2 - node] = "Z"

    observable = Pauli("".join(pauli_string))

    expectation = np.real(
        state.expectation_value(observable)
    )

    score = (expectation + 1) / 2

    quantum_scores[node] = score


# ============================================================
# 5. BELLMAN CALCULATION
# ============================================================

def run_bellman(lambda_q):

    # Create quantum-enhanced rewards
    enhanced_rewards = {}

    for (node1, node2), reward in rewards.items():

        quantum_contribution = (
            lambda_q * quantum_scores[node2]
        )

        enhanced_rewards[(node1, node2)] = (
            reward + quantum_contribution
        )


    # Initial values
    values = {
        0: 0.0,
        1: 0.0,
        2: 0.0
    }


    # Value iteration
    for _ in range(10):

        new_values = {}

        for node in graph:

            if not graph[node]:

                new_values[node] = 0.0
                continue

            best_value = float("-inf")

            for next_node in graph[node]:

                reward = enhanced_rewards[
                    (node, next_node)
                ]

                value = (
                    reward
                    + gamma * values[next_node]
                )

                best_value = max(
                    best_value,
                    value
                )

            new_values[node] = best_value

        values = new_values


    # Find best action from Node 0
    best_node = None
    best_value = float("-inf")

    for next_node in graph[0]:

        reward = enhanced_rewards[
            (0, next_node)
        ]

        value = (
            reward
            + gamma * values[next_node]
        )

        if value > best_value:

            best_value = value
            best_node = next_node


    return values, best_node, best_value


# ============================================================
# 6. RUN SENSITIVITY EXPERIMENT
# ============================================================

lambda_values = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0
]


print("\n==============================================")
print(" QBN QUANTUM CONTRIBUTION SENSITIVITY")
print("==============================================\n")

print("Quantum Node Scores:")

for node, score in quantum_scores.items():

    print(
        f"Node {node}: {score:.4f}"
    )


print("\n----------------------------------------------")

print(
    f"{'Lambda':<10}"
    f"{'V(Node 0)':<15}"
    f"{'Decision':<15}"
)

print("----------------------------------------------")


results = []


for lambda_q in lambda_values:

    values, decision, final_value = (
        run_bellman(lambda_q)
    )

    results.append(
        (
            lambda_q,
            final_value,
            decision
        )
    )

    print(
        f"{lambda_q:<10.2f}"
        f"{final_value:<15.4f}"
        f"Node {decision}"
    )


print("----------------------------------------------")


# ============================================================
# 7. SAVE RESULTS
# ============================================================

with open(
    "results/sensitivity_results.csv",
    "w"
) as file:

    file.write(
        "lambda,node0_value,decision\n"
    )

    for lambda_q, value, decision in results:

        file.write(
            f"{lambda_q},"
            f"{value:.6f},"
            f"{decision}\n"
        )


print(
    "\nResults saved to "
    "results/sensitivity_results.csv"
)