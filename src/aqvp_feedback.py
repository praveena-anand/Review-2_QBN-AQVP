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


# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

gamma = 0.9

lambda_q = 0.5

learning_rate = 0.10

iterations = 10

shift = np.pi / 2


# ============================================================
# 3. INITIAL QUANTUM PARAMETERS
# ============================================================

theta = np.array([
    node_features[0] * np.pi,
    node_features[1] * np.pi,
    node_features[2] * np.pi
])


# ============================================================
# 4. BUILD QUANTUM CIRCUIT
# ============================================================

def build_circuit(parameters):

    qc = QuantumCircuit(3)

    # Feature / trainable parameter encoding
    for node in range(3):

        qc.ry(parameters[node], node)

    # Graph relationships
    for node1, node2 in edges:

        qc.cx(node1, node2)

    return qc


# ============================================================
# 5. QUANTUM NODE SCORES
# ============================================================

def get_quantum_scores(parameters):

    qc = build_circuit(parameters)

    state = Statevector.from_instruction(qc)

    scores = {}

    for node in range(3):

        pauli_string = ["I", "I", "I"]

        pauli_string[2 - node] = "Z"

        observable = Pauli("".join(pauli_string))

        expectation = np.real(
            state.expectation_value(observable)
        )

        # Convert [-1, 1] to [0, 1]
        score = (expectation + 1.0) / 2.0

        scores[node] = score

    return scores


# ============================================================
# 6. PARAMETER-SHIFT GRADIENT
# ============================================================

def parameter_shift_gradient(parameters, node):

    gradients = np.zeros(3)

    for parameter_index in range(3):

        plus_parameters = parameters.copy()

        minus_parameters = parameters.copy()

        plus_parameters[parameter_index] += shift

        minus_parameters[parameter_index] -= shift

        plus_score = get_quantum_scores(
            plus_parameters
        )[node]

        minus_score = get_quantum_scores(
            minus_parameters
        )[node]

        gradients[parameter_index] = (
            plus_score - minus_score
        ) / 2.0

    return gradients


# ============================================================
# 7. BELLMAN TARGETS
# ============================================================

def calculate_targets(values, quantum_scores):

    targets = {}

    for node in graph:

        if not graph[node]:

            targets[node] = 0.0

            continue

        best_target = float("-inf")

        for next_node in graph[node]:

            reward = (
                rewards[(node, next_node)]
                + lambda_q
                * quantum_scores[next_node]
            )

            target = (
                reward
                + gamma
                * values[next_node]
            )

            best_target = max(
                best_target,
                target
            )

        targets[node] = best_target

    return targets


# ============================================================
# 8. INITIAL VALUES
# ============================================================

values = {
    0: 0.0,
    1: 0.0,
    2: 0.0
}


# ============================================================
# 9. TRAINING HISTORY
# ============================================================

history = []


print("\n==============================================")
print(" QBN-AQVP QUANTUM FEEDBACK EXPERIMENT")
print("==============================================\n")


print("Initial quantum parameters:")

for node, value in enumerate(theta):

    print(
        f"Theta {node}: {value:.4f}"
    )


# ============================================================
# 10. ADAPTIVE LEARNING LOOP
# ============================================================

for iteration in range(iterations):

    # --------------------------------------------
    # Quantum forward pass
    # --------------------------------------------

    quantum_scores = get_quantum_scores(theta)


    # --------------------------------------------
    # Bellman target
    # --------------------------------------------

    targets = calculate_targets(
        values,
        quantum_scores
    )


    # --------------------------------------------
    # TD errors
    # --------------------------------------------

    td_errors = {}

    for node in graph:

        td_errors[node] = (
            targets[node]
            - values[node]
        )


    # --------------------------------------------
    # Quantum parameter gradients
    # --------------------------------------------

    parameter_update = np.zeros(3)


    for node in graph:

        if not graph[node]:

            continue

        # Select the action that produced
        # the Bellman target.
        best_next_node = None

        best_target = float("-inf")

        for next_node in graph[node]:

            reward = (
                rewards[(node, next_node)]
                + lambda_q
                * quantum_scores[next_node]
            )

            target = (
                reward
                + gamma
                * values[next_node]
            )

            if target > best_target:

                best_target = target

                best_next_node = next_node


        # Gradient of quantum score
        gradient = parameter_shift_gradient(
            theta,
            best_next_node
        )


        # Bellman error guides the
        # quantum parameter update.
        parameter_update += (
            learning_rate
            * td_errors[node]
            * lambda_q
            * gradient
        )


    # --------------------------------------------
    # Update quantum parameters
    # --------------------------------------------

    theta = theta + parameter_update


    # --------------------------------------------
    # Update Bellman values
    # --------------------------------------------

    values = targets


    # --------------------------------------------
    # Record history
    # --------------------------------------------

    mean_td_error = np.mean(
        np.abs(
            list(td_errors.values())
        )
    )

    history.append(
        {
            "iteration": iteration + 1,
            "td_error": mean_td_error,
            "theta_0": theta[0],
            "theta_1": theta[1],
            "theta_2": theta[2],
            "q0": quantum_scores[0],
            "q1": quantum_scores[1],
            "q2": quantum_scores[2],
            "value0": values[0]
        }
    )


    # --------------------------------------------
    # Print progress
    # --------------------------------------------

    print(
        f"Iteration {iteration + 1}"
    )

    print(
        f"  TD Error      : "
        f"{mean_td_error:.4f}"
    )

    print(
        f"  Theta         : "
        f"[{theta[0]:.4f}, "
        f"{theta[1]:.4f}, "
        f"{theta[2]:.4f}]"
    )

    print(
        f"  Q Scores      : "
        f"[{quantum_scores[0]:.4f}, "
        f"{quantum_scores[1]:.4f}, "
        f"{quantum_scores[2]:.4f}]"
    )

    print(
        f"  V(Node 0)     : "
        f"{values[0]:.4f}"
    )

    print()


# ============================================================
# 11. FINAL QUANTUM SCORES
# ============================================================

final_scores = get_quantum_scores(theta)


# ============================================================
# 12. FINAL DECISION
# ============================================================

best_node = None

best_value = float("-inf")

for next_node in graph[0]:

    reward = (
        rewards[(0, next_node)]
        + lambda_q
        * final_scores[next_node]
    )

    value = (
        reward
        + gamma
        * values[next_node]
    )

    if value > best_value:

        best_value = value

        best_node = next_node


# ============================================================
# 13. FINAL RESULTS
# ============================================================

print("==============================================")
print(" FINAL AQVP RESULTS")
print("==============================================\n")

print("Final quantum parameters:")

for node, value in enumerate(theta):

    print(
        f"Theta {node}: {value:.4f}"
    )


print("\nFinal quantum scores:")

for node, score in final_scores.items():

    print(
        f"Node {node}: {score:.4f}"
    )


print(
    f"\nFinal Node 0 value: "
    f"{values[0]:.4f}"
)

print(
    f"Final decision: "
    f"Node 0 -> Node {best_node}"
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

with open(
    "results/aqvp_feedback_results.csv",
    "w"
) as file:

    file.write(
        "iteration,td_error,"
        "theta_0,theta_1,theta_2,"
        "q0,q1,q2,value0\n"
    )

    for row in history:

        file.write(
            f"{row['iteration']},"
            f"{row['td_error']:.6f},"
            f"{row['theta_0']:.6f},"
            f"{row['theta_1']:.6f},"
            f"{row['theta_2']:.6f},"
            f"{row['q0']:.6f},"
            f"{row['q1']:.6f},"
            f"{row['q2']:.6f},"
            f"{row['value0']:.6f}\n"
        )


print(
    "\nResults saved to "
    "results/aqvp_feedback_results.csv"
)