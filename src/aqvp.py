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
# 2. BUILD QUANTUM GRAPH
# ============================================================

qc = QuantumCircuit(3)

for node, feature in node_features.items():

    angle = feature * np.pi

    qc.ry(angle, node)


for node1, node2 in edges:

    qc.cx(node1, node2)


state = Statevector.from_instruction(qc)


# ============================================================
# 3. QUANTUM NODE SCORES
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


print("\n==============================================")
print(" QUANTUM NODE SCORES")
print("==============================================\n")

for node, score in quantum_scores.items():

    print(
        f"Node {node}: {score:.4f}"
    )


# ============================================================
# 4. BELLMAN SOLVER
# ============================================================

def run_classical_bellman(iterations=10):

    values = {
        0: 0.0,
        1: 0.0,
        2: 0.0
    }

    for _ in range(iterations):

        new_values = {}

        for node in graph:

            if not graph[node]:

                new_values[node] = 0.0
                continue

            best_value = float("-inf")

            for next_node in graph[node]:

                reward = rewards[(node, next_node)]

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

    return values


# ============================================================
# 5. FIXED QBN
# ============================================================

def run_fixed_qbn(lambda_q=0.5, iterations=10):

    values = {
        0: 0.0,
        1: 0.0,
        2: 0.0
    }

    # Quantum-enhanced rewards
    enhanced_rewards = {}

    for (node1, node2), reward in rewards.items():

        enhanced_rewards[(node1, node2)] = (
            reward
            + lambda_q * quantum_scores[node2]
        )

    for _ in range(iterations):

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

    return values


# ============================================================
# 6. ADAPTIVE QBN-AQVP
# ============================================================

def run_aqvp(
    initial_lambda=0.25,
    learning_rate=0.10,
    min_lambda=0.0,
    max_lambda=1.0,
    iterations=10
):

    values = {
        0: 0.0,
        1: 0.0,
        2: 0.0
    }

    lambda_q = initial_lambda

    history = []

    for iteration in range(iterations):

        new_values = {}

        td_errors = []

        # --------------------------------------------
        # Calculate Bellman values
        # --------------------------------------------

        for node in graph:

            if not graph[node]:

                new_values[node] = 0.0
                continue

            best_value = float("-inf")

            best_td_error = 0.0

            for next_node in graph[node]:

                # Quantum-enhanced reward
                reward = (
                    rewards[(node, next_node)]
                    + lambda_q
                    * quantum_scores[next_node]
                )

                # Bellman target
                target = (
                    reward
                    + gamma
                    * values[next_node]
                )

                # Current value estimate
                current_value = values[node]

                # Temporal-difference error
                td_error = (
                    target - current_value
                )

                td_errors.append(
                    abs(td_error)
                )

                if target > best_value:

                    best_value = target
                    best_td_error = td_error

            new_values[node] = best_value

        # --------------------------------------------
        # Adaptive quantum value propagation
        # --------------------------------------------

        if td_errors:

            mean_td_error = np.mean(td_errors)

        else:

            mean_td_error = 0.0

        # Increase quantum contribution when
        # Bellman error is high.
        lambda_q = (
            lambda_q
            + learning_rate * mean_td_error
        )

        # Keep lambda stable
        lambda_q = np.clip(
            lambda_q,
            min_lambda,
            max_lambda
        )

        values = new_values

        history.append(
            (
                iteration + 1,
                lambda_q,
                mean_td_error,
                values[0]
            )
        )

    return values, lambda_q, history


# ============================================================
# 7. RUN ALL THREE MODELS
# ============================================================

print("\n==============================================")
print(" MODEL COMPARISON")
print("==============================================\n")


# Classical
classical_values = run_classical_bellman()

print(
    "Classical Bellman Value:"
)

print(
    f"Node 0 = {classical_values[0]:.4f}"
)


# Fixed QBN
fixed_values = run_fixed_qbn(
    lambda_q=0.5
)

print(
    "\nFixed QBN Value:"
)

print(
    f"Node 0 = {fixed_values[0]:.4f}"
)


# AQVP
aqvp_values, final_lambda, history = run_aqvp()

print(
    "\nQBN-AQVP Value:"
)

print(
    f"Node 0 = {aqvp_values[0]:.4f}"
)

print(
    f"Final Lambda = {final_lambda:.4f}"
)


# ============================================================
# 8. AQVP LEARNING HISTORY
# ============================================================

print("\n==============================================")
print(" AQVP ADAPTATION HISTORY")
print("==============================================\n")

print(
    f"{'Iteration':<12}"
    f"{'Lambda':<12}"
    f"{'TD Error':<12}"
    f"{'V(Node 0)':<12}"
)

print("----------------------------------------------")

for iteration, lambda_q, td_error, value in history:

    print(
        f"{iteration:<12}"
        f"{lambda_q:<12.4f}"
        f"{td_error:<12.4f}"
        f"{value:<12.4f}"
    )


# ============================================================
# 9. FINAL DECISION
# ============================================================

def get_decision(values, lambda_q):

    best_node = None

    best_value = float("-inf")

    for next_node in graph[0]:

        reward = (
            rewards[(0, next_node)]
            + lambda_q
            * quantum_scores[next_node]
        )

        value = (
            reward
            + gamma
            * values[next_node]
        )

        if value > best_value:

            best_value = value
            best_node = next_node

    return best_node, best_value


classical_decision = max(
    graph[0],
    key=lambda n:
    rewards[(0, n)]
    + gamma * classical_values[n]
)

fixed_decision, fixed_decision_value = get_decision(
    fixed_values,
    0.5
)

aqvp_decision, aqvp_decision_value = get_decision(
    aqvp_values,
    final_lambda
)


print("\n==============================================")
print(" FINAL DECISIONS")
print("==============================================\n")

print(
    f"Classical Bellman : Node {classical_decision}"
)

print(
    f"Fixed QBN         : Node {fixed_decision}"
)

print(
    f"QBN-AQVP          : Node {aqvp_decision}"
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

with open(
    "results/aqvp_results.csv",
    "w"
) as file:

    file.write(
        "model,node0_value,decision,lambda\n"
    )

    file.write(
        f"Classical Bellman,"
        f"{classical_values[0]:.6f},"
        f"{classical_decision},"
        f"0.0\n"
    )

    file.write(
        f"Fixed QBN,"
        f"{fixed_values[0]:.6f},"
        f"{fixed_decision},"
        f"0.5\n"
    )

    file.write(
        f"QBN-AQVP,"
        f"{aqvp_values[0]:.6f},"
        f"{aqvp_decision},"
        f"{final_lambda:.6f}\n"
    )


print(
    "\nResults saved to "
    "results/aqvp_results.csv"
)