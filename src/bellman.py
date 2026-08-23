# ==========================================
# BELLman VALUE PROPAGATION
# ==========================================

# Discount factor
gamma = 0.9


# ------------------------------------------
# 1. Temporal graph
# ------------------------------------------

graph = {
    0: [1, 2],
    1: [2],
    2: []
}


# ------------------------------------------
# 2. Rewards
# ------------------------------------------

rewards = {
    (0, 1): 0.7,
    (0, 2): 0.9,
    (1, 2): 1.0
}


# ------------------------------------------
# 3. Initial values
# ------------------------------------------

values = {
    0: 0.0,
    1: 0.0,
    2: 0.0
}


# ------------------------------------------
# 4. Bellman update
# ------------------------------------------

def bellman_update(node):

    next_nodes = graph[node]

    # Terminal state
    if not next_nodes:
        return 0.0

    best_value = float("-inf")

    for next_node in next_nodes:

        reward = rewards[(node, next_node)]

        future_value = values[next_node]

        value = reward + gamma * future_value

        if value > best_value:
            best_value = value

    return best_value


# ------------------------------------------
# 5. Perform value iterations
# ------------------------------------------

print("\n========== BELLMAN VALUE PROPAGATION ==========\n")

for iteration in range(5):

    new_values = {}

    for node in graph:

        new_values[node] = bellman_update(node)

    values = new_values

    print(f"Iteration {iteration + 1}")

    for node, value in values.items():

        print(f"Node {node}: {value:.4f}")

    print()


# ------------------------------------------
# 6. Best decision from Node 0
# ------------------------------------------

best_next_node = None
best_value = float("-inf")

for next_node in graph[0]:

    reward = rewards[(0, next_node)]

    value = reward + gamma * values[next_node]

    if value > best_value:

        best_value = value
        best_next_node = next_node


print("========== FINAL DECISION ==========\n")

print(
    f"From Node 0 → Node {best_next_node}"
)

print(
    f"Best expected value = {best_value:.4f}"
)