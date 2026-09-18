import os
import pickle
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ============================================================
# QBN-AQVP ON METR-LA
# ============================================================
#
# Pipeline:
#
# METR-LA traffic data
#        ↓
# Official sensor graph
#        ↓
# Connected 10-node subgraph
#        ↓
# Temporal traffic snapshots
#        ↓
# 10-qubit quantum graph encoding
#        ↓
# Quantum node scores
#        ↓
# Bellman value propagation
#        ↓
# Adaptive Quantum Value Propagation
#        ↓
# Parameter-shift quantum feedback
#        ↓
# Experimental results
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/metr-la.h5"
GRAPH_PATH = "data/adj_mx.pkl"

RESULTS_DIR = "results"

NUM_NODES = 10

# Number of temporal snapshots used in this experiment
NUM_TIME_STEPS = 100

# Bellman discount factor
GAMMA = 0.90

# Initial adaptive quantum contribution
INITIAL_LAMBDA = 0.25

# Learning rate for adaptive lambda
LAMBDA_LEARNING_RATE = 0.10

# Learning rate for quantum parameters
THETA_LEARNING_RATE = 0.05

# Edge threshold
EDGE_THRESHOLD = 0.0


# ============================================================
# STEP 1
# LOAD METR-LA TRAFFIC DATA
# ============================================================

def load_metr_la():

    print("=" * 70)
    print("STEP 1: LOADING METR-LA")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nMETR-LA file not found:\n{DATA_PATH}"
        )

    # --------------------------------------------------------
    # Direct HDF5 loading.
    #
    # This avoids the Pandas/PyTables compatibility issue
    # encountered earlier.
    # --------------------------------------------------------

    with h5py.File(DATA_PATH, "r") as f:

        values = f["df/block0_values"][:]

        sensor_ids = f["df/axis0"][:]

        timestamps = f["df/axis1"][:]

    # --------------------------------------------------------
    # Decode sensor IDs
    # --------------------------------------------------------

    sensor_ids = [
        x.decode("utf-8") if isinstance(x, bytes)
        else str(x)
        for x in sensor_ids
    ]

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        timestamps,
        unit="ns"
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        values,
        index=timestamps,
        columns=sensor_ids
    )

    print("\nMETR-LA loaded successfully.")

    print(f"Time steps : {df.shape[0]}")
    print(f"Sensors    : {df.shape[1]}")

    print(
        f"Time range : "
        f"{df.index[0]} → {df.index[-1]}"
    )

    print(
        f"Missing values: "
        f"{df.isna().sum().sum()}"
    )

    return df


# ============================================================
# STEP 2
# LOAD OFFICIAL SENSOR GRAPH
# ============================================================

def load_graph():

    print("\n" + "=" * 70)
    print("STEP 2: LOADING METR-LA SENSOR GRAPH")
    print("=" * 70)

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(
            f"\nGraph file not found:\n{GRAPH_PATH}"
        )

    with open(GRAPH_PATH, "rb") as f:

        graph_data = pickle.load(
            f,
            encoding="latin1"
        )

    sensor_ids = graph_data[0]

    sensor_id_to_index = graph_data[1]

    adjacency = np.asarray(
        graph_data[2],
        dtype=float
    )

    print("\nGraph loaded successfully.")

    print(
        f"Adjacency matrix: "
        f"{adjacency.shape}"
    )

    print(
        f"Number of sensors: "
        f"{len(sensor_ids)}"
    )

    return (
        sensor_ids,
        sensor_id_to_index,
        adjacency
    )


# ============================================================
# STEP 3
# FIND CONNECTED 10-NODE SUBGRAPH
# ============================================================

def find_10_node_subgraph(adjacency):

    print("\n" + "=" * 70)
    print("STEP 3: SELECTING CONNECTED 10-NODE SUBGRAPH")
    print("=" * 70)

    num_sensors = adjacency.shape[0]

    # --------------------------------------------------------
    # Convert weighted graph to binary connectivity graph
    # --------------------------------------------------------

    binary_graph = (
        adjacency > EDGE_THRESHOLD
    ).astype(int)

    # Remove self-loops
    np.fill_diagonal(
        binary_graph,
        0
    )

    # --------------------------------------------------------
    # Calculate degree
    # --------------------------------------------------------

    degrees = binary_graph.sum(axis=1)

    # Start from highest-degree node
    start_node = int(
        np.argmax(degrees)
    )

    selected = [start_node]

    # --------------------------------------------------------
    # Grow connected subgraph
    # --------------------------------------------------------

    while len(selected) < NUM_NODES:

        candidates = set()

        for node in selected:

            neighbors = np.where(
                binary_graph[node] > 0
            )[0]

            for neighbor in neighbors:

                if neighbor not in selected:

                    candidates.add(
                        int(neighbor)
                    )

        if not candidates:

            raise RuntimeError(
                "Unable to construct connected "
                "10-node subgraph."
            )

        # ----------------------------------------------------
        # Select candidate with strongest connection
        # to current subgraph.
        # ----------------------------------------------------

        best_candidate = None
        best_strength = -1.0

        for candidate in candidates:

            strength = 0.0

            for node in selected:

                strength += max(
                    adjacency[candidate, node],
                    adjacency[node, candidate]
                )

            if strength > best_strength:

                best_strength = strength
                best_candidate = candidate

        selected.append(
            best_candidate
        )

    selected = np.array(
        selected,
        dtype=int
    )

    print("\nSelected sensors:")

    for local_index, global_index in enumerate(selected):

        print(
            f"Node {local_index} "
            f"→ Sensor {global_index}"
        )

    return selected


# ============================================================
# STEP 4
# BUILD 10 × 10 GRAPH
# ============================================================

def build_subgraph(
    adjacency,
    selected_nodes,
    sensor_ids
):

    print("\n" + "=" * 70)
    print("STEP 4: BUILDING 10 × 10 GRAPH")
    print("=" * 70)

    # Extract selected graph
    subgraph = adjacency[
        np.ix_(
            selected_nodes,
            selected_nodes
        )
    ]

    # --------------------------------------------------------
    # METR-LA adjacency is asymmetric.
    #
    # For our quantum interaction graph, use:
    #
    # A_sym(i,j) = max(A(i,j), A(j,i))
    #
    # This preserves the strongest observed connection
    # while producing an undirected interaction structure.
    # --------------------------------------------------------

    symmetric_graph = np.maximum(
        subgraph,
        subgraph.T
    )

    # Remove self-loops
    np.fill_diagonal(
        symmetric_graph,
        0
    )

    # --------------------------------------------------------
    # Normalize graph weights
    # --------------------------------------------------------

    max_weight = symmetric_graph.max()

    if max_weight > 0:

        normalized_graph = (
            symmetric_graph / max_weight
        )

    else:

        normalized_graph = symmetric_graph

    print("\nSelected 10-node graph:")

    for i, global_index in enumerate(selected_nodes):

        print(
            f"Node {i} "
            f"→ Sensor {sensor_ids[global_index]}"
        )

    print("\n10 × 10 normalized adjacency:")

    print(
        np.round(
            normalized_graph,
            4
        )
    )

    edge_count = 0

    for i in range(NUM_NODES):

        for j in range(i + 1, NUM_NODES):

            if normalized_graph[i, j] > 0:

                edge_count += 1

    print(
        f"\nUndirected connections: "
        f"{edge_count}"
    )

    return normalized_graph


# ============================================================
# STEP 5
# PREPARE TEMPORAL TRAFFIC DATA
# ============================================================

def prepare_temporal_data(
    df,
    selected_nodes,
    sensor_ids
):

    print("\n" + "=" * 70)
    print("STEP 5: PREPARING TEMPORAL TRAFFIC DATA")
    print("=" * 70)

    selected_sensor_ids = [
        sensor_ids[index]
        for index in selected_nodes
    ]

    selected_df = df[
        selected_sensor_ids
    ].copy()

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    selected_df = selected_df.ffill().bfill()

    # --------------------------------------------------------
    # Z-score normalization
    # --------------------------------------------------------

    mean = selected_df.mean()

    std = selected_df.std()

    normalized = (
        selected_df - mean
    ) / (
        std + 1e-8
    )

    # --------------------------------------------------------
    # Convert normalized traffic values to [0,1]
    #
    # This creates a traffic utility representation.
    #
    # Higher speed → higher utility.
    #
    # IMPORTANT:
    # This is our explicitly defined reward proxy,
    # not a native METR-LA reward.
    # --------------------------------------------------------

    min_values = normalized.min()

    max_values = normalized.max()

    utility = (
        normalized - min_values
    ) / (
        max_values - min_values + 1e-8
    )

    utility = utility.iloc[
        :NUM_TIME_STEPS
    ]

    snapshots = utility.to_numpy(
        dtype=float
    )

    print(
        f"\nTemporal snapshots: "
        f"{snapshots.shape}"
    )

    print(
        "\nSnapshot 0:"
    )

    print(
        np.round(
            snapshots[0],
            4
        )
    )

    print(
        "\nSnapshot 1:"
    )

    print(
        np.round(
            snapshots[1],
            4
        )
    )

    return snapshots


# ============================================================
# STEP 6
# QUANTUM GRAPH ENCODING
# ============================================================

def quantum_graph_scores(
    node_features,
    graph,
    theta
):

    # --------------------------------------------------------
    # Create 10-qubit circuit
    # --------------------------------------------------------

    qc = QuantumCircuit(
        NUM_NODES
    )

    # --------------------------------------------------------
    # Encode node features
    #
    # Feature x_i ∈ [0,1]
    #
    # RY angle = π x_i + trainable theta_i
    # --------------------------------------------------------

    for i in range(NUM_NODES):

        angle = (
            np.pi * node_features[i]
            + theta[i]
        )

        qc.ry(
            angle,
            i
        )

    # --------------------------------------------------------
    # Encode graph structure
    #
    # CRY gates use normalized edge weights.
    # --------------------------------------------------------

    for i in range(NUM_NODES):

        for j in range(i + 1, NUM_NODES):

            weight = graph[i, j]

            if weight > 0:

                angle = (
                    np.pi
                    * 0.5
                    * weight
                )

                qc.cry(
                    angle,
                    i,
                    j
                )

    # --------------------------------------------------------
    # Simulate
    # --------------------------------------------------------

    state = Statevector.from_instruction(
        qc
    )

    # --------------------------------------------------------
    # Calculate <Z_i>
    # --------------------------------------------------------

    probabilities = np.abs(
        state.data
    ) ** 2

    scores = np.zeros(
        NUM_NODES
    )

    for basis_state, probability in enumerate(
        probabilities
    ):

        for qubit in range(NUM_NODES):

            # Qiskit basis-state convention:
            # bit position corresponds to qubit.
            bit = (
                basis_state
                >> qubit
            ) & 1

            z_value = (
                1.0
                if bit == 0
                else -1.0
            )

            scores[qubit] += (
                probability
                * z_value
            )

    # --------------------------------------------------------
    # Convert <Z> to [0,1]
    #
    # Q_i = (1 + <Z_i>) / 2
    # --------------------------------------------------------

    scores = (
        1.0 + scores
    ) / 2.0

    return scores


# ============================================================
# STEP 7
# BELLMan NEIGHBOR VALUES
# ============================================================

def bellman_update(
    reward,
    previous_values,
    graph
):

    values = np.zeros(
        NUM_NODES
    )

    # --------------------------------------------------------
    # For every node:
    #
    # V(i) = R(i) +
    #        gamma * max_j[A(i,j)V(j)]
    #
    # where j is a connected neighboring node.
    # --------------------------------------------------------

    for i in range(NUM_NODES):

        neighbors = np.where(
            graph[i] > 0
        )[0]

        if len(neighbors) == 0:

            values[i] = reward[i]

        else:

            candidate_values = []

            for j in neighbors:

                candidate = (
                    graph[i, j]
                    * previous_values[j]
                )

                candidate_values.append(
                    candidate
                )

            values[i] = (
                reward[i]
                + GAMMA
                * max(candidate_values)
            )

    return values


# ============================================================
# STEP 8
# PARAMETER-SHIFT GRADIENT
# ============================================================

def parameter_shift_gradient(
    node_features,
    graph,
    theta,
    node
):

    theta_plus = theta.copy()

    theta_minus = theta.copy()

    theta_plus[node] += (
        np.pi / 2
    )

    theta_minus[node] -= (
        np.pi / 2
    )

    scores_plus = quantum_graph_scores(
        node_features,
        graph,
        theta_plus
    )

    scores_minus = quantum_graph_scores(
        node_features,
        graph,
        theta_minus
    )

    gradient = (
        scores_plus[node]
        - scores_minus[node]
    ) / 2.0

    return gradient


# ============================================================
# STEP 9
# FULL QBN-AQVP EXPERIMENT
# ============================================================

def run_qbn_aqvp(
    snapshots,
    graph
):

    print("\n" + "=" * 70)
    print("STEP 6: RUNNING 10-QUBIT QBN-AQVP")
    print("=" * 70)

    # --------------------------------------------------------
    # Initialize trainable quantum parameters
    # --------------------------------------------------------

    theta = np.zeros(
        NUM_NODES
    )

    # Initial Bellman values
    values = np.zeros(
        NUM_NODES
    )

    lambda_value = INITIAL_LAMBDA

    # --------------------------------------------------------
    # Store experiment history
    # --------------------------------------------------------

    history = []

    initial_scores = None

    final_scores = None

    # --------------------------------------------------------
    # Temporal loop
    # --------------------------------------------------------

    for t in range(
        NUM_TIME_STEPS - 1
    ):

        current_state = snapshots[t]

        next_state = snapshots[t + 1]

        # ----------------------------------------------------
        # Quantum representation
        # ----------------------------------------------------

        q_scores = quantum_graph_scores(
            current_state,
            graph,
            theta
        )

        if initial_scores is None:

            initial_scores = q_scores.copy()

        # ----------------------------------------------------
        # Quantum-enhanced reward
        #
        # R_Q = traffic utility
        #       + lambda * quantum score
        # ----------------------------------------------------

        quantum_reward = (
            current_state
            + lambda_value * q_scores
        )

        # ----------------------------------------------------
        # Bellman update
        # ----------------------------------------------------

        new_values = bellman_update(
            quantum_reward,
            values,
            graph
        )

        # ----------------------------------------------------
        # Select node with largest current value
        # ----------------------------------------------------

        decision_node = int(
            np.argmax(new_values)
        )

        # ----------------------------------------------------
        # One-step TD target
        #
        # T = R + gamma V(next)
        # ----------------------------------------------------

        next_q_scores = quantum_graph_scores(
            next_state,
            graph,
            theta
        )

        next_reward = (
            next_state
            + lambda_value
            * next_q_scores
        )

        next_values = bellman_update(
            next_reward,
            new_values,
            graph
        )

        td_target = (
            quantum_reward[decision_node]
            + GAMMA
            * next_values[decision_node]
        )

        td_error = (
            td_target
            - new_values[decision_node]
        )

        # ----------------------------------------------------
        # Adaptive lambda
        #
        # lambda_(t+1)
        # =
        # clip(
        # lambda_t + eta |delta_t|,
        # 0, 1
        # )
        # ----------------------------------------------------

        lambda_value = np.clip(
            lambda_value
            + LAMBDA_LEARNING_RATE
            * abs(td_error),
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Parameter-shift feedback
        #
        # theta_(t+1)
        # =
        # theta_t
        # +
        # eta * lambda * delta * dQ/dtheta
        # ----------------------------------------------------

        gradient = parameter_shift_gradient(
            current_state,
            graph,
            theta,
            decision_node
        )

        theta[decision_node] += (
            THETA_LEARNING_RATE
            * lambda_value
            * td_error
            * gradient
        )

        # ----------------------------------------------------
        # Store values
        # ----------------------------------------------------

        values = new_values

        history.append(
            {
                "time_step": t,
                "value": float(
                    values[decision_node]
                ),
                "td_error": float(
                    td_error
                ),
                "lambda": float(
                    lambda_value
                ),
                "decision_node": decision_node,
                "quantum_score": float(
                    q_scores[decision_node]
                ),
                "gradient": float(
                    gradient
                )
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            t == 0
            or (t + 1) % 10 == 0
        ):

            print(
                f"Time {t:3d} | "
                f"Value={values[decision_node]:.4f} | "
                f"TD={td_error:.4f} | "
                f"Lambda={lambda_value:.4f} | "
                f"Node={decision_node}"
            )

    # --------------------------------------------------------
    # Final quantum state
    # --------------------------------------------------------

    final_scores = quantum_graph_scores(
        snapshots[-1],
        graph,
        theta
    )

    return (
        history,
        theta,
        initial_scores,
        final_scores,
        values
    )


# ============================================================
# STEP 10
# CLASSICAL BASELINE
# ============================================================

def run_classical_baseline(
    snapshots,
    graph
):

    print("\n" + "=" * 70)
    print("CLASSICAL BELLMAN BASELINE")
    print("=" * 70)

    values = np.zeros(
        NUM_NODES
    )

    history = []

    for t in range(
        NUM_TIME_STEPS - 1
    ):

        reward = snapshots[t]

        values = bellman_update(
            reward,
            values,
            graph
        )

        node = int(
            np.argmax(values)
        )

        history.append(
            {
                "time_step": t,
                "value": float(
                    values[node]
                ),
                "decision_node": node
            }
        )

    return history


# ============================================================
# STEP 11
# SAVE RESULTS
# ============================================================

def save_results(
    history,
    classical_history,
    theta,
    initial_scores,
    final_scores,
    final_values
):

    print("\n" + "=" * 70)
    print("STEP 7: SAVING RESULTS")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # QBN-AQVP history
    # --------------------------------------------------------

    aqvp_df = pd.DataFrame(
        history
    )

    aqvp_path = os.path.join(
        RESULTS_DIR,
        "metrla_qbn_aqvp_results.csv"
    )

    aqvp_df.to_csv(
        aqvp_path,
        index=False
    )

    # --------------------------------------------------------
    # Classical baseline
    # --------------------------------------------------------

    classical_df = pd.DataFrame(
        classical_history
    )

    classical_path = os.path.join(
        RESULTS_DIR,
        "metrla_classical_results.csv"
    )

    classical_df.to_csv(
        classical_path,
        index=False
    )

    # --------------------------------------------------------
    # Quantum parameter results
    # --------------------------------------------------------

    parameter_df = pd.DataFrame(
        {
            "node": np.arange(
                NUM_NODES
            ),
            "initial_quantum_score":
                initial_scores,
            "final_quantum_score":
                final_scores,
            "final_theta":
                theta
        }
    )

    parameter_path = os.path.join(
        RESULTS_DIR,
        "metrla_quantum_parameters.csv"
    )

    parameter_df.to_csv(
        parameter_path,
        index=False
    )

    # --------------------------------------------------------
    # Final values
    # --------------------------------------------------------

    value_df = pd.DataFrame(
        {
            "node": np.arange(
                NUM_NODES
            ),
            "final_value":
                final_values
        }
    )

    value_path = os.path.join(
        RESULTS_DIR,
        "metrla_final_values.csv"
    )

    value_df.to_csv(
        value_path,
        index=False
    )

    print(
        f"\nSaved:\n"
        f"{aqvp_path}\n"
        f"{classical_path}\n"
        f"{parameter_path}\n"
        f"{value_path}"
    )


# ============================================================
# STEP 12
# GENERATE PLOTS
# ============================================================

def generate_plots(
    history,
    classical_history
):

    print("\n" + "=" * 70)
    print("STEP 8: GENERATING EXPERIMENTAL PLOTS")
    print("=" * 70)

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    aqvp_df = pd.DataFrame(
        history
    )

    classical_df = pd.DataFrame(
        classical_history
    )

    # --------------------------------------------------------
    # Plot 1: TD Error
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        aqvp_df["time_step"],
        aqvp_df["td_error"]
    )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "TD Error"
    )

    plt.title(
        "QBN-AQVP Temporal TD Error"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "metrla_td_error.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 2: Adaptive Lambda
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        aqvp_df["time_step"],
        aqvp_df["lambda"]
    )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Adaptive λ"
    )

    plt.title(
        "Adaptive Quantum Contribution"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "metrla_adaptive_lambda.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Plot 3: Value comparison
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        classical_df["time_step"],
        classical_df["value"],
        label="Classical Bellman"
    )

    plt.plot(
        aqvp_df["time_step"],
        aqvp_df["value"],
        label="QBN-AQVP"
    )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Value"
    )

    plt.title(
        "Classical Bellman vs QBN-AQVP"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "metrla_value_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    print(
        "\nPlots saved inside results/."
    )


# ============================================================
# STEP 13
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    selected_nodes,
    sensor_ids,
    history,
    theta,
    initial_scores,
    final_scores,
    final_values
):

    print("\n")
    print("=" * 70)
    print("FINAL QBN-AQVP METR-LA RESULTS")
    print("=" * 70)

    print("\nSelected 10 sensors:")

    for i, global_index in enumerate(
        selected_nodes
    ):

        print(
            f"Node {i}: "
            f"Sensor {sensor_ids[global_index]}"
        )

    print("\nInitial quantum scores:")

    print(
        np.round(
            initial_scores,
            4
        )
    )

    print("\nFinal quantum scores:")

    print(
        np.round(
            final_scores,
            4
        )
    )

    print("\nFinal quantum parameters θ:")

    print(
        np.round(
            theta,
            4
        )
    )

    print("\nFinal Bellman values:")

    print(
        np.round(
            final_values,
            4
        )
    )

    if len(history) > 0:

        print("\nInitial TD error:")

        print(
            f"{history[0]['td_error']:.6f}"
        )

        print("\nFinal TD error:")

        print(
            f"{history[-1]['td_error']:.6f}"
        )

        print("\nFinal adaptive λ:")

        print(
            f"{history[-1]['lambda']:.6f}"
        )

        print("\nFinal decision node:")

        print(
            history[-1]["decision_node"]
        )

    print("\n" + "=" * 70)

    print(
        "QBN-AQVP EXPERIMENT COMPLETE"
    )

    print("=" * 70)

    print(
        "\nPipeline successfully executed:"
    )

    print(
        "METR-LA → 10-node graph → "
        "10 qubits → Quantum scores → "
        "Bellman → TD error → "
        "Adaptive λ → Parameter-shift feedback"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load traffic dataset
    # --------------------------------------------------------

    df = load_metr_la()

    # --------------------------------------------------------
    # 2. Load graph
    # --------------------------------------------------------

    (
        sensor_ids,
        sensor_id_to_index,
        adjacency
    ) = load_graph()

    # --------------------------------------------------------
    # 3. Find connected 10-node subgraph
    # --------------------------------------------------------

    selected_nodes = find_10_node_subgraph(
        adjacency
    )

    # --------------------------------------------------------
    # 4. Build normalized 10-node graph
    # --------------------------------------------------------

    graph = build_subgraph(
        adjacency,
        selected_nodes,
        sensor_ids
    )

    # --------------------------------------------------------
    # 5. Prepare temporal traffic data
    # --------------------------------------------------------

    snapshots = prepare_temporal_data(
        df,
        selected_nodes,
        sensor_ids
    )

    # --------------------------------------------------------
    # 6. Run classical baseline
    # --------------------------------------------------------

    classical_history = run_classical_baseline(
        snapshots,
        graph
    )

    # --------------------------------------------------------
    # 7. Run QBN-AQVP
    # --------------------------------------------------------

    (
        history,
        theta,
        initial_scores,
        final_scores,
        final_values
    ) = run_qbn_aqvp(
        snapshots,
        graph
    )

    # --------------------------------------------------------
    # 8. Save results
    # --------------------------------------------------------

    save_results(
        history,
        classical_history,
        theta,
        initial_scores,
        final_scores,
        final_values
    )

    # --------------------------------------------------------
    # 9. Generate plots
    # --------------------------------------------------------

    generate_plots(
        history,
        classical_history
    )

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print_final_summary(
        selected_nodes,
        sensor_ids,
        history,
        theta,
        initial_scores,
        final_scores,
        final_values
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()