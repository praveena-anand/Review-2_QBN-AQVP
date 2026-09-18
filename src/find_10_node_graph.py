import os
import pickle
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = "data/adj_mx.pkl"

NUM_NODES = 10

# Any adjacency value greater than this is treated as an edge.
# We use > 0 because the METR-LA adjacency matrix already
# contains the graph weights.
EDGE_THRESHOLD = 0.0


# ============================================================
# LOAD METR-LA GRAPH
# ============================================================

def load_graph():

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(
            f"\nGraph file not found:\n{GRAPH_PATH}"
        )

    with open(GRAPH_PATH, "rb") as f:

        graph_data = pickle.load(
            f,
            encoding="latin1"
        )

    # adj_mx.pkl structure:
    #
    # graph_data[0] -> sensor IDs
    # graph_data[1] -> sensor ID to index mapping
    # graph_data[2] -> adjacency matrix

    sensor_ids = graph_data[0]

    sensor_id_to_index = graph_data[1]

    adjacency_matrix = np.asarray(
        graph_data[2],
        dtype=float
    )

    return (
        sensor_ids,
        sensor_id_to_index,
        adjacency_matrix
    )


# ============================================================
# FIND CONNECTED 10-NODE SUBGRAPH
# ============================================================

def find_connected_subgraph(adjacency_matrix):

    num_sensors = adjacency_matrix.shape[0]

    # Convert weighted adjacency matrix to binary graph
    adjacency_binary = (
        adjacency_matrix > EDGE_THRESHOLD
    ).astype(int)

    # Remove self-connections
    np.fill_diagonal(
        adjacency_binary,
        0
    )

    # --------------------------------------------------------
    # Start from the sensor with the highest degree
    # --------------------------------------------------------

    degrees = adjacency_binary.sum(axis=1)

    start_node = int(
        np.argmax(degrees)
    )

    selected_nodes = [start_node]

    # --------------------------------------------------------
    # Expand the subgraph
    # --------------------------------------------------------

    while len(selected_nodes) < NUM_NODES:

        candidate_nodes = set()

        for node in selected_nodes:

            neighbors = np.where(
                adjacency_binary[node] > 0
            )[0]

            for neighbor in neighbors:

                if neighbor not in selected_nodes:

                    candidate_nodes.add(
                        int(neighbor)
                    )

        if not candidate_nodes:

            raise RuntimeError(
                "Could not find enough connected nodes."
            )

        # ----------------------------------------------------
        # Choose the candidate with the largest connection
        # to the current selected subgraph.
        # ----------------------------------------------------

        best_candidate = None
        best_score = -1

        for candidate in candidate_nodes:

            connection_strength = 0.0

            for selected in selected_nodes:

                connection_strength += (
                    adjacency_matrix[
                        candidate,
                        selected
                    ]
                )

                connection_strength += (
                    adjacency_matrix[
                        selected,
                        candidate
                    ]
                )

            if connection_strength > best_score:

                best_score = connection_strength
                best_candidate = candidate

        selected_nodes.append(
            best_candidate
        )

    return selected_nodes


# ============================================================
# DISPLAY GRAPH
# ============================================================

def display_graph(
    selected_nodes,
    sensor_ids,
    adjacency_matrix
):

    print("=" * 60)
    print("SELECTED 10-NODE METR-LA SUBGRAPH")
    print("=" * 60)

    print("\nSelected nodes:\n")

    for local_index, global_index in enumerate(
        selected_nodes
    ):

        sensor_id = sensor_ids[global_index]

        print(
            f"Node {local_index} "
            f"-> Sensor {sensor_id} "
            f"(Global index {global_index})"
        )

    # --------------------------------------------------------
    # Extract 10 x 10 adjacency matrix
    # --------------------------------------------------------

    subgraph = adjacency_matrix[
        np.ix_(
            selected_nodes,
            selected_nodes
        )
    ]

    # Remove self-loops for graph analysis
    subgraph_no_self = subgraph.copy()

    np.fill_diagonal(
        subgraph_no_self,
        0
    )

    print("\n" + "=" * 60)
    print("10 × 10 ADJACENCY MATRIX")
    print("=" * 60)

    print(
        np.round(
            subgraph,
            4
        )
    )

    # --------------------------------------------------------
    # Display edges
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRAPH EDGES")
    print("=" * 60)

    edge_count = 0

    for i in range(NUM_NODES):

        for j in range(i + 1, NUM_NODES):

            weight_ij = subgraph[i, j]
            weight_ji = subgraph[j, i]

            if (
                weight_ij > EDGE_THRESHOLD
                or weight_ji > EDGE_THRESHOLD
            ):

                print(
                    f"Node {i} "
                    f"({sensor_ids[selected_nodes[i]]})"
                    f" <-> "
                    f"Node {j} "
                    f"({sensor_ids[selected_nodes[j]]})"
                    f" | "
                    f"w(i,j)={weight_ij:.4f}"
                    f" | "
                    f"w(j,i)={weight_ji:.4f}"
                )

                edge_count += 1

    print(
        f"\nTotal undirected connections: "
        f"{edge_count}"
    )

    return subgraph


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("METR-LA 10-NODE GRAPH CONSTRUCTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    (
        sensor_ids,
        sensor_id_to_index,
        adjacency_matrix
    ) = load_graph()

    print(
        f"\nFull graph size: "
        f"{adjacency_matrix.shape}"
    )

    print(
        f"Number of sensors: "
        f"{len(sensor_ids)}"
    )

    # --------------------------------------------------------
    # Find connected subgraph
    # --------------------------------------------------------

    selected_nodes = find_connected_subgraph(
        adjacency_matrix
    )

    # --------------------------------------------------------
    # Display selected graph
    # --------------------------------------------------------

    subgraph = display_graph(
        selected_nodes,
        sensor_ids,
        adjacency_matrix
    )

    print("\n" + "=" * 60)
    print("10-NODE GRAPH CONSTRUCTION COMPLETE")
    print("=" * 60)

    print("\nThe graph is now ready for:")

    print("METR-LA traffic data")
    print("        ↓")
    print("10-node temporal graph")
    print("        ↓")
    print("10 qubits")
    print("        ↓")
    print("Quantum graph encoding")
    print("        ↓")
    print("Bellman value propagation")
    print("        ↓")
    print("AQVP")
    print("        ↓")
    print("Quantum feedback")


if __name__ == "__main__":

    main()