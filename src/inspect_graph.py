import os
import pickle
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_PATH = "data/adj_mx.pkl"


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph():

    if not os.path.exists(GRAPH_PATH):

        raise FileNotFoundError(
            f"\nGraph file not found:\n{GRAPH_PATH}"
        )

    print("=" * 60)
    print("METR-LA SENSOR GRAPH")
    print("=" * 60)

    # --------------------------------------------------------
    # Load old-format pickle file
    # --------------------------------------------------------

    with open(GRAPH_PATH, "rb") as f:

        graph_data = pickle.load(
            f,
            encoding="latin1"
        )

    print("\nGraph file loaded successfully!")

    print("\nPython object type:")
    print(type(graph_data))

    # --------------------------------------------------------
    # Inspect contents
    # --------------------------------------------------------

    print("\nGraph contents:")

    if isinstance(graph_data, tuple):

        print(
            f"Tuple length: {len(graph_data)}"
        )

        for i, item in enumerate(graph_data):

            print(f"\nItem {i}:")
            print(f"Type: {type(item)}")

            if hasattr(item, "shape"):

                print(f"Shape: {item.shape}")

                if isinstance(item, np.ndarray):

                    print("\nFirst few values:")

                    print(item[:5])

            else:

                print(item)

    elif isinstance(graph_data, dict):

        print(
            "Dictionary keys:"
        )

        print(
            list(graph_data.keys())
        )

    elif hasattr(graph_data, "shape"):

        print(
            f"Shape: {graph_data.shape}"
        )

        if isinstance(graph_data, np.ndarray):

            print("\nFirst 5 rows:")

            print(graph_data[:5])

    else:

        print(graph_data)

    return graph_data


# ============================================================
# MAIN
# ============================================================

def main():

    graph_data = load_graph()

    print("\n" + "=" * 60)
    print("GRAPH INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    main()