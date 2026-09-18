import os
import numpy as np
import pandas as pd
import h5py


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/metr-la.h5"

# Number of sensors we will use for the quantum experiment
NUM_NODES = 10

# Start with a small number of temporal samples
# We will increase this later
NUM_TIME_STEPS = 100


# ============================================================
# LOAD METR-LA DIRECTLY USING H5PY
# ============================================================

def load_metr_la():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            "\nMETR-LA dataset not found.\n"
            f"Expected file:\n{DATA_PATH}\n\n"
            "Make sure metr-la.h5 is inside the data folder."
        )

    print("=" * 60)
    print("METR-LA DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Open HDF5 file
    # --------------------------------------------------------

    with h5py.File(DATA_PATH, "r") as f:

        # The dataset structure we found is:
        #
        # df/axis0
        # df/axis1
        # df/block0_values

        sensor_ids = f["df/axis0"][:]

        timestamps = f["df/axis1"][:]

        values = f["df/block0_values"][:]

    # --------------------------------------------------------
    # Decode sensor IDs
    # --------------------------------------------------------

    sensor_ids = [
        sensor.decode("utf-8") if isinstance(sensor, bytes)
        else str(sensor)
        for sensor in sensor_ids
    ]

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    timestamps = pd.to_datetime(
        timestamps,
        unit="ns"
    )

    # --------------------------------------------------------
    # Create Pandas DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        values,
        index=timestamps,
        columns=sensor_ids
    )

    # --------------------------------------------------------
    # Display dataset information
    # --------------------------------------------------------

    print("\nDataset loaded successfully!")

    print("\nDataset shape:")
    print(df.shape)

    print("\nNumber of time steps:")
    print(df.shape[0])

    print("\nNumber of sensors:")
    print(df.shape[1])

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSensor IDs:")
    print(list(df.columns))

    print("\nTime range:")
    print(df.index[0])
    print("to")
    print(df.index[-1])

    print("\nMissing values:")
    print(df.isna().sum().sum())

    return df


# ============================================================
# SELECT 10 SENSORS
# ============================================================

def select_nodes(df):

    if df.shape[1] < NUM_NODES:

        raise ValueError(
            f"Dataset contains only {df.shape[1]} sensors, "
            f"but {NUM_NODES} nodes were requested."
        )

    # --------------------------------------------------------
    # TEMPORARY selection
    #
    # We use the first 10 sensors only for preprocessing.
    #
    # Later we will use adj_mx.pkl to select a meaningful
    # connected 10-node subgraph.
    # --------------------------------------------------------

    selected_columns = list(df.columns[:NUM_NODES])

    selected_df = df[selected_columns].copy()

    print("\n" + "=" * 60)
    print("10-NODE SELECTION")
    print("=" * 60)

    print("\nSelected sensors:")

    for i, sensor in enumerate(selected_columns):

        print(
            f"Node {i} -> Sensor {sensor}"
        )

    print("\nSelected data shape:")

    print(selected_df.shape)

    return selected_df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    print("\n" + "=" * 60)
    print("MISSING VALUE HANDLING")
    print("=" * 60)

    missing_before = df.isna().sum().sum()

    print(
        f"\nMissing values before processing: "
        f"{missing_before}"
    )

    # Forward fill using previous traffic observation
    df = df.ffill()

    # If missing values remain at the beginning,
    # use backward fill
    df = df.bfill()

    missing_after = df.isna().sum().sum()

    print(
        f"Missing values after processing: "
        f"{missing_after}"
    )

    return df


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(df):

    print("\n" + "=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    # Calculate mean and standard deviation
    mean = df.mean()
    std = df.std()

    # Z-score normalization
    normalized = (df - mean) / (std + 1e-8)

    print("\nMean after normalization:")

    print(
        normalized.mean()
        .round(4)
        .values
    )

    print("\nStandard deviation after normalization:")

    print(
        normalized.std()
        .round(4)
        .values
    )

    return normalized


# ============================================================
# CREATE TEMPORAL SNAPSHOTS
# ============================================================

def create_snapshots(data):

    # --------------------------------------------------------
    # Use only first NUM_TIME_STEPS initially
    # --------------------------------------------------------

    data = data.iloc[:NUM_TIME_STEPS]

    # Convert DataFrame to NumPy array
    snapshots = data.to_numpy(dtype=float)

    print("\n" + "=" * 60)
    print("TEMPORAL SNAPSHOTS")
    print("=" * 60)

    print(
        f"\nNumber of time steps: "
        f"{snapshots.shape[0]}"
    )

    print(
        f"Number of nodes:     "
        f"{snapshots.shape[1]}"
    )

    # --------------------------------------------------------
    # Display first snapshot
    # --------------------------------------------------------

    print("\nFirst temporal snapshot:")

    for node in range(NUM_NODES):

        print(
            f"Node {node}: "
            f"{snapshots[0, node]:.4f}"
        )

    # --------------------------------------------------------
    # Display second snapshot
    # --------------------------------------------------------

    print("\nSecond temporal snapshot:")

    for node in range(NUM_NODES):

        print(
            f"Node {node}: "
            f"{snapshots[1, node]:.4f}"
        )

    return snapshots


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # STEP 1
    # Load METR-LA
    # --------------------------------------------------------

    df = load_metr_la()

    # --------------------------------------------------------
    # STEP 2
    # Select 10 sensors
    # --------------------------------------------------------

    selected_df = select_nodes(df)

    # --------------------------------------------------------
    # STEP 3
    # Handle missing values
    # --------------------------------------------------------

    cleaned_df = handle_missing_values(selected_df)

    # --------------------------------------------------------
    # STEP 4
    # Normalize traffic data
    # --------------------------------------------------------

    normalized_df = normalize_data(cleaned_df)

    # --------------------------------------------------------
    # STEP 5
    # Create temporal snapshots
    # --------------------------------------------------------

    snapshots = create_snapshots(normalized_df)

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("METR-LA PREPROCESSING COMPLETE")
    print("=" * 60)

    print("\nCurrent pipeline:")

    print("METR-LA")
    print("   ↓")
    print("207 traffic sensors")
    print("   ↓")
    print("10 selected sensors")
    print("   ↓")
    print("Missing-value handling")
    print("   ↓")
    print("Normalization")
    print("   ↓")
    print("Temporal snapshots")
    print("   ↓")
    print("Ready for graph construction")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()