import pandas as pd

# Load dataset
df = pd.read_csv("Dataset_Machine_tool_wear_mm.csv")

# Create sequence time: 1 row = 1 second
df["Time_sec"] = df.groupby("tool").cumcount()

WEAR_LIMIT = 0.250  # mm

# Store valid tool failure times
failure_times = {}

for tool, group in df.groupby("tool"):

    failure = group[group["Vb (mm)"] >= WEAR_LIMIT]

    if len(failure) > 0:
        failure_times[tool] = failure["Time_sec"].iloc[0]

# Keep only tools that actually reached the wear limit
df_rul = df[df["tool"].isin(failure_times.keys())].copy()

# Calculate RUL
df_rul["Failure_Time_sec"] = df_rul["tool"].map(failure_times)

df_rul["RUL_sec"] = (
    df_rul["Failure_Time_sec"] - df_rul["Time_sec"]
)

# Keep only observations before or at failure
df_rul = df_rul[df_rul["RUL_sec"] >= 0]

# Remove unnecessary columns
df_rul = df_rul.drop(
    columns=["Failure_Time_sec"]
)

# Save
df_rul.to_csv(
    "Tool_Wear_RUL_Ready.csv",
    index=False
)

print("RUL DATASET CREATED")
print("--------------------")

print("Rows:", len(df_rul))
print("Columns:", len(df_rul.columns))

print("\nTools included:")
print(sorted(df_rul["tool"].unique()))

print("\nRUL range:")
print(
    df_rul["RUL_sec"].min(),
    "to",
    df_rul["RUL_sec"].max(),
    "seconds"
)

print("\nMissing values:")
print(df_rul.isnull().sum().sum())

print("\nSaved as:")
print("Tool_Wear_RUL_Ready.csv")