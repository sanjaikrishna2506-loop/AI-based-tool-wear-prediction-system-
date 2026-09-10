import pandas as pd

# Load dataset
df = pd.read_csv("Dataset_Machine_tool_wear.csv")

# Sort each tool in its original sequence
df["Time_sec"] = df.groupby("tool").cumcount()

# Tool-life criterion from the published dataset methodology
WEAR_LIMIT = 250.0  # micrometers

# Keep only data up to the tool-life limit
df_rul = df[df["Vb"] <= WEAR_LIMIT].copy()

# Find the final time before reaching the wear limit for each tool
end_time = df_rul.groupby("tool")["Time_sec"].max()

# Calculate remaining useful life in seconds
df_rul["RUL_sec"] = df_rul.apply(
    lambda row: end_time[row["tool"]] - row["Time_sec"],
    axis=1
)

# Remove columns with excessive missing values
df_rul = df_rul.drop(columns=["CV3.X_SK", "CV3.X_KURT"])

# Save processed dataset
df_rul.to_csv("Tool_Wear_RUL_Preprocessed.csv", index=False)

print("Original dataset:", df.shape)
print("RUL dataset:", df_rul.shape)

print("\nWear range:")
print(df_rul["Vb"].min(), "to", df_rul["Vb"].max(), "µm")

print("\nTools:")
print(df_rul["tool"].value_counts().sort_index())

print("\nRUL range:")
print(df_rul["RUL_sec"].min(), "to", df_rul["RUL_sec"].max(), "seconds")

print("\nSaved: Tool_Wear_RUL_Preprocessed.csv")