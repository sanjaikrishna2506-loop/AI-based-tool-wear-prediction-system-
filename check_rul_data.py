import pandas as pd

df = pd.read_csv("Dataset_Machine_tool_wear_mm.csv")

# Each row represents 1 second
df["Time_sec"] = df.groupby("tool").cumcount()

# End-of-life criterion
WEAR_LIMIT = 0.250  # mm

print("Tool-wise wear progression:\n")

for tool, group in df.groupby("tool"):
    
    max_wear = group["Vb (mm)"].max()
    total_time = group["Time_sec"].max()
    
    reached_limit = max_wear >= WEAR_LIMIT
    
    print(
        f"Tool {tool}: "
        f"Records={len(group)}, "
        f"Max wear={max_wear:.4f} mm, "
        f"Duration={total_time} sec, "
        f"Reached 0.25 mm={reached_limit}"
    )

print("\nFeed range:")
print(df["CON.G.FREAL"].min(), "to", df["CON.G.FREAL"].max())

print("\nProgrammed spindle speed range:")
print(df["CON.A.SREAL.S"].min(), "to", df["CON.A.SREAL.S"].max())

print("\nWear limit:")
print("0.250 mm")