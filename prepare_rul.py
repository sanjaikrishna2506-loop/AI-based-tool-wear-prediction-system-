import pandas as pd

df = pd.read_csv("Dataset_Machine_tool_wear_mm.csv")

# Each row represents 1 second of machining
df["Time_sec"] = df.groupby("tool").cumcount()

WEAR_LIMIT = 0.250  # mm

print("\nEND-OF-LIFE TIME FOR EACH TOOL\n")

for tool, group in df.groupby("tool"):

    # First point where flank wear reaches 0.25 mm
    failure = group[group["Vb (mm)"] >= WEAR_LIMIT]

    if len(failure) > 0:
        failure_time = failure["Time_sec"].iloc[0]
        failure_wear = failure["Vb (mm)"].iloc[0]

        print(
            f"Tool {tool}: "
            f"Failure time = {failure_time} sec, "
            f"Wear = {failure_wear:.4f} mm"
        )
    else:
        print(
            f"Tool {tool}: "
            f"Did NOT reach 0.25 mm"
        )