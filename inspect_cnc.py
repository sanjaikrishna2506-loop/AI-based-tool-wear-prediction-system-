import pandas as pd

df = pd.read_csv("Dataset_Machine_tool_wear_mm.csv")

columns = [
    "CON.G.FREAL",
    "CON.A.SREAL.S",
    "CON.SV2.S",
    "CON.SV2.X",
    "CON.SV2.Z"
]

print("\nCNC SIGNALS\n")

for col in columns:
    print(f"\n{col}")
    print("Min :", df[col].min())
    print("Max :", df[col].max())
    print("Mean:", df[col].mean())
    print("First 10:")
    print(df[col].head(10).to_list())