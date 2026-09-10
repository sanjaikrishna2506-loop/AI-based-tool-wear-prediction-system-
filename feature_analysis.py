import pandas as pd

df = pd.read_csv("Dataset_Machine_tool_wear_mm.csv")

# Remove columns that are not sensor features
exclude = [
    "Vb (mm)",
    "tool"
]

numeric = df.select_dtypes(include="number")

correlation = numeric.corr()["Vb (mm)"].drop(exclude)

correlation = correlation.abs().sort_values(ascending=False)

print("\nTOP SENSOR FEATURES RELATED TO FLANK WEAR")
print("------------------------------------------")

print(correlation.head(20))