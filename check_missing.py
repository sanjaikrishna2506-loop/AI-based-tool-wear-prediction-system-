import pandas as pd

df = pd.read_csv("Tool_Wear_RUL_Ready.csv")

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

print("\nCOLUMNS WITH MISSING VALUES")
print("--------------------------")
print(missing)

print("\nTotal missing values:", missing.sum())