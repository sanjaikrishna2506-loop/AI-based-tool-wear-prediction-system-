import pandas as pd

df = pd.read_csv("Dataset_Machine_tool_wear.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nData types:")
print(df.dtypes.value_counts())

print("\nTool distribution:")
print(df["tool"].value_counts().sort_index())

print("\nFlank wear statistics:")
print(df["Vb"].describe())

print("\nFirst 10 flank wear values:")
print(df["Vb"].head(10).to_list())

print("\nLast 10 flank wear values:")
print(df["Vb"].tail(10).to_list())