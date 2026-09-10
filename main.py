import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load dataset
df = pd.read_csv("CNC_Lathe_Flank_Wear_Cleaned.csv")

# 2. Define inputs and target
X = df[[
    "Feed rate (mm/min)",
    "Depth of cut (mm)",
    "Speed (RPM)"
]]

y = df["Flank wear (mm)"]

# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 4. Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# 5. Train the model
model.fit(X_train, y_train)

print("Random Forest model trained successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. Predict flank wear for the test data
y_pred = model.predict(X_test)

print("\nFirst 10 predictions:")
for actual, predicted in zip(y_test.head(10), y_pred[:10]):
    print(f"Actual: {actual:.6f} mm | Predicted: {predicted:.6f} mm")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 7. Calculate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE  : {mae:.6f} mm")
print(f"RMSE : {rmse:.6f} mm")
print(f"R²   : {r2:.6f}")

import joblib

joblib.dump(model, "random_forest_tool_wear_model.pkl")

print("Model saved successfully!")

feed = float(input("Enter feed rate: "))
doc = float(input("Enter depth of cut: "))
speed = float(input("Enter speed (RPM): "))

new_data = pd.DataFrame(
    [[feed, doc, speed]],
    columns=[
        "Feed rate (mm/min)",
        "Depth of cut (mm)",
        "Speed (RPM)"
    ]
)

prediction = model.predict(new_data)

print(f"Predicted flank wear: {prediction[0]:.6f} mm")