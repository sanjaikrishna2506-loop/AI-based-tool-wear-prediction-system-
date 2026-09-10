import joblib

model = joblib.load("sensor_tool_condition_model.pkl")

print("\nMODEL INPUT INFORMATION")
print("========================")

print("Number of inputs:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("\nExact features expected by model:")
    for i, feature in enumerate(model.feature_names_in_, 1):
        print(i, ":", feature)
else:
    print("\nThis model does not contain feature names.")