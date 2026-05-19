import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Load processed dataset
df = pd.read_csv("data/processed/processed_telco_data.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow tracking
mlflow.start_run()

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")

# Log metrics in MLflow
mlflow.log_metric("accuracy", accuracy)

# Save model
joblib.dump(model, "models/churn_model.pkl")

# Log model in MLflow
mlflow.sklearn.log_model(model, "random_forest_model")

# End MLflow run
mlflow.end_run()

print("Training completed successfully!")


