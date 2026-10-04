
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# ============================================================
# MECORA V5 - Neural Network Training
# ============================================================

DATASET = "robot_training_data_v5.csv"
MODEL_FILE = "robot_ml_model_v5.joblib"


# ============================================================
# Load dataset
# ============================================================

print("=" * 65)
print("🤖 MECORA V5 - MODEL TRAINING")
print("=" * 65)

df = pd.read_csv(DATASET)

print()
print(f"Loaded {len(df)} samples")


# ============================================================
# Features
# ============================================================

features = [
    "left_sensor",
    "front_sensor",
    "right_sensor",
    "target_distance",
    "target_angle",
    "previous_action"
]

X = df[features]

y = df["action"]


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# Neural network
# ============================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),

            activation="relu",

            solver="adam",

            learning_rate_init=0.001,

            max_iter=700,

            early_stopping=True,

            validation_fraction=0.15,

            n_iter_no_change=30,

            random_state=42
        )
    )
])


# ============================================================
# Train
# ============================================================

print()
print("Training MECORA V5 brain...")
print()

model.fit(
    X_train,
    y_train
)


# ============================================================
# Evaluation
# ============================================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("=" * 65)
print("V5 MODEL RESULTS")
print("=" * 65)

print()
print(
    f"MODEL ACCURACY: {accuracy * 100:.2f}%"
)

print()
print("Classification report:")
print()

print(
    classification_report(
        y_test,
        predictions
    )
)


# ============================================================
# Save model
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print()
print("=" * 65)
print(f"Model saved as: {MODEL_FILE}")
print("=" * 65)

print()
print("🤖 MECORA V5 Brain is ready!")