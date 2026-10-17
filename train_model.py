"""
Antenna ML Model Trainer
--------------------------
Trains and compares multiple multi-output regression models to predict:
    - patch_width_mm
    - patch_length_mm
    - gain_dBi
    - bandwidth_percent

from inputs:
    - frequency_GHz
    - dielectric_constant
    - substrate_height_mm

Models compared: Linear Regression, Random Forest, Gradient Boosting (via MultiOutputRegressor),
and a small Neural Network (MLPRegressor).

Best model (by average R2 across outputs) is saved to disk for use in the FastAPI backend.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DATA_PATH = "antenna_synthetic_dataset.csv"
MODEL_OUTPUT_PATH = "best_antenna_model.pkl"
SCALER_OUTPUT_PATH = "input_scaler.pkl"
RANDOM_SEED = 42

INPUT_COLS = ["frequency_GHz", "dielectric_constant", "substrate_height_mm"]
OUTPUT_COLS = ["patch_width_mm", "patch_length_mm", "gain_dBi", "bandwidth_percent", "S11_min_dB"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df[INPUT_COLS].values
    y = df[OUTPUT_COLS].values
    return X, y, df


def get_models():
    """Return a dict of candidate models, all wrapped for multi-output regression."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=RANDOM_SEED)
        ),
        "Neural Network (MLP)": MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),
            activation="relu",
            max_iter=2000,
            random_state=RANDOM_SEED,
            early_stopping=True,
        ),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Per-output metrics
    results = {}
    for i, col in enumerate(OUTPUT_COLS):
        rmse = np.sqrt(mean_squared_error(y_test[:, i], preds[:, i]))
        mae = mean_absolute_error(y_test[:, i], preds[:, i])
        r2 = r2_score(y_test[:, i], preds[:, i])
        results[col] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    avg_r2 = np.mean([results[c]["R2"] for c in OUTPUT_COLS])
    return model, results, avg_r2


def print_results_table(all_results):
    print("\n" + "=" * 90)
    print(f"{'Model':<22} | " + " | ".join([f"{c[:12]:>14}" for c in OUTPUT_COLS]) + " | Avg R2")
    print("-" * 90)
    for name, (results, avg_r2) in all_results.items():
        row = f"{name:<22} | "
        row += " | ".join([f"{results[c]['R2']:>14.4f}" for c in OUTPUT_COLS])
        row += f" | {avg_r2:.4f}"
        print(row)
    print("=" * 90)


def main():
    print("Loading dataset...")
    X, y, df = load_data()
    print(f"Loaded {len(df)} samples with inputs {INPUT_COLS} -> outputs {OUTPUT_COLS}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # Scale inputs (helps Linear Regression + Neural Network especially)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    trained_models = {}
    all_results = {}

    print("\nTraining and evaluating models...")
    for name, model in models.items():
        print(f"  -> {name}...")
        trained_model, results, avg_r2 = evaluate_model(
            name, model, X_train_scaled, X_test_scaled, y_train, y_test
        )
        trained_models[name] = trained_model
        all_results[name] = (results, avg_r2)

    print_results_table(all_results)

    # Pick best model by average R2
    best_name = max(all_results, key=lambda k: all_results[k][1])
    best_model = trained_models[best_name]
    best_avg_r2 = all_results[best_name][1]

    print(f"\nBest model: {best_name} (Avg R2 = {best_avg_r2:.4f})")

    # Save best model + scaler for use in the web backend
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    joblib.dump({"input_cols": INPUT_COLS, "output_cols": OUTPUT_COLS, "model_name": best_name},
                "model_metadata.pkl")

    print(f"Saved best model to '{MODEL_OUTPUT_PATH}'")
    print(f"Saved input scaler to '{SCALER_OUTPUT_PATH}'")
    print(f"Saved metadata to 'model_metadata.pkl'")


if __name__ == "__main__":
    main()
