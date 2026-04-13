import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression


# Create folder for saved outputs
os.makedirs("outputs", exist_ok=True)

# STEP 1: Load dataset
df = pd.read_csv("data/used_car_price_dataset_extended.csv")

# STEP 2: Quick look at data
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# STEP 3: Cleaning
df = df.drop_duplicates()
df = df.dropna()

# STEP 4: Feature engineering
current_year = 2026
df["car_age"] = current_year - df["make_year"]

# STEP 5: Select features
selected_features = [
    "mileage_kmpl",
    "engine_cc",
    "fuel_type",
    "owner_count",
    "car_age"
]

df = df[selected_features + ["price_usd"]]

# Keep a copy before encoding for scatter plot use
df_plot = df.copy()

# STEP 6: Encoding categorical data
df = pd.get_dummies(df, columns=["fuel_type"], drop_first=True)

# STEP 7: Define X and y
X = df.drop("price_usd", axis=1)
y = df["price_usd"]

# STEP 8: Feature selection
selector = SelectKBest(score_func=f_regression, k="all")
selector.fit(X, y)

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.scores_
}).sort_values(by="Score", ascending=False)

print("\nFeature Selection Scores:")
print(feature_scores)

# STEP 9: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)

# MODEL 2: Decision Tree
dt_params = {
    "max_depth": [3, 5, 10, None]
}

dt_grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    dt_params,
    cv=5,
    scoring="neg_mean_absolute_error"
)
dt_grid.fit(X_train, y_train)
dt_model = dt_grid.best_estimator_

dt_predictions = dt_model.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_predictions)

# MODEL 3: Random Forest
rf_params = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=5,
    scoring="neg_mean_absolute_error"
)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_

rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)

# SAMPLE PREDICTIONS
results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": rf_predictions
})

print("\nSample Predictions (Random Forest):")
print(results.head())

# FINAL METRICS
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)

dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
dt_r2 = r2_score(y_test, dt_predictions)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\nModel Performance Comparison:")

print("\nLinear Regression:")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R2:", lr_r2)

print("\nDecision Tree:")
print("MAE:", dt_mae)
print("RMSE:", dt_rmse)
print("R2:", dt_r2)

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)

# =========================
# BETTER VISUALIZATIONS FOR SLIDES
# ALL 5 GRAPHS IN ONE WINDOW
# =========================

models = ["Linear Regression", "Decision Tree", "Random Forest"]
mae_values = [lr_mae, dt_mae, rf_mae]

# Train vs Test MAE for overfitting check
lr_train_pred = lr_model.predict(X_train)
dt_train_pred = dt_model.predict(X_train)
rf_train_pred = rf_model.predict(X_train)

lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
dt_train_mae = mean_absolute_error(y_train, dt_train_pred)
rf_train_mae = mean_absolute_error(y_train, rf_train_pred)

train_mae = [lr_train_mae, dt_train_mae, rf_train_mae]
test_mae = [lr_mae, dt_mae, rf_mae]

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": np.abs(lr_model.coef_)
}).sort_values(by="Importance", ascending=True)

# Create one figure with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Used Car Price Prediction - Model Visualizations", fontsize=20)

# 1) Model Comparison by MAE
bars1 = axs[0, 0].bar(models, mae_values)
axs[0, 0].set_title("Model Comparison by MAE", fontsize=14)
axs[0, 0].set_xlabel("Model", fontsize=12)
axs[0, 0].set_ylabel("Mean Absolute Error (USD)", fontsize=12)
axs[0, 0].tick_params(axis="x", labelrotation=10)

for bar in bars1:
    height = bar.get_height()
    axs[0, 0].text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# 2) Feature Importance
bars2 = axs[0, 1].barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)
axs[0, 1].set_title("Feature Importance", fontsize=14)
axs[0, 1].set_xlabel("Coefficient Magnitude", fontsize=12)
axs[0, 1].set_ylabel("Feature", fontsize=12)

for bar in bars2:
    width = bar.get_width()
    axs[0, 1].text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}",
        va="center",
        fontsize=9
    )

# 3) Mileage vs Price, Colored by Car Age
scatter = axs[0, 2].scatter(
    df_plot["mileage_kmpl"],
    df_plot["price_usd"],
    c=df_plot["car_age"],
    alpha=0.35
)
axs[0, 2].set_title("Mileage vs Price", fontsize=14)
axs[0, 2].set_xlabel("Mileage (km per liter)", fontsize=12)
axs[0, 2].set_ylabel("Price (USD)", fontsize=12)

cbar = fig.colorbar(scatter, ax=axs[0, 2])
cbar.set_label("Car Age", fontsize=10)

# 4) Actual vs Predicted Prices
axs[1, 0].scatter(y_test, rf_predictions, alpha=0.5)
min_val = min(y_test.min(), rf_predictions.min())
max_val = max(y_test.max(), rf_predictions.max())
axs[1, 0].plot([min_val, max_val], [min_val, max_val])
axs[1, 0].set_title("Actual vs Predicted Prices", fontsize=14)
axs[1, 0].set_xlabel("Actual Price (USD)", fontsize=12)
axs[1, 0].set_ylabel("Predicted Price (USD)", fontsize=12)

# 5) Train vs Test MAE for Overfitting Check
x = np.arange(len(models))
width = 0.35

axs[1, 1].bar(x - width / 2, train_mae, width, label="Train MAE")
axs[1, 1].bar(x + width / 2, test_mae, width, label="Test MAE")
axs[1, 1].set_title("Train vs Test Error", fontsize=14)
axs[1, 1].set_xlabel("Model", fontsize=12)
axs[1, 1].set_ylabel("Mean Absolute Error (USD)", fontsize=12)
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(models)
axs[1, 1].legend()

# Empty last subplot since there are only 5 graphs
axs[1, 2].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/all_5_graphs.png", dpi=300)
plt.show()

print("\nCombined graph saved as:")
print("outputs/all_5_graphs.png")