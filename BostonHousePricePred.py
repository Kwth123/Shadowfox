import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv(r'Beginner Boston House price prediction\HousingData.csv')

# Impute missing values with the median of each feature column
df.fillna(df.median(), inplace=True)

# Split the data into features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest': RandomForestRegressor()
}

# Define hyperparameter grids for tuning
param_grids = {
    'Ridge': {'alpha': [0.1, 1, 10]},
    'Lasso': {'alpha': [0.01, 0.1, 1]},
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    }
}

# Perform model selection with hyperparameter tuning
best_models = {}
for model_name, model in models.items():
    if model_name in param_grids:
        grid = GridSearchCV(model, param_grids[model_name], cv=5, scoring='neg_mean_squared_error')
        grid.fit(X_train_scaled, y_train)
        best_models[model_name] = grid.best_estimator_
        print(f"Best {model_name}: {grid.best_params_}")
    else:
        model.fit(X_train_scaled, y_train)
        best_models[model_name] = model

# Evaluate all models on test data
best_model_name = None
best_mse = float('inf')
best_r2 = float('-inf')

for model_name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model: {model_name}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print("-" * 30)
    
    # Track the best model based on MSE
    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        best_model_name = model_name

print(f"Best Model: {best_model_name} with MSE: {best_mse} and R^2: {best_r2}")

# Feature importance for Random Forest
if best_model_name == 'Random Forest':
    best_rf = best_models['Random Forest']
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Visualization of predicted vs actual prices
best_model = best_models[best_model_name]
y_pred = best_model.predict(X_test_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title(f"Actual vs Predicted Prices ({best_model_name})")
plt.show()

# Cross-validation scores for best model
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"{best_model_name} Cross-Validation MSE: {-cv_scores.mean()} ± {cv_scores.std()}")


