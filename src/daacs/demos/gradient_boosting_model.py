# In this file, we will be testing the gradient boosting model
# It should also be accurate at predicting scores
# To run this, you need to install the package xgboost (pip install xgboost)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt


# Load the Boston Housing dataset for demonstration
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 100, random_state=42)

# Fit the model to the training data
xg_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = xg_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Optional: Plot feature importance
#xgb.plot_importance(xg_reg, feature_names=california_housing.feature_names)
#plt.show()

xgb.plot_importance(xg_reg)
plt.xticks(range(X.shape[1]), california_housing.feature_names, rotation=45, ha='right')  # Set feature names as x-axis ticks
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(xg_reg, ax=ax)
ax.set_yticklabels(california_housing.feature_names)  # Set feature names as y-axis tick labels
plt.show()
