# Support Vector Machine model
# This is a test with a different data set
# File takes so long to run

# Training time: 3388.1081483364105 seconds = 56 minutes
# Mean Squared Error (SVM): 1.1799737794338485

import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import numpy as np

#       Load the California Housing dataset for regression
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

#       Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#       Create a Support Vector Machine regressor
svm_model = SVR(kernel='linear')  # You can experiment with different kernels (linear, rbf, etc.)

start_time = time.time()


#       Fit the model to the training data
svm_model.fit(X_train, y_train)


# Measure the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time} seconds")

#       Make predictions on the test set
y_pred_svm = svm_model.predict(X_test)

#       Evaluate the model using Mean Squared Error
mse_svm = mean_squared_error(y_test, y_pred_svm)
print(f'Mean Squared Error (SVM): {mse_svm}')

#       Compare with the Random Forest model (assuming you already have rf_model)
#y_pred_rf = rf_model.predict(X_test)
#mse_rf = mean_squared_error(y_test, y_pred_rf)
#print(f'Mean Squared Error (Random Forest): {mse_rf}')
