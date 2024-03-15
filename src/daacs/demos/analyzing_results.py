# Test the random forest model 1000 times

# See the result, and compare the results to the actual answer
# Test the model on the data we already have, test and train
# Then try to apply it to new essays without scores
# Create partial dependence plots
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

#       Load your data

#       Create a Random Forest model if not already created
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#rf_model.fit(X, y)

#       Choose the feature for which you want to create a partial dependence plot
feature_index = 5  # Change this to the index of the feature you're interested in

#       Calculate partial dependence
pdp_values, feature_values = partial_dependence(rf_model, X, features=[feature_index])

#       Plot the PDP
plt.figure(figsize=(10, 6))
plt.plot(feature_values[0], pdp_values[0], marker='o', color='r')
plt.title(f'Partial Dependence Plot for Feature {feature_index}')
plt.xlabel(f'Feature {feature_index} Values')
plt.ylabel('Partial Dependence')
plt.grid(True)
plt.show()
#       Repeat for all the indexes you are interested in looking at

# Create confusion matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_iris

#       Load your data
#X = data
#y = target data

#       Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#       Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#       Make predictions on the test set
predictions = rf_model.predict(X_test)

#       Create a confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Print the confusion matrix and accuracy score
print("Confusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")



# ROC/AUC curves
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#       Load the data
#X = data
#y = result of data (test scores)

#       Use only two classes for binary classification (like 0-1, 2-3 scores)
X_binary = X[y != 0]
y_binary = y[y != 0]

#       Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

#       Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#       Get predicted probabilities for the positive class
y_probs = rf_model.predict_proba(X_test)[:, 1]

#       Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

#       Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Model')
plt.legend(loc='lower right')
plt.show()


# Error analysis plots
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris

#       Load the data

#       Use only two classes for binary classification


#       Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

#       Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#       Make predictions on the test set
y_pred = rf_model.predict(X_test)
y_probs = rf_model.predict_proba(X_test)[:, 1]

#       Create an error analysis plot
plt.figure(figsize=(12, 6))

#       Plot 1: Confusion Matrix
plt.subplot(1, 2, 1)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Class 1', 'Class 2'])
plt.yticks([0, 1], ['Class 1', 'Class 2'])
plt.tight_layout()

#       Plot 2: Predicted Probabilities vs. True Labels
plt.subplot(1, 2, 2)
plt.scatter(y_probs, y_test, color='b', alpha=0.5)
plt.title('Predicted Probabilities vs. True Labels')
plt.xlabel('Predicted Probability')
plt.ylabel('True Label')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks([0, 1], ['Class 1', 'Class 2'])
plt.grid(True)
plt.tight_layout()

plt.show()

#       Print classification report for detailed metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Model calibration curve

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_iris

#       Load the dataset


#       Use only two classes for binary classification


#       Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

#       Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#       Obtain predicted probabilities for the positive class
y_probs = rf_model.predict_proba(X_test)[:, 1]

#       Create a calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy='uniform')

#       Plot the calibration curve
plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', linestyle='-', color='blue', label='Random Forest')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve for Random Forest Model')
plt.legend(loc='best')
plt.show()



