
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load dataset
data = pd.read_csv('breast-cancer.csv')

# Print columns for verification
print("Columns in dataset:", data.columns.tolist())

# Prepare features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})  # Map M to 1 (malignant), B to 0 (benign)

# Select two features for 2D visualization (use correct column names)
feature_cols = ['radius_mean', 'texture_mean']
X_2d = X[feature_cols]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM with linear kernel
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

# Predictions
y_pred_linear = svm_linear.predict(X_test_scaled)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

# Cross-validation scores
cv_linear = cross_val_score(svm_linear, X_train_scaled, y_train, cv=5)
cv_rbf = cross_val_score(svm_rbf, X_train_scaled, y_train, cv=5)

# Hyperparameter tuning for RBF kernel
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_

# Print results
print("Linear SVM CV scores:", cv_linear)
print("RBF SVM CV scores:", cv_rbf)
print("Best parameters for RBF SVM:", grid_search.best_params_)
print("\nClassification Report (Linear SVM):\n", classification_report(y_test, y_pred_linear))
print("\nClassification Report (RBF SVM):\n", classification_report(y_test, y_pred_rbf))
print("\nConfusion Matrix (Linear SVM):\n", confusion_matrix(y_test, y_pred_linear))
print("\nConfusion Matrix (RBF SVM):\n", confusion_matrix(y_test, y_pred_rbf))

# Visualize decision boundary for SVM models
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.title(title)
    plt.colorbar(scatter)
    plt.show()

plot_decision_boundary(svm_linear, X_train_scaled, y_train.values, "Linear SVM Decision Boundary")
plot_decision_boundary(svm_rbf, X_train_scaled, y_train.values, "RBF SVM Decision Boundary")
