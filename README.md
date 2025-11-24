# Support_Vector_Machines_-SVM-
Breast cancer classification using SVM effectively distinguishes benign and malignant tumors based on key features like tumor size and texture. Linear and RBF kernel SVMs achieve high accuracy and precision, supported by hyperparameter tuning, making them reliable tools for early breast cancer detection and diagnosis.
Breast Cancer Classification using SVM
Project Overview
This project implements breast cancer classification using Support Vector Machines (SVM) to distinguish between benign and malignant tumors based on key tumor features. The dataset contains attributes like radius and texture means, and the SVM models use linear and RBF kernels with hyperparameter tuning to achieve high accuracy.

Features
Data preprocessing and feature selection

SVM classification with linear and RBF kernels

Hyperparameter tuning using GridSearchCV

Model evaluation with classification reports and confusion matrices

Decision boundary visualization for 2D feature space

Dataset
The breast cancer dataset contains tumor characteristics including radius_mean, texture_mean, and others. Diagnosis labels are encoded as 0 (benign) and 1 (malignant).

Requirements
Python 3.x

scikit-learn

pandas

numpy

matplotlib

seaborn

Install dependencies with:

text
pip install -r requirements.txt
Usage
Clone the repository

Place the breast-cancer.csv dataset in the project directory

Run the script SVM.py to train, tune, and evaluate the model

Visualize decision boundaries and view model metrics

Results
Achieved over 90% accuracy on test data using both linear and RBF SVM

Effective classification with strong precision and recall for both tumor classes

Future Work
Extend to use all features for improved prediction

Experiment with ensemble methods or deep learning models

Implement automated model deployment and real-time predictions
