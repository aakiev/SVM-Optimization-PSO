import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from pyswarm import pso  # PSO library

def load_and_prepare_data(file_path):
    """
    Loads the dataset, standardizes the features, converts the target variable into numerical values,
    and returns the standardized features and labels.
    """
    df = pd.read_csv(file_path)

    # Convert target variable (low=1, mid=2, high=3)
    label_mapping = {"low risk": 1, "mid risk": 2, "high risk": 3}
    df["RiskLevel"] = df["RiskLevel"].map(label_mapping)

    # Separate features and target variable
    X = df.drop(columns=["RiskLevel"])
    y = df["RiskLevel"]

    # Standardize features
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    return pd.DataFrame(X_standardized, columns=X.columns), y

def svm_cross_val_score(params, X_train, y_train):
    """
    Objective function for PSO. Returns the negative cross-validation accuracy for given parameters.
    """
    C, gamma = params
    model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return -np.mean(scores)  # Minimize negative accuracy

def optimize_hyperparameters(X_train, y_train, swarmsize=10, maxiter=30):
    """
    Optimizes SVM hyperparameters (C and gamma) using PSO and measures performance.
    """
    # Bounds for C and gamma
    lb = [0.01, 0.001]  # Lower bounds for C and gamma
    ub = [32000.0, 128.0]  # Upper bounds for C and gamma

    # Measure time
    start_time = time.time()
    best_params, convergence = pso(svm_cross_val_score, lb, ub, args=(X_train, y_train), swarmsize=swarmsize, maxiter=maxiter)
    end_time = time.time()
    optimization_time = end_time - start_time
    print(f"✅ PSO Optimization Time (swarmsize={swarmsize}, maxiter={maxiter}): {optimization_time:.2f} seconds")
    return best_params, convergence, optimization_time

def evaluate_model(svm_model, X_test, y_test):
    """
    Evaluates the SVM model on test data and returns accuracy.
    """
    y_pred = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    return test_accuracy

if __name__ == "__main__":
    # 🔹 1️⃣ Adjust dataset path (Load your CSV file here!)
    file_path = "MaternalHealthRisk.csv" 

    # 🔹 2️⃣ Load data
    X, y = load_and_prepare_data(file_path)

    # 🔹 3️⃣ Split data into training & test sets (70% training, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)



    # 🔹 4️⃣ Optimize hyperparameters using PSO with performance measurement
    best_params, convergence, optimization_time = optimize_hyperparameters(X_train, y_train, swarmsize=5, maxiter=10)
    best_C, best_gamma = best_params
    print(f"✅ Optimized Parameters: C={best_C:.4f}, gamma={best_gamma:.4f}")

    # 🔹 5️⃣ Train SVM with optimized parameters
    svm_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    svm_model.fit(X_train, y_train)

    # 🔹 6️⃣ Evaluate the optimized SVM model
    test_accuracy = evaluate_model(svm_model, X_test, y_test)
    print(f"✅ Test accuracy with optimized parameters: {test_accuracy * 100:.2f}%")




    best_params, convergence, optimization_time = optimize_hyperparameters(X_train, y_train, swarmsize=10, maxiter=10)  #Best performance in shortest time!
    best_C, best_gamma = best_params
    print(f"✅ Optimized Parameters: C={best_C:.4f}, gamma={best_gamma:.4f}")

    # 🔹 5️⃣ Train SVM with optimized parameters
    svm_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    svm_model.fit(X_train, y_train)

    # 🔹 6️⃣ Evaluate the optimized SVM model
    test_accuracy = evaluate_model(svm_model, X_test, y_test)
    print(f"✅ Test accuracy with optimized parameters: {test_accuracy * 100:.2f}%")




    best_params, convergence, optimization_time = optimize_hyperparameters(X_train, y_train, swarmsize=5, maxiter=20)
    best_C, best_gamma = best_params
    print(f"✅ Optimized Parameters: C={best_C:.4f}, gamma={best_gamma:.4f}")

    # 🔹 5️⃣ Train SVM with optimized parameters
    svm_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    svm_model.fit(X_train, y_train)

    # 🔹 6️⃣ Evaluate the optimized SVM model
    test_accuracy = evaluate_model(svm_model, X_test, y_test)
    print(f"✅ Test accuracy with optimized parameters: {test_accuracy * 100:.2f}%")




    best_params, convergence, optimization_time = optimize_hyperparameters(X_train, y_train, swarmsize=20, maxiter=20)
    best_C, best_gamma = best_params
    print(f"✅ Optimized Parameters: C={best_C:.4f}, gamma={best_gamma:.4f}")

    # 🔹 5️⃣ Train SVM with optimized parameters
    svm_model = SVC(kernel="rbf", C=best_C, gamma=best_gamma, random_state=42)
    svm_model.fit(X_train, y_train)

    # 🔹 6️⃣ Evaluate the optimized SVM model
    test_accuracy = evaluate_model(svm_model, X_test, y_test)
    print(f"✅ Test accuracy with optimized parameters: {test_accuracy * 100:.2f}%")
