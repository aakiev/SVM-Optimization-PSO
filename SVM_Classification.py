import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(file_path):
    """
    Loads the dataset, converts the target variable into numerical values,
    and returns the features and labels.
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

def train_svm(X_train, y_train):
    """
    Creates and trains an SVM model with an RBF kernel.
    """
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)  # Standard values for later optimization
    svm_model.fit(X_train, y_train)
    
    # Cross-validation (5-fold CV)
    cross_val_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
    cross_val_accuracy = np.mean(cross_val_scores)  # Average cross-validation accuracy

    return svm_model, cross_val_accuracy

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

    # 🔹 4️⃣ Train SVM model
    svm_model, cross_val_accuracy = train_svm(X_train, y_train)

    # 🔹 5️⃣ Evaluate model
    test_accuracy = evaluate_model(svm_model, X_test, y_test)

    # 📊 Print results
    print(f"✅ 5-fold cross-validation accuracy: {cross_val_accuracy * 100:.2f}%")
    print(f"✅ Test accuracy: {test_accuracy * 100:.2f}%")
