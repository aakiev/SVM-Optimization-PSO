# SVM Hyperparameter Optimization with PSO

## Overview  
This repository contains an implementation of **Support Vector Machine (SVM) hyperparameter optimization** using **Particle Swarm Optimization (PSO)**. The goal is to optimize the `C` and `gamma` parameters of an **RBF-kernel SVM** to improve classification accuracy on the **Maternal Health Risk dataset**. The project integrates **data preprocessing, cross-validation, and performance benchmarking** to evaluate the impact of different PSO configurations.  

## Key Objectives  
- Implement a **fully functional SVM model** with hyperparameter tuning.  
- Utilize **Particle Swarm Optimization (PSO)** to optimize `C` and `gamma`.  
- Measure **performance impact** based on different PSO configurations.  
- Ensure scalability for future enhancements, such as additional datasets or optimization techniques.  

## Core Features  

### 1. Data Preprocessing & Feature Scaling  
- The dataset is **standardized using `StandardScaler`** to improve SVM performance.  
- The **target labels are encoded numerically** (`low risk = 1`, `mid risk = 2`, `high risk = 3`).  

### 2. SVM with RBF Kernel  
- The model is trained using a **Radial Basis Function (RBF) kernel**.  
- **5-fold cross-validation** ensures robust accuracy estimation.  

### 3. Hyperparameter Optimization with PSO  
- `C` and `gamma` are optimized within a defined range (`C: 0.01 – 32,000`, `gamma: 0.001 – 128`).  
- The **best parameters are selected based on cross-validation accuracy**.  
- **Execution time for different PSO configurations** is measured to analyze optimization performance.  

### 4. Performance Benchmarking & Visualization  
- **Training & test accuracy are evaluated** after PSO optimization.  
- The **PSO convergence plot** visualizes the optimization process.  

## Technologies Used  
- **Programming Language**: Python  
- **Libraries**: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `pyswarm`  
- **Development Environment**: Jupyter Notebook / VS Code  

## Getting Started  

### Prerequisites  
- Install required dependencies using:  
  ```bash
  pip install numpy pandas matplotlib scikit-learn pyswarm
  ```

### Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/aakiev/SVM-Optimization-PSO.git
   ```
2. Run the Python script:  
   ```bash
   python SVM_Classification_Optimized.py
   ```

## Usage  

### Training & Optimization  
- Load the **Maternal Health Risk dataset** and split it into training & test sets.  
- Run **PSO optimization** to determine the best `C` and `gamma`.  
- Train the **SVM model** using the optimized hyperparameters.  

### Evaluation  
- The script prints **cross-validation accuracy** and **test accuracy**.  
- The **execution time for PSO optimization** is displayed.   

## Performance Comparison  
- The script can be modified to compare **different PSO configurations** (`swarmsize` and `maxiter`).  
- Performance can be evaluated in terms of **accuracy vs. execution time**.  

## Future Enhancements  
- Adding **Grid Search & Genetic Algorithm (GA) support** for comparison.  
- Extending optimization to **multi-class classification problems**.  
- Improving visualization with **3D decision boundaries**.  
- Testing with **additional datasets** for broader applicability.  

---

