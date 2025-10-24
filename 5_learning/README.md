# Week 5: Learning (Machine Learning)

Welcome to Week 5 of the Introduction to AI course! This week focuses on machine learning, where we'll explore how computers can learn from data and make predictions or decisions without being explicitly programmed.

## Overview

Machine learning is one of the most practical and widely-used branches of AI. This week covers the fundamental concepts of supervised learning, from basic algorithms to advanced ensemble methods, with hands-on implementations and real-world applications.

## Learning Objectives

By the end of this week, you will be able to:

- Understand the difference between supervised, unsupervised, and reinforcement learning
- Implement core supervised learning algorithms from scratch
- Apply linear regression, logistic regression, k-NN, and decision trees
- Evaluate model performance using appropriate metrics
- Use cross-validation and hyperparameter tuning
- Understand and prevent overfitting
- Implement and apply ensemble methods like Random Forests and Gradient Boosting
- Use scikit-learn for real-world machine learning tasks

## Prerequisites

- Python programming fundamentals
- Basic understanding of NumPy and Pandas
- Knowledge of probability and statistics (Week 3 helpful)
- Understanding of optimization (Week 4 helpful)

## Labs

### Lab 1: Introduction to Machine Learning
**File:** `1_lab1.ipynb`

Introduction to machine learning concepts and your first ML models.

**Topics:**
- Types of machine learning (supervised, unsupervised, reinforcement)
- The machine learning workflow
- Linear regression from scratch
- Polynomial regression and feature engineering
- Gradient descent optimization
- Introduction to scikit-learn
- Practical example: Housing price prediction

**Key Concepts:**
- Training vs testing data
- Loss functions and optimization
- Feature scaling and normalization
- Model parameters vs hyperparameters

### Lab 2: Supervised Learning Algorithms
**File:** `2_lab2.ipynb`

Core supervised learning algorithms for classification and regression.

**Topics:**
- Logistic regression for binary classification
- k-Nearest Neighbors (k-NN) algorithm
- Decision trees (CART algorithm)
- Support Vector Machines (SVM) basics
- Naive Bayes classifier
- Algorithm selection and comparison
- Practical example: Medical diagnosis classifier

**Key Concepts:**
- Classification vs regression
- Decision boundaries
- Distance metrics
- Entropy and information gain
- Kernel trick

### Lab 3: Model Evaluation and Validation
**File:** `3_lab3.ipynb`

Techniques for evaluating model performance and ensuring generalization.

**Topics:**
- Training, validation, and test sets
- Cross-validation (k-fold, stratified, leave-one-out)
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Regression metrics (MSE, RMSE, MAE, R²)
- Confusion matrices
- Bias-variance tradeoff
- Overfitting and underfitting
- Regularization (L1, L2, Elastic Net)
- Hyperparameter tuning (Grid Search, Random Search)
- Learning curves

**Key Concepts:**
- Generalization
- Model selection
- Statistical significance
- Trade-offs in model complexity

### Lab 4: Ensemble Methods
**File:** `4_lab4.ipynb`

Advanced techniques that combine multiple models for improved performance.

**Topics:**
- Ensemble learning principles
- Bagging and Bootstrap Aggregating
- Random Forests
- Boosting algorithms
- AdaBoost
- Gradient Boosting
- XGBoost and LightGBM
- Stacking and blending
- Feature importance
- Practical example: Credit risk assessment

**Key Concepts:**
- Wisdom of crowds
- Variance reduction
- Bias reduction through boosting
- Weak vs strong learners

## Interactive Application

**File:** `ml_app.py`

A comprehensive Gradio application for experimenting with machine learning:

1. **Regression Playground**: Visualize different regression models on various datasets
2. **Classification Visualizer**: Compare classification algorithms with interactive decision boundaries
3. **Model Evaluation Dashboard**: Explore metrics, confusion matrices, and learning curves

Run with:
```bash
python ml_app.py
```

## Key Concepts Summary

- **Supervised Learning**: Learning from labeled examples to make predictions
- **Features**: Input variables used to make predictions
- **Labels**: Output variables we're trying to predict
- **Training**: Process of learning patterns from data
- **Inference**: Making predictions on new data
- **Generalization**: Ability to perform well on unseen data
- **Overfitting**: Model performs well on training but poorly on test data
- **Underfitting**: Model is too simple to capture patterns in data
- **Cross-Validation**: Technique for assessing model performance
- **Ensemble**: Combining multiple models to improve predictions

## Real-World Applications

Machine learning powers countless modern applications:

- **Healthcare**: Disease diagnosis, medical image analysis, drug discovery
- **Finance**: Credit scoring, fraud detection, algorithmic trading
- **E-commerce**: Recommendation systems, price optimization, customer segmentation
- **Transportation**: Autonomous vehicles, traffic prediction, route optimization
- **Marketing**: Customer churn prediction, sentiment analysis, ad targeting
- **Manufacturing**: Quality control, predictive maintenance, supply chain optimization
- **Agriculture**: Crop yield prediction, disease detection, precision farming

## Installation

Ensure you have the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn gradio xgboost lightgbm
```

## Tips for Success

1. **Practice with Real Data**: Use datasets from Kaggle, UCI ML Repository, or scikit-learn's built-in datasets
2. **Visualize Everything**: Plot your data, decision boundaries, and learning curves
3. **Start Simple**: Begin with simple models before trying complex ones
4. **Validate Properly**: Always use proper train/test splits and cross-validation
5. **Understand Metrics**: Choose appropriate metrics for your problem
6. **Iterate**: Machine learning is iterative - try different features, models, and hyperparameters
7. **Watch for Overfitting**: Regularize and validate on held-out data
8. **Feature Engineering**: Good features often matter more than complex models

## Algorithm Selection Guide

**Use Linear/Logistic Regression when:**
- You need interpretability
- You have limited training data
- Features have linear relationships
- You need fast training and prediction

**Use k-NN when:**
- You have small to medium datasets
- Decision boundaries are irregular
- You need a simple baseline
- Training time is not critical

**Use Decision Trees when:**
- You need interpretability
- Features are mixed (categorical and numerical)
- Relationships are non-linear
- You need to handle missing values easily

**Use Random Forests when:**
- You want high accuracy
- You can afford longer training time
- You need feature importance
- You want to reduce overfitting

**Use Gradient Boosting when:**
- You need state-of-the-art performance
- You have sufficient training data
- You can tune hyperparameters carefully
- Prediction speed is acceptable

**Use SVMs when:**
- You have high-dimensional data
- You need strong theoretical guarantees
- Your dataset is small to medium
- Classes are well-separated

## Common Pitfalls

1. **Data Leakage**: Information from test set influencing training
2. **Imbalanced Classes**: Not handling class imbalance properly
3. **Feature Scaling**: Forgetting to scale features for distance-based algorithms
4. **Overfitting**: Creating overly complex models
5. **Wrong Metrics**: Using accuracy when classes are imbalanced
6. **Not Using Cross-Validation**: Relying on a single train/test split
7. **Ignoring Domain Knowledge**: Not incorporating expert knowledge into features

## Next Steps

After completing this week:
- Week 6: Neural Networks - Deep learning and neural architectures
- Week 7: Language - Natural language processing and transformers
- Explore advanced topics: Unsupervised learning, reinforcement learning, deep learning
- Practice on Kaggle competitions
- Build your own ML projects

## Resources

- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Kaggle Learn**: https://www.kaggle.com/learn
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/
- **Google ML Crash Course**: https://developers.google.com/machine-learning/crash-course

## Community Contributions

Have you created additional exercises, datasets, or applications? Share them in the `community/` folder!

Good luck, and enjoy exploring the power of machine learning!
