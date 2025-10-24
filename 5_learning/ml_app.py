"""
Machine Learning Interactive Application

This Gradio app provides interactive demonstrations of:
1. Regression Playground - Visualize different regression models
2. Classification Visualizer - Compare classification algorithms
3. Model Evaluation Dashboard - Explore metrics and learning curves
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression, make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    confusion_matrix, classification_report
)
import pandas as pd
from io import StringIO

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================================
# TAB 1: Regression Playground
# ============================================================================

def regression_playground(dataset_type, model_type, polynomial_degree,
                         alpha, noise_level, n_samples):
    """
    Interactive regression visualization.
    """
    try:
        # Generate data
        np.random.seed(42)

        if dataset_type == "Linear":
            X, y = make_regression(n_samples=n_samples, n_features=1,
                                  noise=noise_level, random_state=42)
        elif dataset_type == "Polynomial":
            X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
            y = 0.5 * X**3 - 2 * X**2 + X + 5 + np.random.randn(n_samples, 1) * noise_level
            y = y.ravel()
        else:  # Sinusoidal
            X = np.linspace(0, 10, n_samples).reshape(-1, 1)
            y = np.sin(X).ravel() + np.random.randn(n_samples) * noise_level * 0.1

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Apply polynomial features if needed
        if polynomial_degree > 1:
            poly = PolynomialFeatures(degree=polynomial_degree)
            X_train_transformed = poly.fit_transform(X_train)
            X_test_transformed = poly.transform(X_test)
        else:
            X_train_transformed = X_train
            X_test_transformed = X_test

        # Create and train model
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Ridge":
            model = Ridge(alpha=alpha)
        elif model_type == "Lasso":
            model = Lasso(alpha=alpha, max_iter=5000)
        else:  # Decision Tree
            model = DecisionTreeRegressor(max_depth=5, random_state=42)

        model.fit(X_train_transformed, y_train)

        # Predictions
        y_train_pred = model.predict(X_train_transformed)
        y_test_pred = model.predict(X_test_transformed)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Data and predictions
        X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        if polynomial_degree > 1:
            X_plot_transformed = poly.transform(X_plot)
        else:
            X_plot_transformed = X_plot
        y_plot = model.predict(X_plot_transformed)

        axes[0].scatter(X_train, y_train, alpha=0.6, label='Training data', s=30)
        axes[0].scatter(X_test, y_test, alpha=0.6, color='orange', label='Test data', s=30)
        axes[0].plot(X_plot, y_plot, 'r-', linewidth=2, label='Model')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'{model_type}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Residuals
        axes[1].scatter(y_test_pred, y_test - y_test_pred, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Metrics text
        metrics_text = f"""
        Model Performance Metrics:

        Training Set:
        - MSE: {train_mse:.3f}
        - RMSE: {np.sqrt(train_mse):.3f}
        - R² Score: {train_r2:.3f}

        Test Set:
        - MSE: {test_mse:.3f}
        - RMSE: {np.sqrt(test_mse):.3f}
        - R² Score: {test_r2:.3f}

        Overfitting Check:
        - Train-Test R² Gap: {abs(train_r2 - test_r2):.3f}
        """

        if abs(train_r2 - test_r2) > 0.1:
            metrics_text += "\n⚠️ Warning: Possible overfitting detected!"

        return fig, metrics_text

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        return fig, f"Error: {str(e)}"

# ============================================================================
# TAB 2: Classification Visualizer
# ============================================================================

def classification_visualizer(dataset_type, model_type, n_samples, test_size):
    """
    Interactive classification algorithm comparison.
    """
    try:
        # Generate data
        np.random.seed(42)

        if dataset_type == "Linearly Separable":
            X, y = make_classification(
                n_samples=n_samples, n_features=2, n_redundant=0,
                n_informative=2, n_clusters_per_class=1, random_state=42
            )
        elif dataset_type == "Moons":
            X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
        else:  # Circles
            X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create model
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        elif model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "k-NN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_type == "SVM (RBF)":
            model = SVC(kernel='rbf', random_state=42)
        else:  # Gradient Boosting
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Decision boundary
        h = 0.02
        x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
        y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        axes[0].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test,
                       cmap='RdYlBu', edgecolors='black', s=50)
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].set_title(f'{model_type} - Decision Boundary\\nAccuracy: {accuracy:.3f}')

        # Plot 2: Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Confusion Matrix')

        plt.tight_layout()

        # Classification report
        report = classification_report(y_test, y_pred,
                                      target_names=['Class 0', 'Class 1'])

        metrics_text = f"""
        Classification Metrics:

        Overall Accuracy: {accuracy:.3f}

        {report}

        Training Set Size: {len(X_train)}
        Test Set Size: {len(X_test)}
        """

        return fig, metrics_text

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        return fig, f"Error: {str(e)}"

# ============================================================================
# TAB 3: Model Evaluation Dashboard
# ============================================================================

def evaluation_dashboard(model_type, max_depth, n_estimators):
    """
    Model evaluation with learning curves and cross-validation.
    """
    try:
        # Load data
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create model
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        elif model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth, random_state=42)
        else:  # Gradient Boosting
            model = GradientBoostingClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth, random_state=42)

        # Train
        model.fit(X_train_scaled, y_train)

        # Calculate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train_scaled, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        # Create visualization
        fig = plt.figure(figsize=(14, 10))

        # Plot 1: Learning curves
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
        ax1.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1)
        ax1.plot(train_sizes, val_mean, 'o-', label='Validation Score', linewidth=2)
        ax1.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1)
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Learning Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Confusion matrix
        ax2 = plt.subplot(2, 2, 2)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=data.target_names, yticklabels=data.target_names)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title(f'Confusion Matrix\\nAccuracy: {accuracy:.3f}')

        # Plot 3: Feature importance (if available)
        ax3 = plt.subplot(2, 2, 3)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': data.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            ax3.barh(feature_importance['feature'], feature_importance['importance'])
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importances')
            ax3.invert_yaxis()
        else:
            ax3.text(0.5, 0.5, 'Feature importance not available\\nfor this model',
                    ha='center', va='center')
            ax3.set_title('Feature Importance')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Train vs test scores
        ax4 = plt.subplot(2, 2, 4)
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)

        bars = ax4.bar(['Training', 'Test'], [train_acc, test_acc])
        bars[0].set_color('skyblue')
        bars[1].set_color('lightcoral')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Training vs Test Performance')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        # Detailed metrics
        report = classification_report(y_test, y_pred, target_names=data.target_names)

        metrics_text = f"""
        Model: {model_type}

        Performance Summary:
        - Training Accuracy: {train_acc:.3f}
        - Test Accuracy: {test_acc:.3f}
        - Overfit Gap: {abs(train_acc - test_acc):.3f}

        Cross-Validation Results:
        - Mean CV Score: {val_mean[-1]:.3f}
        - Std Dev: {val_std[-1]:.3f}

        Detailed Classification Report:
        {report}

        Model Complexity:
        """

        if model_type == "Decision Tree":
            metrics_text += f"- Max Depth: {max_depth}"
        elif model_type in ["Random Forest", "Gradient Boosting"]:
            metrics_text += f"- Number of Estimators: {n_estimators}\\n- Max Depth: {max_depth}"

        return fig, metrics_text

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        return fig, f"Error: {str(e)}"

# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Machine Learning Playground") as app:
    gr.Markdown("""
    # 🤖 Machine Learning Interactive Playground

    Explore machine learning algorithms with interactive visualizations!

    ## Features:
    1. **Regression Playground**: Experiment with different regression models and datasets
    2. **Classification Visualizer**: Compare classification algorithms with visual decision boundaries
    3. **Model Evaluation**: Understand learning curves, metrics, and overfitting
    """)

    with gr.Tabs():
        # TAB 1: Regression Playground
        with gr.Tab("🔷 Regression Playground"):
            gr.Markdown("""
            ### Regression Model Comparison

            Experiment with different regression algorithms and observe how they fit various datasets.
            Try changing the polynomial degree and regularization to see their effects!
            """)

            with gr.Row():
                with gr.Column():
                    reg_dataset = gr.Dropdown(
                        choices=["Linear", "Polynomial", "Sinusoidal"],
                        value="Polynomial",
                        label="Dataset Type"
                    )
                    reg_model = gr.Dropdown(
                        choices=["Linear Regression", "Ridge", "Lasso", "Decision Tree"],
                        value="Ridge",
                        label="Model Type"
                    )
                    poly_degree = gr.Slider(1, 10, value=3, step=1,
                                          label="Polynomial Degree")
                    alpha = gr.Slider(0.01, 10, value=1.0, step=0.1,
                                    label="Regularization (Alpha)")
                    noise = gr.Slider(1, 50, value=10, step=1,
                                    label="Noise Level")
                    n_samples_reg = gr.Slider(50, 500, value=200, step=50,
                                            label="Number of Samples")

                    reg_button = gr.Button("Train Model", variant="primary")

                with gr.Column():
                    reg_plot = gr.Plot(label="Visualization")
                    reg_metrics = gr.Textbox(label="Metrics", lines=15)

            reg_button.click(
                regression_playground,
                inputs=[reg_dataset, reg_model, poly_degree, alpha, noise, n_samples_reg],
                outputs=[reg_plot, reg_metrics]
            )

            gr.Markdown("""
            **Tips:**
            - Try increasing polynomial degree to see overfitting
            - Compare Ridge vs Lasso regularization
            - Observe residual patterns for model diagnostics
            """)

        # TAB 2: Classification Visualizer
        with gr.Tab("🔶 Classification Visualizer"):
            gr.Markdown("""
            ### Classification Algorithm Comparison

            Visualize how different algorithms create decision boundaries for various dataset types.
            Some algorithms work better on certain types of data!
            """)

            with gr.Row():
                with gr.Column():
                    class_dataset = gr.Dropdown(
                        choices=["Linearly Separable", "Moons", "Circles"],
                        value="Moons",
                        label="Dataset Type"
                    )
                    class_model = gr.Dropdown(
                        choices=["Logistic Regression", "Decision Tree", "Random Forest",
                               "k-NN", "SVM (RBF)", "Gradient Boosting"],
                        value="Random Forest",
                        label="Model Type"
                    )
                    n_samples_class = gr.Slider(100, 1000, value=300, step=100,
                                               label="Number of Samples")
                    test_size = gr.Slider(10, 50, value=30, step=5,
                                        label="Test Size (%)")

                    class_button = gr.Button("Train Classifier", variant="primary")

                with gr.Column():
                    class_plot = gr.Plot(label="Visualization")
                    class_metrics = gr.Textbox(label="Metrics", lines=15)

            class_button.click(
                classification_visualizer,
                inputs=[class_dataset, class_model, n_samples_class, test_size],
                outputs=[class_plot, class_metrics]
            )

            gr.Markdown("""
            **Observations:**
            - Linear models struggle with non-linear boundaries
            - Tree-based models create rectangular boundaries
            - SVM with RBF kernel handles complex shapes well
            - Ensemble methods (RF, GB) often perform best
            """)

        # TAB 3: Model Evaluation Dashboard
        with gr.Tab("📊 Model Evaluation"):
            gr.Markdown("""
            ### Model Evaluation Dashboard

            Comprehensive evaluation using the Breast Cancer dataset.
            Explore learning curves, confusion matrices, and feature importance!
            """)

            with gr.Row():
                with gr.Column():
                    eval_model = gr.Dropdown(
                        choices=["Logistic Regression", "Decision Tree",
                               "Random Forest", "Gradient Boosting"],
                        value="Random Forest",
                        label="Model Type"
                    )
                    max_depth = gr.Slider(1, 20, value=5, step=1,
                                        label="Max Depth (Tree-based)")
                    n_estimators = gr.Slider(10, 200, value=100, step=10,
                                           label="Number of Estimators (Ensemble)")

                    eval_button = gr.Button("Evaluate Model", variant="primary")

                with gr.Column():
                    eval_plot = gr.Plot(label="Evaluation Dashboard")
                    eval_metrics = gr.Textbox(label="Detailed Metrics", lines=20)

            eval_button.click(
                evaluation_dashboard,
                inputs=[eval_model, max_depth, n_estimators],
                outputs=[eval_plot, eval_metrics]
            )

            gr.Markdown("""
            **Key Metrics to Watch:**
            - **Learning Curves**: Diagnose overfitting/underfitting
            - **Train-Test Gap**: Large gap indicates overfitting
            - **Confusion Matrix**: Understand prediction errors
            - **Feature Importance**: See what the model learned
            """)

    gr.Markdown("""
    ---
    ### 📚 Learning Resources

    - **Week 5 Labs**: Complete hands-on implementations
    - **Scikit-learn Docs**: https://scikit-learn.org/
    - **ML Fundamentals**: Review linear algebra and probability

    ### 💡 Next Steps

    After mastering these concepts:
    1. Week 6: Neural Networks and Deep Learning
    2. Week 7: Natural Language Processing
    3. Practice on real datasets (Kaggle, UCI ML Repository)

    **Created for Week 5: Machine Learning**
    """)

if __name__ == "__main__":
    app.launch()
