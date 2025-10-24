"""
Neural Networks Interactive Application

This Gradio app provides interactive demonstrations of:
1. Neural Network Playground - Visualize training dynamics
2. CNN Filter Visualizer - Explore convolutional filters
3. Architecture Comparator - Compare different network types
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================================
# TAB 1: Neural Network Playground
# ============================================================================

def neural_network_playground(dataset_type, n_hidden_layers, neurons_per_layer,
                              learning_rate, epochs, activation):
    """
    Interactive neural network training visualization.
    """
    try:
        # Generate data
        np.random.seed(42)

        if dataset_type == "Moons":
            X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
        elif dataset_type == "Circles":
            X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
        else:  # Linear
            X, y = make_classification(
                n_samples=500, n_features=2, n_redundant=0,
                n_informative=2, n_clusters_per_class=1, random_state=42
            )

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build model
        model = keras.Sequential()
        model.add(layers.Dense(neurons_per_layer, activation=activation, input_shape=(2,)))

        for _ in range(n_hidden_layers - 1):
            model.add(layers.Dense(neurons_per_layer, activation=activation))

        model.add(layers.Dense(1, activation='sigmoid'))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = model.fit(
            X_train_scaled, y_train,
            epochs=epochs,
            validation_split=0.2,
            verbose=0
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

        # Create visualization
        fig = plt.figure(figsize=(14, 10))

        # Plot 1: Decision boundary
        ax1 = plt.subplot(2, 2, 1)
        h = 0.02
        x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
        y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
        Z = Z.reshape(xx.shape)

        ax1.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        ax1.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test,
                   cmap='RdYlBu', edgecolors='black', s=50)
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title(f'Decision Boundary\\nTest Accuracy: {test_acc:.3f}')

        # Plot 2: Training history
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Accuracy
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(history.history['accuracy'], label='Training Accuracy')
        ax3.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Layer weights visualization
        ax4 = plt.subplot(2, 2, 4)
        first_layer_weights = model.layers[0].get_weights()[0]
        im = ax4.imshow(first_layer_weights.T, aspect='auto', cmap='viridis')
        ax4.set_xlabel('Input Features')
        ax4.set_ylabel('Neurons')
        ax4.set_title('First Layer Weights')
        plt.colorbar(im, ax=ax4)

        plt.tight_layout()

        # Metrics text
        metrics_text = f"""
        Model Architecture:
        - Input: 2 features
        - Hidden Layers: {n_hidden_layers} x {neurons_per_layer} neurons
        - Activation: {activation}
        - Output: 1 neuron (sigmoid)
        - Total Parameters: {model.count_params():,}

        Training Configuration:
        - Learning Rate: {learning_rate}
        - Epochs: {epochs}
        - Dataset: {dataset_type}

        Performance:
        - Training Accuracy: {history.history['accuracy'][-1]:.4f}
        - Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}
        - Test Accuracy: {test_acc:.4f}
        - Test Loss: {test_loss:.4f}

        Observations:
        """

        # Add observations
        if abs(history.history['accuracy'][-1] - history.history['val_accuracy'][-1]) > 0.1:
            metrics_text += "\n⚠️ Possible overfitting detected (train-val gap > 0.1)"
        else:
            metrics_text += "\n✓ Good generalization (train-val gap < 0.1)"

        if test_acc > 0.9:
            metrics_text += "\n✓ Excellent test performance"
        elif test_acc > 0.7:
            metrics_text += "\n✓ Good test performance"
        else:
            metrics_text += "\n⚠️ Consider more training or different architecture"

        return fig, metrics_text

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        return fig, f"Error: {str(e)}"

# ============================================================================
# TAB 2: CNN Filter Visualizer
# ============================================================================

def cnn_filter_visualizer(n_filters, filter_size, pooling_type):
    """
    Visualize CNN filters and feature maps.
    """
    try:
        # Load sample MNIST image
        (X_train, _), _ = keras.datasets.mnist.load_data()
        sample_image = X_train[0].astype('float32') / 255.0
        sample_image = sample_image.reshape(1, 28, 28, 1)

        # Build simple CNN
        model = keras.Sequential([
            layers.Conv2D(n_filters, (filter_size, filter_size),
                         activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)) if pooling_type == "Max" else layers.AveragePooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax')
        ])

        # Get intermediate layer output
        layer_model = keras.Model(inputs=model.input,
                                 outputs=model.layers[0].output)
        feature_maps = layer_model.predict(sample_image, verbose=0)

        # Create visualization
        n_cols = min(8, n_filters)
        n_rows = (n_filters + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(14, 3 * n_rows))

        # Original image
        plt.subplot(n_rows + 1, n_cols, 1)
        plt.imshow(sample_image[0, :, :, 0], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Feature maps
        for i in range(n_filters):
            ax = plt.subplot(n_rows + 1, n_cols, i + n_cols + 1)
            ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
            ax.set_title(f'Filter {i+1}')
            ax.axis('off')

        plt.suptitle(f'CNN Feature Maps ({n_filters} filters, {filter_size}x{filter_size})',
                    fontsize=14)
        plt.tight_layout()

        info_text = f"""
        CNN Configuration:
        - Number of Filters: {n_filters}
        - Filter Size: {filter_size}x{filter_size}
        - Pooling: {pooling_type} Pooling (2x2)
        - Input Shape: 28x28x1
        - Output Shape: {feature_maps.shape[1]}x{feature_maps.shape[2]}x{n_filters}

        What are we seeing?
        - Each filter detects different features (edges, textures, patterns)
        - Brighter areas indicate strong activations
        - Different filters activate on different parts of the image

        Observations:
        - Filter size {filter_size}x{filter_size} creates receptive fields
        - {pooling_type} pooling reduces spatial dimensions
        - Total parameters in conv layer: {n_filters * (filter_size * filter_size + 1):,}
        """

        return fig, info_text

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        return fig, f"Error: {str(e)}"

# ============================================================================
# TAB 3: Architecture Comparator
# ============================================================================

def architecture_comparator(architecture_type, dataset, epochs_comp):
    """
    Compare different neural network architectures.
    """
    try:
        # Load data based on selection
        if dataset == "MNIST":
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            X_train = X_train.reshape(-1, 784) / 255.0
            X_test = X_test.reshape(-1, 784) / 255.0
            input_shape = (784,)
            n_classes = 10
        else:  # Fashion-MNIST
            (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
            X_train = X_train.reshape(-1, 784) / 255.0
            X_test = X_test.reshape(-1, 784) / 255.0
            input_shape = (784,)
            n_classes = 10

        # Use subset for speed
        X_train, y_train = X_train[:10000], y_train[:10000]
        X_test, y_test = X_test[:2000], y_test[:2000]

        # Build model based on architecture
        if architecture_type == "Simple MLP":
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=input_shape),
                layers.Dense(64, activation='relu'),
                layers.Dense(n_classes, activation='softmax')
            ])
        elif architecture_type == "Deep MLP":
            model = keras.Sequential([
                layers.Dense(256, activation='relu', input_shape=input_shape),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(n_classes, activation='softmax')
            ])
        else:  # CNN
            # Reshape for CNN
            X_train_cnn = X_train.reshape(-1, 28, 28, 1)
            X_test_cnn = X_test.reshape(-1, 28, 28, 1)

            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(n_classes, activation='softmax')
            ])

            X_train = X_train_cnn
            X_test = X_test_cnn

        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train
        history = model.fit(
            X_train, y_train,
            batch_size=128,
            epochs=epochs_comp,
            validation_split=0.2,
            verbose=0
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Training loss
        axes[0, 0].plot(history.history['loss'], label='Training')
        axes[0, 0].plot(history.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Training accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Final performance comparison
        metrics = ['Train Acc', 'Val Acc', 'Test Acc']
        values = [history.history['accuracy'][-1],
                 history.history['val_accuracy'][-1],
                 test_acc]

        bars = axes[1, 0].bar(metrics, values)
        bars[0].set_color('skyblue')
        bars[1].set_color('lightcoral')
        bars[2].set_color('lightgreen')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Performance Comparison')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}', ha='center', va='bottom')

        # Plot 4: Model complexity
        axes[1, 1].text(0.5, 0.7, f'Total Parameters: {model.count_params():,}',
                       ha='center', fontsize=14, weight='bold')
        axes[1, 1].text(0.5, 0.5, f'Trainable Parameters: {model.count_params():,}',
                       ha='center', fontsize=12)
        axes[1, 1].text(0.5, 0.3, f'Layers: {len(model.layers)}',
                       ha='center', fontsize=12)
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Model Complexity')

        plt.tight_layout()

        # Metrics text
        metrics_text = f"""
        Architecture: {architecture_type}
        Dataset: {dataset}

        Model Summary:
        - Total Parameters: {model.count_params():,}
        - Number of Layers: {len(model.layers)}
        - Training Samples: {len(X_train)}
        - Test Samples: {len(X_test)}

        Performance:
        - Final Training Accuracy: {history.history['accuracy'][-1]:.4f}
        - Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}
        - Test Accuracy: {test_acc:.4f}
        - Test Loss: {test_loss:.4f}

        Training Time:
        - Epochs: {epochs_comp}
        - Samples per Epoch: {len(X_train)}

        Analysis:
        """

        # Add analysis
        gap = abs(history.history['accuracy'][-1] - history.history['val_accuracy'][-1])
        if gap > 0.1:
            metrics_text += f"\n⚠️ Overfitting detected (gap: {gap:.3f})"
            metrics_text += "\n   Consider: more data, regularization, or simpler model"
        else:
            metrics_text += f"\n✓ Good generalization (gap: {gap:.3f})"

        if test_acc > 0.95:
            metrics_text += "\n✓ Excellent test performance!"
        elif test_acc > 0.85:
            metrics_text += "\n✓ Good test performance"
        else:
            metrics_text += "\n⚠️ Room for improvement"
            metrics_text += "\n   Consider: more training, data augmentation, or better architecture"

        return fig, metrics_text

    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        return fig, f"Error: {str(e)}"

# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Neural Networks Playground") as app:
    gr.Markdown("""
    # 🧠 Neural Networks Interactive Playground

    Explore deep learning concepts with interactive visualizations!

    ## Features:
    1. **Neural Network Playground**: Train networks and visualize decision boundaries
    2. **CNN Filter Visualizer**: Explore convolutional filters and feature maps
    3. **Architecture Comparator**: Compare MLP vs CNN performance
    """)

    with gr.Tabs():
        # TAB 1: Neural Network Playground
        with gr.Tab("🎮 Neural Network Playground"):
            gr.Markdown("""
            ### Train Neural Networks Interactively

            Experiment with different architectures and see how they learn to classify data!
            """)

            with gr.Row():
                with gr.Column():
                    dataset_nn = gr.Dropdown(
                        choices=["Moons", "Circles", "Linear"],
                        value="Moons",
                        label="Dataset Type"
                    )
                    n_layers = gr.Slider(1, 5, value=2, step=1,
                                        label="Number of Hidden Layers")
                    neurons = gr.Slider(4, 128, value=16, step=4,
                                       label="Neurons per Layer")
                    lr_nn = gr.Slider(0.0001, 0.1, value=0.01, step=0.0001,
                                     label="Learning Rate")
                    epochs_nn = gr.Slider(10, 200, value=50, step=10,
                                         label="Epochs")
                    activation_nn = gr.Dropdown(
                        choices=["relu", "tanh", "sigmoid"],
                        value="relu",
                        label="Activation Function"
                    )

                    nn_button = gr.Button("Train Network", variant="primary")

                with gr.Column():
                    nn_plot = gr.Plot(label="Training Visualization")
                    nn_metrics = gr.Textbox(label="Metrics and Analysis", lines=20)

            nn_button.click(
                neural_network_playground,
                inputs=[dataset_nn, n_layers, neurons, lr_nn, epochs_nn, activation_nn],
                outputs=[nn_plot, nn_metrics]
            )

            gr.Markdown("""
            **Tips:**
            - Try different datasets to see how complexity affects learning
            - Increase layers/neurons for more complex decision boundaries
            - Watch for overfitting (train-val gap)
            - Experiment with activation functions
            """)

        # TAB 2: CNN Filter Visualizer
        with gr.Tab("🔍 CNN Filter Visualizer"):
            gr.Markdown("""
            ### Visualize Convolutional Filters

            See what CNN filters learn to detect in images!
            """)

            with gr.Row():
                with gr.Column():
                    n_filters_cnn = gr.Slider(4, 32, value=16, step=4,
                                             label="Number of Filters")
                    filter_size_cnn = gr.Slider(3, 7, value=3, step=2,
                                               label="Filter Size")
                    pooling_cnn = gr.Dropdown(
                        choices=["Max", "Average"],
                        value="Max",
                        label="Pooling Type"
                    )

                    cnn_button = gr.Button("Generate Feature Maps", variant="primary")

                with gr.Column():
                    cnn_plot = gr.Plot(label="Feature Maps Visualization")
                    cnn_info = gr.Textbox(label="CNN Information", lines=15)

            cnn_button.click(
                cnn_filter_visualizer,
                inputs=[n_filters_cnn, filter_size_cnn, pooling_cnn],
                outputs=[cnn_plot, cnn_info]
            )

            gr.Markdown("""
            **Observations:**
            - Each filter detects different features (edges, corners, textures)
            - Early layers detect simple patterns
            - Later layers detect complex features
            - Filter visualization helps understand what the network learns
            """)

        # TAB 3: Architecture Comparator
        with gr.Tab("⚖️ Architecture Comparator"):
            gr.Markdown("""
            ### Compare Neural Network Architectures

            See how different architectures perform on the same dataset!
            """)

            with gr.Row():
                with gr.Column():
                    arch_type = gr.Dropdown(
                        choices=["Simple MLP", "Deep MLP", "CNN"],
                        value="Simple MLP",
                        label="Architecture Type"
                    )
                    dataset_comp = gr.Dropdown(
                        choices=["MNIST", "Fashion-MNIST"],
                        value="MNIST",
                        label="Dataset"
                    )
                    epochs_comp = gr.Slider(5, 20, value=10, step=1,
                                           label="Training Epochs")

                    comp_button = gr.Button("Train and Compare", variant="primary")

                with gr.Column():
                    comp_plot = gr.Plot(label="Comparison Results")
                    comp_metrics = gr.Textbox(label="Performance Analysis", lines=20)

            comp_button.click(
                architecture_comparator,
                inputs=[arch_type, dataset_comp, epochs_comp],
                outputs=[comp_plot, comp_metrics]
            )

            gr.Markdown("""
            **Architecture Insights:**
            - **Simple MLP**: Fast, good for simple patterns
            - **Deep MLP**: More capacity, better for complex data
            - **CNN**: Best for images, spatial feature extraction

            Try both datasets to see how architecture choice affects performance!
            """)

    gr.Markdown("""
    ---
    ### 📚 Learning Resources

    - **Week 6 Labs**: Deep dive into neural network concepts
    - **TensorFlow**: https://www.tensorflow.org/
    - **PyTorch**: https://pytorch.org/
    - **Deep Learning Book**: deeplearningbook.org

    ### 🎯 Key Takeaways

    1. **Architecture matters**: Choose based on your data type
    2. **Regularization helps**: Dropout, batch norm prevent overfitting
    3. **Visualization aids understanding**: Always plot training curves
    4. **Experimentation is key**: Try different hyperparameters

    **Created for Week 6: Neural Networks and Deep Learning**
    """)

if __name__ == "__main__":
    app.launch()
