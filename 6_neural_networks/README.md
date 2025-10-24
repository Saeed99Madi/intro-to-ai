# Week 6: Neural Networks and Deep Learning

Welcome to Week 6 of the Introduction to AI course! This week dives into neural networks and deep learning, the technology behind many modern AI breakthroughs including image recognition, natural language processing, and game-playing agents.

## Overview

Neural networks are computational models inspired by biological neurons. Deep learning refers to neural networks with multiple layers that can learn hierarchical representations of data. This week, you'll build neural networks from scratch and use modern frameworks like TensorFlow/Keras and PyTorch.

## Learning Objectives

By the end of this week, you will be able to:

- Understand the architecture and mathematics of neural networks
- Implement feedforward networks and backpropagation from scratch
- Build and train deep neural networks using Keras/PyTorch
- Apply convolutional neural networks (CNNs) for image tasks
- Understand recurrent neural networks (RNNs) for sequences
- Implement modern architectures (ResNet, Attention mechanisms)
- Diagnose and fix common training issues
- Apply transfer learning and fine-tuning
- Build practical deep learning applications

## Prerequisites

- Python programming fundamentals
- NumPy for numerical computing
- Understanding of machine learning basics (Week 5)
- Linear algebra (matrices, vectors, derivatives)
- Basic calculus (chain rule, gradients)

## Labs

### Lab 1: Introduction to Neural Networks
**File:** `1_lab1.ipynb`

Build your first neural networks from scratch and understand the fundamentals.

**Topics:**
- Biological inspiration and artificial neurons
- Perceptrons and activation functions
- Feedforward neural networks
- Backpropagation algorithm from scratch
- Gradient descent variants (SGD, Momentum, Adam)
- Multi-layer perceptrons (MLPs)
- Introduction to Keras and PyTorch
- Practical example: Handwritten digit recognition (MNIST)

**Key Concepts:**
- Forward propagation
- Loss functions (MSE, Cross-Entropy)
- Backpropagation and chain rule
- Weight initialization
- Learning rate scheduling

### Lab 2: Deep Learning Fundamentals
**File:** `2_lab2.ipynb`

Master the techniques for training deep neural networks effectively.

**Topics:**
- Deep network architectures
- Activation functions (ReLU, LeakyReLU, ELU, Swish)
- Batch normalization and layer normalization
- Dropout and regularization techniques
- Weight initialization strategies (Xavier, He)
- Optimizers (SGD, Momentum, RMSprop, Adam, AdamW)
- Learning rate schedules and warmup
- Gradient clipping and exploding/vanishing gradients
- Residual connections and skip connections
- Practical example: Deep classifier for Fashion-MNIST

**Key Concepts:**
- Overfitting in deep networks
- Normalization techniques
- Optimization landscape
- Hyperparameter tuning for deep learning

### Lab 3: Convolutional Neural Networks (CNNs)
**File:** `3_lab3.ipynb`

Learn specialized architectures for computer vision tasks.

**Topics:**
- Convolutional layers and feature maps
- Pooling layers (max, average, global)
- CNN architectures from scratch
- Famous architectures (LeNet, AlexNet, VGG, ResNet)
- Transfer learning and fine-tuning
- Data augmentation techniques
- Image classification with CNNs
- Object detection basics
- Visualizing CNN features and activations
- Practical example: Custom image classifier

**Key Concepts:**
- Spatial hierarchies in vision
- Parameter sharing and translation invariance
- Receptive fields
- Feature extraction and representation learning

### Lab 4: Advanced Architectures and Techniques
**File:** `4_lab4.ipynb`

Explore cutting-edge architectures and techniques.

**Topics:**
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM) networks
- Gated Recurrent Units (GRUs)
- Attention mechanisms
- Transformer architecture basics
- Autoencoders and representation learning
- Generative models introduction
- Multi-task learning
- Neural architecture search concepts
- Practical example: Sequence prediction and text generation

**Key Concepts:**
- Sequential data processing
- Memory and gating mechanisms
- Self-attention
- Encoder-decoder architectures

## Interactive Application

**File:** `nn_app.py`

A comprehensive Gradio application for experimenting with neural networks:

1. **Neural Network Playground**: Visualize how neural networks learn with interactive weight updates
2. **CNN Visualizer**: Explore convolutional filters and feature maps on images
3. **Architecture Comparator**: Compare different network architectures on various datasets

Run with:
```bash
python nn_app.py
```

## Key Concepts Summary

- **Neuron**: Basic computational unit that applies weights, bias, and activation
- **Layer**: Collection of neurons operating in parallel
- **Depth**: Number of layers in the network
- **Forward Pass**: Computing predictions from inputs
- **Backward Pass**: Computing gradients via backpropagation
- **Epoch**: One complete pass through the training data
- **Batch**: Subset of training data processed together
- **Learning Rate**: Step size for parameter updates
- **Activation Function**: Non-linear transformation in neurons
- **Loss Function**: Measure of prediction error
- **Optimizer**: Algorithm for updating weights
- **Regularization**: Techniques to prevent overfitting
- **Batch Normalization**: Normalizing layer inputs
- **Dropout**: Randomly dropping neurons during training
- **Convolution**: Sliding filter operation for spatial features
- **Pooling**: Downsampling operation
- **Transfer Learning**: Using pre-trained models
- **Fine-tuning**: Adapting pre-trained models to new tasks

## Real-World Applications

Deep learning powers many modern AI systems:

- **Computer Vision**: Image classification, object detection, facial recognition, medical imaging, autonomous vehicles
- **Natural Language Processing**: Translation, sentiment analysis, question answering, chatbots
- **Speech**: Speech recognition, text-to-speech, voice assistants
- **Recommendation Systems**: Netflix, YouTube, Spotify recommendations
- **Gaming**: AlphaGo, game-playing AI agents
- **Healthcare**: Disease diagnosis, drug discovery, protein folding
- **Finance**: Fraud detection, algorithmic trading, risk assessment
- **Creative AI**: Art generation, music composition, style transfer
- **Robotics**: Robot control, manipulation, navigation
- **Scientific Research**: Climate modeling, particle physics, astronomy

## Installation

Ensure you have the required deep learning frameworks:

```bash
# TensorFlow and Keras
pip install tensorflow

# PyTorch
pip install torch torchvision

# Other dependencies
pip install numpy matplotlib seaborn scikit-learn gradio pillow
```

## Framework Choice: TensorFlow vs PyTorch

Both frameworks are excellent. This course covers both:

**TensorFlow/Keras:**
- High-level, beginner-friendly API
- Great for production deployment
- Excellent documentation
- Used in industry widely

**PyTorch:**
- More Pythonic and flexible
- Preferred in research
- Dynamic computational graphs
- Growing industry adoption

**Recommendation:** Learn both! We'll use Keras for quick prototyping and PyTorch for understanding internals.

## Tips for Success

1. **Start Simple**: Begin with small networks, then scale up
2. **Visualize**: Plot training curves, activations, and gradients
3. **Experiment**: Try different architectures, learning rates, and hyperparameters
4. **Debug Systematically**: Check shapes, ranges, and gradients
5. **Use Pre-trained Models**: Leverage transfer learning when possible
6. **Monitor Training**: Watch for overfitting, underfitting, and convergence issues
7. **Data Matters**: Good data beats fancy architectures
8. **Be Patient**: Deep learning requires computational resources and time
9. **Read Papers**: Stay current with latest architectures and techniques
10. **Hands-on Practice**: Build projects beyond course exercises

## Common Issues and Solutions

| Problem | Possible Causes | Solutions |
|---------|----------------|-----------|
| **Loss not decreasing** | Learning rate too high/low, bad initialization | Adjust learning rate, try different initialization |
| **Overfitting** | Model too complex, insufficient data | Add regularization, dropout, more data |
| **Vanishing gradients** | Deep network, sigmoid activation | Use ReLU, batch norm, residual connections |
| **Exploding gradients** | Learning rate too high, unstable training | Gradient clipping, lower learning rate |
| **Slow training** | Large model, inefficient code | Use GPU, batch processing, smaller model |
| **Poor generalization** | Memorizing training data | Cross-validation, regularization, more data |

## Architecture Selection Guide

**Use Feedforward Networks (MLPs) when:**
- Working with tabular/structured data
- Features are not spatially or temporally related
- Need simple, interpretable models
- Examples: Classification, regression on feature vectors

**Use CNNs when:**
- Working with images or spatial data
- Need translation invariance
- Want to learn hierarchical features
- Examples: Image classification, object detection, segmentation

**Use RNNs/LSTMs when:**
- Working with sequential data
- Order matters (time series, text, audio)
- Need to remember long-term dependencies
- Examples: Language modeling, time series prediction, speech

**Use Transformers when:**
- Need attention over sequences
- Parallelizable training is important
- Working with long sequences
- Examples: NLP tasks, machine translation, document understanding

## Optimization Best Practices

1. **Start with Adam optimizer** (good default)
2. **Use learning rate warmup** for stable training
3. **Apply learning rate scheduling** (decay over time)
4. **Batch size**: Start with 32-128, adjust based on memory
5. **Monitor validation metrics** to detect overfitting early
6. **Use early stopping** to prevent overtraining
7. **Checkpoint best models** during training
8. **Try different random seeds** to ensure robustness

## Data Preparation Tips

1. **Normalize inputs** (zero mean, unit variance)
2. **Shuffle training data** each epoch
3. **Use data augmentation** (especially for images)
4. **Balance classes** or use weighted loss
5. **Split properly**: Train/Validation/Test (70/15/15 or 80/10/10)
6. **Create meaningful validation sets** that represent test distribution
7. **Use stratified splits** for classification
8. **Check for data leakage** between splits

## GPU Utilization

Deep learning is computationally intensive. GPUs provide massive speedups:

**Using GPUs:**
```python
# TensorFlow - automatic GPU detection
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# PyTorch - move model to GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

**Cloud Options:**
- Google Colab (free GPU/TPU)
- Kaggle Kernels (free GPU)
- AWS/GCP/Azure (paid)
- Paperspace Gradient (paid)

## Model Deployment

After training, deploy your models:

1. **Save Models**: TensorFlow SavedModel, PyTorch state_dict
2. **Optimize**: Quantization, pruning, knowledge distillation
3. **Serve**: TensorFlow Serving, TorchServe, ONNX
4. **Edge Deployment**: TensorFlow Lite, PyTorch Mobile
5. **Web**: ONNX.js, TensorFlow.js
6. **APIs**: FastAPI, Flask with model serving

## Next Steps

After completing this week:
- Week 7: Language - Advanced NLP and Transformers
- Explore specialized topics: GANs, Reinforcement Learning, Graph Networks
- Read seminal papers: AlexNet, ResNet, Attention is All You Need
- Participate in Kaggle competitions
- Build your own deep learning projects
- Contribute to open-source ML projects

## Resources

### Documentation
- **TensorFlow**: https://www.tensorflow.org/
- **PyTorch**: https://pytorch.org/
- **Keras**: https://keras.io/

### Courses
- **Deep Learning Specialization** (Andrew Ng)
- **Fast.ai Practical Deep Learning**
- **Stanford CS231n** (Computer Vision)
- **Stanford CS224n** (NLP)

### Books
- *Deep Learning* by Goodfellow, Bengio, Courville
- *Deep Learning with Python* by François Chollet
- *Dive into Deep Learning* (d2l.ai)

### Papers
- ImageNet Classification (AlexNet)
- Very Deep Networks (VGG)
- Deep Residual Learning (ResNet)
- Batch Normalization
- Dropout
- Adam Optimizer

### Communities
- r/MachineLearning
- Papers with Code
- Hugging Face Community
- PyTorch Forums
- TensorFlow Community

## Community Contributions

Have you created additional exercises, architectures, or applications? Share them in the `community/` folder!

## Ethics and Responsible AI

Deep learning is powerful. Use it responsibly:
- Consider bias in training data
- Evaluate fairness across demographics
- Ensure privacy and data protection
- Be transparent about limitations
- Consider environmental impact of large models
- Document model behavior and failure modes

Good luck, and enjoy exploring the fascinating world of deep learning!
