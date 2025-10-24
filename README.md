# Introduction to AI: Practical Python Course

A comprehensive, hands-on course covering the foundational concepts of Artificial Intelligence with practical Python implementations. This course is designed for complete beginners with basic Python knowledge who want to understand AI from the ground up.

## Course Overview

This 7-week course takes you from basic AI concepts to advanced neural networks and natural language processing. Each week combines theory with hands-on coding, implementing algorithms from scratch before exploring professional libraries.

### What You'll Build

- **Interactive pathfinding visualizers** showing search algorithms in action
- **Logic puzzle solvers** and expert systems
- **Probabilistic reasoning systems** for real-world uncertainty
- **Optimization engines** for scheduling and constraint satisfaction
- **Machine learning models** for classification and prediction
- **Neural networks from scratch** and deep learning applications
- **Natural language processing** tools for text analysis and generation

## Weekly Modules

### Week 1: Search
Learn how AI agents find solutions by exploring problem spaces.

**Topics:**
- Depth-First Search (DFS) and Breadth-First Search (BFS)
- Uniform Cost Search
- A* Search and Heuristics
- Adversarial Search (Minimax, Alpha-Beta Pruning)

**Projects:**
- Interactive maze solver with visualization
- Route planning application
- Game-playing AI (Tic-Tac-Toe)

**Key Libraries:** NetworkX, Pygame/Matplotlib

---

### Week 2: Knowledge
Represent and reason with knowledge using logic.

**Topics:**
- Propositional Logic
- Inference and Resolution
- First-Order Logic
- Knowledge Representation

**Projects:**
- Logic puzzle solver (Knights and Knaves)
- Simple expert system
- Rule-based reasoning engine

**Key Libraries:** SymPy, custom implementations

---

### Week 3: Uncertainty
Handle probabilistic reasoning and uncertain information.

**Topics:**
- Probability Theory
- Bayes' Theorem and Bayesian Networks
- Markov Models
- Hidden Markov Models

**Projects:**
- Medical diagnosis system
- Weather prediction model
- Spam filter with naive Bayes

**Key Libraries:** pgmpy, NumPy

---

### Week 4: Optimization
Find optimal solutions to complex problems.

**Topics:**
- Local Search (Hill Climbing, Simulated Annealing)
- Genetic Algorithms
- Constraint Satisfaction Problems
- Linear Programming

**Projects:**
- Course scheduling system
- N-Queens solver
- Resource allocation optimizer

**Key Libraries:** SciPy, custom implementations

---

### Week 5: Learning
Build systems that learn from data.

**Topics:**
- Supervised Learning (Classification, Regression)
- k-Nearest Neighbors
- Decision Trees and Random Forests
- Support Vector Machines
- Model Evaluation

**Projects:**
- Movie recommendation system
- Handwriting recognition (using features)
- Housing price predictor

**Key Libraries:** scikit-learn, pandas

---

### Week 6: Neural Networks
Understand and build neural networks and deep learning models.

**Topics:**
- Perceptrons and Neural Network Architecture
- Backpropagation
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transfer Learning

**Projects:**
- Neural network from scratch
- MNIST digit classifier
- Image recognition app with Gradio UI
- Style transfer demo

**Key Libraries:** TensorFlow/Keras, PyTorch

---

### Week 7: Language
Process and generate human language with AI.

**Topics:**
- Tokenization and Text Processing
- N-grams and Language Models
- Word Embeddings (Word2Vec, GloVe)
- Transformers and Attention
- Sentiment Analysis

**Projects:**
- Sentiment analyzer
- Text generator
- Question answering system
- Chatbot with context

**Key Libraries:** NLTK, spaCy, Transformers

---

## Course Structure

Each week contains:

```
week_folder/
├── 1_lab1.ipynb          # Introduction and basic concepts
├── 2_lab2.ipynb          # Core algorithms (from scratch)
├── 3_lab3.ipynb          # Advanced techniques and libraries
├── 4_lab4.ipynb          # Integration and experimentation
├── project_app.py        # Interactive Gradio application
├── utils.py              # Helper functions
├── data/                 # Sample datasets
└── community/            # Student submissions
```

## Prerequisites

- **Python**: Basic Python programming (variables, functions, loops, classes)
- **Math**: High school algebra (calculus and linear algebra helpful but not required)
- **Tools**: Command line basics, text editor or IDE

No prior AI or machine learning experience needed!

## Getting Started

### 1. Choose Your Setup Guide

Select the guide for your operating system:

- [Mac Setup](setup/SETUP-mac.md)
- [Windows Setup](setup/SETUP-windows.md)
- [Linux Setup](setup/SETUP-linux.md)
- [WSL Setup](setup/SETUP-wsl.md) (Windows Subsystem for Linux)

### 2. Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd intro-to-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Mac/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -e .

# Download NLTK data (for Week 7)
python -c "import nltk; nltk.download('popular')"

# Download spaCy model (for Week 7)
python -m spacy download en_core_web_sm
```

### 3. Start Learning

Open Jupyter Lab and start with Week 1:

```bash
jupyter lab
```

Navigate to `1_search/1_lab1.ipynb` to begin!

## Additional Resources

### Guides

The `guides/` directory contains essential background material:

- **01_python_refresher.ipynb** - Python essentials review
- **02_numpy_basics.ipynb** - NumPy for scientific computing
- **03_math_foundations.ipynb** - Linear algebra and calculus primer
- **04_probability_basics.ipynb** - Probability theory fundamentals
- **05_jupyter_guide.ipynb** - Using Jupyter notebooks effectively
- **06_visualization.ipynb** - Creating effective visualizations
- **07_debugging.ipynb** - Debugging AI code

### Datasets

Common datasets are provided in `data/` directory:

- MNIST handwritten digits
- Iris flower classification
- Movie ratings for recommendations
- Text corpora for NLP
- Sample images for computer vision

### Community

Share your projects in the `community/` directory! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Learning Tips

1. **Code along**: Type the code yourself rather than copy-pasting
2. **Experiment**: Modify parameters and see what happens
3. **Implement first**: Try the from-scratch versions before using libraries
4. **Visualize**: Use the visualization tools to understand algorithms
5. **Debug**: When stuck, use print statements and visualization
6. **Share**: Post your projects in the community folder

## Project Philosophy

This course follows a **hybrid learning approach**:

1. **Understand the concept** through clear explanations and visualizations
2. **Implement from scratch** to see how algorithms really work
3. **Use professional libraries** to build production-ready applications
4. **Build real projects** that demonstrate practical applications

## Tools and Technologies

- **Python 3.10+**: Primary programming language
- **Jupyter Notebooks**: Interactive learning environment
- **NumPy/SciPy**: Scientific computing
- **scikit-learn**: Machine learning library
- **TensorFlow/Keras**: Deep learning framework
- **Gradio**: Building interactive web UIs
- **Matplotlib/Plotly**: Data visualization

## Getting Help

### Documentation

- Check the [FAQ](docs/FAQ.md) for common questions
- Review [Troubleshooting Guide](setup/troubleshooting.ipynb)
- Read the [Python Refresher](guides/01_python_refresher.ipynb)

### Community

- Open an issue on GitHub for bugs or questions
- Share your projects in `community/`
- Collaborate with other students

## Course Requirements

**Time Commitment**: 8-10 hours per week
- 2-3 hours: Video lectures and reading
- 3-4 hours: Lab notebooks and exercises
- 3-4 hours: Projects and experimentation

**Hardware**: Any modern computer (GPU helpful for Week 6-7 but not required)

**Software**: Python 3.10+, modern web browser, text editor/IDE

## Acknowledgments

This course draws inspiration from:
- CS50's Introduction to AI with Python (Harvard)
- UC Berkeley CS188: Artificial Intelligence
- Stanford CS229: Machine Learning
- Fast.ai courses

Special thanks to the open-source community for the amazing libraries that make this course possible.

## License

MIT License - see [LICENSE](LICENSE) for details

Course materials are free for educational use. Please attribute when sharing or adapting.

---

**Ready to start your AI journey?** Head to [Week 1: Search](1_search/1_lab1.ipynb) and let's build something amazing!

---

*Questions or feedback? Open an issue or reach out at [your.email@example.com](mailto:your.email@example.com)*
