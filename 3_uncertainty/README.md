# Week 3: Uncertainty - Probabilistic Reasoning

Welcome to Week 3! This week we explore how AI systems reason under uncertainty using probability theory and Bayesian methods.

## Overview

Real-world AI systems rarely have complete information. This week, you'll learn how to:
- Represent and reason with uncertain knowledge
- Use probability theory for decision-making
- Build Bayesian networks for complex reasoning
- Apply probabilistic models to real problems

## Learning Objectives

By the end of this week, you will be able to:

1. **Apply probability theory** to AI problems
2. **Use Bayes' theorem** for inference and updating beliefs
3. **Build and query Bayesian networks** for structured reasoning
4. **Implement Markov models** for sequential data
5. **Design probabilistic systems** for real-world applications

## Prerequisites

- Completion of Weeks 1-2 (Search, Knowledge)
- Basic Python programming
- Familiarity with NumPy arrays
- Understanding of logical reasoning (from Week 2)
- Basic algebra and probability concepts (helpful but not required)

## Labs

### Lab 1: Introduction to Probability
**Duration:** 2-3 hours | **Difficulty:** Beginner

Learn the foundations of probability theory for AI:
- Basic probability rules and axioms
- Conditional probability and independence
- Random variables and distributions
- Joint and marginal probabilities
- Real-world probability calculations

**Key Concepts:** Sample spaces, events, probability axioms, conditional probability, Bayes' rule introduction

### Lab 2: Bayes' Theorem and Inference
**Duration:** 2-3 hours | **Difficulty:** Intermediate

Master Bayesian inference for AI applications:
- Derivation and intuition of Bayes' theorem
- Medical diagnosis systems
- Spam filtering with naive Bayes
- Updating beliefs with new evidence
- Introduction to pgmpy library

**Key Concepts:** Prior/posterior probabilities, likelihood, evidence, naive Bayes classifier

### Lab 3: Bayesian Networks
**Duration:** 3-4 hours | **Difficulty:** Intermediate

Build structured probabilistic models:
- Graphical representation of dependencies
- Conditional probability tables (CPTs)
- Building networks from scratch
- Inference algorithms (variable elimination, sampling)
- Real-world network examples

**Key Concepts:** DAGs, conditional independence, d-separation, exact and approximate inference

### Lab 4: Markov Models and Applications
**Duration:** 3-4 hours | **Difficulty:** Advanced

Work with sequential probabilistic models:
- Markov chains and properties
- Hidden Markov Models (HMMs)
- Forward-backward algorithm
- Viterbi algorithm for sequence decoding
- Time series prediction

**Key Concepts:** Markov property, transition matrices, emission probabilities, sequence prediction

## Interactive Application

### `diagnosis_app.py` - Probabilistic Reasoning System

An interactive Gradio application featuring:

1. **Medical Diagnosis Simulator**
   - Enter symptoms to get disease probabilities
   - See how Bayes' theorem updates beliefs
   - Visualize probability distributions

2. **Bayesian Network Builder**
   - Create custom Bayesian networks
   - Define conditional probabilities
   - Query networks for inference
   - Visualize network structure

3. **Weather Prediction System**
   - HMM-based weather forecasting
   - Predict sequences based on observations
   - Visualize state transitions

**Run it:** `python diagnosis_app.py`

## Key Concepts Covered

### Probability Fundamentals
- Sample spaces and events
- Probability axioms
- Conditional probability
- Independence
- Random variables

### Bayesian Reasoning
- Bayes' theorem
- Prior and posterior probabilities
- Likelihood and evidence
- Belief updating
- Naive Bayes classifier

### Graphical Models
- Bayesian networks
- Directed acyclic graphs (DAGs)
- Conditional probability tables
- Inference algorithms
- Variable elimination

### Sequential Models
- Markov chains
- Hidden Markov Models
- Forward-backward algorithm
- Viterbi algorithm
- Sequence prediction

## Real-World Applications

1. **Medical Diagnosis**
   - Symptom-based disease prediction
   - Treatment recommendation systems
   - Patient risk assessment

2. **Spam Filtering**
   - Email classification
   - Content filtering
   - Text categorization

3. **Weather Forecasting**
   - Meteorological prediction
   - Climate modeling
   - Agricultural planning

4. **Finance**
   - Risk assessment
   - Portfolio optimization
   - Fraud detection

5. **Robotics**
   - Sensor fusion
   - Localization (where am I?)
   - Probabilistic planning

## Python Libraries Used

- **NumPy** - Numerical computations
- **SciPy** - Statistical distributions
- **pgmpy** - Probabilistic graphical models
- **NetworkX** - Graph visualization
- **Matplotlib** - Plotting and visualization
- **Gradio** - Interactive applications

## Installation

All dependencies are in the main `pyproject.toml`:

```bash
# From the root directory
pip install -e .
```

Or install specific packages:

```bash
pip install numpy scipy pgmpy networkx matplotlib gradio
```

## Exercises and Projects

### Practice Exercises
Each lab includes:
- Guided exercises with solutions
- Challenge problems
- Real-world scenarios
- Debugging exercises

### Mini Projects
1. **Disease Diagnosis System** - Build a medical reasoning system
2. **Email Spam Filter** - Implement naive Bayes classifier
3. **Weather Predictor** - Create HMM-based forecasting
4. **Game AI** - Use probability for decision-making
5. **Sensor Fusion** - Combine uncertain measurements

### Community Projects
Share your work in `community/`:
- Custom Bayesian networks
- Novel applications
- Improved algorithms
- Educational resources

## Tips for Success

1. **Build Intuition First**
   - Draw probability trees
   - Work through examples by hand
   - Use visualizations to understand concepts

2. **Start Simple**
   - Begin with small networks
   - Test with known examples
   - Gradually increase complexity

3. **Verify Results**
   - Check probabilities sum to 1
   - Test edge cases
   - Compare with analytical solutions

4. **Experiment**
   - Modify network structures
   - Try different parameters
   - Test on real data

5. **Connect to Reality**
   - Think about real applications
   - Consider where uncertainty matters
   - Design practical systems

## Common Pitfalls

- **Confusing P(A|B) with P(B|A)** - Always clarify what's given
- **Ignoring independence assumptions** - Check if variables are truly independent
- **Forgetting to normalize** - Probabilities must sum to 1
- **Overcomplicated networks** - Start simple, add complexity as needed
- **Not checking CPT consistency** - Each row of CPT should sum to 1

## Additional Resources

### Books
- "Artificial Intelligence: A Modern Approach" (Russell & Norvig) - Chapter 13-15
- "Probabilistic Graphical Models" (Koller & Friedman)
- "Pattern Recognition and Machine Learning" (Bishop)

### Online Resources
- [pgmpy Documentation](https://pgmpy.org/)
- [Khan Academy: Probability](https://www.khanacademy.org/math/probability)
- [3Blue1Brown: Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)

### Papers
- Pearl, J. (1988). "Probabilistic Reasoning in Intelligent Systems"
- Lauritzen, S. L., & Spiegelhalter, D. J. (1988). "Local computations with probabilities"

## What's Next?

After completing this week:
- **Week 4: Optimization** - Local search, genetic algorithms, constraint satisfaction
- Apply probability to learning (Week 5)
- Use uncertainty in neural networks (Week 6)
- Probabilistic language models (Week 7)

## Getting Help

- Review the theory sections in each lab
- Run the interactive Gradio app
- Check the FAQ in main documentation
- Post questions in community discussions
- Review solutions to exercises

## Assessment

You should be able to:
- [ ] Calculate conditional probabilities
- [ ] Apply Bayes' theorem to real problems
- [ ] Build and query Bayesian networks
- [ ] Implement basic Markov models
- [ ] Explain when to use probabilistic reasoning
- [ ] Design systems that handle uncertainty

---

**Time Investment:** 10-14 hours for complete mastery

**Difficulty Progression:** Beginner → Intermediate → Advanced

**Hands-on Focus:** 70% coding, 30% theory

Happy learning! Uncertainty is where AI meets the real world.
