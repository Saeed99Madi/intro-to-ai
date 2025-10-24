# Week 4: Optimization - Finding the Best Solutions

Welcome to Week 4! This week we explore optimization techniques that help AI systems find the best solutions in complex search spaces.

## Overview

Many AI problems involve finding optimal or near-optimal solutions from a vast number of possibilities. This week, you'll learn:
- Local search algorithms for large state spaces
- Constraint satisfaction techniques
- Genetic algorithms and evolutionary computation
- Real-world optimization applications

## Learning Objectives

By the end of this week, you will be able to:

1. **Apply local search algorithms** (Hill Climbing, Simulated Annealing)
2. **Solve constraint satisfaction problems** (CSP)
3. **Implement genetic algorithms** for complex optimization
4. **Choose appropriate optimization techniques** for different problems
5. **Design and tune optimization systems** for real applications

## Prerequisites

- Completion of Weeks 1-3 (Search, Knowledge, Uncertainty)
- Basic Python programming
- Understanding of search algorithms (from Week 1)
- Familiarity with probability (from Week 3)
- Basic calculus concepts (helpful but not required)

## Labs

### Lab 1: Introduction to Optimization
**Duration:** 2-3 hours | **Difficulty:** Beginner

Learn optimization fundamentals:
- What is optimization?
- Objective functions and constraints
- Global vs local optima
- Optimization landscapes
- Gradient-free vs gradient-based methods

**Key Concepts:** Objective functions, constraints, search spaces, local/global optima, optimization strategies

### Lab 2: Local Search Algorithms
**Duration:** 3-4 hours | **Difficulty:** Intermediate

Master local search techniques:
- Hill Climbing and variants
- Simulated Annealing
- Local Beam Search
- Escaping local optima
- Applications: TSP, N-Queens, scheduling

**Key Concepts:** Neighborhood search, temperature schedules, random restarts, diversification

### Lab 3: Constraint Satisfaction Problems
**Duration:** 3-4 hours | **Difficulty:** Intermediate

Solve CSPs efficiently:
- CSP formulation and representation
- Backtracking search
- Constraint propagation
- Arc consistency
- Forward checking and heuristics
- Applications: Sudoku, map coloring, scheduling

**Key Concepts:** Variables, domains, constraints, backtracking, inference, heuristics

### Lab 4: Genetic Algorithms and Evolutionary Computation
**Duration:** 3-4 hours | **Difficulty:** Advanced

Evolutionary optimization:
- Genetic algorithm fundamentals
- Representation and encoding
- Selection, crossover, mutation
- Fitness functions
- Multi-objective optimization
- Applications: Feature selection, neural architecture search

**Key Concepts:** Population, generations, genetic operators, fitness, evolution strategies

## Interactive Application

### `scheduler_app.py` - Optimization Playground

An interactive Gradio application featuring:

1. **Course Scheduler**
   - Optimize class schedules with constraints
   - Minimize conflicts and gaps
   - Interactive constraint editing
   - Real-time optimization

2. **Traveling Salesman Problem Solver**
   - Compare different optimization algorithms
   - Visualize solution evolution
   - Adjust algorithm parameters
   - Performance comparison

3. **N-Queens Puzzle Solver**
   - Multiple solution strategies
   - Step-by-step visualization
   - Performance metrics
   - Scalability testing

**Run it:** `python scheduler_app.py`

## Key Concepts Covered

### Local Search
- Hill Climbing
- Simulated Annealing
- Tabu Search
- Random restarts
- Local beam search

### Constraint Satisfaction
- Backtracking
- Forward checking
- Arc consistency
- Variable ordering heuristics
- Value ordering heuristics
- Constraint propagation

### Evolutionary Algorithms
- Genetic algorithms
- Genetic programming
- Evolution strategies
- Differential evolution
- Multi-objective optimization

### Optimization Problems
- Traveling Salesman Problem (TSP)
- N-Queens problem
- Knapsack problem
- Scheduling problems
- Resource allocation

## Real-World Applications

1. **Scheduling**
   - Course timetabling
   - Employee shift scheduling
   - Project planning
   - Resource allocation

2. **Route Optimization**
   - Vehicle routing
   - Delivery optimization
   - Network design
   - Path planning

3. **Resource Allocation**
   - Budget optimization
   - Portfolio management
   - Task assignment
   - Load balancing

4. **Design Optimization**
   - Engineering design
   - Circuit design
   - Architecture optimization
   - Hyperparameter tuning

5. **Operations Research**
   - Supply chain optimization
   - Inventory management
   - Production scheduling
   - Facility location

## Python Libraries Used

- **NumPy** - Numerical computations
- **SciPy** - Optimization algorithms
- **Matplotlib** - Visualization
- **NetworkX** - Graph problems
- **DEAP** - Evolutionary algorithms
- **python-constraint** - CSP solving
- **Gradio** - Interactive applications

## Installation

All dependencies are in the main `pyproject.toml`:

```bash
# From the root directory
pip install -e .
```

Or install specific packages:

```bash
pip install numpy scipy matplotlib networkx gradio deap python-constraint
```

## Exercises and Projects

### Practice Exercises
Each lab includes:
- Guided implementations
- Algorithm comparisons
- Parameter tuning challenges
- Real-world problem solving

### Mini Projects
1. **Smart Scheduler** - Build an intelligent class scheduler
2. **Route Optimizer** - Solve delivery routing problems
3. **Puzzle Solver** - Implement constraint-based puzzle solving
4. **Feature Selector** - Use GA for ML feature selection
5. **Game AI** - Evolve game-playing strategies

### Community Projects
Share your work in `community/`:
- Novel optimization problems
- Algorithm improvements
- Hybrid approaches
- Performance comparisons

## Tips for Success

1. **Understand the Problem**
   - Define objective clearly
   - Identify all constraints
   - Consider solution representation

2. **Start Simple**
   - Test on small instances
   - Validate correctness first
   - Then optimize for performance

3. **Tune Parameters**
   - Temperature schedules matter
   - Population size affects quality
   - Balance exploration vs exploitation

4. **Visualize**
   - Plot objective function evolution
   - Watch solution improvements
   - Understand convergence behavior

5. **Compare Algorithms**
   - No single best algorithm
   - Different problems need different approaches
   - Benchmark systematically

## Common Pitfalls

- **Premature convergence** - Population loses diversity too quickly
- **Poor representation** - Encoding doesn't match problem structure
- **Wrong objective function** - Optimizing the wrong metric
- **Ignoring constraints** - Feasibility is critical
- **Inadequate exploration** - Getting stuck in local optima
- **Over-tuning** - Parameters specific to one instance

## Algorithm Selection Guide

### When to Use Hill Climbing
- Quick local improvements needed
- Simple problems with clear gradient
- Starting point is good
- Real-time optimization

### When to Use Simulated Annealing
- Need to escape local optima
- Have time budget
- Solution quality matters more than speed
- Discrete optimization

### When to Use CSP Techniques
- Well-defined constraints
- Satisfiability is key
- Combinatorial problems
- Need all solutions or proof of infeasibility

### When to Use Genetic Algorithms
- Complex fitness landscape
- No gradient information
- Multiple objectives
- Need diverse solutions
- Black-box optimization

## Additional Resources

### Books
- "Artificial Intelligence: A Modern Approach" (Russell & Norvig) - Chapters 4, 6
- "Introduction to Evolutionary Computing" (Eiben & Smith)
- "Constraint Processing" (Dechter)

### Online Resources
- [SciPy Optimization](https://docs.scipy.org/doc/scipy/reference/optimize.html)
- [DEAP Documentation](https://deap.readthedocs.io/)
- [OR-Tools by Google](https://developers.google.com/optimization)

### Papers
- Kirkpatrick et al. (1983) - "Optimization by Simulated Annealing"
- Holland (1975) - "Adaptation in Natural and Artificial Systems"
- Mackworth (1977) - "Consistency in Networks of Relations"

## What's Next?

After completing this week:
- **Week 5: Learning** - Machine learning fundamentals
- Apply optimization to learning algorithms
- Use evolutionary algorithms for AutoML
- Optimize neural network architectures

## Getting Help

- Review theory sections in each lab
- Run the interactive Gradio app
- Check the FAQ in main documentation
- Post questions in community discussions
- Compare your solutions with provided ones

## Assessment

You should be able to:
- [ ] Implement hill climbing and simulated annealing
- [ ] Formulate and solve constraint satisfaction problems
- [ ] Design and implement genetic algorithms
- [ ] Choose appropriate optimization techniques
- [ ] Tune algorithm parameters effectively
- [ ] Apply optimization to real-world problems

## Performance Benchmarks

### Expected Performance (N-Queens)
- 8-Queens: < 1 second
- 16-Queens: < 10 seconds
- 32-Queens: < 1 minute (with good heuristics)

### TSP Benchmark (50 cities)
- Greedy: ~30% worse than optimal
- Hill Climbing: ~20% worse than optimal
- Simulated Annealing: ~10% worse than optimal
- Genetic Algorithm: ~5-15% worse than optimal

### Sudoku Solving
- Easy puzzles: < 0.1 seconds
- Medium puzzles: < 1 second
- Hard puzzles: < 10 seconds
- Expert puzzles: < 1 minute

---

**Time Investment:** 12-16 hours for complete mastery

**Difficulty Progression:** Beginner → Intermediate → Advanced

**Hands-on Focus:** 70% implementation, 30% theory

Happy optimizing! Finding the best solution is what AI is all about.
