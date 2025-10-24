# Week 1: Search

## Overview

Search is one of the most fundamental problems in AI. This week, you'll learn how AI agents find solutions by exploring different possibilities systematically.

## Learning Objectives

By the end of this week, you will:
- Understand how to formulate problems as search problems
- Implement uninformed search algorithms (DFS, BFS)
- Implement informed search algorithms (Greedy, A*)
- Understand the trade-offs between different search strategies
- Apply search algorithms to real-world problems

## Lab Notebooks

### Lab 1: Introduction to Search
**File:** `1_lab1.ipynb`

- What is search in AI?
- Problem formulation (states, actions, goals)
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- Comparing search strategies

**Hands-on:**
- Implement DFS and BFS from scratch
- Solve maze navigation problems
- Visualize search processes

### Lab 2: Informed Search
**File:** `2_lab2.ipynb`

- Uniform Cost Search
- Greedy Best-First Search
- A* Search algorithm
- Heuristics and admissibility

**Hands-on:**
- Implement A* from scratch
- Design heuristic functions
- Compare performance with uninformed search

### Lab 3: Adversarial Search
**File:** `3_lab3.ipynb`

- Game playing AI
- Minimax algorithm
- Alpha-Beta pruning
- Evaluation functions

**Hands-on:**
- Build a Tic-Tac-Toe AI
- Implement Minimax with alpha-beta pruning
- Create evaluation functions

### Lab 4: Advanced Search & Applications
**File:** `4_lab4.ipynb`

- Search in continuous spaces
- Local search algorithms
- Real-world applications
- Using NetworkX library

**Hands-on:**
- Route planning with real maps
- Graph search applications
- Performance optimization

## Interactive Application

### Pathfinding Visualizer
**File:** `pathfinding_app.py`

An interactive Gradio web application where you can:
- Generate random mazes
- Choose between DFS, BFS, and A* algorithms
- Visualize the pathfinding process
- Compare algorithm performance

**To run:**
```bash
python pathfinding_app.py
```

Then open your browser to the URL shown in the terminal (typically http://localhost:7860)

## Key Concepts

### Search Problem Components
1. **Initial State**: Where we start
2. **Actions**: What we can do
3. **Transition Model**: Results of actions
4. **Goal Test**: Are we done?
5. **Path Cost**: Cost of solution (optional)

### Search Algorithm Properties
- **Completeness**: Will it find a solution if one exists?
- **Optimality**: Will it find the best solution?
- **Time Complexity**: How long does it take?
- **Space Complexity**: How much memory does it use?

### Uninformed Search
- **DFS**: Goes deep, uses stack, memory efficient
- **BFS**: Goes wide, uses queue, finds shortest path

### Informed Search
- **Greedy**: Uses heuristic, fast but not optimal
- **A***: Combines cost and heuristic, optimal with good heuristic

## Real-World Applications

- **GPS Navigation**: Finding routes between locations
- **Game AI**: Chess, Go, video games
- **Robotics**: Path planning for robots
- **Puzzles**: Solving Sudoku, Rubik's cube, etc.
- **Web Crawling**: Exploring website structures
- **Network Routing**: Finding paths in computer networks

## Prerequisites

- Basic Python (functions, loops, conditionals)
- Basic data structures (lists, sets, dictionaries)
- Understanding of stacks and queues (will be reviewed)

## Additional Resources

### Recommended Reading
- Russell & Norvig, "Artificial Intelligence: A Modern Approach" - Chapter 3
- [Wikipedia: Search Algorithms](https://en.wikipedia.org/wiki/Search_algorithm)

### Videos
- [Search Algorithms Visualized](https://www.youtube.com/watch?v=19h1g22hby8)
- [A* Pathfinding Explained](https://www.youtube.com/watch?v=-L-WgKMFuhE)

### Interactive Tools
- [PathFinding.js](https://qiao.github.io/PathFinding.js/visual/)
- [Algorithm Visualizer](https://algorithm-visualizer.org/)

## Exercises & Challenges

### Basic Exercises (Complete these first)
1. Implement BFS and DFS for tree traversal
2. Solve the 8-puzzle problem using A*
3. Create custom heuristic functions
4. Compare algorithm performance on different maze sizes

### Intermediate Challenges
1. Implement Dijkstra's algorithm
2. Solve the traveling salesman problem (small instances)
3. Add diagonal movement to maze solving
4. Implement iterative deepening DFS

### Advanced Projects
1. Build a Pac-Man AI using search
2. Implement MCTS (Monte Carlo Tree Search)
3. Create a route planner using real map data
4. Build a Sudoku solver using constraint satisfaction

## Common Issues & Tips

### Issue: Stack overflow with DFS
**Solution**: Use iterative version with explicit stack instead of recursion

### Issue: BFS too slow on large mazes
**Solution**: Try A* with Manhattan distance heuristic

### Issue: A* not finding optimal path
**Solution**: Check if your heuristic is admissible (never overestimates)

### Tip: Visualize!
Always visualize your search process to understand what's happening

### Tip: Start Simple
Test algorithms on small problems before scaling up

## Assessment

After completing this week, you should be able to:
- [ ] Explain the difference between informed and uninformed search
- [ ] Implement BFS, DFS, and A* from scratch
- [ ] Choose appropriate search algorithms for different problems
- [ ] Design and evaluate heuristic functions
- [ ] Apply search algorithms to real-world problems

## Next Week Preview

**Week 2: Knowledge** - Learn how to represent and reason with logical knowledge, building expert systems that can make inferences and solve logic puzzles.

---

**Questions?** Open an issue on GitHub or check the [FAQ](../docs/FAQ.md)

Happy searching! 🎯
