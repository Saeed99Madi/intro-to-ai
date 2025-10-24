# Week 2: Knowledge Representation and Reasoning

## Overview

How do we represent what we know? How do we reason with that knowledge? This week explores how AI systems use logic to represent facts, draw conclusions, and solve problems that require reasoning.

## Learning Objectives

By the end of this week, you will:
- Understand propositional and first-order logic
- Represent knowledge using logical statements
- Implement inference algorithms
- Build rule-based expert systems
- Solve logic puzzles programmatically
- Apply knowledge representation to real-world problems

## Lab Notebooks

### Lab 1: Introduction to Logic and Knowledge
**File:** `1_lab1.ipynb`

- What is knowledge representation?
- Propositional logic basics
- Logical connectives (AND, OR, NOT, IMPLIES)
- Truth tables and logical equivalence
- Knowledge bases

**Hands-on:**
- Represent facts in propositional logic
- Evaluate logical expressions
- Build simple knowledge bases
- Solve logic puzzles

### Lab 2: Propositional Logic and Inference
**File:** `2_lab2.ipynb`

- Inference rules (Modus Ponens, Resolution)
- Model checking
- Resolution algorithm
- Horn clauses
- Forward and backward chaining

**Hands-on:**
- Implement inference from scratch
- Build an inference engine
- Solve "Knights and Knaves" puzzles
- Create a simple reasoning system

### Lab 3: First-Order Logic
**File:** `3_lab3.ipynb`

- Predicates, variables, and quantifiers
- Universal and existential quantification
- Unification algorithm
- Forward chaining with first-order logic
- Prolog-style reasoning

**Hands-on:**
- Represent complex relationships
- Implement unification
- Build family tree reasoner
- Query knowledge bases

### Lab 4: Knowledge-Based Systems and Applications
**File:** `4_lab4.ipynb`

- Expert systems architecture
- Rule-based systems
- Semantic networks
- Ontologies and knowledge graphs
- Using SymPy for symbolic reasoning

**Hands-on:**
- Build a medical diagnosis expert system
- Create knowledge graphs
- Use SymPy for logical reasoning
- Real-world applications

## Interactive Application

### Logic Puzzle Solver
**File:** `logic_solver.py`

An interactive Gradio web application where you can:
- Solve classic logic puzzles (Knights and Knaves, Zebra puzzle)
- Visualize knowledge bases
- See inference steps
- Create custom logic problems

**To run:**
```bash
python logic_solver.py
```

## Key Concepts

### Propositional Logic
- **Propositions**: Statements that are true or false
- **Connectives**: AND (∧), OR (∨), NOT (¬), IMPLIES (→), IFF (↔)
- **Inference**: Deriving new facts from known facts
- **Satisfiability**: Finding truth assignments that make formulas true

### First-Order Logic
- **Predicates**: Properties and relations
- **Quantifiers**: ∀ (for all), ∃ (there exists)
- **Terms**: Objects, variables, functions
- **Unification**: Matching patterns to facts

### Knowledge Representation
- **Facts**: Things that are true
- **Rules**: If-then relationships
- **Queries**: Questions to answer
- **Inference**: Deriving answers from facts and rules

## Real-World Applications

- **Expert Systems**: Medical diagnosis, troubleshooting, recommendations
- **Semantic Web**: Knowledge graphs (Google, Amazon)
- **Natural Language Understanding**: Parsing meaning from text
- **Robotics**: Planning and reasoning about actions
- **Legal Reasoning**: Analyzing contracts and regulations
- **Database Query Optimization**: Query planning

## Prerequisites

- Week 1: Search (helpful but not required)
- Basic Python (functions, classes, loops)
- High school logic (understanding of "if-then" statements)

## Tools and Libraries

- **SymPy**: Symbolic mathematics and logic
- **itertools**: Combinations and permutations
- **NetworkX**: Visualizing knowledge graphs
- **Gradio**: Interactive applications

## Classic Problems

### Knights and Knaves
On an island, knights always tell the truth and knaves always lie. Can you determine who is who?

### The Zebra Puzzle
Five people live in five houses. Given clues about their nationalities, drinks, pets, etc., who owns the zebra?

### Wumpus World
Navigate a cave with pits and a monster using logical reasoning about what you sense.

## Example: Simple Knowledge Base

```python
# Facts
knowledge_base = [
    "Socrates is a man",
    "All men are mortal"
]

# Query
query = "Is Socrates mortal?"

# Inference
# From "Socrates is a man" and "All men are mortal"
# We can infer: "Socrates is mortal"
answer = True
```

## Common Logic Puzzles

### Puzzle 1: The Liar Paradox
A says: "B is a knight"
B says: "A and I are of opposite types"

Who is the knight and who is the knave?

### Puzzle 2: Three Gods
Three gods (Truth, False, Random) answer yes/no questions. You can ask 3 questions to identify them.

### Puzzle 3: The Muddy Children
N children are told "at least one of you has mud on your forehead." When will they figure out their own status?

## Assessment

After completing this week, you should be able to:
- [ ] Represent knowledge using propositional logic
- [ ] Implement basic inference algorithms
- [ ] Solve logic puzzles programmatically
- [ ] Use first-order logic for complex relationships
- [ ] Build simple expert systems
- [ ] Apply knowledge representation to new problems

## Resources

### Books
- Russell & Norvig: "Artificial Intelligence: A Modern Approach" (Chapters 6-9)
- Smullyan: "What Is the Name of This Book?" (Logic puzzles)
- Nilsson: "Artificial Intelligence: A New Synthesis" (Chapter 15)

### Online
- [Stanford Logic Course](http://logic.stanford.edu/)
- [SymPy Logic Documentation](https://docs.sympy.org/latest/modules/logic.html)
- [Prolog Tutorial](https://www.cpp.edu/~jrfisher/www/prolog_tutorial/contents.html)

### Interactive
- [Logic Puzzle Solver](https://www.logic-puzzles.org/)
- [Knights and Knaves Generator](https://philosophy.hku.hk/think/logic/knights.php)

## Tips for Success

### Understanding Logic
- Draw truth tables to understand connectives
- Practice translating English to logical statements
- Verify your reasoning with examples

### Debugging Logic
- Check your premises (are facts correct?)
- Trace inference steps manually
- Test with simple cases first

### Building Systems
- Start with a small knowledge base
- Add rules incrementally
- Test each rule individually

## Exercises & Challenges

### Basic Exercises
1. Create truth tables for all logical connectives
2. Translate natural language statements to logic
3. Solve 5 Knights and Knaves puzzles
4. Implement Modus Ponens inference rule

### Intermediate Challenges
1. Build a family tree reasoner with first-order logic
2. Solve the Zebra puzzle programmatically
3. Create an expert system for a domain of your choice
4. Implement the DPLL SAT solver algorithm

### Advanced Projects
1. Build a Sudoku solver using logical constraints
2. Create a natural language to logic translator
3. Implement a Prolog-like interpreter
4. Build a knowledge graph for a real domain

## Common Issues & Solutions

### Issue: Logical statements seem ambiguous
**Solution**: Be explicit! "All birds fly" should be "For all X, if X is a bird, then X flies"

### Issue: Inference taking too long
**Solution**: Use Horn clauses and forward chaining for efficiency

### Issue: Can't represent certain statements
**Solution**: You might need first-order logic instead of propositional logic

### Issue: Getting circular reasoning
**Solution**: Check for cycles in your rules, use depth limits

## Next Week Preview

**Week 3: Uncertainty** - Learn how to reason with probabilities, Bayes' theorem, and Bayesian networks when information is uncertain.

---

**Questions?** Check the [FAQ](../docs/FAQ.md) or open an issue on GitHub

Happy reasoning! 🧠✨
