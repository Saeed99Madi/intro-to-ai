"""
Optimization Playground - Interactive Gradio Application
Week 4: Optimization

Features:
1. TSP Solver with multiple algorithms
2. N-Queens Puzzle Solver
3. Sudoku Solver with CSP
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict
import io
from PIL import Image

# Set random seed
np.random.seed(42)
random.seed(42)


# ============================================================================
# Part 1: TSP Solver
# ============================================================================

class TSP:
    """Traveling Salesman Problem solver."""

    def __init__(self, cities: np.ndarray):
        self.cities = cities
        self.n_cities = len(cities)

        # Precompute distance matrix
        self.distances = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                self.distances[i, j] = np.linalg.norm(cities[i] - cities[j])

    def tour_length(self, tour: List[int]) -> float:
        """Calculate tour length."""
        length = 0
        for i in range(len(tour)):
            length += self.distances[tour[i], tour[(i + 1) % len(tour)]]
        return length

    def random_tour(self) -> List[int]:
        """Generate random tour."""
        tour = list(range(self.n_cities))
        random.shuffle(tour)
        return tour

    def simulated_annealing(self, max_iterations: int = 5000) -> Tuple[List[int], List[float]]:
        """Solve using simulated annealing."""
        current_tour = self.random_tour()
        current_length = self.tour_length(current_tour)

        best_tour = current_tour[:]
        best_length = current_length

        temperature = 100.0
        cooling_rate = 0.995
        history = [best_length]

        for _ in range(max_iterations):
            # Generate neighbor (2-opt)
            i, j = sorted(random.sample(range(self.n_cities), 2))
            new_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
            new_length = self.tour_length(new_tour)

            # Accept or reject
            delta = new_length - current_length
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_tour = new_tour
                current_length = new_length

                if current_length < best_length:
                    best_tour = current_tour[:]
                    best_length = current_length

            temperature *= cooling_rate
            history.append(best_length)

        return best_tour, history


def solve_tsp(n_cities: int, algorithm: str) -> Tuple[str, Image.Image]:
    """Solve TSP and visualize."""

    # Generate random cities
    np.random.seed(42)
    cities = np.random.rand(n_cities, 2) * 100

    tsp = TSP(cities)

    # Solve
    if algorithm == "Simulated Annealing":
        best_tour, history = tsp.simulated_annealing(max_iterations=5000)
    else:
        # Random baseline
        best_tour = tsp.random_tour()
        history = [tsp.tour_length(best_tour)]

    best_length = tsp.tour_length(best_tour)

    # Create result text
    result = f"**{algorithm} Results**\n\n"
    result += f"**Number of cities:** {n_cities}\n"
    result += f"**Best tour length:** {best_length:.2f}\n"
    result += f"**Iterations:** {len(history)}\n\n"

    if algorithm == "Simulated Annealing":
        initial_length = history[0]
        improvement = (initial_length - best_length) / initial_length * 100
        result += f"**Improvement:** {improvement:.1f}%\n"

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot tour
    tour_coords = np.array([cities[i] for i in best_tour + [best_tour[0]]])
    ax1.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', linewidth=2, alpha=0.6)
    ax1.scatter(cities[:, 0], cities[:, 1], s=200, c='red',
               zorder=5, edgecolors='black', linewidths=2)

    # Mark start city
    ax1.scatter(cities[best_tour[0]][0], cities[best_tour[0]][1],
               s=400, c='green', marker='*', zorder=6, edgecolors='black', linewidths=2,
               label='Start')

    ax1.set_xlabel('X', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Y', fontweight='bold', fontsize=12)
    ax1.set_title(f'Best Tour (Length: {best_length:.2f})',
                 fontweight='bold', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.axis('equal')

    # Plot convergence
    if len(history) > 1:
        ax2.plot(history, linewidth=2, color='blue')
        ax2.set_xlabel('Iteration', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Tour Length', fontweight='bold', fontsize=12)
        ax2.set_title('Convergence', fontweight='bold', fontsize=13)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Random Tour\n(No Optimization)',
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return result, Image.open(buf)


# ============================================================================
# Part 2: N-Queens Solver
# ============================================================================

class NQueens:
    """N-Queens puzzle solver."""

    def __init__(self, n: int):
        self.n = n

    def conflicts(self, state: List[int]) -> int:
        """Count attacking pairs."""
        conflicts = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Same column or diagonal
                if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                    conflicts += 1
        return conflicts

    def random_state(self) -> List[int]:
        """Generate random state."""
        return [random.randint(0, self.n - 1) for _ in range(self.n)]

    def solve_simulated_annealing(self) -> Tuple[List[int], List[int]]:
        """Solve using simulated annealing."""
        current = self.random_state()
        current_conflicts = self.conflicts(current)

        best = current[:]
        best_conflicts = current_conflicts

        temperature = 100.0
        cooling_rate = 0.99
        history = [best_conflicts]

        for _ in range(5000):
            if current_conflicts == 0:
                break

            # Generate neighbor
            neighbor = current[:]
            row = random.randint(0, self.n - 1)
            col = random.randint(0, self.n - 1)
            neighbor[row] = col

            neighbor_conflicts = self.conflicts(neighbor)
            delta = neighbor_conflicts - current_conflicts

            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current = neighbor
                current_conflicts = neighbor_conflicts

                if current_conflicts < best_conflicts:
                    best = current[:]
                    best_conflicts = current_conflicts

            temperature *= cooling_rate
            history.append(best_conflicts)

        return best, history


def solve_n_queens(n: int) -> Tuple[str, Image.Image]:
    """Solve N-Queens puzzle."""

    nqueens = NQueens(n)
    solution, history = nqueens.solve_simulated_annealing()
    conflicts = nqueens.conflicts(solution)

    # Result text
    result = f"**{n}-Queens Puzzle Solution**\n\n"
    result += f"**Conflicts:** {conflicts}\n"
    result += f"**Iterations:** {len(history)}\n\n"

    if conflicts == 0:
        result += "✅ **Solution found!** All queens are safe.\n"
    else:
        result += f"⚠️ **Partial solution** with {conflicts} conflicts.\n"
        result += "Try running again or increase iterations.\n"

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Draw board
    for i in range(n):
        for j in range(n):
            color = 'wheat' if (i + j) % 2 == 0 else 'tan'
            ax1.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1,
                                       facecolor=color, edgecolor='black'))

    # Draw queens
    for row, col in enumerate(solution):
        ax1.text(col + 0.5, n - 1 - row + 0.5, '♛',
                fontsize=max(300/n, 20), ha='center', va='center')

    ax1.set_xlim(0, n)
    ax1.set_ylim(0, n)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f'{n}-Queens Board (Conflicts: {conflicts})',
                 fontweight='bold', fontsize=13)

    # Plot convergence
    ax2.plot(history, linewidth=2, color='blue')
    ax2.axhline(y=0, color='r', linestyle='--', label='Goal', linewidth=2)
    ax2.set_xlabel('Iteration', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Conflicts', fontweight='bold', fontsize=12)
    ax2.set_title('Solution Progress', fontweight='bold', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return result, Image.open(buf)


# ============================================================================
# Part 3: Sudoku Solver (Simple CSP)
# ============================================================================

class SimpleSudoku:
    """Simple Sudoku solver using backtracking."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid.copy()
        self.size = 9

    def is_valid(self, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in self.grid[row]:
            return False

        # Check column
        if num in self.grid[:, col]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in self.grid[box_row:box_row+3, box_col:box_col+3]:
            return False

        return True

    def solve(self) -> bool:
        """Solve using backtracking."""
        # Find empty cell
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:
                    # Try numbers 1-9
                    for num in range(1, 10):
                        if self.is_valid(i, j, num):
                            self.grid[i, j] = num

                            if self.solve():
                                return True

                            self.grid[i, j] = 0

                    return False

        return True


def solve_sudoku(puzzle_difficulty: str) -> Tuple[str, Image.Image, Image.Image]:
    """Solve Sudoku puzzle."""

    # Predefined puzzles
    puzzles = {
        "Easy": np.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]),
        "Medium": np.array([
            [0, 0, 0, 6, 0, 0, 4, 0, 0],
            [7, 0, 0, 0, 0, 3, 6, 0, 0],
            [0, 0, 0, 0, 9, 1, 0, 8, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 1, 8, 0, 0, 0, 3],
            [0, 0, 0, 3, 0, 6, 0, 4, 5],
            [0, 4, 0, 2, 0, 0, 0, 6, 0],
            [9, 0, 3, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 1, 0, 0]
        ])
    }

    puzzle = puzzles[puzzle_difficulty].copy()

    # Visualize puzzle
    def visualize_grid(grid, title, original_grid=None):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw grid
        for i in range(10):
            linewidth = 3 if i % 3 == 0 else 1
            ax.axhline(i, color='black', linewidth=linewidth)
            ax.axvline(i, color='black', linewidth=linewidth)

        # Fill numbers
        for i in range(9):
            for j in range(9):
                if grid[i, j] != 0:
                    # Check if original (given) number
                    if original_grid is not None and original_grid[i, j] != 0:
                        color = 'black'
                        weight = 'bold'
                    else:
                        color = 'blue'
                        weight = 'normal'

                    ax.text(j + 0.5, 8.5 - i, str(int(grid[i, j])),
                           ha='center', va='center', fontsize=20,
                           color=color, fontweight=weight)

        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return Image.open(buf)

    puzzle_img = visualize_grid(puzzle, f"Sudoku Puzzle ({puzzle_difficulty})")

    # Solve
    solver = SimpleSudoku(puzzle)
    solved = solver.solve()

    if solved:
        result = f"**✅ Sudoku Solved Successfully!**\n\n"
        result += f"**Difficulty:** {puzzle_difficulty}\n"
        result += f"**Algorithm:** Backtracking with Constraint Propagation\n\n"
        result += "**How it works:**\n"
        result += "- Systematically tries values 1-9 for empty cells\n"
        result += "- Checks row, column, and 3×3 box constraints\n"
        result += "- Backtracks when conflicts occur\n"

        solution_img = visualize_grid(solver.grid, "Sudoku Solution", puzzle)
    else:
        result = "**❌ Could not solve puzzle**\n\nThis shouldn't happen with valid puzzles!"
        solution_img = puzzle_img

    return result, puzzle_img, solution_img


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Optimization Playground", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎯 Optimization Playground
    ## Week 4: Optimization Algorithms

    Explore different optimization techniques through interactive demos!
    """)

    with gr.Tabs():

        # Tab 1: TSP Solver
        with gr.Tab("🗺️ Traveling Salesman Problem"):
            gr.Markdown("""
            ### Traveling Salesman Problem (TSP)

            Find the shortest route visiting all cities exactly once.

            **Algorithms**:
            - **Simulated Annealing**: Accepts worse solutions probabilistically to escape local optima
            - **Random**: Baseline comparison
            """)

            with gr.Row():
                with gr.Column():
                    tsp_cities = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=15,
                        step=1,
                        label="Number of Cities"
                    )
                    tsp_algorithm = gr.Dropdown(
                        choices=["Simulated Annealing", "Random"],
                        value="Simulated Annealing",
                        label="Algorithm"
                    )
                    tsp_solve_btn = gr.Button("🚀 Solve TSP", variant="primary")

                with gr.Column():
                    tsp_result = gr.Markdown()
                    tsp_plot = gr.Image(label="TSP Solution")

            tsp_solve_btn.click(
                fn=solve_tsp,
                inputs=[tsp_cities, tsp_algorithm],
                outputs=[tsp_result, tsp_plot]
            )

            gr.Examples(
                examples=[[10, "Simulated Annealing"],
                         [20, "Simulated Annealing"],
                         [15, "Random"]],
                inputs=[tsp_cities, tsp_algorithm],
                label="Try These Examples"
            )

        # Tab 2: N-Queens
        with gr.Tab("♛ N-Queens Puzzle"):
            gr.Markdown("""
            ### N-Queens Puzzle

            Place N queens on an N×N chessboard so no queen attacks another.

            **Constraints**:
            - No two queens in same row, column, or diagonal

            **Algorithm**: Simulated Annealing
            """)

            with gr.Row():
                with gr.Column():
                    nqueens_n = gr.Slider(
                        minimum=4,
                        maximum=20,
                        value=8,
                        step=1,
                        label="Board Size (N×N)"
                    )
                    nqueens_solve_btn = gr.Button("🎯 Solve N-Queens", variant="primary")

                with gr.Column():
                    nqueens_result = gr.Markdown()
                    nqueens_plot = gr.Image(label="N-Queens Solution")

            nqueens_solve_btn.click(
                fn=solve_n_queens,
                inputs=nqueens_n,
                outputs=[nqueens_result, nqueens_plot]
            )

            gr.Examples(
                examples=[[4], [8], [12], [16]],
                inputs=nqueens_n,
                label="Try Different Board Sizes"
            )

        # Tab 3: Sudoku
        with gr.Tab("🔢 Sudoku Solver"):
            gr.Markdown("""
            ### Sudoku Puzzle Solver

            Solve Sudoku puzzles using Constraint Satisfaction Problem (CSP) techniques.

            **Algorithm**: Backtracking with constraint propagation
            - Systematically assigns values 1-9
            - Checks row, column, and box constraints
            - Backtracks when conflicts occur
            """)

            with gr.Row():
                with gr.Column():
                    sudoku_difficulty = gr.Radio(
                        choices=["Easy", "Medium"],
                        value="Easy",
                        label="Puzzle Difficulty"
                    )
                    sudoku_solve_btn = gr.Button("🧩 Solve Sudoku", variant="primary")

                with gr.Column():
                    sudoku_result = gr.Markdown()

            with gr.Row():
                sudoku_puzzle = gr.Image(label="Puzzle")
                sudoku_solution = gr.Image(label="Solution")

            sudoku_solve_btn.click(
                fn=solve_sudoku,
                inputs=sudoku_difficulty,
                outputs=[sudoku_result, sudoku_puzzle, sudoku_solution]
            )

    gr.Markdown("""
    ---
    ### 📚 Learn More

    - Complete Week 4 lab notebooks for detailed implementations
    - Experiment with different parameters
    - Try creating your own optimization problems!

    **Key Concepts**:
    - **Local Search**: Hill climbing, simulated annealing
    - **CSP**: Backtracking, constraint propagation
    - **Evolutionary**: Genetic algorithms, evolution strategies
    """)


if __name__ == "__main__":
    demo.launch(share=False)
