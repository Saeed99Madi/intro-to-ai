"""
Interactive Pathfinding Visualizer
Week 1: Search

This Gradio app allows you to:
- Draw your own maze
- Choose between DFS, BFS, and A* algorithms
- Visualize the pathfinding process
"""

import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple, Optional
import io
from PIL import Image


def get_neighbors(maze: np.ndarray, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Get valid neighboring positions."""
    rows, cols = maze.shape
    row, col = pos
    neighbors = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if (0 <= new_row < rows and
            0 <= new_col < cols and
            maze[new_row, new_col] == 0):
            neighbors.append((new_row, new_col))

    return neighbors


def bfs(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """Breadth-First Search."""
    queue = deque([(start, [start])])
    visited = set([start])
    nodes_explored = 0

    while queue:
        current, path = queue.popleft()
        nodes_explored += 1

        if current == goal:
            return path, nodes_explored

        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None, nodes_explored


def dfs(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """Depth-First Search."""
    stack = [(start, [start])]
    visited = set([start])
    nodes_explored = 0

    while stack:
        current, path = stack.pop()
        nodes_explored += 1

        if current == goal:
            return path, nodes_explored

        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))

    return None, nodes_explored


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance heuristic."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def a_star(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* Search with Manhattan distance heuristic."""
    from heapq import heappush, heappop

    # Priority queue: (f_score, counter, current, path, g_score)
    counter = 0
    pq = [(manhattan_distance(start, goal), counter, start, [start], 0)]
    visited = set()
    nodes_explored = 0

    while pq:
        f, _, current, path, g = heappop(pq)
        nodes_explored += 1

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path, nodes_explored

        for neighbor in get_neighbors(maze, current):
            if neighbor not in visited:
                new_g = g + 1
                new_f = new_g + manhattan_distance(neighbor, goal)
                counter += 1
                heappush(pq, (new_f, counter, neighbor, path + [neighbor], new_g))

    return None, nodes_explored


def visualize_path(maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                   path: Optional[List[Tuple[int, int]]] = None):
    """Create visualization of maze and path."""
    fig, ax = plt.subplots(figsize=(10, 10))

    display = np.copy(maze).astype(float)

    # Color the path if it exists
    if path:
        for pos in path[1:-1]:
            display[pos] = 0.5

    # Special colors for start and goal
    display[start] = 0.3
    display[goal] = 0.7

    im = ax.imshow(display, cmap='RdYlGn_r', interpolation='nearest')

    # Add grid
    for i in range(maze.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1)
    for j in range(maze.shape[1] + 1):
        ax.axvline(j - 0.5, color='black', linewidth=1)

    # Add labels
    ax.text(start[1], start[0], 'S', ha='center', va='center',
            fontsize=16, fontweight='bold', color='blue')
    ax.text(goal[1], goal[0], 'G', ha='center', va='center',
            fontsize=16, fontweight='bold', color='green')

    # Draw path with arrows
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_title('Maze Pathfinding (S=Start, G=Goal, Red=Path)', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


def solve_maze(maze_size: int, wall_density: float, algorithm: str, random_seed: int):
    """Generate a random maze and solve it with the selected algorithm."""

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Generate random maze
    maze = (np.random.random((maze_size, maze_size)) < wall_density).astype(int)

    # Ensure start and goal are open
    start = (0, 0)
    goal = (maze_size - 1, maze_size - 1)
    maze[start] = 0
    maze[goal] = 0

    # Select algorithm
    if algorithm == "BFS (Breadth-First Search)":
        result = bfs(maze, start, goal)
    elif algorithm == "DFS (Depth-First Search)":
        result = dfs(maze, start, goal)
    else:  # A* Search
        result = a_star(maze, start, goal)

    path, nodes_explored = result if result[0] is not None else (None, result[1])

    # Create visualization
    img = visualize_path(maze, start, goal, path)

    # Create result message
    if path:
        message = f"""
        ✅ **Path Found!**

        - **Algorithm**: {algorithm}
        - **Path Length**: {len(path)} steps
        - **Nodes Explored**: {nodes_explored}
        - **Maze Size**: {maze_size}x{maze_size}
        - **Wall Density**: {wall_density:.0%}
        """
    else:
        message = f"""
        ❌ **No Path Found**

        - **Algorithm**: {algorithm}
        - **Nodes Explored**: {nodes_explored}
        - **Maze Size**: {maze_size}x{maze_size}
        - **Wall Density**: {wall_density:.0%}

        Try reducing wall density or changing the random seed.
        """

    return img, message


# Create Gradio interface
with gr.Blocks(title="Pathfinding Visualizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 Interactive Pathfinding Visualizer

    ## Week 1: Search Algorithms

    Explore different search algorithms by generating random mazes and watching them find paths!

    **Algorithms:**
    - **BFS (Breadth-First Search)**: Guarantees shortest path, explores layer by layer
    - **DFS (Depth-First Search)**: Memory efficient, explores deeply first
    - **A\* Search**: Uses heuristics to find optimal path efficiently

    **Controls:**
    - Adjust maze size and wall density
    - Choose your algorithm
    - Change random seed for different mazes
    """)

    with gr.Row():
        with gr.Column():
            maze_size = gr.Slider(
                minimum=5, maximum=30, value=15, step=1,
                label="Maze Size (NxN)"
            )

            wall_density = gr.Slider(
                minimum=0.0, maximum=0.5, value=0.25, step=0.05,
                label="Wall Density",
                info="Proportion of cells that are walls"
            )

            algorithm = gr.Radio(
                choices=["BFS (Breadth-First Search)",
                        "DFS (Depth-First Search)",
                        "A* Search"],
                value="BFS (Breadth-First Search)",
                label="Search Algorithm"
            )

            random_seed = gr.Number(
                value=42,
                label="Random Seed",
                info="Change for different mazes"
            )

            solve_btn = gr.Button("🔍 Solve Maze!", variant="primary", size="lg")

        with gr.Column():
            output_image = gr.Image(label="Maze Solution", type="pil")
            output_text = gr.Markdown()

    # Examples
    gr.Markdown("### Try These Examples:")
    gr.Examples(
        examples=[
            [10, 0.2, "BFS (Breadth-First Search)", 42],
            [15, 0.25, "A* Search", 123],
            [20, 0.3, "DFS (Depth-First Search)", 456],
            [25, 0.15, "A* Search", 789],
        ],
        inputs=[maze_size, wall_density, algorithm, random_seed],
    )

    solve_btn.click(
        fn=solve_maze,
        inputs=[maze_size, wall_density, algorithm, random_seed],
        outputs=[output_image, output_text]
    )

    gr.Markdown("""
    ---
    ### 📚 Learn More

    - Complete the Week 1 lab notebooks to understand how these algorithms work
    - Experiment with different parameters to see how they affect performance
    - Try creating your own maze variations!

    **Challenge**: Can you predict which algorithm will explore fewer nodes?
    """)


if __name__ == "__main__":
    demo.launch(share=False)
