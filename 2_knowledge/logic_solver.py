"""
Logic Puzzle Solver - Interactive Gradio Application
Week 2: Knowledge Representation

This app allows you to:
- Solve Knights and Knaves puzzles
- Evaluate logical expressions
- Check satisfiability
- Visualize truth tables
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Tuple, Dict, Optional
import io
from PIL import Image


# Logic functions
def NOT(p: bool) -> bool:
    return not p


def AND(p: bool, q: bool) -> bool:
    return p and q


def OR(p: bool, q: bool) -> bool:
    return p or q


def IMPLIES(p: bool, q: bool) -> bool:
    return (not p) or q


def IFF(p: bool, q: bool) -> bool:
    return IMPLIES(p, q) and IMPLIES(q, p)


# Knights and Knaves Solver
def solve_knights_knaves(statement_a: str, statement_b: str = "") -> Tuple[str, str]:
    """
    Solve a Knights and Knaves puzzle.

    Knights always tell the truth.
    Knaves always lie.
    """

    results = []

    # Try all possibilities
    for a_is_knight, b_is_knight in product([False, True], repeat=2):

        # Parse A's statement
        a_consistent = check_statement_consistency(
            statement_a, a_is_knight, b_is_knight, "A", "B"
        )

        # Parse B's statement if provided
        if statement_b:
            b_consistent = check_statement_consistency(
                statement_b, b_is_knight, a_is_knight, "B", "A"
            )
        else:
            b_consistent = True

        if a_consistent and b_consistent:
            a_type = "Knight" if a_is_knight else "Knave"
            b_type = "Knight" if b_is_knight else "Knave"

            explanation = f"**Solution Found!**\n\n"
            explanation += f"- **A** is a **{a_type}**\n"
            if statement_b:
                explanation += f"- **B** is a **{b_type}**\n"

            explanation += f"\n**Verification:**\n"
            explanation += f"- A's statement: \"{statement_a}\"\n"
            explanation += f"  - A is {'knight (truth-teller)' if a_is_knight else 'knave (liar)'}\n"

            if statement_b:
                explanation += f"- B's statement: \"{statement_b}\"\n"
                explanation += f"  - B is {'knight (truth-teller)' if b_is_knight else 'knave (liar)'}\n"

            results.append(explanation)

    if results:
        return "\n\n---\n\n".join(results)
    else:
        return "**No solution found!** The statements may be contradictory."


def check_statement_consistency(statement: str, speaker_is_knight: bool,
                               other_is_knight: bool, speaker_name: str,
                               other_name: str) -> bool:
    """Check if a statement is consistent with the speaker's type."""

    statement = statement.lower().strip()

    # Simple pattern matching for common statements
    if "both" in statement and "knave" in statement:
        statement_value = (not speaker_is_knight) and (not other_is_knight)
    elif "both" in statement and "knight" in statement:
        statement_value = speaker_is_knight and other_is_knight
    elif other_name.lower() in statement and "knight" in statement:
        statement_value = other_is_knight
    elif other_name.lower() in statement and "knave" in statement:
        statement_value = not other_is_knight
    elif "i am" in statement and "knave" in statement:
        statement_value = not speaker_is_knight
    elif "i am" in statement and "knight" in statement:
        statement_value = speaker_is_knight
    elif "opposite" in statement or "different" in statement:
        statement_value = (speaker_is_knight != other_is_knight)
    elif "same" in statement:
        statement_value = (speaker_is_knight == other_is_knight)
    else:
        # Can't parse - assume consistent
        return True

    # If knight, statement must be true
    # If knave, statement must be false
    return speaker_is_knight == statement_value


# Truth Table Generator
def generate_truth_table(expression: str) -> Tuple[str, Optional[Image.Image]]:
    """Generate truth table for a logical expression."""

    try:
        # Extract variables
        variables = sorted(set([c for c in expression.upper() if c.isalpha()]))

        if len(variables) > 4:
            return "Too many variables! Maximum is 4.", None

        if not variables:
            return "No variables found in expression!", None

        # Build truth table
        rows = []
        header = " | ".join(variables) + " | Result"
        separator = "-" * (len(header) + 10)

        rows.append(header)
        rows.append(separator)

        results = []
        for values in product([False, True], repeat=len(variables)):
            # Create context for evaluation
            context = dict(zip(variables, values))

            # Evaluate expression
            try:
                result = eval_logic_expression(expression.upper(), context)
                results.append(result)
            except:
                return f"Error evaluating expression: {expression}", None

            # Format row
            value_str = " | ".join(["T" if v else "F" for v in values])
            result_str = "T" if result else "F"
            rows.append(f"{value_str} | {result_str}")

        table_text = "\n".join(rows)

        # Create visualization
        fig = create_truth_table_visual(variables, results, expression)

        return f"**Truth Table for:** {expression}\n\n```\n{table_text}\n```", fig

    except Exception as e:
        return f"Error: {str(e)}", None


def eval_logic_expression(expr: str, context: Dict[str, bool]) -> bool:
    """Evaluate a logical expression given variable values."""

    # Replace operators
    expr = expr.replace("AND", " and ")
    expr = expr.replace("OR", " or ")
    expr = expr.replace("NOT", " not ")
    expr = expr.replace("→", " <= ")  # Implication in Python
    expr = expr.replace("->", " <= ")

    # Replace variables with values
    for var, value in context.items():
        expr = expr.replace(var, str(value))

    # Evaluate
    return eval(expr)


def create_truth_table_visual(variables: List[str], results: List[bool],
                              expression: str) -> Image.Image:
    """Create visual representation of truth table."""

    n_vars = len(variables)
    n_rows = 2 ** n_vars

    fig, ax = plt.subplots(figsize=(max(8, n_vars * 2), max(6, n_rows * 0.5)))

    # Create grid
    data = []
    for i, values in enumerate(product([0, 1], repeat=n_vars)):
        row = list(values) + [1 if results[i] else 0]
        data.append(row)

    data = np.array(data)

    # Display as heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(range(n_vars + 1))
    ax.set_xticklabels(variables + ['Result'])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"Row {i+1}" for i in range(n_rows)])

    # Add text annotations
    for i in range(n_rows):
        for j in range(n_vars + 1):
            text = ax.text(j, i, 'T' if data[i, j] else 'F',
                         ha="center", va="center", color="black",
                         fontsize=12, fontweight='bold')

    ax.set_title(f'Truth Table: {expression}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return Image.open(buf)


# Satisfiability Checker
def check_satisfiability(expression: str) -> str:
    """Check if a logical expression is satisfiable."""

    try:
        # Extract variables
        variables = sorted(set([c for c in expression.upper() if c.isalpha()]))

        if not variables:
            return "No variables found in expression!"

        # Try all possible truth assignments
        satisfying_models = []

        for values in product([False, True], repeat=len(variables)):
            context = dict(zip(variables, values))

            try:
                if eval_logic_expression(expression.upper(), context):
                    satisfying_models.append(context)
            except:
                return f"Error evaluating expression: {expression}"

        if satisfying_models:
            result = f"**✓ SATISFIABLE**\n\n"
            result += f"Found {len(satisfying_models)} satisfying model(s):\n\n"

            for i, model in enumerate(satisfying_models[:5], 1):  # Show first 5
                model_str = ", ".join([f"{k}={'T' if v else 'F'}" for k, v in model.items()])
                result += f"{i}. {model_str}\n"

            if len(satisfying_models) > 5:
                result += f"\n... and {len(satisfying_models) - 5} more"

            return result
        else:
            return f"**✗ UNSATISFIABLE**\n\nNo truth assignment makes this expression true.\nThis is a contradiction!"

    except Exception as e:
        return f"Error: {str(e)}"


# Build Gradio interface
with gr.Blocks(title="Logic Puzzle Solver", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧩 Logic Puzzle Solver
    ## Week 2: Knowledge Representation

    Explore propositional logic through interactive puzzles and tools!
    """)

    with gr.Tabs():

        # Tab 1: Knights and Knaves
        with gr.Tab("Knights & Knaves"):
            gr.Markdown("""
            ### Knights and Knaves Puzzle Solver

            **Rules:**
            - **Knights** always tell the truth
            - **Knaves** always lie

            Enter what each person says, and the solver will determine who is who!
            """)

            with gr.Row():
                with gr.Column():
                    kk_statement_a = gr.Textbox(
                        label="A says:",
                        placeholder="e.g., 'We are both knaves' or 'B is a knight'",
                        lines=2
                    )
                    kk_statement_b = gr.Textbox(
                        label="B says (optional):",
                        placeholder="e.g., 'A and I are opposite types'",
                        lines=2
                    )
                    kk_solve_btn = gr.Button("🔍 Solve Puzzle", variant="primary")

                with gr.Column():
                    kk_output = gr.Markdown(label="Solution")

            kk_solve_btn.click(
                fn=solve_knights_knaves,
                inputs=[kk_statement_a, kk_statement_b],
                outputs=kk_output
            )

            gr.Markdown("### Try These Examples:")
            gr.Examples(
                examples=[
                    ["We are both knaves", ""],
                    ["B is a knight", "A and I are of opposite types"],
                    ["I am a knave", ""],
                    ["At least one of us is a knave", "A is a knave"],
                ],
                inputs=[kk_statement_a, kk_statement_b]
            )

        # Tab 2: Truth Tables
        with gr.Tab("Truth Tables"):
            gr.Markdown("""
            ### Truth Table Generator

            Generate truth tables for logical expressions!

            **Operators:**
            - `AND`, `OR`, `NOT`
            - `->` or `→` for implication

            **Example:** `(P AND Q) -> R`
            """)

            with gr.Row():
                with gr.Column():
                    tt_expression = gr.Textbox(
                        label="Logical Expression",
                        placeholder="e.g., (P AND Q) OR (NOT R)",
                        lines=2
                    )
                    tt_generate_btn = gr.Button("📊 Generate Table", variant="primary")

                with gr.Column():
                    tt_output_text = gr.Markdown()
                    tt_output_img = gr.Image(label="Visual Table", type="pil")

            tt_generate_btn.click(
                fn=generate_truth_table,
                inputs=tt_expression,
                outputs=[tt_output_text, tt_output_img]
            )

            gr.Examples(
                examples=[
                    ["P AND Q"],
                    ["P OR Q"],
                    ["(P AND Q) -> R"],
                    ["(P -> Q) AND (Q -> R)"],
                    ["NOT (P AND Q)"],
                ],
                inputs=tt_expression
            )

        # Tab 3: Satisfiability
        with gr.Tab("Satisfiability"):
            gr.Markdown("""
            ### Satisfiability Checker

            Check if a logical expression can be satisfied (made true).

            An expression is:
            - **Satisfiable** if there exists a truth assignment that makes it true
            - **Unsatisfiable** if no truth assignment makes it true (contradiction)

            **Example:** `P AND NOT P` is unsatisfiable
            """)

            with gr.Row():
                with gr.Column():
                    sat_expression = gr.Textbox(
                        label="Logical Expression",
                        placeholder="e.g., (P AND Q) OR (NOT P AND R)",
                        lines=3
                    )
                    sat_check_btn = gr.Button("✓ Check Satisfiability", variant="primary")

                with gr.Column():
                    sat_output = gr.Markdown()

            sat_check_btn.click(
                fn=check_satisfiability,
                inputs=sat_expression,
                outputs=sat_output
            )

            gr.Examples(
                examples=[
                    ["P AND NOT P"],  # Unsatisfiable
                    ["P OR NOT P"],   # Tautology (always satisfiable)
                    ["(P -> Q) AND P AND NOT Q"],  # Unsatisfiable
                    ["(P AND Q) OR R"],  # Satisfiable
                ],
                inputs=sat_expression
            )

    gr.Markdown("""
    ---
    ### 📚 Learn More

    - Complete the Week 2 lab notebooks to understand the theory
    - Experiment with different logical expressions
    - Try to stump the solver with tricky puzzles!

    **Challenge:** Can you create a Knights and Knaves puzzle with 3 people?
    """)


if __name__ == "__main__":
    demo.launch(share=False)
