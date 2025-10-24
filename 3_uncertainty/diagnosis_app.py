"""
Probabilistic Reasoning Diagnosis App
Week 3: Uncertainty

Interactive Gradio application featuring:
1. Medical Diagnosis System using Bayesian inference
2. Bayesian Network Builder and Query Tool
3. Weather Prediction with Hidden Markov Models
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple
import io
from PIL import Image

# Set random seed
np.random.seed(42)


# ============================================================================
# Part 1: Medical Diagnosis System
# ============================================================================

class BayesianDiagnosisSystem:
    """Simple Bayesian diagnosis system."""

    def __init__(self):
        # Disease prior probabilities
        self.disease_priors = {
            'Flu': 0.05,
            'Cold': 0.15,
            'COVID': 0.02,
            'Healthy': 0.78
        }

        # P(symptom | disease)
        self.symptom_likelihoods = {
            'Flu': {'fever': 0.90, 'cough': 0.70, 'fatigue': 0.85, 'headache': 0.75},
            'Cold': {'fever': 0.20, 'cough': 0.80, 'fatigue': 0.40, 'headache': 0.30},
            'COVID': {'fever': 0.85, 'cough': 0.75, 'fatigue': 0.80, 'headache': 0.60},
            'Healthy': {'fever': 0.01, 'cough': 0.05, 'fatigue': 0.10, 'headache': 0.05}
        }

    def diagnose(self, symptoms: List[str]) -> Dict[str, float]:
        """
        Calculate disease probabilities given symptoms.

        Args:
            symptoms: List of observed symptoms

        Returns:
            Dictionary mapping diseases to posterior probabilities
        """
        posteriors = {}

        for disease, prior in self.disease_priors.items():
            # Start with prior
            likelihood = prior

            # Multiply by likelihoods for each symptom
            for symptom in symptoms:
                if symptom in self.symptom_likelihoods[disease]:
                    likelihood *= self.symptom_likelihoods[disease][symptom]
                else:
                    likelihood *= 0.01  # Unknown symptom

            posteriors[disease] = likelihood

        # Normalize
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v/total for k, v in posteriors.items()}

        return posteriors


def medical_diagnosis(fever: bool, cough: bool, fatigue: bool, headache: bool) -> Tuple[str, Image.Image]:
    """Medical diagnosis interface."""

    # Collect symptoms
    symptoms = []
    if fever: symptoms.append('fever')
    if cough: symptoms.append('cough')
    if fatigue: symptoms.append('fatigue')
    if headache: symptoms.append('headache')

    if not symptoms:
        return "Please select at least one symptom.", None

    # Diagnose
    system = BayesianDiagnosisSystem()
    posteriors = system.diagnose(symptoms)

    # Format results
    result = "**Diagnosis Results**\n\n"
    result += f"**Symptoms:** {', '.join(symptoms)}\n\n"
    result += "**Disease Probabilities:**\n\n"

    sorted_diseases = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)

    for disease, prob in sorted_diseases:
        result += f"- **{disease}**: {prob*100:.2f}%\n"

    most_likely = sorted_diseases[0][0]
    result += f"\n**Most Likely Diagnosis:** {most_likely}\n"

    if posteriors[most_likely] > 0.5:
        result += "\n⚠️ **Recommendation:** Consult a healthcare professional."
    else:
        result += "\n💡 **Note:** Probabilities are close. More symptoms needed for certainty."

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))
    diseases = [d for d, _ in sorted_diseases]
    probs = [p for _, p in sorted_diseases]

    colors = ['red' if p == max(probs) else 'skyblue' for p in probs]
    bars = ax.barh(diseases, probs, color=colors, edgecolor='navy', linewidth=2, alpha=0.7)

    ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Disease Probability Distribution', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{prob*100:.1f}%', ha='left', va='center',
               fontsize=11, fontweight='bold', color='black')

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return result, Image.open(buf)


# ============================================================================
# Part 2: Bayesian Network Builder
# ============================================================================

class SimpleBayesianNetwork:
    """Simple Bayesian network for demonstration."""

    def __init__(self, network_type: str):
        self.network_type = network_type
        self.setup_network()

    def setup_network(self):
        """Set up predefined networks."""
        if self.network_type == "Alarm Network":
            self.nodes = ['Burglary', 'Earthquake', 'Alarm', 'JohnCalls', 'MaryCalls']
            self.edges = [
                ('Burglary', 'Alarm'),
                ('Earthquake', 'Alarm'),
                ('Alarm', 'JohnCalls'),
                ('Alarm', 'MaryCalls')
            ]
            self.description = """
**Alarm Network**

A burglar alarm can be triggered by:
- Burglary
- Earthquake

If the alarm goes off:
- John might call
- Mary might call
"""

        elif self.network_type == "Medical Network":
            self.nodes = ['Age', 'Smoking', 'Disease', 'Symptom1', 'Symptom2', 'TestResult']
            self.edges = [
                ('Age', 'Disease'),
                ('Smoking', 'Disease'),
                ('Disease', 'Symptom1'),
                ('Disease', 'Symptom2'),
                ('Disease', 'TestResult')
            ]
            self.description = """
**Medical Diagnosis Network**

Risk factors influence disease:
- Age
- Smoking

Disease manifests as:
- Symptoms
- Test results
"""

        else:  # Weather Network
            self.nodes = ['Season', 'Temperature', 'Humidity', 'Rain', 'Traffic']
            self.edges = [
                ('Season', 'Temperature'),
                ('Season', 'Rain'),
                ('Temperature', 'Humidity'),
                ('Rain', 'Traffic')
            ]
            self.description = """
**Weather Impact Network**

Seasonal patterns affect:
- Temperature
- Rain probability

Weather impacts:
- Humidity
- Traffic conditions
"""

    def visualize(self) -> Image.Image:
        """Visualize the Bayesian network."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Use hierarchical layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=3000, alpha=0.9,
                              edgecolors='navy', linewidths=2.5, ax=ax)

        nx.draw_networkx_labels(G, pos, font_size=11,
                               font_weight='bold', ax=ax)

        nx.draw_networkx_edges(G, pos, edge_color='gray',
                              arrows=True, arrowsize=20,
                              arrowstyle='->', width=2.5,
                              connectionstyle='arc3,rad=0.1', ax=ax)

        ax.set_title(f'{self.network_type} Structure',
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return Image.open(buf)


def bayesian_network_explorer(network_type: str) -> Tuple[str, Image.Image]:
    """Explore Bayesian network structures."""

    network = SimpleBayesianNetwork(network_type)
    image = network.visualize()

    info = f"{network.description}\n\n"
    info += f"**Nodes:** {len(network.nodes)}\n"
    info += f"**Edges:** {len(network.edges)}\n\n"
    info += "**Structure Properties:**\n"
    info += "- Directed Acyclic Graph (DAG)\n"
    info += "- Each node represents a random variable\n"
    info += "- Edges show direct dependencies\n"
    info += "- Each node has a Conditional Probability Table (CPT)\n\n"
    info += "**Inference:**\n"
    info += "- Can query any variable given evidence\n"
    info += "- Exploits conditional independence\n"
    info += "- Efficient for complex probability calculations\n"

    return info, image


# ============================================================================
# Part 3: Weather Prediction with HMM
# ============================================================================

class WeatherHMM:
    """Simple weather prediction HMM."""

    def __init__(self):
        # Hidden states: actual weather
        self.states = ['Sunny', 'Rainy']

        # Transition probabilities
        self.transition = np.array([
            [0.8, 0.2],  # Sunny → Sunny, Rainy
            [0.4, 0.6]   # Rainy → Sunny, Rainy
        ])

        # Observation probabilities (temperature)
        self.observations = ['Hot', 'Warm', 'Cold']
        self.emission = np.array([
            [0.6, 0.3, 0.1],  # Sunny → Hot, Warm, Cold
            [0.1, 0.4, 0.5]   # Rainy → Hot, Warm, Cold
        ])

        # Initial distribution
        self.initial = np.array([0.6, 0.4])

    def predict_sequence(self, length: int) -> Tuple[List[str], List[str]]:
        """Generate a weather sequence."""
        states_seq = []
        obs_seq = []

        # Initial state
        state = np.random.choice(len(self.states), p=self.initial)
        states_seq.append(self.states[state])

        # Initial observation
        obs = np.random.choice(len(self.observations), p=self.emission[state])
        obs_seq.append(self.observations[obs])

        # Generate sequence
        for _ in range(length - 1):
            # Next state
            state = np.random.choice(len(self.states), p=self.transition[state])
            states_seq.append(self.states[state])

            # Observation
            obs = np.random.choice(len(self.observations), p=self.emission[state])
            obs_seq.append(self.observations[obs])

        return states_seq, obs_seq

    def viterbi(self, observations: List[str]) -> List[str]:
        """Infer most likely state sequence."""
        T = len(observations)
        n_states = len(self.states)

        delta = np.zeros((T, n_states))
        psi = np.zeros((T, n_states), dtype=int)

        # Initialization
        obs_idx = self.observations.index(observations[0])
        delta[0] = self.initial * self.emission[:, obs_idx]

        # Recursion
        for t in range(1, T):
            obs_idx = self.observations.index(observations[t])
            for j in range(n_states):
                probs = delta[t-1] * self.transition[:, j]
                psi[t, j] = np.argmax(probs)
                delta[t, j] = np.max(probs) * self.emission[j, obs_idx]

        # Backtracking
        path = [0] * T
        path[T-1] = np.argmax(delta[T-1])

        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return [self.states[i] for i in path]


def weather_prediction(days: int) -> Tuple[str, Image.Image]:
    """Weather prediction interface."""

    if days < 1 or days > 30:
        return "Please select between 1 and 30 days.", None

    hmm = WeatherHMM()

    # Generate sequence
    true_weather, observations = hmm.predict_sequence(days)

    # Infer weather using Viterbi
    inferred_weather = hmm.viterbi(observations)

    # Calculate accuracy
    correct = sum(1 for t, i in zip(true_weather, inferred_weather) if t == i)
    accuracy = correct / days * 100

    # Format results
    result = f"**Weather Forecast for {days} Days**\n\n"
    result += "**Observations:** Temperature readings\n\n"
    result += "Day | Temp | True Weather | Predicted | Match\n"
    result += "-" * 50 + "\n"

    for day, (obs, true, pred) in enumerate(zip(observations, true_weather, inferred_weather), 1):
        match = "✓" if true == pred else "✗"
        result += f"{day:2d}  | {obs:4s} | {true:6s}      | {pred:6s}   | {match}\n"

    result += f"\n**Prediction Accuracy:** {accuracy:.1f}%\n"
    result += f"\n**How it works:**\n"
    result += "- Hidden Markov Model tracks actual weather (hidden)\n"
    result += "- We only observe temperature (noisy observation)\n"
    result += "- Viterbi algorithm infers most likely weather states\n"

    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    day_range = range(1, days + 1)

    # Map weather to numbers
    weather_map = {'Sunny': 1, 'Rainy': 0}
    true_vals = [weather_map[w] for w in true_weather]
    pred_vals = [weather_map[w] for w in inferred_weather]

    # True weather
    ax1.plot(day_range, true_vals, 'o-', linewidth=2, markersize=8,
            color='orange', label='True Weather')
    ax1.fill_between(day_range, true_vals, alpha=0.3, color='orange')
    ax1.set_ylabel('Weather State', fontweight='bold')
    ax1.set_title('True Weather (Hidden)', fontweight='bold', fontsize=13)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Rainy', 'Sunny'])
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Predicted weather
    ax2.plot(day_range, pred_vals, 's-', linewidth=2, markersize=8,
            color='blue', label='Predicted Weather')
    ax2.fill_between(day_range, pred_vals, alpha=0.3, color='blue')
    ax2.set_xlabel('Day', fontweight='bold')
    ax2.set_ylabel('Weather State', fontweight='bold')
    ax2.set_title('Predicted Weather (Viterbi)', fontweight='bold', fontsize=13)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Rainy', 'Sunny'])
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return result, Image.open(buf)


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="Probabilistic Reasoning System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🎲 Probabilistic Reasoning System
    ## Week 3: Uncertainty

    Explore probabilistic AI through interactive demonstrations!
    """)

    with gr.Tabs():

        # Tab 1: Medical Diagnosis
        with gr.Tab("🏥 Medical Diagnosis"):
            gr.Markdown("""
            ### Bayesian Medical Diagnosis System

            Select symptoms to get probabilistic disease diagnosis using Bayes' theorem.

            **How it works:**
            - Prior probabilities: Base rates of diseases in population
            - Likelihoods: P(symptom | disease) from medical data
            - Posterior: P(disease | symptoms) calculated using Bayes' rule
            """)

            with gr.Row():
                with gr.Column():
                    fever_check = gr.Checkbox(label="Fever", value=False)
                    cough_check = gr.Checkbox(label="Cough", value=False)
                    fatigue_check = gr.Checkbox(label="Fatigue", value=False)
                    headache_check = gr.Checkbox(label="Headache", value=False)
                    diagnose_btn = gr.Button("🔍 Diagnose", variant="primary")

                with gr.Column():
                    diagnosis_text = gr.Markdown()
                    diagnosis_plot = gr.Image(label="Probability Distribution")

            diagnose_btn.click(
                fn=medical_diagnosis,
                inputs=[fever_check, cough_check, fatigue_check, headache_check],
                outputs=[diagnosis_text, diagnosis_plot]
            )

            gr.Examples(
                examples=[
                    [True, True, True, False],
                    [True, False, False, True],
                    [False, True, True, False],
                    [True, True, True, True]
                ],
                inputs=[fever_check, cough_check, fatigue_check, headache_check],
                label="Example Symptom Combinations"
            )

        # Tab 2: Bayesian Networks
        with gr.Tab("🕸️ Bayesian Networks"):
            gr.Markdown("""
            ### Bayesian Network Explorer

            Visualize and understand Bayesian network structures.

            **Key Concepts:**
            - Nodes represent random variables
            - Edges show probabilistic dependencies
            - Structure encodes conditional independence
            - Enables efficient inference
            """)

            with gr.Row():
                with gr.Column():
                    network_dropdown = gr.Dropdown(
                        choices=["Alarm Network", "Medical Network", "Weather Network"],
                        value="Alarm Network",
                        label="Select Network"
                    )
                    explore_btn = gr.Button("📊 Explore Network", variant="primary")

                with gr.Column():
                    network_info = gr.Markdown()
                    network_viz = gr.Image(label="Network Structure")

            explore_btn.click(
                fn=bayesian_network_explorer,
                inputs=network_dropdown,
                outputs=[network_info, network_viz]
            )

            # Auto-load first network
            demo.load(
                fn=bayesian_network_explorer,
                inputs=network_dropdown,
                outputs=[network_info, network_viz]
            )

        # Tab 3: Weather Prediction
        with gr.Tab("🌤️ Weather Prediction"):
            gr.Markdown("""
            ### Hidden Markov Model Weather Predictor

            Predict weather states from temperature observations using HMM.

            **Hidden Markov Model:**
            - Hidden states: Actual weather (Sunny/Rainy)
            - Observations: Temperature (Hot/Warm/Cold)
            - Viterbi algorithm finds most likely weather sequence
            """)

            with gr.Row():
                with gr.Column():
                    days_slider = gr.Slider(
                        minimum=7,
                        maximum=30,
                        value=14,
                        step=1,
                        label="Number of Days"
                    )
                    predict_btn = gr.Button("🔮 Predict Weather", variant="primary")

                with gr.Column():
                    weather_text = gr.Markdown()
                    weather_plot = gr.Image(label="Weather Sequence")

            predict_btn.click(
                fn=weather_prediction,
                inputs=days_slider,
                outputs=[weather_text, weather_plot]
            )

    gr.Markdown("""
    ---
    ### 📚 Learn More

    - Complete Week 3 lab notebooks for in-depth theory
    - Experiment with different inputs and parameters
    - Build your own probabilistic reasoning systems!

    **Key Takeaways:**
    - Bayes' theorem enables rational belief updating
    - Bayesian networks compactly represent complex distributions
    - HMMs handle sequential data with hidden structure
    """)


if __name__ == "__main__":
    demo.launch(share=False)
