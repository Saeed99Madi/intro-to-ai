"""
NLP Interactive Application
Week 7: Natural Language Processing and Transformers

This Gradio application provides interactive demonstrations of NLP concepts:
1. Text Analyzer: Preprocessing, tokenization, POS tagging, NER
2. Sentiment Analysis: Train and test sentiment classifiers
3. Text Generation: Generate text with LSTM and GPT-style models
4. Transformer Playground: Use pre-trained models for various tasks

Run: python nlp_app.py
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk import pos_tag, ne_chunk

    # Download required NLTK data (silent)
    for package in ['punkt', 'stopwords', 'averaged_perceptron_tagger',
                    'maxent_ne_chunker', 'words', 'wordnet']:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Machine learning
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Set style
sns.set_style('whitegrid')

# ============================================================================
# TAB 1: Text Analyzer
# ============================================================================

def analyze_text(text):
    """Analyze text with various NLP techniques."""

    if not NLTK_AVAILABLE:
        return "Error: NLTK not installed. Install with: pip install nltk"

    if not text.strip():
        return "Please enter some text to analyze."

    results = []
    results.append("=" * 70)
    results.append("TEXT ANALYSIS")
    results.append("=" * 70)

    # Basic stats
    results.append("\n1. BASIC STATISTICS")
    results.append("-" * 70)
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    results.append(f"Characters: {len(text)}")
    results.append(f"Words: {len(words)}")
    results.append(f"Sentences: {len(sentences)}")
    results.append(f"Average word length: {np.mean([len(w) for w in words]):.2f}")
    results.append(f"Average sentence length: {np.mean([len(word_tokenize(s)) for s in sentences]):.2f}")

    # Tokenization
    results.append("\n2. TOKENIZATION")
    results.append("-" * 70)
    results.append(f"Tokens: {words[:20]}")
    if len(words) > 20:
        results.append(f"... and {len(words) - 20} more")

    # Stopwords removal
    results.append("\n3. STOPWORDS REMOVAL")
    results.append("-" * 70)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w.lower() not in stop_words and w.isalnum()]
    results.append(f"Words after stopwords removal: {len(filtered_words)}")
    results.append(f"Filtered: {filtered_words[:15]}")

    # Stemming
    results.append("\n4. STEMMING")
    results.append("-" * 70)
    stemmer = PorterStemmer()
    stems = [(w, stemmer.stem(w)) for w in filtered_words[:10]]
    for word, stem in stems:
        if word.lower() != stem:
            results.append(f"{word:15s} -> {stem}")

    # Lemmatization
    results.append("\n5. LEMMATIZATION")
    results.append("-" * 70)
    lemmatizer = WordNetLemmatizer()
    lemmas = [(w, lemmatizer.lemmatize(w.lower())) for w in filtered_words[:10]]
    for word, lemma in lemmas:
        if word.lower() != lemma:
            results.append(f"{word:15s} -> {lemma}")

    # POS tagging
    results.append("\n6. PART-OF-SPEECH TAGGING")
    results.append("-" * 70)
    pos_tags = pos_tag(words)
    results.append("Word            | POS Tag")
    results.append("-" * 35)
    for word, tag in pos_tags[:15]:
        results.append(f"{word:15s} | {tag}")

    # Word frequency
    results.append("\n7. WORD FREQUENCY")
    results.append("-" * 70)
    word_freq = Counter([w.lower() for w in filtered_words])
    results.append("Top 10 most frequent words:")
    for word, count in word_freq.most_common(10):
        results.append(f"{word:15s}: {count}")

    # Named Entity Recognition
    results.append("\n8. NAMED ENTITY RECOGNITION")
    results.append("-" * 70)
    try:
        ne_tree = ne_chunk(pos_tags)
        entities = []
        for subtree in ne_tree:
            if hasattr(subtree, 'label'):
                entity = " ".join([word for word, tag in subtree.leaves()])
                entities.append((entity, subtree.label()))

        if entities:
            results.append("Entities found:")
            for entity, label in entities:
                results.append(f"{entity:20s} -> {label}")
        else:
            results.append("No named entities found.")
    except Exception as e:
        results.append(f"NER error: {e}")

    results.append("\n" + "=" * 70)

    return "\n".join(results)


# ============================================================================
# TAB 2: Sentiment Analysis
# ============================================================================

# Global model storage
sentiment_model = None
sentiment_vectorizer = None

def train_sentiment_model(model_type, sample_data):
    """Train a sentiment classifier."""

    global sentiment_model, sentiment_vectorizer

    if not SKLEARN_AVAILABLE:
        return "Error: scikit-learn not installed. Install with: pip install scikit-learn"

    # Sample dataset
    if sample_data == "Movie Reviews":
        texts = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible film. Waste of time and money.",
            "An amazing performance by the lead actor. Highly recommended!",
            "Boring and predictable. Would not watch again.",
            "One of the best movies I've seen this year!",
            "Awful screenplay and poor direction.",
            "Brilliant cinematography and storytelling.",
            "Not worth watching. Very disappointed.",
            "A masterpiece! Everyone should see this.",
            "Complete disaster. Worst movie ever.",
            "Entertaining and well-made. Enjoyed it thoroughly.",
            "Dull and uninspiring. Fell asleep halfway.",
            "Excellent cast and gripping plot.",
            "Poor acting and weak storyline.",
            "Must-watch! Absolutely loved it!"
        ]
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1=positive, 0=negative
    else:  # Product Reviews
        texts = [
            "Great product! Works perfectly as described.",
            "Poor quality. Broke after one use.",
            "Excellent value for money. Very satisfied!",
            "Complete waste of money. Do not buy.",
            "Amazing! Exceeded my expectations.",
            "Terrible. Would not recommend to anyone.",
            "High quality and fast delivery.",
            "Cheap material. Very disappointed.",
            "Best purchase I've made this year!",
            "Defective product. Returning immediately.",
            "Works great! Very happy with it.",
            "Not as advertised. Very poor.",
            "Fantastic product. Worth every penny!",
            "Low quality. Not worth the price.",
            "Perfect! Exactly what I needed."
        ]
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # Vectorize text
    sentiment_vectorizer = TfidfVectorizer(max_features=100)
    X = sentiment_vectorizer.fit_transform(texts)
    y = np.array(labels)

    # Train model
    if model_type == "Naive Bayes":
        sentiment_model = MultinomialNB()
    else:  # Logistic Regression
        sentiment_model = LogisticRegression(max_iter=1000)

    sentiment_model.fit(X, y)

    # Evaluate on training data (for demo purposes)
    train_pred = sentiment_model.predict(X)
    train_acc = np.mean(train_pred == y)

    result = []
    result.append(f"Model trained: {model_type}")
    result.append(f"Dataset: {sample_data}")
    result.append(f"Training samples: {len(texts)}")
    result.append(f"Training accuracy: {train_acc:.2%}")
    result.append("\nModel ready for predictions!")

    return "\n".join(result)


def predict_sentiment(text):
    """Predict sentiment of input text."""

    global sentiment_model, sentiment_vectorizer

    if sentiment_model is None or sentiment_vectorizer is None:
        return "Please train a model first!"

    if not text.strip():
        return "Please enter some text."

    # Vectorize and predict
    X = sentiment_vectorizer.transform([text])
    prediction = sentiment_model.predict(X)[0]
    probability = sentiment_model.predict_proba(X)[0]

    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    confidence = probability[prediction]

    result = []
    result.append("=" * 50)
    result.append("SENTIMENT ANALYSIS RESULT")
    result.append("=" * 50)
    result.append(f"\nText: {text}")
    result.append(f"\nSentiment: {sentiment}")
    result.append(f"Confidence: {confidence:.2%}")
    result.append(f"\nProbability distribution:")
    result.append(f"  Negative: {probability[0]:.2%}")
    result.append(f"  Positive: {probability[1]:.2%}")
    result.append("=" * 50)

    return "\n".join(result)


# ============================================================================
# TAB 3: Text Generation
# ============================================================================

# Global language model
language_model = None
word2idx_lm = None
idx2word_lm = None
seq_length = 3

def build_language_model(corpus_choice):
    """Build and train LSTM language model."""

    global language_model, word2idx_lm, idx2word_lm, seq_length

    if not TF_AVAILABLE:
        return "Error: TensorFlow not installed. Install with: pip install tensorflow"

    # Choose corpus
    if corpus_choice == "Simple Stories":
        corpus = [
            "the cat sat on the mat",
            "the dog ran in the park",
            "a bird flew over the tree",
            "the sun shines in the sky",
            "children play in the garden",
            "the moon glows at night",
            "fish swim in the ocean",
            "flowers bloom in spring",
            "snow falls in winter",
            "rain drops on the ground"
        ]
    else:  # Technical Text
        corpus = [
            "machine learning models process data efficiently",
            "neural networks learn complex patterns automatically",
            "deep learning requires large datasets",
            "transformers revolutionized natural language processing",
            "attention mechanisms improve model performance",
            "embeddings capture semantic relationships effectively",
            "supervised learning uses labeled training data",
            "reinforcement learning optimizes sequential decisions",
            "unsupervised learning discovers hidden patterns",
            "transfer learning leverages pretrained models"
        ]

    # Prepare data
    text = " ".join(corpus)
    words = text.lower().split()

    vocab = sorted(set(words))
    word2idx_lm = {word: idx for idx, word in enumerate(vocab)}
    idx2word_lm = {idx: word for word, idx in word2idx_lm.items()}
    vocab_size = len(vocab)

    # Create sequences
    X_seq, y_seq = [], []
    seq_length = 3

    for i in range(len(words) - seq_length):
        seq_in = words[i:i+seq_length]
        seq_out = words[i+seq_length]

        X_seq.append([word2idx_lm[w] for w in seq_in])
        y_seq.append(word2idx_lm[seq_out])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Build model
    language_model = keras.Sequential([
        layers.Embedding(vocab_size, 32, input_length=seq_length),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(vocab_size, activation='softmax')
    ])

    language_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history = language_model.fit(
        X_seq, y_seq,
        epochs=100,
        batch_size=8,
        verbose=0
    )

    result = []
    result.append(f"Language Model trained on: {corpus_choice}")
    result.append(f"Vocabulary size: {vocab_size}")
    result.append(f"Training sequences: {len(X_seq)}")
    result.append(f"Final accuracy: {history.history['accuracy'][-1]:.2%}")
    result.append("\nModel ready for text generation!")

    return "\n".join(result)


def generate_text_lm(seed_text, num_words, temperature):
    """Generate text using trained language model."""

    global language_model, word2idx_lm, idx2word_lm, seq_length

    if language_model is None:
        return "Please train a language model first!"

    if not seed_text.strip():
        return "Please provide seed text."

    # Prepare seed
    seed_words = seed_text.lower().split()
    if len(seed_words) < seq_length:
        return f"Seed text must have at least {seq_length} words."

    generated = seed_words[-seq_length:]

    # Generate
    for _ in range(num_words):
        # Prepare input
        try:
            seq = [word2idx_lm[w] for w in generated[-seq_length:]]
        except KeyError:
            return "Error: Some words in seed text not in vocabulary. Try different seed text."

        seq_encoded = np.array([seq])

        # Predict
        predictions = language_model.predict(seq_encoded, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions + 1e-7) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        # Sample
        next_idx = np.random.choice(len(predictions), p=predictions)
        next_word = idx2word_lm[next_idx]
        generated.append(next_word)

    return " ".join(generated)


# ============================================================================
# TAB 4: Transformer Playground
# ============================================================================

def use_transformers(task, text, context=""):
    """Use pre-trained transformer models for various tasks."""

    if not TRANSFORMERS_AVAILABLE:
        return "Error: transformers not installed. Install with: pip install transformers"

    try:
        result = []
        result.append("=" * 70)
        result.append(f"TASK: {task}")
        result.append("=" * 70)

        if task == "Sentiment Analysis":
            pipe = pipeline("sentiment-analysis")
            output = pipe(text)[0]
            result.append(f"\nInput: {text}")
            result.append(f"\nSentiment: {output['label']}")
            result.append(f"Confidence: {output['score']:.2%}")

        elif task == "Named Entity Recognition":
            pipe = pipeline("ner", grouped_entities=True)
            entities = pipe(text)
            result.append(f"\nInput: {text}")
            result.append(f"\nEntities found: {len(entities)}")
            result.append("\nEntity            | Type       | Score")
            result.append("-" * 50)
            for entity in entities:
                result.append(f"{entity['word']:17s} | {entity['entity_group']:10s} | {entity['score']:.2f}")

        elif task == "Question Answering":
            if not context:
                return "Please provide context for question answering."
            pipe = pipeline("question-answering")
            output = pipe(question=text, context=context)
            result.append(f"\nQuestion: {text}")
            result.append(f"Context: {context}")
            result.append(f"\nAnswer: {output['answer']}")
            result.append(f"Confidence: {output['score']:.2%}")

        elif task == "Text Generation":
            pipe = pipeline("text-generation", model="gpt2")
            output = pipe(text, max_length=100, num_return_sequences=1)[0]
            result.append(f"\nPrompt: {text}")
            result.append(f"\nGenerated text:")
            result.append(output['generated_text'])

        elif task == "Fill Mask":
            pipe = pipeline("fill-mask")
            # Ensure [MASK] token is in text
            if "[MASK]" not in text:
                text = text.replace("___", "[MASK]")
            if "[MASK]" not in text:
                return "Please include [MASK] token in your text."

            outputs = pipe(text)
            result.append(f"\nInput: {text}")
            result.append(f"\nTop predictions:")
            for i, output in enumerate(outputs[:5], 1):
                result.append(f"{i}. {output['token_str']:15s} (score: {output['score']:.3f})")

        result.append("\n" + "=" * 70)
        return "\n".join(result)

    except Exception as e:
        return f"Error: {str(e)}\n\nNote: First run downloads models (may take time)."


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create Gradio interface."""

    with gr.Blocks(title="NLP Interactive App", theme=gr.themes.Soft()) as app:

        gr.Markdown("""
        # Natural Language Processing Interactive Application

        Explore NLP concepts interactively! Choose a tab below:
        - **Text Analyzer**: Tokenization, POS tagging, NER
        - **Sentiment Analysis**: Train and test classifiers
        - **Text Generation**: Generate text with LSTM
        - **Transformer Playground**: Use pre-trained models
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1: Text Analyzer
            # ================================================================

            with gr.Tab("Text Analyzer"):
                gr.Markdown("""
                ### Text Analysis Tool
                Analyze text with various NLP techniques: tokenization, stemming,
                lemmatization, POS tagging, and NER.
                """)

                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text to analyze...",
                            lines=8,
                            value="Apple Inc. was founded by Steve Jobs in Cupertino. "
                                  "The company revolutionized personal computing with the Macintosh."
                        )
                        analyze_btn = gr.Button("Analyze Text", variant="primary")

                    with gr.Column():
                        analysis_output = gr.Textbox(
                            label="Analysis Results",
                            lines=25
                        )

                analyze_btn.click(
                    fn=analyze_text,
                    inputs=[text_input],
                    outputs=[analysis_output]
                )

                gr.Examples(
                    examples=[
                        ["The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."],
                        ["Machine learning is a subset of artificial intelligence. Deep learning uses neural networks with many layers."],
                        ["Barack Obama was the 44th President of the United States. He served from 2009 to 2017 in Washington, D.C."]
                    ],
                    inputs=[text_input]
                )

            # ================================================================
            # TAB 2: Sentiment Analysis
            # ================================================================

            with gr.Tab("Sentiment Analysis"):
                gr.Markdown("""
                ### Sentiment Analysis
                Train a classifier on sample data, then test it on your own text.
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 1. Train Model")
                        model_type = gr.Radio(
                            choices=["Naive Bayes", "Logistic Regression"],
                            value="Naive Bayes",
                            label="Model Type"
                        )
                        dataset_type = gr.Radio(
                            choices=["Movie Reviews", "Product Reviews"],
                            value="Movie Reviews",
                            label="Training Dataset"
                        )
                        train_btn = gr.Button("Train Model", variant="primary")
                        train_output = gr.Textbox(label="Training Results", lines=8)

                    with gr.Column():
                        gr.Markdown("#### 2. Test Model")
                        sentiment_input = gr.Textbox(
                            label="Test Text",
                            placeholder="Enter text to classify...",
                            lines=4,
                            value="This is absolutely fantastic! I love it!"
                        )
                        predict_btn = gr.Button("Predict Sentiment", variant="secondary")
                        sentiment_output = gr.Textbox(label="Prediction", lines=12)

                train_btn.click(
                    fn=train_sentiment_model,
                    inputs=[model_type, dataset_type],
                    outputs=[train_output]
                )

                predict_btn.click(
                    fn=predict_sentiment,
                    inputs=[sentiment_input],
                    outputs=[sentiment_output]
                )

                gr.Examples(
                    examples=[
                        ["This is the best thing I've ever seen!"],
                        ["Terrible experience. Very disappointed."],
                        ["It's okay, nothing special."],
                        ["Absolutely amazing! Highly recommend!"],
                        ["Waste of time and money."]
                    ],
                    inputs=[sentiment_input]
                )

            # ================================================================
            # TAB 3: Text Generation
            # ================================================================

            with gr.Tab("Text Generation"):
                gr.Markdown("""
                ### Text Generation with LSTM
                Train a language model and generate text.
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 1. Train Language Model")
                        corpus_choice = gr.Radio(
                            choices=["Simple Stories", "Technical Text"],
                            value="Simple Stories",
                            label="Training Corpus"
                        )
                        train_lm_btn = gr.Button("Train Model", variant="primary")
                        train_lm_output = gr.Textbox(label="Training Results", lines=6)

                    with gr.Column():
                        gr.Markdown("#### 2. Generate Text")
                        seed_input = gr.Textbox(
                            label="Seed Text (at least 3 words)",
                            value="the cat sat",
                            lines=2
                        )
                        num_words = gr.Slider(
                            minimum=5,
                            maximum=30,
                            value=10,
                            step=1,
                            label="Number of Words to Generate"
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            label="Temperature (randomness)"
                        )
                        generate_btn = gr.Button("Generate Text", variant="secondary")
                        generated_output = gr.Textbox(label="Generated Text", lines=6)

                train_lm_btn.click(
                    fn=build_language_model,
                    inputs=[corpus_choice],
                    outputs=[train_lm_output]
                )

                generate_btn.click(
                    fn=generate_text_lm,
                    inputs=[seed_input, num_words, temperature],
                    outputs=[generated_output]
                )

            # ================================================================
            # TAB 4: Transformer Playground
            # ================================================================

            with gr.Tab("Transformer Playground"):
                gr.Markdown("""
                ### Pre-trained Transformers
                Use state-of-the-art models for various NLP tasks.

                **Note**: First run downloads models (may take time).
                """)

                task_choice = gr.Radio(
                    choices=[
                        "Sentiment Analysis",
                        "Named Entity Recognition",
                        "Question Answering",
                        "Text Generation",
                        "Fill Mask"
                    ],
                    value="Sentiment Analysis",
                    label="Task"
                )

                with gr.Row():
                    with gr.Column():
                        transformer_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text or question...",
                            lines=4,
                            value="I love using transformers for NLP tasks!"
                        )
                        context_input = gr.Textbox(
                            label="Context (for Q&A only)",
                            placeholder="Provide context for question answering...",
                            lines=4,
                            visible=False
                        )
                        transform_btn = gr.Button("Run", variant="primary")

                    with gr.Column():
                        transformer_output = gr.Textbox(
                            label="Output",
                            lines=15
                        )

                # Show/hide context based on task
                def update_interface(task):
                    if task == "Question Answering":
                        return gr.update(visible=True)
                    return gr.update(visible=False)

                task_choice.change(
                    fn=update_interface,
                    inputs=[task_choice],
                    outputs=[context_input]
                )

                transform_btn.click(
                    fn=use_transformers,
                    inputs=[task_choice, transformer_input, context_input],
                    outputs=[transformer_output]
                )

                # Task-specific examples
                gr.Markdown("### Examples by Task")

                with gr.Accordion("Sentiment Analysis Examples", open=False):
                    gr.Examples(
                        examples=[
                            ["This movie is absolutely fantastic!"],
                            ["I'm very disappointed with this product."]
                        ],
                        inputs=[transformer_input]
                    )

                with gr.Accordion("NER Examples", open=False):
                    gr.Examples(
                        examples=[
                            ["Apple Inc. was founded by Steve Jobs in Cupertino, California."],
                            ["Barack Obama was born in Hawaii and became the 44th President."]
                        ],
                        inputs=[transformer_input]
                    )

                with gr.Accordion("Question Answering Example", open=False):
                    gr.Markdown("Question: `When was the transformer introduced?`")
                    gr.Markdown("Context: `The transformer was introduced in 2017...`")

                with gr.Accordion("Text Generation Examples", open=False):
                    gr.Examples(
                        examples=[
                            ["Once upon a time in a faraway land"],
                            ["The future of artificial intelligence is"]
                        ],
                        inputs=[transformer_input]
                    )

                with gr.Accordion("Fill Mask Examples", open=False):
                    gr.Examples(
                        examples=[
                            ["The [MASK] is shining brightly today."],
                            ["Paris is the [MASK] of France."]
                        ],
                        inputs=[transformer_input]
                    )

        # Footer
        gr.Markdown("""
        ---
        ### About
        This application demonstrates key NLP concepts from Week 7 of the Introduction to AI course.

        **Topics covered:**
        - Text preprocessing and analysis
        - Traditional ML for sentiment analysis
        - Neural language models (LSTM)
        - Pre-trained transformers (BERT, GPT-2)

        **Requirements:**
        ```bash
        pip install gradio nltk scikit-learn tensorflow transformers
        ```
        """)

    return app


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NLP Interactive Application")
    print("=" * 70)
    print("\nChecking dependencies...")

    if not NLTK_AVAILABLE:
        print("⚠️  NLTK not available (install: pip install nltk)")
    else:
        print("✓ NLTK available")

    if not SKLEARN_AVAILABLE:
        print("⚠️  scikit-learn not available (install: pip install scikit-learn)")
    else:
        print("✓ scikit-learn available")

    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available (install: pip install tensorflow)")
    else:
        print("✓ TensorFlow available")

    if not TRANSFORMERS_AVAILABLE:
        print("⚠️  transformers not available (install: pip install transformers)")
    else:
        print("✓ transformers available")

    print("\nLaunching application...")
    print("=" * 70)

    app = create_interface()
    app.launch(share=False)
