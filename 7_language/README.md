# Week 7: Language (Natural Language Processing and Transformers)

Welcome to Week 7, the final week of the Introduction to AI course! This week explores Natural Language Processing (NLP), from traditional techniques to modern transformer architectures that power ChatGPT, BERT, and other state-of-the-art language models.

## Overview

Natural Language Processing enables computers to understand, interpret, and generate human language. This week covers the full spectrum from text preprocessing to cutting-edge transformer models, with hands-on implementations and practical applications.

## Learning Objectives

By the end of this week, you will be able to:

- Preprocess and tokenize text data effectively
- Implement traditional NLP techniques (Bag of Words, TF-IDF)
- Understand and use word embeddings (Word2Vec, GloVe)
- Build recurrent language models (LSTM, GRU)
- Implement attention mechanisms
- Understand transformer architecture in depth
- Use pre-trained models (BERT, GPT) with Hugging Face
- Fine-tune transformers for specific tasks
- Build practical NLP applications (sentiment analysis, text generation, Q&A)
- Evaluate language models properly

## Prerequisites

- Python programming fundamentals
- Understanding of neural networks (Week 6)
- Familiarity with RNNs and attention (Week 6, Lab 4)
- Basic probability and statistics
- Linear algebra (vectors, matrices)

## Labs

### Lab 1: Introduction to NLP
**File:** `1_lab1.ipynb`

Fundamentals of text processing and traditional NLP techniques.

**Topics:**
- Text preprocessing (tokenization, stemming, lemmatization)
- Bag of Words (BoW) representation
- TF-IDF (Term Frequency-Inverse Document Frequency)
- N-grams and language modeling
- Text classification with traditional methods
- Named Entity Recognition (NER) basics
- Part-of-speech (POS) tagging
- Practical example: Spam detection

**Key Concepts:**
- Tokenization strategies
- Stop words and normalization
- Feature extraction from text
- Sparse representations

### Lab 2: Word Embeddings and Language Models
**File:** `2_lab2.ipynb`

Dense representations of words and neural language models.

**Topics:**
- Limitations of one-hot encoding
- Word2Vec (Skip-gram and CBOW)
- GloVe (Global Vectors)
- FastText for subword embeddings
- Word analogies and semantic relationships
- Building vocabulary and embedding matrices
- Character-level models
- RNN language models
- LSTM for text generation
- Practical example: Text generation and word similarity

**Key Concepts:**
- Distributed representations
- Semantic and syntactic relationships
- Embedding space geometry
- Context windows

### Lab 3: Sequence Models and Attention
**File:** `3_lab3.ipynb`

Advanced sequence modeling with attention mechanisms.

**Topics:**
- Sequence-to-sequence (Seq2Seq) models
- Encoder-decoder architecture
- Attention mechanisms (Bahdanau, Luong)
- Self-attention fundamentals
- Multi-head attention
- Positional encoding
- Bidirectional models
- Beam search decoding
- Practical example: Neural machine translation

**Key Concepts:**
- Alignment in translation
- Query, Key, Value paradigm
- Attention weights visualization
- Context vectors

### Lab 4: Transformers and Modern NLP
**File:** `4_lab4.ipynb`

State-of-the-art transformer models and applications.

**Topics:**
- Transformer architecture deep dive
- BERT (Bidirectional Encoder Representations)
- GPT (Generative Pre-trained Transformer)
- T5, RoBERTa, and variants
- Transfer learning in NLP
- Fine-tuning strategies
- Prompt engineering
- Using Hugging Face Transformers
- Model evaluation (BLEU, ROUGE, perplexity)
- Practical example: Text classification, Q&A, summarization

**Key Concepts:**
- Pre-training and fine-tuning
- Masked language modeling
- Causal language modeling
- Zero-shot and few-shot learning

## Interactive Application

**File:** `nlp_app.py`

A comprehensive Gradio application for NLP experimentation:

1. **Text Analyzer**: Tokenization, POS tagging, NER visualization
2. **Sentiment Analysis**: Train and test sentiment classifiers
3. **Text Generation**: Generate text with different models and temperatures
4. **Transformer Playground**: Fine-tune and use pre-trained models

Run with:
```bash
python nlp_app.py
```

## Key Concepts Summary

- **Token**: Basic unit of text (word, subword, character)
- **Vocabulary**: Set of all unique tokens
- **Embedding**: Dense vector representation of tokens
- **Context Window**: Surrounding tokens used for prediction
- **Attention**: Mechanism to focus on relevant parts of input
- **Transformer**: Architecture using self-attention exclusively
- **Pre-training**: Learning general language understanding
- **Fine-tuning**: Adapting pre-trained models to specific tasks
- **Tokenizer**: Converts text to tokens (and vice versa)
- **Language Model**: Predicts probability of sequences
- **Perplexity**: Measure of language model quality
- **BLEU**: Translation quality metric
- **ROUGE**: Summarization quality metric

## Real-World Applications

Modern NLP powers countless applications:

- **Search Engines**: Google, Bing query understanding and ranking
- **Virtual Assistants**: Siri, Alexa, Google Assistant
- **Machine Translation**: Google Translate, DeepL
- **Content Moderation**: Detecting harmful content
- **Sentiment Analysis**: Brand monitoring, customer feedback
- **Question Answering**: Customer support, information retrieval
- **Text Summarization**: News aggregation, document summary
- **Text Generation**: Creative writing, code generation (GitHub Copilot)
- **Named Entity Recognition**: Information extraction, knowledge graphs
- **Speech Recognition**: Voice-to-text systems
- **Autocomplete**: Email, search suggestions
- **Grammar Checking**: Grammarly, MS Word
- **Chatbots**: Customer service, conversational AI

## Installation

Install required packages:

```bash
# Core NLP libraries
pip install nltk spacy

# Download NLTK data
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger

# Download spaCy model
python -m spacy download en_core_web_sm

# Transformers and tokenizers
pip install transformers tokenizers datasets

# Other dependencies
pip install numpy pandas matplotlib seaborn scikit-learn gradio torch
```

## Popular NLP Libraries

### NLTK (Natural Language Toolkit)
- Comprehensive NLP library
- Great for learning and education
- Traditional NLP tools
- Good documentation

### spaCy
- Industrial-strength NLP
- Fast and efficient
- Pre-trained models
- Production-ready

### Hugging Face Transformers
- State-of-the-art models
- Easy-to-use API
- Model hub with thousands of models
- Active community

### Gensim
- Topic modeling
- Word embeddings
- Document similarity

## Tips for Success

1. **Start with Clean Data**: Text preprocessing is crucial
2. **Understand Tokenization**: Different tasks need different tokenizers
3. **Use Pre-trained Models**: Don't train from scratch unless necessary
4. **Fine-tune Carefully**: Small learning rates, monitor validation
5. **Evaluate Properly**: Use appropriate metrics for your task
6. **Handle OOV**: Out-of-vocabulary words need special handling
7. **Consider Context**: Language is context-dependent
8. **Experiment with Prompts**: For generative models, prompt engineering matters
9. **Monitor for Bias**: Language models can perpetuate biases
10. **Use Version Control**: Track model versions and hyperparameters

## Transformer Model Timeline

- **2017**: Transformer introduced ("Attention is All You Need")
- **2018**: BERT (Bidirectional understanding)
- **2018**: GPT (Generative pre-training)
- **2019**: GPT-2 (Larger scale generation)
- **2019**: RoBERTa, ALBERT (BERT improvements)
- **2019**: T5 (Text-to-Text Transfer Transformer)
- **2020**: GPT-3 (175B parameters, few-shot learning)
- **2021**: CLIP, DALL-E (Multimodal models)
- **2022**: ChatGPT (Conversational AI)
- **2023**: GPT-4, LLaMA (More capable models)

## Task Selection Guide

**Use Traditional Methods (BoW, TF-IDF) when:**
- You have limited data
- You need interpretability
- Computational resources are limited
- Baseline performance is sufficient

**Use Word Embeddings (Word2Vec, GloVe) when:**
- You need semantic similarity
- You have medium-sized datasets
- You want better than BoW performance
- You need word analogies

**Use RNNs/LSTMs when:**
- You need to model sequences
- Order matters critically
- You have moderate computational resources
- You want to understand sequential dependencies

**Use Transformers when:**
- You want state-of-the-art performance
- You can use pre-trained models
- You have sufficient computational resources
- You need to capture long-range dependencies

## Common NLP Tasks

### Classification Tasks
- Sentiment analysis
- Topic classification
- Spam detection
- Intent recognition

### Sequence Labeling
- Named Entity Recognition (NER)
- Part-of-speech tagging
- Chunking

### Generation Tasks
- Text generation
- Machine translation
- Summarization
- Dialogue generation

### Understanding Tasks
- Question answering
- Reading comprehension
- Natural language inference
- Semantic similarity

## Evaluation Metrics

**Classification:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix

**Language Models:**
- Perplexity (lower is better)
- Cross-entropy loss

**Translation:**
- BLEU (Bilingual Evaluation Understudy)
- METEOR
- TER (Translation Error Rate)

**Summarization:**
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- ROUGE-1, ROUGE-2, ROUGE-L

**Generation:**
- Human evaluation
- Diversity metrics
- Fluency and coherence

## Common Challenges

| Challenge | Description | Solutions |
|-----------|-------------|-----------|
| **Data Sparsity** | Not enough training data | Transfer learning, data augmentation |
| **OOV Words** | Unknown words at test time | Subword tokenization, character models |
| **Long Sequences** | Handling long documents | Hierarchical models, efficient attention |
| **Ambiguity** | Multiple meanings | Context modeling, disambiguation |
| **Bias** | Model perpetuates biases | Bias detection, fair datasets |
| **Evaluation** | Hard to evaluate generation | Multiple metrics, human evaluation |
| **Computational Cost** | Large models expensive | Model distillation, efficient architectures |

## Best Practices

### Data Preprocessing
1. Lowercase text (task-dependent)
2. Remove or handle special characters
3. Tokenization appropriate for task
4. Handle contractions consistently
5. Remove or mark URLs, emails
6. Consider domain-specific preprocessing

### Model Training
1. Start with pre-trained models
2. Use appropriate tokenizer
3. Monitor validation metrics
4. Use learning rate scheduling
5. Apply gradient clipping
6. Save checkpoints regularly
7. Use mixed precision training for speed

### Model Deployment
1. Optimize model size (quantization, pruning)
2. Batch predictions when possible
3. Cache common queries
4. Monitor inference time
5. Handle edge cases gracefully
6. Log predictions for analysis

## Ethical Considerations

Language models raise important ethical questions:

- **Bias**: Models reflect biases in training data
- **Misinformation**: Can generate false information
- **Privacy**: May memorize training data
- **Environmental Impact**: Large models consume significant energy
- **Accessibility**: Not all languages are well-supported
- **Transparency**: Model decisions can be opaque

**Responsible AI Practices:**
- Audit for bias regularly
- Provide uncertainty estimates
- Document limitations clearly
- Use diverse training data
- Consider model size vs. performance trade-offs
- Respect user privacy

## Advanced Topics

Beyond this course:
- **Multimodal Models**: CLIP, DALL-E (text + images)
- **Reinforcement Learning from Human Feedback (RLHF)**
- **Prompt Engineering**: Optimizing prompts for LLMs
- **Chain-of-Thought Reasoning**
- **Retrieval-Augmented Generation (RAG)**
- **Efficient Transformers**: Linear attention, sparse attention
- **Multilingual Models**: mBERT, XLM-R
- **Domain Adaptation**: Adapting models to specific domains
- **Model Compression**: Distillation, quantization, pruning

## Resources

### Documentation
- **Hugging Face**: https://huggingface.co/docs
- **NLTK**: https://www.nltk.org/
- **spaCy**: https://spacy.io/
- **PyTorch**: https://pytorch.org/

### Courses
- **Stanford CS224N**: Natural Language Processing with Deep Learning
- **Fast.ai NLP**: Practical NLP
- **Hugging Face Course**: Free transformer course

### Books
- *Speech and Language Processing* by Jurafsky & Martin
- *Natural Language Processing with Transformers* by Tunstall et al.
- *Deep Learning for NLP* by Palash Goyal et al.

### Papers (Must-Reads)
- "Attention is All You Need" (Transformers)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "Language Models are Few-Shot Learners" (GPT-3)
- "Efficient Estimation of Word Representations" (Word2Vec)
- "GloVe: Global Vectors for Word Representation"

### Websites
- **Papers with Code NLP**: Track SOTA results
- **Hugging Face Model Hub**: Thousands of pre-trained models
- **ArXiv NLP**: Latest research papers
- **ACL Anthology**: NLP conference papers

## Community Contributions

Share your NLP projects in the `community/` folder:
- Custom tokenizers
- Domain-specific models
- Novel applications
- Dataset contributions

## Course Completion

Congratulations on completing the Introduction to AI course! You've covered:

- **Week 1**: Search algorithms and heuristics
- **Week 2**: Knowledge representation and logic
- **Week 3**: Uncertainty and probabilistic reasoning
- **Week 4**: Optimization and constraint satisfaction
- **Week 5**: Machine learning fundamentals
- **Week 6**: Neural networks and deep learning
- **Week 7**: Natural language processing and transformers

You're now equipped with a comprehensive understanding of AI fundamentals and modern techniques!

## Next Steps After This Course

1. **Specialize**: Choose an area (CV, NLP, RL) and go deeper
2. **Build Projects**: Apply what you learned to real problems
3. **Contribute**: Open-source projects, Kaggle competitions
4. **Stay Current**: Follow research, read papers
5. **Network**: Join AI communities, attend conferences
6. **Advanced Courses**: Specialized deep dives
7. **Research**: Consider graduate studies or research roles

## Career Paths

With these skills, you can pursue:
- Machine Learning Engineer
- Data Scientist
- AI Research Scientist
- NLP Engineer
- Computer Vision Engineer
- AI Product Manager
- ML Infrastructure Engineer
- Applied AI Scientist

Good luck on your AI journey! 🚀
