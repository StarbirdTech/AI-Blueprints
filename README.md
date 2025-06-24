<h1 style="text-align: center; font-size: 45px;"> AI Blueprint Projects for HP AI Studio 🚀 </h1>

# Content  
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Science](#data-science)
- [Deep Learning](#deep-learning)
- [Generative AI](#generative-ai)
- [NVIDIA GPU Cloud (NGC) Integration](#nvidia-gpu-cloud-integration)
- [Contact and Support](#contact-and-support)

---

# Overview 

This repository contains a collection of sample projects that you can run quickly and effortlessly, designed to integrate seamlessly with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html). Each project runs end-to-end, offering out-of-the-box, ready-to-use solutions across various domains, including data science, machine learning, deep learning, and generative AI.  

The projects leverage local open-source models such as **LLaMA** (Meta), **BERT** (Google), and **CitriNet** (NVIDIA), alongside selected online models accessible via **Hugging Face**. These examples cover a wide range of use cases, including **data visualization**, **stock analysis**, **audio translation**, **agentic RAG applications**, and much more.  

We are continuously expanding this collection with new projects. If you have suggestions or would like to see a specific sample project integrated with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html), please feel free to open a new issue in this repository — we welcome your feedback!

---

# Repository Structure 

- data-science
  - classification-with-svm
  - data-analysis-with-var
- deep-learning
  - classification-with-keras
  - question-answering-with-bert
  - recommendation-system-with-tensorflow
  - spam-detection-with-nlp
  - super-resolution-with-fsrcnn
  - text-generation-with-rnn
- generative-ai
  - automated-evaluation-with-structured-outputs
  - code-generation-with-langchain
  - fine-tuning-with-orpo
  - image-generation-with-stablediffusion
  - text-generation-with-langchain
  - text-summarization-with-langchain
  - vanilla-rag-with-langchain
- ngc-integration
  - agentic-rag-with-tensorrtllm
  - audio-translation-with-nemo
  - data-analysis-with-cudf
  - data-visualization-with-cudf
  - vacation-recommendation-with-bert

---

# Data Science

The sample projects in this folder demonstrate how to build data science applications with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provide **2 blueprint projects**, each designed for quick and easy use to help you get started efficiently.

### 🌸 Classification with SVM

This project is a simple **classification** experiment focused on predicting species of **Iris flowers**.  

It runs on the **Data Science Workspace**, demonstrating basic supervised learning techniques for multi-class classification tasks.

### 🏙️ Data Analysis with VAR

This project explores a **regression** experiment using **mobility data** collected during the COVID-19 pandemic.  

It highlights how city-level movement patterns changed during the crisis. The experiment runs on the **Data Science Workspace**.

---

# Deep Learning

The sample projects in this folder demonstrate how to build deep learning applications with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provide **6 blueprint projects**, each designed for quick and easy use to help you get started efficiently.


### 🖌️ Classification with Keras

This project performs basic **image classification** using the **TensorFlow** framework.  

It trains a model to classify handwritten digits from the **MNIST** dataset and runs on the **Deep Learning Workspace**.


### 🧠 Question Answering with BERT

This project demonstrates a simple **BERT Question Answering (QA)** experiment. It provides code to train a BERT-based model, as well as instructions to load a pretrained model from **Hugging Face**.  

The model is deployed using **MLflow** to expose an inference service capable of answering questions based on input text.


### 🎬 Recommendation System with TensorFlow

This project builds a simple **recommender system** for movies using **TensorFlow**.  

It trains on user-item interaction data to predict movie preferences and runs on the **Deep Learning Workspace**.


### 🚫 Spam Detection with NLP

This project implements a **text classification** system to detect **spam** messages.  

It uses deep learning techniques and requires the **Deep Learning Workspace** for training and inference.


### 🖼️ Super Resolution with FSRCNN

This project showcases a **Computer Vision** experiment that applies convolutional neural networks for **image super-resolution** — enhancing the quality and resolution of input images.  


### ✍️ Text Generation with RNN

This project illustrates how to build a simple **character-by-character text generation** model.  

It trains on a dataset containing **Shakespeare's texts**, demonstrating the fundamentals of text generation by predicting one character at a time.

---

# Generative AI

The sample projects in this folder demonstrate how to build generative AI applications with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provide **7 blueprint projects**, each designed for quick and easy use to help you get started efficiently.


### 📊 Automated Evaluation with Structured Outputs

**Automated Evaluation with Structured Outputs** turns a local **Meta‑Llama‑2** model into an MLflow‑served scorer that rates any batch of texts (e.g., project abstracts) against arbitrary rubric criteria.

* Generates scores locally via `llama.cpp` (no data leaves your machine)
* Registers the evaluator as a **pyfunc** model in MLflow
* Exposes a REST `/invocations` endpoint
* Ships two front‑ends — a **Streamlit** dashboard and a pure **HTML/JS** UI — for instant human‑friendly interaction and CSV download.


### Code Generation with Langchain

This notebook performs automatic code explanation by extracting code snippets from Jupyter notebooks and generating natural language descriptions using LLMs. It supports contextual enrichment based on adjacent markdown cells, enables configurable prompt templating, and integrates with PromptQuality and Galileo for evaluation and tracking. The pipeline is modular, supports local or hosted model inference, and is compatible with LLaMA, Mistral, and Hugging Face-based models. It also includes GitHub notebook crawling, metadata structuring, and vector store integration for downstream tasks like RAG and semantic search.


### Fine Tuning with ORPO

This project demonstrates a full-stack LLM fine-tuning experiment using ORPO (Open-Source Reinforcement Pretraining Objective) to align a base language model with human preference data. It leverages the Z by HP AI Studio Local GenAI environment, and uses models such as LLaMA 3, Gemma 1B, and Mistral 7B as foundations.

We incorporate:

Galileo PromptQuality for evaluating model responses with human-like scorers (e.g., context adherence)
TensorBoard for human feedback visualization before fine-tuning
A flexible model selector and inference runner architecture
A comparative setup to benchmark base vs fine-tuned models on the same prompts


### Image Generation with Stable Diffusion

This notebook performs image generation inference using the Stable Diffusion architecture, with support for both standard and DreamBooth fine-tuned models. It loads configuration and secrets from YAML files, enables local or deployed inference execution, and calculates custom image quality metrics such as entropy and complexity. The pipeline is modular, supports Hugging Face model loading, and integrates with PromptQuality for evaluation and tracking.


### Text Generation with LangChain

This notebook implements a full Retrieval-Augmented Generation (RAG) pipeline for automatically generating a scientific presentation script. It integrates paper retrieval from arXiv, text extraction and chunking, embedding generation with HuggingFace, vector storage with ChromaDB, and context-aware generation using LLMs. It also integrates Galileo Prompt Quality for evaluation and logging, and supports multi-source model loading including local Llama.cpp, HuggingFace-hosted, and HuggingFace-cloud models like Mistral or DeepSeek.


### Text Summarization with LangChain

This project demonstrates how to build a semantic chunking and summarization pipeline for texts using LangChain, Sentence Transformers, and Galileo for model evaluation, protection, and observability. It leverages the Z by HP AI Studio Local GenAI image and the LLaMA2-7B model to generate concise and contextually accurate summaries from text data.


### Vanilla RAG with LangChain

This project is an AI-powered vanilla RAG (Retrieval-Augmented Generation) chatbot built using LangChain and Galileo for model evaluation, protection, and observability. It leverages the Z by HP AI Studio Local GenAI image and the LLaMA2-7B model to generate contextual and document-grounded answers to user queries about Z by HP AI Studio.


---

# NVIDIA GPU Cloud Integration

The sample projects in this folder demonstrate how to integrate **NVIDIA NGC (NVIDIA GPU Cloud)** resources with [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).

We provide **5 blueprint projects**, each designed for quick and easy use to help you get started efficiently.

### 🤖 Agentic RAG with TensorRT-LLM

This project contains a single integrated pipeline—Agentic RAG for AI Studio with TRT-LLM and LangGraph—that implements a Retrieval-Augmented Generation (RAG) workflow using:

TensorRT-backed Llama-3.1-Nano (TRT-LLM): for fast, GPU-accelerated inference.
LangGraph: to orchestrate an agentic, multi-step decision flow (relevance check, memory lookup, query rewriting, retrieval, answer generation, and memory update).
ChromaDB: as a local vector store over Markdown context files (about AI Studio).
SimpleKVMemory: a lightweight on-disk key-value store to cache query-answer pairs.


### 🎙️ Audio Translation with NeMo

This project demonstrates an end-to-end **audio translation pipeline** using **NVIDIA NeMo models**. It takes an English audio sample and performs:

1. **Speech-to-Text (STT)** conversion using Citrinet  
2. **Text Translation (TT)** from English to Spanish using NMT  
3. **Text-to-Speech (TTS)** synthesis in Spanish using FastPitch and HiFiGAN  

All steps are GPU-accelerated, and the full workflow is integrated with **MLflow** for experiment tracking and model registration.


### 📈 Data Analysis with cuDF  

In this project, we provide notebooks to compare the execution time of dataset operations using traditional **Pandas** (CPU) versus **NVIDIA’s cuDF**, a GPU-accelerated drop-in replacement for Pandas. This example is presented in two different formats:

- **Original Example Notebook**: This version, created by NVIDIA, runs the entire evaluation within a single notebook. It includes downloading the data and restarting the kernel to activate the cuDF extension.

- **Data Analysis Notebooks**: These notebooks use preprocessed datasets of varying sizes from **datafabric** folder in AI Studio. The evaluation is split across two notebooks—one using Pandas (CPU) and the other using cuDF (GPU)—with performance metrics logged to **MLflow**.


### 📡 Data Visualization with cuDF  

This project is a GPU-accelerated, interactive **exploratory data analysis (EDA)** dashboard for the [OpenCellID](https://www.opencellid.org/) dataset. It uses **Panel** and **cuDF** to deliver lightning-fast geospatial analysis and visualization.

You can explore cell tower distributions by radio type, operator, country, and time window — rendered live on an interactive map with full GPU acceleration.


### 🌍 Vacation Recommendation with BERT

This project implements an **AI-powered recommendation agent** that delivers personalized travel suggestions based on user queries. 

It leverages the **NVIDIA NeMo Framework** and **BERT embeddings** to understand user intent and generate highly relevant, tailored vacation recommendations.

---

# Contact and Support  

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://www.hp.com/us-en/workstations/ai-studio.html).
