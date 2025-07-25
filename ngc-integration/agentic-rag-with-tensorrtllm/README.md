# 🤖 Agentic RAG for AI Studio with TRT-LLM and LangGraph

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-supported-orange.svg?logo=jupyter)
![TensorRT](https://img.shields.io/badge/TensorRT-optimized-green.svg?logo=TensorRT)
![LangChain](https://img.shields.io/badge/LangChain-used-lightgreen.svg?logo=langchain)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-blue.svg)

</div>

# 📚 Contents

- [🧠 Overview](#overview)
- [🗂 Project Structure](#project-structure)
- [⚙️ Setup](#setup)
- [🚀 Usage](#usage)
- [📞 Contact and Support](#contact-and-support)

---

## Overview  
This project contains a single integrated pipeline—**Agentic RAG for AI Studio with TRT-LLM and LangGraph**—that implements a Retrieval-Augmented Generation (RAG) workflow using:

- **TensorRT-backed Llama-3.1-Nano (TRT-LLM)**: for fast, GPU-accelerated inference.
- **LangGraph**: to orchestrate an agentic, multi-step decision flow (relevance check, memory lookup, query rewriting, retrieval, answer generation, and memory update).
- **ChromaDB**: as a local vector store over Markdown context files (about AI Studio).
- **SimpleKVMemory**: a lightweight on-disk key-value store to cache query-answer pairs.

---

## Project Structure
```
agentic-rag-with-tensorrtllm/
├── data/                                             # Data assets used in the project
│   └── context/
│       └── aistudio
├── docs/
|   ├── architecture-for-agentic-rag.png              # Architecture screenshot of the agentic RAG system
|   └── Build Custom Agentic RAG Systems.pptx         # Powerpoint walkthrough slides for building general agentic RAG systems
|   
├── notebooks/
|   ├── register-model.ipynb                          # Notebook for registering trained models to MLflow
│   └── run-workflow.ipynb                            # Notebook for executing the pipeline using custom inputs and configurations
├── src/                                              # Core Python modules
│   ├── __init__.py
│   ├── trt_llm_langchain.py
|   └── workspace.sh
├── README.md                                         # Project documentation
└── requirements.txt                                  # Python dependencies
```  

---

## Setup  

### Step 0: Minimum Hardware Requirements
To ensure smooth execution and reliable model deployment, make sure your system meets the following minimum hardware specifications:

- GPU: NVIDIA GPU with at least 32 GB VRAM (for TensorRT-LLM engine)

- RAM: ≥ 64 GB system memory

- Disk: ≥ 32 GB free

- CUDA: Compatible CUDA toolkit (11.8 or 12.x) installed on your system

### Step 1: Create an AI Studio Project  
1. Create a **New Project** in AI Studio.   

### Step 2: Create a Workspace  
1. Select **NeMo Framework (version 25.04)** as the base image.    
2. To use this specific image version in AI Studio, add the following two lines to your `config.yaml` file located in `C:\Users\<user-name>\AppData\Local\HP\AIStudio` on Windows (or the corresponding directory on Ubuntu):
   
   ```
   ngcconfig:
	   nemoversionpin: "25.04"
   ```
3. To use this specific image version with all necessary root user permissions in AI Studio and avoid errors when running the notebook, replace the existing `workspace.sh` file in your AI Studio app with the one provided in the `src/` folder.

- On **Windows**, the file is located at:  
  `C:\Program Files\HP\AIStudio\util\container-setup\workspace.sh`

- On **Ubuntu**, replace the corresponding `workspace.sh` file in the equivalent directory.

   
### Step 3: Verify Project Files  
1. Clone the GitHub repository:
   
   ```
   git clone https://github.com/HPInc/AI-Blueprints.git
   ```  
3. Navigate to `ngc-integration/agentic-rag-with-tensorrtllm` to ensure all files are cloned correctly after workspace creation.  

---



## Usage 

### Step 1: Run the Agentic RAG Workflow

Execute the following notebook located in the `notebooks/` folder to see the Agentic RAG workflow in action:  
- **`run-workflow.ipynb`**

### Step 2: Register the Agentic RAG Model in MLflow

Run the following notebook in the `notebooks/` folder to register the Agentic RAG model in MLflow:  
- **`register-model.ipynb`**

### Step 3: Deploy the Agentic RAG Service Locally

Currently, deploying this service locally in AI Studio is not possible due to limitations in the version of the NeMo framework image used in this blueprint. We are actively working on resolving this issue.



---



## Contact and Support

- Issues: Open a new issue in our [**AI-Blueprints GitHub repo**](https://github.com/HPInc/AI-Blueprints).

- Docs: Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting. 

- Community: Join the [**HP AI Creator Community**](https://community.datascience.hp.com/) for questions and help.

---

> Built with ❤️ using [**HP AI Studio**](https://hp.com/ai-studio).
