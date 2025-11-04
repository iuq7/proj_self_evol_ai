# System Manual

This manual provides a detailed explanation of the Self-Evolving Agentic Framework's architecture, modules, and how they interact.

## 1. System Architecture Overview

The framework is composed of four main modules that work together to enable the autonomous adaptation, learning, and optimization of AI agents:

- **Auto-RL:** For automated reinforcement learning and policy evolution.
- **Dynamic Context Engineering:** For managing memory and context for LLM-based agents.
- **Auto-Fine-Tuning:** For automatically fine-tuning models based on data drift.
- **Auto-Quantization:** For compressing and optimizing models for efficient deployment.

These modules are orchestrated by a **Main Orchestrator** that coordinates their execution in a continuous cycle.

## 2. Modules

### 2.1. Auto-RL Module (`auto_rl`)

- **Purpose:** To automate the process of training and tuning RL agents.
- **Key Components:**
    - `orchestrator.py`: The main entry point for the Auto-RL module. It uses Ray Tune for hyperparameter search and RLlib for training.
- **Functionality:**
    - **Hyperparameter Search:** Uses Ray Tune with HyperOpt to find the optimal hyperparameters for the RL agent.
    - **Training:** Uses RLlib to train the agent in the specified environment.
    - **Evaluation:** Evaluates the trained agent based on the mean episode reward.
    - **Policy Promotion:** Promotes the policy to the model registry if it meets the predefined criteria.

### 2.2. Dynamic Context Engineering Module (`dynamic_context`)

- **Purpose:** To provide a memory-aware context for LLM-based agents.
- **Key Components:**
    - `rag_baseline.py`: The main entry point for the Dynamic Context Engineering module. It uses a RAG-like architecture for retrieval and generation.
    - `SentenceTransformerRetriever`: A custom retriever that uses sentence-transformers to find relevant documents.
    - `Reranker`: A cross-encoder based reranker to improve the relevance of retrieved documents.
    - `MemoryPolicyEngine`: A MemGPT-style memory engine to manage short-term and long-term memory.
- **Functionality:**
    - **Retrieval:** Retrieves relevant documents from a knowledge base using a sentence-transformer model.
    - **Reranking:** Reranks the retrieved documents using a cross-encoder model.
    - **Memory Management:** Manages short-term and long-term memory to provide context for the agent.
    - **Generation:** Generates a response based on the retrieved context and the user's query.

### 2.3. Auto-Fine-Tuning Module (`auto_fine_tuning`)

- **Purpose:** To automatically fine-tune models based on data drift.
- **Key Components:**
    - `drift_detector.py`: Detects data drift by training a classifier to distinguish between reference and new data.
    - `lora_finetuner.py`: Fine-tunes a model using LoRA (Low-Rank Adaptation).
- **Functionality:**
    - **Drift Detection:** Detects drift in the data distribution.
    - **Fine-Tuning:** Fine-tunes a model using LoRA to adapt to the new data distribution.
    - **Evaluation:** Evaluates the fine-tuned model.
    - **Canary Deployment:** Performs a canary deployment of the fine-tuned model.

### 2.4. Auto-Quantization Module (`auto_quantization`)

- **Purpose:** To compress and optimize models for efficient deployment.
- **Key Components:**
    - `gptq_quantizer.py`: Quantizes a model using GPTQ (Generalized Post-Training Quantization).
    - `model_analyzer.py`: Analyzes the sensitivity of each layer to quantization.
- **Functionality:**
    - **Quantization:** Quantizes a model to a lower bit precision (e.g., 4-bit, 8-bit).
    - **ONNX Export:** Exports the quantized model to ONNX format for faster inference.
    - **Latency Benchmarking:** Benchmarks the latency of the quantized model.

## 3. Main Orchestrator (`orchestrator`)

- **Purpose:** To coordinate the execution of all modules in a continuous cycle.
- **Key Components:**
    - `main_orchestrator.py`: The main entry point for the entire framework.
- **Functionality:**
    - **MLflow Integration:** Integrates with MLflow for experiment tracking and model registry.
    - **End-to-End Cycle:** Runs the complete end-to-end cycle, including fine-tuning, quantization, and deployment.

## 4. How to Extend the Framework

The framework is designed to be extensible. Here are some ways you can extend it:

- **Add new RL environments:** You can add new environments to the `auto_rl` module by modifying the `orchestrator.py` file.
- **Use different language models:** You can use different language models in the `dynamic_context` module by modifying the `rag_baseline.py` file.
- **Implement different fine-tuning techniques:** You can implement different fine-tuning techniques in the `auto_fine_tuning` module.
- **Use different quantization algorithms:** You can use different quantization algorithms in the `auto_quantization` module.
