# Self-Evolving Agentic Framework

This project is a Self-Evolving Agentic Framework that enables autonomous adaptation, learning, and optimization of AI agents. It integrates four intelligent subsystems that run in parallel yet synergistically:

- **Auto-RL:** Automatically trains and optimizes policies for agentic behavior.
- **Dynamic Context Engineering:** Dynamically manages memory and retrieval for LLM-based agents.
- **Auto-Fine-Tuning:** Detects model drift and retrains models adaptively using LoRA/QLoRA.
- **Auto-Quantization:** Compresses and optimizes models for efficient deployment.

The framework is self-improving, continuously cycling through learning, fine-tuning, compression, and deployment.

## Features

- **Automated Reinforcement Learning:** Automates RL agent experimentation, tuning, and evaluation using Ray/RLlib.
- **Dynamic Memory and Context:** Enables memory-aware LLM prompting and retrieval, inspired by MemGPT.
- **Automatic Fine-Tuning:** Automatically fine-tunes models using LoRA in response to data drift.
- **Automatic Quantization:** Reduces model size and latency using GPTQ while maintaining accuracy.
- **Continuous Self-Evolution:** A continuous feedback loop integrates all modules with unified orchestration and evaluation.
- **MLflow Integration:** Tracks experiments, logs metrics, and registers models.

## Architecture

The framework is composed of four main modules orchestrated by a Main Orchestrator:

- **Auto-RL (`auto_rl`):** Handles RL agent training and policy optimization.
- **Dynamic Context Engineering (`dynamic_context`):** Manages memory and context for the agent.
- **Auto-Fine-Tuning (`auto_fine_tuning`):** Fine-tunes models when data drift is detected.
- **Auto-Quantization (`auto_quantization`):** Quantizes models for efficient deployment.
- **Main Orchestrator (`orchestrator`):** Coordinates the execution of all modules.

For a more detailed explanation of the architecture, please refer to the [SYSTEM_MANUAL.md](SYSTEM_MANUAL.md).

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Framework Locally

1.  **Start the infrastructure services:**

    ```bash
    docker-compose up -d
    ```

    This will start the following services:
    - **MLflow:** For experiment tracking and model registry (UI at `http://localhost:5000`).
    - **MinIO:** For artifact storage (Console at `http://localhost:9001`).
    - **PostgreSQL:** As the backend store for MLflow.

2.  **Run the end-to-end cycle:**

    To run the full end-to-end cycle, execute the main orchestrator:

    ```bash
    python orchestrator/main_orchestrator.py
    ```

    This will run the complete pipeline, including fine-tuning, quantization, and logging to MLflow. You can monitor the progress and view the results in the MLflow UI.

3.  **Run individual modules:**

    You can also run each module individually for testing and debugging purposes:

    - **Auto-RL:**
      ```bash
      python auto_rl/orchestrator.py
      ```

    - **Dynamic Context Engineering:**
      ```bash
      python dynamic_context/rag_baseline.py
      ```

    - **Auto-Fine-Tuning:**
      ```bash
      python auto_fine_tuning/lora_finetuner.py
      ```

    - **Auto-Quantization:**
      ```bash
      python auto_quantization/gptq_quantizer.py
      ```

4.  **Run regression tests:**

    To run the regression tests for all modules, execute the test script:

    ```bash
    python tests/test_modules.py
    ```

## Documentation

- **Deployment Guide:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **System Manual:** [SYSTEM_MANUAL.md](SYSTEM_MANUAL.md)
- **Performance Report:** [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)
- **Evaluation Metrics:** [EVALUATION.md](EVALUATION.md)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
