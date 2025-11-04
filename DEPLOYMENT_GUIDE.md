# Deployment Guide

This guide provides instructions on how to deploy the Self-Evolving Agentic Framework.

## 1. Prerequisites

- **Docker and Docker Compose:** Required for running the infrastructure services (MLflow, MinIO).
- **Python 3.8+:** Required for running the application code.
- **Git:** Required for version control.

## 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 3. Infrastructure Setup

1.  **Start the infrastructure services:**
    ```bash
    docker-compose up -d
    ```
    This will start the following services:
    - **MLflow:** For experiment tracking and model registry (UI at `http://localhost:5000`).
    - **MinIO:** For artifact storage (Console at `http://localhost:9001`).
    - **PostgreSQL:** As the backend store for MLflow.

## 4. Running the Framework

The framework can be run in two modes: **end-to-end cycle** and **individual module execution**.

### 4.1. End-to-End Cycle

To run the full end-to-end cycle, execute the main orchestrator:

```bash
python orchestrator/main_orchestrator.py
```

This will run the complete pipeline, including fine-tuning, quantization, and logging to MLflow.

### 4.2. Individual Module Execution

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

## 5. Deployment to Production (Placeholder)

This section provides a high-level overview of how to deploy the framework to a production environment. The actual steps will depend on your specific infrastructure and requirements.

1.  **Containerize the application:** Create a Dockerfile for the main application to package it as a container image.
2.  **Set up a production-grade infrastructure:**
    - **Kubernetes:** For container orchestration and scaling.
    - **Cloud Storage:** For storing artifacts (e.g., AWS S3, Google Cloud Storage).
    - **Managed Database:** For the MLflow backend (e.g., AWS RDS, Google Cloud SQL).
3.  **CI/CD Pipeline:** Set up a CI/CD pipeline (e.g., using GitHub Actions, Jenkins) to automate the building, testing, and deployment of the application.
4.  **Monitoring and Logging:** Set up monitoring and logging tools (e.g., Prometheus, Grafana, ELK stack) to monitor the health and performance of the deployed application.
