# Performance Report

This report summarizes the expected performance metrics for the Self-Evolving Agentic Framework. The actual metrics can be retrieved from MLflow.

## 1. Auto-RL Module

- **Metric:** `episode_reward_mean`
- **Expected Outcome:** The `episode_reward_mean` should show a positive trend over training iterations, indicating that the agent is learning effectively. The final mean reward should be above a predefined threshold (e.g., 150 for CartPole-v0).
- **MLflow Tag:** `auto_rl_status` (will be `simulated_run` for now, but will be `completed` with actual runs)

## 2. Dynamic Context Engineering Module

- **Metric:** Retrieval Accuracy, Latency
- **Expected Outcome:**
    - **Retrieval Accuracy:** The reranker and improved embedding model should lead to higher precision and recall of relevant documents.
    - **Latency:** The retrieval and generation process should be efficient, with latency below a certain threshold (e.g., < 150ms).
- **MLflow Tags:** `rag_query`, `rag_response`

## 3. Auto-Fine-Tuning Module

- **Metric:** Model Drift, Accuracy Improvement
- **Expected Outcome:**
    - **Model Drift:** The drift detector should accurately identify data drift, triggering fine-tuning.
    - **Accuracy Improvement:** Fine-tuning with LoRA should lead to a measurable improvement in model accuracy on new data distributions.
- **MLflow Tag:** `fine_tuning_status` (will be `simulated_run` for now, but will be `completed` with actual runs)

## 4. Auto-Quantization Module

- **Metric:** Model Size, Latency, Accuracy
- **Expected Outcome:**
    - **Model Size:** Significant reduction in model size (e.g., 4-bit or 8-bit quantization).
    - **Latency:** Reduced inference latency, especially with ONNX Runtime.
    - **Accuracy:** Minimal degradation in accuracy after quantization (within 1-2% of the original model).
- **MLflow Tag:** `quantization_status` (will be `simulated_run` for now, but will be `completed` with actual runs)

## How to Retrieve Actual Metrics from MLflow

1.  **Start MLflow UI:** Ensure your MLflow UI is running by executing `docker-compose up -d` in your project root.
2.  **Access UI:** Open your web browser and navigate to `http://localhost:5000`.
3.  **Explore Experiments:** In the MLflow UI, you can browse through your experiments, view runs, and inspect logged parameters, metrics, and artifacts.

    - **Parameters:** Look for logged parameters like `lr`, `gamma`, `rag_query`, `rag_response`, `drift_detected`, `fine_tuning_status`, `quantization_status`, etc.
    - **Metrics:** If actual training and evaluation runs are performed, you will see metrics like `episode_reward_mean`, `accuracy`, `latency`, etc.
    - **Artifacts:** You can download logged artifacts such as `lora_model_integrated` and `quantized_model_integrated`.

This report serves as a template. The actual values will depend on the execution of the full pipeline and the specific configurations used.
